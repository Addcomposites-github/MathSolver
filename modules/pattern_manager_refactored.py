"""
Enhanced Pattern Manager for Refactored Trajectory System

This module provides intelligent pattern selection and optimization for COPV filament winding,
integrating seamlessly with the refactored trajectory planning architecture.
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple
from .trajectories_refactored import TrajectoryPlannerRefactored
from .geometry import VesselGeometry


class PatternManagerRefactored:
    """
    Enhanced pattern manager that works with the refactored trajectory system.
    Provides intelligent pattern selection, optimization, and coverage analysis.
    """
    
    def __init__(self, vessel_geometry: VesselGeometry):
        """
        Initialize pattern manager with vessel geometry.
        
        Parameters:
        -----------
        vessel_geometry : VesselGeometry
            Vessel geometry for pattern optimization
        """
        self.vessel = vessel_geometry
        self.R_equator_m = self.vessel.inner_radius * 1e-3  # Convert mm to m
        
    def analyze_pattern_options(self, 
                              roving_width_mm: float = 3.0,
                              target_angle_deg: Optional[float] = None) -> Dict:
        """
        Analyze available pattern options for the given parameters.
        
        Returns comprehensive analysis of pattern feasibility and recommendations.
        """
        # Create temporary planner for analysis
        planner = TrajectoryPlannerRefactored(
            vessel_geometry=self.vessel,
            dry_roving_width_m=roving_width_mm * 1e-3,
            target_cylinder_angle_deg=target_angle_deg
        )
        
        # Get basic validation
        validation = planner.get_validation_results()
        
        # Calculate pattern options
        pattern_options = {
            'geodesic_spiral': self._analyze_geodesic_options(planner, validation),
            'non_geodesic_spiral': self._analyze_non_geodesic_options(planner, validation),
            'multi_circuit': self._analyze_multi_circuit_options(planner, validation)
        }
        
        # Determine recommendations
        recommendations = self._generate_recommendations(pattern_options, validation)
        
        return {
            'validation': validation,
            'pattern_options': pattern_options,
            'recommendations': recommendations,
            'vessel_parameters': {
                'equatorial_radius_mm': self.vessel.inner_radius,
                'polar_opening_mm': validation.get('effective_polar_opening_mm', 0),
                'clairaut_constant_mm': validation.get('clairaut_constant_mm', 0)
            }
        }
        
    def _analyze_geodesic_options(self, planner: TrajectoryPlannerRefactored, validation: Dict) -> Dict:
        """Analyze geodesic pattern options."""
        if not validation['is_valid']:
            return {'feasible': False, 'reason': validation.get('error_message', 'Validation failed')}
            
        # Calculate coverage metrics
        if planner.phi_advancement_rad_per_pass:
            passes_for_coverage = math.ceil(2 * math.pi / planner.phi_advancement_rad_per_pass)
            circuits_for_coverage = math.ceil(passes_for_coverage / 2)
        else:
            passes_for_coverage = 20  # Default estimate
            circuits_for_coverage = 10
        
        return {
            'feasible': True,
            'coverage_options': {
                'single_circuit': {'passes': 2, 'coverage_percent': 100 * planner.phi_advancement_rad_per_pass / math.pi if planner.phi_advancement_rad_per_pass else 5},
                'full_coverage': {'passes': passes_for_coverage, 'circuits': circuits_for_coverage},
                'recommended_circuits': min(circuits_for_coverage, 8)  # Practical limit
            },
            'angle_achievable_deg': math.degrees(math.asin(planner.clairauts_constant_for_path_m / planner.R_cyl_m)) if planner.clairauts_constant_for_path_m and planner.R_cyl_m else 45,
            'safety_margin_mm': validation.get('safety_margin_mm', 0)
        }
        
    def _analyze_non_geodesic_options(self, planner: TrajectoryPlannerRefactored, validation: Dict) -> Dict:
        """Analyze non-geodesic pattern options."""
        return {
            'feasible': True,
            'status': 'experimental',
            'requires_friction': True,
            'recommended_mu': 0.3,
            'note': 'Non-geodesic patterns require friction coefficient and are currently experimental'
        }
        
    def _analyze_multi_circuit_options(self, planner: TrajectoryPlannerRefactored, validation: Dict) -> Dict:
        """Analyze multi-circuit pattern options."""
        if not validation['is_valid']:
            return {'feasible': False, 'reason': validation.get('error_message', 'Validation failed')}
            
        # Multi-circuit uses proven geodesic foundation
        base_analysis = self._analyze_geodesic_options(planner, validation)
        
        if base_analysis['feasible']:
            return {
                'feasible': True,
                'based_on': 'geodesic_spiral',
                'recommended_circuits': [2, 4, 6, 8],
                'coverage_efficiency': base_analysis['coverage_options'],
                'note': 'Multi-circuit patterns use robust geodesic generation with pattern advancement'
            }
        else:
            return base_analysis
            
    def _generate_recommendations(self, pattern_options: Dict, validation: Dict) -> Dict:
        """Generate intelligent pattern recommendations."""
        recommendations = {
            'primary_recommendation': 'geodesic_spiral',
            'coverage_strategy': 'single_circuit',
            'reasoning': []
        }
        
        # Analyze options and provide reasoning
        if pattern_options['geodesic_spiral']['feasible']:
            if validation.get('safety_margin_mm', 0) > 5:
                recommendations['reasoning'].append("✅ Geodesic pattern has good safety margin")
                recommendations['coverage_strategy'] = 'user_defined'
                recommendations['recommended_circuits'] = pattern_options['geodesic_spiral']['coverage_options']['recommended_circuits']
            else:
                recommendations['reasoning'].append("⚠️ Limited safety margin - single circuit recommended")
                
        if pattern_options['multi_circuit']['feasible']:
            recommendations['reasoning'].append("✅ Multi-circuit patterns available for full coverage")
            
        if not pattern_options['geodesic_spiral']['feasible']:
            recommendations['primary_recommendation'] = 'non_geodesic_spiral'
            recommendations['reasoning'].append("🔬 Geodesic not feasible - trying experimental non-geodesic")
            
        return recommendations
        
    def generate_optimal_trajectory(self, 
                                  pattern_preference: str = 'auto',
                                  coverage_preference: str = 'auto',
                                  roving_width_mm: float = 3.0,
                                  target_angle_deg: Optional[float] = None) -> Optional[Dict]:
        """
        Generate optimal trajectory based on preferences and analysis.
        
        Parameters:
        -----------
        pattern_preference : str
            'auto', 'geodesic_spiral', 'non_geodesic_spiral', 'multi_circuit'
        coverage_preference : str  
            'auto', 'single_circuit', 'full_coverage', 'user_defined'
        roving_width_mm : float
            Roving width in millimeters
        target_angle_deg : Optional[float]
            Target winding angle in degrees
            
        Returns:
        --------
        Dict containing optimized trajectory data
        """
        # Analyze options first
        analysis = self.analyze_pattern_options(roving_width_mm, target_angle_deg)
        
        # Determine optimal pattern
        if pattern_preference == 'auto':
            pattern_name = analysis['recommendations']['primary_recommendation']
        else:
            pattern_name = pattern_preference
            
        # Determine optimal coverage
        if coverage_preference == 'auto':
            coverage_option = analysis['recommendations']['coverage_strategy']
        else:
            coverage_option = coverage_preference
            
        # Create optimized planner
        planner = TrajectoryPlannerRefactored(
            vessel_geometry=self.vessel,
            dry_roving_width_m=roving_width_mm * 1e-3,
            target_cylinder_angle_deg=target_angle_deg
        )
        
        # Determine circuit count
        user_circuits = 1
        if coverage_option == 'user_defined':
            user_circuits = analysis['recommendations'].get('recommended_circuits', 2)
        elif coverage_option == 'full_coverage':
            user_circuits = analysis['pattern_options']['geodesic_spiral']['coverage_options'].get('circuits', 4)
            
        # Generate trajectory
        print(f"🎯 Generating optimal trajectory: {pattern_name} with {coverage_option} coverage")
        
        trajectory_result = planner.generate_trajectory(
            pattern_name=pattern_name,
            coverage_option=coverage_option,
            user_circuits=user_circuits
        )
        
        if trajectory_result:
            # Add optimization metadata
            trajectory_result['optimization_analysis'] = analysis
            trajectory_result['selected_pattern'] = pattern_name
            trajectory_result['selected_coverage'] = coverage_option
            trajectory_result['optimization_reasoning'] = analysis['recommendations']['reasoning']
            
        return trajectory_result


class SmartPatternSelector:
    """
    Intelligent pattern selector that provides automatic optimization.
    """
    
    @staticmethod
    def select_best_pattern(vessel_geometry: VesselGeometry, 
                          requirements: Dict) -> Dict:
        """
        Automatically select the best pattern based on requirements.
        
        Parameters:
        -----------
        vessel_geometry : VesselGeometry
            Vessel geometry
        requirements : Dict
            Requirements including target_angle, coverage_type, manufacturing_constraints
            
        Returns:
        --------
        Dict containing selected pattern and reasoning
        """
        manager = PatternManagerRefactored(vessel_geometry)
        
        # Extract requirements
        target_angle = requirements.get('target_angle_deg')
        coverage_type = requirements.get('coverage_type', 'optimal')
        roving_width = requirements.get('roving_width_mm', 3.0)
        
        # Analyze and generate optimal solution
        return manager.generate_optimal_trajectory(
            pattern_preference='auto',
            coverage_preference='auto' if coverage_type == 'optimal' else coverage_type,
            roving_width_mm=roving_width,
            target_angle_deg=target_angle
        )
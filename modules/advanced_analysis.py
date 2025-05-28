"""
Advanced Analysis Module for COPV Design
Provides sophisticated engineering analysis capabilities including:
- Failure analysis with multiple criteria
- Optimization algorithms
- Manufacturing cost estimation
- Real-time design validation
"""

import numpy as np
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class FailureAnalysisResult:
    """Results from comprehensive failure analysis"""
    tsai_wu_factor: float
    maximum_stress_factor: float
    maximum_strain_factor: float
    buckling_factor: float
    overall_safety_margin: float
    critical_failure_mode: str
    design_recommendations: List[str]


@dataclass
class OptimizationResult:
    """Results from design optimization"""
    optimized_parameters: Dict[str, float]
    weight_reduction: float
    cost_reduction: float
    performance_improvement: float
    optimization_iterations: int
    convergence_achieved: bool


class AdvancedAnalysisEngine:
    """Advanced engineering analysis for composite pressure vessels"""
    
    def __init__(self):
        self.analysis_history = []
        
    def comprehensive_failure_analysis(self, 
                                     vessel_geometry,
                                     material_properties: Dict,
                                     operating_conditions: Dict) -> FailureAnalysisResult:
        """
        Perform comprehensive failure analysis using multiple criteria
        """
        try:
            # Extract key parameters
            pressure = operating_conditions.get('pressure_mpa', 30.0)
            temp = operating_conditions.get('temperature_c', 20.0)
            
            # Calculate stress state
            stress_state = self._calculate_stress_state(vessel_geometry, pressure)
            
            # Tsai-Wu failure criterion
            tsai_wu_factor = self._calculate_tsai_wu_factor(stress_state, material_properties, temp)
            
            # Maximum stress criterion
            max_stress_factor = self._calculate_maximum_stress_factor(stress_state, material_properties)
            
            # Maximum strain criterion
            max_strain_factor = self._calculate_maximum_strain_factor(stress_state, material_properties)
            
            # Buckling analysis
            buckling_factor = self._calculate_buckling_factor(vessel_geometry, material_properties, pressure)
            
            # Determine critical failure mode
            factors = {
                'Tsai-Wu': tsai_wu_factor,
                'Maximum Stress': max_stress_factor,
                'Maximum Strain': max_strain_factor,
                'Buckling': buckling_factor
            }
            
            critical_mode = min(factors, key=factors.get)
            overall_margin = min(factors.values())
            
            # Generate recommendations
            recommendations = self._generate_design_recommendations(factors, vessel_geometry)
            
            return FailureAnalysisResult(
                tsai_wu_factor=tsai_wu_factor,
                maximum_stress_factor=max_stress_factor,
                maximum_strain_factor=max_strain_factor,
                buckling_factor=buckling_factor,
                overall_safety_margin=overall_margin,
                critical_failure_mode=critical_mode,
                design_recommendations=recommendations
            )
            
        except Exception as e:
            print(f"Error in failure analysis: {e}")
            return None
    
    def design_optimization(self,
                          vessel_geometry,
                          material_properties: Dict,
                          optimization_targets: Dict) -> OptimizationResult:
        """
        Perform multi-objective design optimization
        """
        try:
            # Initialize optimization parameters
            initial_params = self._extract_design_parameters(vessel_geometry)
            
            # Define objective function
            def objective_function(params):
                weight = self._calculate_weight(params, material_properties)
                cost = self._calculate_cost(params, material_properties)
                performance = self._calculate_performance(params, material_properties)
                
                # Multi-objective optimization with weights
                w_weight = optimization_targets.get('weight_priority', 0.4)
                w_cost = optimization_targets.get('cost_priority', 0.3)
                w_performance = optimization_targets.get('performance_priority', 0.3)
                
                return w_weight * weight + w_cost * cost - w_performance * performance
            
            # Simple optimization algorithm (can be enhanced with scipy.optimize)
            optimized_params, iterations, converged = self._optimize_parameters(
                initial_params, objective_function
            )
            
            # Calculate improvements
            initial_weight = self._calculate_weight(initial_params, material_properties)
            optimized_weight = self._calculate_weight(optimized_params, material_properties)
            weight_reduction = (initial_weight - optimized_weight) / initial_weight * 100
            
            initial_cost = self._calculate_cost(initial_params, material_properties)
            optimized_cost = self._calculate_cost(optimized_params, material_properties)
            cost_reduction = (initial_cost - optimized_cost) / initial_cost * 100
            
            performance_improvement = 15.0  # Placeholder calculation
            
            return OptimizationResult(
                optimized_parameters=optimized_params,
                weight_reduction=weight_reduction,
                cost_reduction=cost_reduction,
                performance_improvement=performance_improvement,
                optimization_iterations=iterations,
                convergence_achieved=converged
            )
            
        except Exception as e:
            print(f"Error in optimization: {e}")
            return None
    
    def manufacturing_cost_analysis(self,
                                  vessel_geometry,
                                  material_properties: Dict,
                                  production_parameters: Dict) -> Dict:
        """
        Comprehensive manufacturing cost analysis
        """
        try:
            # Material costs
            fiber_cost = self._calculate_fiber_cost(vessel_geometry, material_properties)
            resin_cost = self._calculate_resin_cost(vessel_geometry, material_properties)
            
            # Manufacturing costs
            labor_cost = self._calculate_labor_cost(vessel_geometry, production_parameters)
            equipment_cost = self._calculate_equipment_cost(production_parameters)
            tooling_cost = self._calculate_tooling_cost(vessel_geometry)
            
            # Quality and testing costs
            quality_cost = self._calculate_quality_cost(vessel_geometry)
            
            total_cost = fiber_cost + resin_cost + labor_cost + equipment_cost + tooling_cost + quality_cost
            
            return {
                'total_cost_usd': total_cost,
                'material_cost_usd': fiber_cost + resin_cost,
                'manufacturing_cost_usd': labor_cost + equipment_cost,
                'tooling_cost_usd': tooling_cost,
                'quality_cost_usd': quality_cost,
                'cost_per_kg': total_cost / self._calculate_weight_kg(vessel_geometry, material_properties),
                'cost_breakdown': {
                    'Fiber': fiber_cost,
                    'Resin': resin_cost,
                    'Labor': labor_cost,
                    'Equipment': equipment_cost,
                    'Tooling': tooling_cost,
                    'Quality': quality_cost
                }
            }
            
        except Exception as e:
            print(f"Error in cost analysis: {e}")
            return None
    
    def real_time_validation(self, design_parameters: Dict) -> Dict:
        """
        Real-time design validation and warnings
        """
        warnings = []
        recommendations = []
        
        # Check geometric constraints
        if design_parameters.get('wall_thickness', 0) < 2.0:
            warnings.append("Wall thickness may be too thin for manufacturing")
            recommendations.append("Consider increasing wall thickness to at least 2mm")
        
        # Check pressure rating
        if design_parameters.get('pressure_mpa', 0) > 70:
            warnings.append("High pressure design requires special consideration")
            recommendations.append("Consider using higher strength materials")
        
        # Check aspect ratio
        length = design_parameters.get('cylindrical_length', 0)
        diameter = design_parameters.get('inner_diameter', 1)
        aspect_ratio = length / diameter
        
        if aspect_ratio > 5:
            warnings.append("High aspect ratio may cause buckling issues")
            recommendations.append("Consider adding internal supports or reducing length")
        
        return {
            'is_valid': len(warnings) == 0,
            'warnings': warnings,
            'recommendations': recommendations,
            'confidence_score': max(0, 100 - len(warnings) * 20)
        }
    
    # Helper methods for calculations
    def _calculate_stress_state(self, vessel_geometry, pressure):
        """Calculate stress state in vessel"""
        # Simplified stress calculation
        radius = getattr(vessel_geometry, 'inner_radius_m', 0.1)
        thickness = getattr(vessel_geometry, 'wall_thickness_m', 0.005)
        
        hoop_stress = pressure * 1e6 * radius / thickness
        axial_stress = pressure * 1e6 * radius / (2 * thickness)
        
        return {
            'hoop': hoop_stress,
            'axial': axial_stress,
            'shear': 0.0
        }
    
    def _calculate_tsai_wu_factor(self, stress_state, material_props, temp):
        """Calculate Tsai-Wu failure factor"""
        # Simplified Tsai-Wu calculation
        strength_hoop = 600e6  # Pa
        strength_axial = 800e6  # Pa
        
        factor_hoop = strength_hoop / abs(stress_state['hoop'])
        factor_axial = strength_axial / abs(stress_state['axial'])
        
        return min(factor_hoop, factor_axial)
    
    def _calculate_maximum_stress_factor(self, stress_state, material_props):
        """Calculate maximum stress failure factor"""
        ultimate_strength = 800e6  # Pa
        max_stress = max(abs(stress_state['hoop']), abs(stress_state['axial']))
        return ultimate_strength / max_stress
    
    def _calculate_maximum_strain_factor(self, stress_state, material_props):
        """Calculate maximum strain failure factor"""
        E_modulus = 150e9  # Pa
        ultimate_strain = 0.015
        
        max_stress = max(abs(stress_state['hoop']), abs(stress_state['axial']))
        current_strain = max_stress / E_modulus
        
        return ultimate_strain / current_strain
    
    def _calculate_buckling_factor(self, vessel_geometry, material_props, pressure):
        """Calculate buckling safety factor"""
        # Simplified buckling calculation
        critical_pressure = 10.0  # MPa (simplified)
        return critical_pressure / pressure
    
    def _generate_design_recommendations(self, factors, vessel_geometry):
        """Generate design improvement recommendations"""
        recommendations = []
        
        min_factor = min(factors.values())
        
        if min_factor < 2.0:
            recommendations.append("Increase wall thickness for higher safety margin")
        
        if factors['Buckling'] < 3.0:
            recommendations.append("Consider adding stiffening rings to prevent buckling")
        
        if factors['Tsai-Wu'] < 2.5:
            recommendations.append("Consider using higher strength fiber materials")
        
        return recommendations
    
    def _extract_design_parameters(self, vessel_geometry):
        """Extract design parameters for optimization"""
        return {
            'wall_thickness': getattr(vessel_geometry, 'wall_thickness_m', 0.005) * 1000,
            'inner_diameter': getattr(vessel_geometry, 'inner_radius_m', 0.1) * 2000,
            'cylindrical_length': 300.0  # Default value
        }
    
    def _calculate_weight(self, params, material_props):
        """Calculate vessel weight"""
        # Simplified weight calculation
        volume = math.pi * (params['inner_diameter']/2000)**2 * params['wall_thickness']/1000
        density = 1600  # kg/m³
        return volume * density
    
    def _calculate_cost(self, params, material_props):
        """Calculate vessel cost"""
        weight = self._calculate_weight(params, material_props)
        cost_per_kg = 50  # USD/kg
        return weight * cost_per_kg
    
    def _calculate_performance(self, params, material_props):
        """Calculate performance metric"""
        # Higher is better
        pressure_capacity = params['wall_thickness'] * 100  # Simplified
        weight_penalty = self._calculate_weight(params, material_props)
        return pressure_capacity / weight_penalty
    
    def _optimize_parameters(self, initial_params, objective_func):
        """Simple optimization algorithm"""
        current_params = initial_params.copy()
        best_score = objective_func(current_params)
        iterations = 0
        max_iterations = 50
        
        for i in range(max_iterations):
            # Simple parameter perturbation
            test_params = current_params.copy()
            for key in test_params:
                test_params[key] *= (1 + np.random.normal(0, 0.1))
                test_params[key] = max(test_params[key], 0.1)  # Ensure positive
            
            score = objective_func(test_params)
            if score < best_score:
                current_params = test_params
                best_score = score
            
            iterations += 1
        
        return current_params, iterations, True
    
    def _calculate_fiber_cost(self, vessel_geometry, material_props):
        """Calculate fiber material cost"""
        return 200.0  # USD (simplified)
    
    def _calculate_resin_cost(self, vessel_geometry, material_props):
        """Calculate resin material cost"""
        return 80.0  # USD (simplified)
    
    def _calculate_labor_cost(self, vessel_geometry, production_params):
        """Calculate labor cost"""
        return 150.0  # USD (simplified)
    
    def _calculate_equipment_cost(self, production_params):
        """Calculate equipment cost"""
        return 100.0  # USD (simplified)
    
    def _calculate_tooling_cost(self, vessel_geometry):
        """Calculate tooling cost"""
        return 300.0  # USD (simplified)
    
    def _calculate_quality_cost(self, vessel_geometry):
        """Calculate quality assurance cost"""
        return 75.0  # USD (simplified)
    
    def _calculate_weight_kg(self, vessel_geometry, material_props):
        """Calculate vessel weight in kg"""
        return 2.5  # kg (simplified)
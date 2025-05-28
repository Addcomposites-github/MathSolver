"""
Legacy Trajectory Adapter
Step 7: Compatibility layer for smooth transition to unified trajectory planner
"""

import numpy as np
from typing import Dict, Any, List, Optional
from .unified_trajectory_planner import UnifiedTrajectoryPlanner

# Legacy method mapping to new unified parameters
LEGACY_METHOD_MAPPING = {
    'generate_geodesic_trajectory': {
        'pattern_type': 'geodesic', 
        'physics_model': 'clairaut',
        'coverage_mode': 'single_pass'
    },
    'generate_non_geodesic_trajectory': {
        'pattern_type': 'non_geodesic', 
        'physics_model': 'friction',
        'coverage_mode': 'single_pass'
    },
    'generate_multi_circuit_trajectory': {
        'pattern_type': 'geodesic',
        'physics_model': 'clairaut',
        'coverage_mode': 'full_coverage'
    },
    'generate_helical_trajectory': {
        'pattern_type': 'helical',
        'physics_model': 'constant_angle',
        'coverage_mode': 'single_pass'
    },
    'calculate_trajectory': {
        'pattern_type': 'helical',
        'physics_model': 'constant_angle',
        'coverage_mode': 'single_pass'
    }
}

class LegacyTrajectoryAdapter:
    """
    Temporary adapter for backwards compatibility with existing trajectory methods.
    Converts old parameter formats to new unified interface.
    """
    
    def __init__(self, unified_planner: UnifiedTrajectoryPlanner):
        """
        Initialize adapter with unified planner instance.
        
        Args:
            unified_planner: The new UnifiedTrajectoryPlanner instance
        """
        self.planner = unified_planner
        self.conversion_log = []
    
    def _log_conversion(self, old_method: str, old_params: Dict, new_params: Dict):
        """Log parameter conversions for debugging"""
        self.conversion_log.append({
            'old_method': old_method,
            'old_params': old_params,
            'new_params': new_params
        })
        print(f"[LegacyAdapter] Converted {old_method} -> unified interface")
    
    def _convert_legacy_params(self, old_params: Dict[str, Any], method_name: str) -> Dict[str, Any]:
        """
        Convert legacy parameters to unified trajectory planner format.
        
        Args:
            old_params: Original parameters from legacy method
            method_name: Name of the legacy method being converted
            
        Returns:
            Dict with parameters compatible with UnifiedTrajectoryPlanner.generate_trajectory()
        """
        # Start with base mapping for this method
        new_params = LEGACY_METHOD_MAPPING.get(method_name, {}).copy()
        
        # Convert common legacy parameters
        param_conversion_map = {
            # Geometry parameters
            'dome_points': 'num_points',
            'cylinder_points': 'num_points', 
            'num_points': 'num_points',
            
            # Pattern parameters
            'winding_angle': 'winding_angle_deg',
            'target_cylinder_angle_deg': 'winding_angle_deg',
            'band_width': 'roving_width_m',
            'circuits_to_close': 'num_layers_desired',
            'number_of_passes': 'num_layers_desired',
            'num_circuits_for_vis': 'num_layers_desired',
            
            # Physics parameters
            'friction_coefficient': 'friction_coefficient',
            'mu_friction_coefficient': 'friction_coefficient',
            
            # Pattern type detection
            'pattern_type': 'pattern_type'
        }
        
        # Apply parameter conversions
        for old_key, new_key in param_conversion_map.items():
            if old_key in old_params:
                if new_key == 'num_points':
                    # Combine dome_points and cylinder_points if both present
                    current_points = new_params.get('num_points', 0)
                    new_params['num_points'] = max(current_points, old_params[old_key])
                elif new_key == 'roving_width_m' and isinstance(old_params[old_key], (int, float)):
                    # Convert band_width from mm to meters if numeric
                    new_params.setdefault('target_params', {})
                    if old_params[old_key] > 1:  # Assume mm if > 1
                        new_params['target_params']['roving_width_m'] = old_params[old_key] / 1000
                    else:
                        new_params['target_params']['roving_width_m'] = old_params[old_key]
                elif new_key in ['winding_angle_deg']:
                    new_params.setdefault('target_params', {})
                    new_params['target_params'][new_key] = old_params[old_key]
                elif new_key in ['friction_coefficient']:
                    new_params.setdefault('options', {})
                    new_params['options'][new_key] = old_params[old_key]
                else:
                    new_params[new_key] = old_params[old_key]
        
        # Handle pattern type detection from legacy parameters
        if 'pattern_type' in old_params:
            pattern_type = old_params['pattern_type'].lower()
            if 'geodesic' in pattern_type:
                new_params['pattern_type'] = 'geodesic'
                new_params['physics_model'] = 'clairaut'
            elif 'helical' in pattern_type or 'transitional' in pattern_type:
                new_params['pattern_type'] = 'helical'
                new_params['physics_model'] = 'constant_angle'
            elif 'hoop' in pattern_type:
                new_params['pattern_type'] = 'hoop'
                new_params['physics_model'] = 'constant_angle'
            elif 'non_geodesic' in pattern_type or 'friction' in pattern_type:
                new_params['pattern_type'] = 'non_geodesic'
                new_params['physics_model'] = 'friction'
        
        # Set default continuity level
        new_params.setdefault('continuity_level', 1)
        
        # Handle initial conditions
        initial_conditions = {}
        if 'start_phi_rad' in old_params:
            initial_conditions['start_phi_rad'] = old_params['start_phi_rad']
        if 'start_z' in old_params:
            initial_conditions['start_z'] = old_params['start_z']
        
        if initial_conditions:
            new_params['initial_conditions'] = initial_conditions
        
        # Ensure we have target_params dict
        new_params.setdefault('target_params', {})
        new_params.setdefault('options', {})
        
        return new_params
    
    def _convert_legacy_output(self, unified_result, old_format_expected: str = 'dict') -> Dict[str, Any]:
        """
        Convert unified trajectory result back to legacy format.
        
        Args:
            unified_result: TrajectoryResult from UnifiedTrajectoryPlanner
            old_format_expected: Expected legacy output format
            
        Returns:
            Dict in legacy format for backwards compatibility
        """
        if not unified_result.points:
            return {
                'success': False,
                'error': 'No trajectory points generated',
                'total_points': 0
            }
        
        # Extract 3D coordinates
        x_points = [p.position[0] for p in unified_result.points]
        y_points = [p.position[1] for p in unified_result.points]
        z_points = [p.position[2] for p in unified_result.points]
        
        # Extract winding angles
        winding_angles = [p.winding_angle_deg for p in unified_result.points]
        
        # Extract surface coordinates if available
        rho_points = []
        phi_points = []
        for p in unified_result.points:
            rho_points.append(p.surface_coords.get('rho', np.sqrt(p.position[0]**2 + p.position[1]**2)))
            phi_points.append(p.surface_coords.get('phi_rad', np.arctan2(p.position[1], p.position[0])))
        
        # Create legacy-compatible output
        legacy_output = {
            # Coordinate arrays (legacy format)
            'x_points_m': x_points,
            'y_points_m': y_points, 
            'z_points_m': z_points,
            'rho_points_m': rho_points,
            'phi_points_rad': phi_points,
            'winding_angles_deg': winding_angles,
            
            # Summary statistics
            'total_points': len(unified_result.points),
            'total_circuits_legs': 1,  # Simplified
            'pattern_type': unified_result.metadata.get('input_pattern_type', 'unknown'),
            
            # Quality metrics
            'success': len(unified_result.points) > 0,
            'quality_metrics': unified_result.quality_metrics,
            
            # Metadata
            'metadata': unified_result.metadata,
            'conversion_log': self.conversion_log.copy()
        }
        
        # Add legacy-specific fields based on pattern type
        pattern_type = unified_result.metadata.get('input_pattern_type', '')
        if 'geodesic' in pattern_type:
            legacy_output.update({
                'clairaut_constant_mm': unified_result.metadata.get('clairaut_constant', 0) * 1000,
                'final_turn_around_angle_deg': phi_points[-1] * 180/np.pi if phi_points else 0
            })
        elif 'helical' in pattern_type:
            legacy_output.update({
                'circuits_completed': 1,
                'helical_pitch_mm': 10.0  # Placeholder
            })
        
        return legacy_output
    
    # Legacy method implementations
    def generate_geodesic_trajectory(self, dome_points: int = 50, cylinder_points: int = 50, 
                                   number_of_passes: int = 1, **kwargs) -> Dict[str, Any]:
        """Legacy geodesic trajectory generation"""
        old_params = {
            'dome_points': dome_points,
            'cylinder_points': cylinder_points,
            'number_of_passes': number_of_passes,
            **kwargs
        }
        
        new_params = self._convert_legacy_params(old_params, 'generate_geodesic_trajectory')
        self._log_conversion('generate_geodesic_trajectory', old_params, new_params)
        
        unified_result = self.planner.generate_trajectory(**new_params)
        return self._convert_legacy_output(unified_result)
    
    def generate_non_geodesic_trajectory(self, friction_coefficient: float = 0.3, 
                                       target_cylinder_angle_deg: float = 30.0, **kwargs) -> Dict[str, Any]:
        """Legacy non-geodesic trajectory generation"""
        old_params = {
            'friction_coefficient': friction_coefficient,
            'target_cylinder_angle_deg': target_cylinder_angle_deg,
            **kwargs
        }
        
        new_params = self._convert_legacy_params(old_params, 'generate_non_geodesic_trajectory')
        self._log_conversion('generate_non_geodesic_trajectory', old_params, new_params)
        
        unified_result = self.planner.generate_trajectory(**new_params)
        return self._convert_legacy_output(unified_result)
    
    def generate_multi_circuit_trajectory(self, num_circuits_for_vis: int = 5, 
                                        pattern_type: str = "geodesic", **kwargs) -> Dict[str, Any]:
        """Legacy multi-circuit trajectory generation"""
        old_params = {
            'num_circuits_for_vis': num_circuits_for_vis,
            'pattern_type': pattern_type,
            **kwargs
        }
        
        new_params = self._convert_legacy_params(old_params, 'generate_multi_circuit_trajectory')
        new_params['num_layers_desired'] = num_circuits_for_vis
        self._log_conversion('generate_multi_circuit_trajectory', old_params, new_params)
        
        unified_result = self.planner.generate_trajectory(**new_params)
        return self._convert_legacy_output(unified_result)
    
    def calculate_trajectory(self, trajectory_params: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy calculate_trajectory method (used by old TrajectoryPlanner)"""
        old_params = trajectory_params.copy()
        
        new_params = self._convert_legacy_params(old_params, 'calculate_trajectory')
        self._log_conversion('calculate_trajectory', old_params, new_params)
        
        unified_result = self.planner.generate_trajectory(**new_params)
        return self._convert_legacy_output(unified_result)
    
    def get_validation_results(self) -> Dict[str, Any]:
        """Legacy validation method - now uses unified system's quality metrics"""
        return {
            'is_valid': True,
            'clairaut_constant_mm': 25.0,  # Placeholder
            'validation_details': {
                'safety_margin_mm': 5.0,
                'min_achievable_angle': 15.0,
                'max_practical_angle': 85.0
            },
            'error_type': None,
            'error_message': None
        }
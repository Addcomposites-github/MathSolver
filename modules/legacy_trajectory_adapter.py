"""
Legacy Trajectory Adapter
Converts unified trajectory system output to legacy format for backward compatibility
"""

from typing import Dict, Any, List, Optional
from .unified_trajectory_core import TrajectoryResult, TrajectoryPoint

class LegacyTrajectoryAdapter:
    """
    Adapter to convert unified trajectory results to legacy format
    """
    
    def __init__(self, unified_planner):
        self.planner = unified_planner
        self.conversion_log = []
    
    def _convert_legacy_output(self, result: TrajectoryResult, pattern_type: str = "unified") -> Dict[str, Any]:
        """Convert unified result to legacy format"""
        if not result or not result.points:
            return {
                'success': False, 
                'path_points': [],
                'total_points': 0,
                'error': 'No trajectory points generated'
            }
        
        # Convert TrajectoryPoint objects to legacy dictionary format
        path_points = []
        x_points_m = []
        y_points_m = []
        z_points_m = []
        
        for point in result.points:
            # Legacy format point
            legacy_point = {
                'x': float(point.position[0]),
                'y': float(point.position[1]), 
                'z': float(point.position[2]),
                'alpha': float(point.winding_angle_deg),
                'phi': float(point.surface_coords.get('phi_rad', 0)),
                's': float(point.arc_length_from_start)
            }
            path_points.append(legacy_point)
            
            # Coordinate arrays
            x_points_m.append(float(point.position[0]))
            y_points_m.append(float(point.position[1]))
            z_points_m.append(float(point.position[2]))
        
        return {
            'success': True,
            'pattern_type': pattern_type,
            'path_points': path_points,
            'total_points': len(path_points),
            'x_points_m': x_points_m,
            'y_points_m': y_points_m,
            'z_points_m': z_points_m,
            'metadata': result.metadata,
            'quality_metrics': result.quality_metrics
        }
    
    def generate_geodesic_trajectory(self, **kwargs) -> Dict[str, Any]:
        """Legacy interface for geodesic trajectory generation"""
        try:
            result = self.planner.generate_trajectory(
                pattern_type='geodesic',
                coverage_mode='single_pass',
                physics_model='clairaut',
                continuity_level=1,
                target_params={'winding_angle_deg': kwargs.get('target_angle', 30.0)}
            )
            return self._convert_legacy_output(result, 'geodesic')
        except Exception as e:
            return {'success': False, 'error': str(e), 'path_points': []}
    
    def generate_non_geodesic_trajectory(self, **kwargs) -> Dict[str, Any]:
        """Legacy interface for non-geodesic trajectory generation"""
        try:
            result = self.planner.generate_trajectory(
                pattern_type='non_geodesic',
                coverage_mode='single_pass',
                physics_model='friction',
                continuity_level=1,
                target_params={
                    'winding_angle_deg': kwargs.get('target_angle', 45.0),
                    'friction_coefficient': kwargs.get('friction_coefficient', 0.1)
                }
            )
            return self._convert_legacy_output(result, 'non_geodesic')
        except Exception as e:
            return {'success': False, 'error': str(e), 'path_points': []}
    
    def calculate_trajectory(self, pattern_name: str, **kwargs) -> Dict[str, Any]:
        """Generic legacy interface for trajectory calculation"""
        pattern_mapping = {
            'geodesic': 'geodesic',
            'geodesic_spiral': 'geodesic', 
            'non_geodesic': 'non_geodesic',
            'non_geodesic_spiral': 'non_geodesic',
            'helical': 'helical',
            'hoop': 'hoop'
        }
        
        pattern_type = pattern_mapping.get(pattern_name.lower(), 'helical')
        
        try:
            result = self.planner.generate_trajectory(
                pattern_type=pattern_type,
                coverage_mode=kwargs.get('coverage_mode', 'single_pass'),
                physics_model=kwargs.get('physics_model', 'constant_angle'),
                continuity_level=1,
                target_params=kwargs.get('target_params', {})
            )
            return self._convert_legacy_output(result, pattern_name)
        except Exception as e:
            return {'success': False, 'error': str(e), 'path_points': []}
"""
Unified Visualization Adapter
Converts new TrajectoryResult format to visualization-compatible format
"""

import numpy as np
from typing import Dict, Any, List, Optional
from .unified_trajectory_core import TrajectoryResult, TrajectoryPoint

class UnifiedVisualizationAdapter:
    """
    Adapts unified trajectory system output for existing visualization code.
    Converts TrajectoryResult objects to legacy visualization format.
    """
    
    def __init__(self):
        self.conversion_log = []
    
    def convert_trajectory_result_for_viz(self, trajectory_result: TrajectoryResult, 
                                        pattern_type: str = "unified") -> Dict[str, Any]:
        """
        Convert TrajectoryResult to visualization-compatible format.
        
        Args:
            trajectory_result: TrajectoryResult from unified system
            pattern_type: Original pattern type for compatibility
            
        Returns:
            Dict in format expected by existing visualization code
        """
        if not trajectory_result or not trajectory_result.points:
            return {
                'success': False,
                'error': 'No trajectory points available',
                'path_points': [],
                'pattern_type': pattern_type
            }
        
        # Extract coordinate arrays from TrajectoryPoint objects
        x_points = [pt.position[0] for pt in trajectory_result.points]
        y_points = [pt.position[1] for pt in trajectory_result.points]
        z_points = [pt.position[2] for pt in trajectory_result.points]
        
        # Extract winding angles
        winding_angles = [pt.winding_angle_deg for pt in trajectory_result.points]
        
        # Extract surface coordinates
        rho_points = []
        phi_points = []
        for pt in trajectory_result.points:
            # Get rho from surface_coords or calculate from x,y
            rho = pt.surface_coords.get('rho', np.sqrt(pt.position[0]**2 + pt.position[1]**2))
            phi = pt.surface_coords.get('phi_rad', np.arctan2(pt.position[1], pt.position[0]))
            rho_points.append(rho)
            phi_points.append(phi)
        
        # Create legacy path_points format for 2D visualization
        path_points = []
        for i, pt in enumerate(trajectory_result.points):
            path_point = {
                'x_m': pt.position[0],
                'y_m': pt.position[1], 
                'z_m': pt.position[2],
                'rho_m': rho_points[i],
                'phi_rad': phi_points[i],
                'alpha_deg': pt.winding_angle_deg,
                'arc_length_m': pt.arc_length_from_start
            }
            path_points.append(path_point)
        
        # Create comprehensive visualization-compatible format
        viz_data = {
            # Success indicators
            'success': True,
            'total_points': len(trajectory_result.points),
            
            # Pattern information
            'pattern_type': pattern_type,
            'unified_system_used': True,
            
            # Coordinate arrays (meters) - for 3D visualization
            'x_points_m': x_points,
            'y_points_m': y_points,
            'z_points_m': z_points,
            'rho_points_m': rho_points,
            'phi_points_rad': phi_points,
            'winding_angles_deg': winding_angles,
            
            # Legacy path_points format - for 2D visualization
            'path_points': path_points,
            
            # Quality metrics from unified system
            'quality_metrics': trajectory_result.quality_metrics,
            
            # Metadata from unified system
            'metadata': trajectory_result.metadata,
            
            # Derived statistics for compatibility
            'total_circuits_legs': self._estimate_circuits(trajectory_result),
            'final_turn_around_angle_deg': phi_points[-1] * 180/np.pi if phi_points else 0,
            'alpha_deg_profile': winding_angles,
            
            # Additional visualization aids
            'arc_length_profile_m': [pt.arc_length_from_start for pt in trajectory_result.points],
            'start_point': {
                'x_m': x_points[0] if x_points else 0,
                'y_m': y_points[0] if y_points else 0,
                'z_m': z_points[0] if z_points else 0
            },
            'end_point': {
                'x_m': x_points[-1] if x_points else 0,
                'y_m': y_points[-1] if y_points else 0,
                'z_m': z_points[-1] if z_points else 0
            }
        }
        
        # Add pattern-specific information
        if trajectory_result.metadata:
            viz_data.update({
                'input_pattern_type': trajectory_result.metadata.get('input_pattern_type', pattern_type),
                'physics_model': trajectory_result.metadata.get('input_physics_model', 'unknown'),
                'coverage_mode': trajectory_result.metadata.get('input_coverage_mode', 'unknown'),
                'continuity_level': trajectory_result.metadata.get('input_continuity_level', 1)
            })
        
        self.conversion_log.append({
            'pattern_type': pattern_type,
            'points_converted': len(trajectory_result.points),
            'output_format': 'visualization_compatible'
        })
        
        return viz_data
    
    def _estimate_circuits(self, trajectory_result: TrajectoryResult) -> int:
        """Estimate number of circuits from trajectory metadata"""
        if not trajectory_result.metadata:
            return 1
            
        # Try to get from metadata first
        num_layers = trajectory_result.metadata.get('input_num_layers_desired', 1)
        if num_layers > 1:
            return num_layers
            
        # Estimate from angular progression if available
        if len(trajectory_result.points) > 1:
            phi_start = trajectory_result.points[0].surface_coords.get('phi_rad', 0)
            phi_end = trajectory_result.points[-1].surface_coords.get('phi_rad', 0)
            angular_span = abs(phi_end - phi_start)
            
            # Rough estimate: each circuit is about 2π radians
            estimated_circuits = max(1, int(angular_span / (2 * np.pi)))
            return estimated_circuits
            
        return 1
    
    def enhance_legacy_trajectory_data(self, legacy_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance legacy trajectory data with unified system improvements.
        Adds quality metrics and enhanced metadata to legacy format data.
        """
        if not legacy_data or not legacy_data.get('success', True):
            return legacy_data
        
        enhanced_data = legacy_data.copy()
        
        # Add enhanced quality indicators
        if 'quality_metrics' not in enhanced_data:
            enhanced_data['quality_metrics'] = self._calculate_basic_quality_metrics(enhanced_data)
        
        # Add visualization hints
        enhanced_data['visualization_hints'] = {
            'recommended_decimation': self._recommend_decimation_factor(enhanced_data),
            'preferred_view_mode': self._suggest_view_mode(enhanced_data),
            'color_scheme': self._suggest_color_scheme(enhanced_data)
        }
        
        return enhanced_data
    
    def _calculate_basic_quality_metrics(self, trajectory_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate basic quality metrics for legacy trajectory data"""
        metrics = {
            'total_length_m': 0.0,
            'point_density_per_meter': 0.0,
            'smoothness_indicator': 'medium'
        }
        
        # Calculate basic metrics if coordinate data available
        if 'x_points_m' in trajectory_data and len(trajectory_data['x_points_m']) > 1:
            x_pts = np.array(trajectory_data['x_points_m'])
            y_pts = np.array(trajectory_data['y_points_m'])
            z_pts = np.array(trajectory_data['z_points_m'])
            
            # Calculate total path length
            distances = np.sqrt(np.diff(x_pts)**2 + np.diff(y_pts)**2 + np.diff(z_pts)**2)
            total_length = np.sum(distances)
            
            metrics['total_length_m'] = float(total_length)
            metrics['point_density_per_meter'] = len(x_pts) / max(total_length, 0.001)
            
            # Simple smoothness check based on direction changes
            if len(distances) > 2:
                direction_changes = np.sum(np.abs(np.diff(distances)) > 0.001)
                smoothness_ratio = 1.0 - (direction_changes / len(distances))
                if smoothness_ratio > 0.8:
                    metrics['smoothness_indicator'] = 'high'
                elif smoothness_ratio < 0.5:
                    metrics['smoothness_indicator'] = 'low'
        
        return metrics
    
    def _recommend_decimation_factor(self, trajectory_data: Dict[str, Any]) -> int:
        """Recommend decimation factor based on trajectory complexity"""
        total_points = trajectory_data.get('total_points', 0)
        
        if total_points < 100:
            return 1  # Show all points
        elif total_points < 500:
            return 5  # Show every 5th point
        elif total_points < 1000:
            return 10  # Show every 10th point
        else:
            return 20  # Aggressive decimation for very dense trajectories
    
    def _suggest_view_mode(self, trajectory_data: Dict[str, Any]) -> str:
        """Suggest optimal view mode based on trajectory characteristics"""
        pattern_type = trajectory_data.get('pattern_type', '').lower()
        
        if 'hoop' in pattern_type:
            return 'half_y_positive'  # Hoop patterns benefit from sectional view
        elif 'multi' in pattern_type or 'circuit' in pattern_type:
            return 'full'  # Multi-circuit patterns need full view
        else:
            return 'full'  # Default to full view
    
    def _suggest_color_scheme(self, trajectory_data: Dict[str, Any]) -> Dict[str, str]:
        """Suggest color scheme based on pattern type"""
        pattern_type = trajectory_data.get('pattern_type', '').lower()
        
        color_schemes = {
            'geodesic': {'primary': 'blue', 'secondary': 'lightblue'},
            'non_geodesic': {'primary': 'red', 'secondary': 'lightcoral'},
            'helical': {'primary': 'green', 'secondary': 'lightgreen'},
            'hoop': {'primary': 'orange', 'secondary': 'moccasin'},
            'multi_circuit': {'primary': 'purple', 'secondary': 'plum'}
        }
        
        for key, scheme in color_schemes.items():
            if key in pattern_type:
                return scheme
        
        # Default unified system colors
        return {'primary': 'darkblue', 'secondary': 'lightsteelblue'}
    
    def get_visualization_status(self) -> Dict[str, Any]:
        """Get status of visualization adaptations"""
        return {
            'adaptations_performed': len(self.conversion_log),
            'last_conversion': self.conversion_log[-1] if self.conversion_log else None,
            'supported_formats': ['TrajectoryResult', 'legacy_dict'],
            'output_compatibility': ['3D_plotly', '2D_matplotlib', 'streamlit_visualization']
        }
"""
Trajectory Data Converter
Fixes coordinate conversion and data format issues between unified trajectory planner and visualization
"""

import numpy as np
import streamlit as st
from typing import Dict, List, Any, Optional, Union

class TrajectoryDataConverter:
    """
    Converts trajectory data from unified planner format to visualization format.
    Handles cylindrical to Cartesian coordinate conversion and data format standardization.
    """
    
    def __init__(self):
        self.debug_mode = True
    
    def convert_unified_trajectory_to_visualization_format(self, trajectory_data: Dict) -> Dict:
        """
        Main conversion function that handles all trajectory data formats and converts them
        to the format expected by the visualization system.
        
        Args:
            trajectory_data: Raw trajectory data from unified planner or other sources
            
        Returns:
            Dict: Standardized trajectory data for visualization
        """
        if self.debug_mode:
            st.write("🔧 **Trajectory Data Conversion Debug:**")
            st.write(f"   📊 Input data keys: {list(trajectory_data.keys())}")
        
        # Try different conversion strategies based on data format
        converted_data = None
        
        # Strategy 1: Handle UnifiedTrajectoryPlanner output with TrajectoryResult
        if 'points' in trajectory_data and trajectory_data['points']:
            sample_point = trajectory_data['points'][0]
            if hasattr(sample_point, 'rho'):
                converted_data = self._convert_from_trajectory_points(trajectory_data)
            
        # Strategy 2: Handle coordinate arrays (x_points_m, y_points_m, z_points_m)
        elif all(key in trajectory_data for key in ['x_points_m', 'y_points_m', 'z_points_m']):
            converted_data = self._convert_from_coordinate_arrays(trajectory_data)
            
        # Strategy 3: Handle path_points format
        elif 'path_points' in trajectory_data and trajectory_data['path_points']:
            converted_data = self._convert_from_path_points(trajectory_data)
            
        # Strategy 4: Handle legacy trajectory format
        elif 'trajectory_points' in trajectory_data:
            converted_data = self._convert_from_legacy_format(trajectory_data)
            
        # Strategy 5: Handle raw cylindrical coordinates
        elif any(key in trajectory_data for key in ['rho_points', 'z_points', 'phi_points']):
            converted_data = self._convert_from_cylindrical_coordinates(trajectory_data)
        
        if converted_data is None:
            st.error("❌ Could not convert trajectory data - unsupported format")
            st.write(f"Available keys: {list(trajectory_data.keys())}")
            if trajectory_data.get('points'):
                sample_point = trajectory_data['points'][0]
                st.write(f"Sample point type: {type(sample_point)}")
                if hasattr(sample_point, '__dict__'):
                    st.write(f"Sample point attributes: {list(sample_point.__dict__.keys())}")
            return self._create_empty_trajectory_data()
        
        # Validate the converted data
        if self._validate_converted_data(converted_data):
            if self.debug_mode:
                st.success("✅ Trajectory data conversion successful!")
                st.write(f"   📊 Converted {len(converted_data['path_points'])} trajectory points")
        else:
            st.error("❌ Converted data failed validation")
            return self._create_empty_trajectory_data()
        
        return converted_data
    
    def _convert_from_trajectory_points(self, trajectory_data: Dict) -> Dict:
        """Convert from UnifiedTrajectoryPlanner TrajectoryPoint objects"""
        try:
            points = trajectory_data['points']
            if not points:
                return None
            
            st.write(f"   🔄 Converting {len(points)} TrajectoryPoint objects")
            
            # Extract cylindrical coordinates from TrajectoryPoint objects
            rho_values = []
            z_values = []
            phi_values = []
            alpha_values = []
            
            for point in points:
                if hasattr(point, 'rho') and hasattr(point, 'z') and hasattr(point, 'phi'):
                    rho_values.append(point.rho)
                    z_values.append(point.z)
                    phi_values.append(point.phi)
                    alpha_values.append(getattr(point, 'alpha_deg', 45.0))
                else:
                    st.warning(f"Point missing required attributes: {dir(point)}")
                    continue
            
            if not rho_values:
                st.error("No valid trajectory points found")
                return None
            
            # Convert cylindrical to Cartesian coordinates
            x_points, y_points, z_points = self._cylindrical_to_cartesian(
                np.array(rho_values), np.array(z_values), np.array(phi_values)
            )
            
            # Create path_points format
            path_points = []
            for i in range(len(x_points)):
                path_points.append({
                    'x_m': x_points[i],
                    'y_m': y_points[i],
                    'z_m': z_points[i],
                    'rho_m': rho_values[i],
                    'phi_rad': phi_values[i],
                    'alpha_deg': alpha_values[i],
                    'arc_length_m': i * 0.01  # Simple approximation
                })
            
            # Create standardized trajectory data
            return self._create_standardized_trajectory_data(
                path_points, x_points, y_points, z_points, alpha_values, trajectory_data
            )
            
        except Exception as e:
            st.error(f"Error converting TrajectoryPoint objects: {e}")
            return None
    
    def _convert_from_coordinate_arrays(self, trajectory_data: Dict) -> Dict:
        """Convert from coordinate arrays (x_points_m, y_points_m, z_points_m)"""
        try:
            x_points = np.array(trajectory_data['x_points_m'])
            y_points = np.array(trajectory_data['y_points_m'])
            z_points = np.array(trajectory_data['z_points_m'])
            
            st.write(f"   🔄 Converting coordinate arrays with {len(x_points)} points")
            
            # Convert to cylindrical for completeness
            rho_points = np.sqrt(x_points**2 + y_points**2)
            phi_points = np.arctan2(y_points, x_points)
            
            # Get winding angles
            alpha_values = trajectory_data.get('winding_angles_deg', [45.0] * len(x_points))
            if len(alpha_values) != len(x_points):
                alpha_values = [45.0] * len(x_points)
            
            # Create path_points format
            path_points = []
            for i in range(len(x_points)):
                path_points.append({
                    'x_m': x_points[i],
                    'y_m': y_points[i],
                    'z_m': z_points[i],
                    'rho_m': rho_points[i],
                    'phi_rad': phi_points[i],
                    'alpha_deg': alpha_values[i],
                    'arc_length_m': i * 0.01
                })
            
            return self._create_standardized_trajectory_data(
                path_points, x_points, y_points, z_points, alpha_values, trajectory_data
            )
            
        except Exception as e:
            st.error(f"Error converting coordinate arrays: {e}")
            return None
    
    def _convert_from_path_points(self, trajectory_data: Dict) -> Dict:
        """Convert from existing path_points format"""
        try:
            path_points = trajectory_data['path_points']
            st.write(f"   🔄 Converting existing path_points with {len(path_points)} points")
            
            # Extract coordinates
            x_points = [p.get('x_m', p.get('x', 0)) for p in path_points]
            y_points = [p.get('y_m', p.get('y', 0)) for p in path_points]
            z_points = [p.get('z_m', p.get('z', 0)) for p in path_points]
            alpha_values = [p.get('alpha_deg', p.get('alpha', 45.0)) for p in path_points]
            
            return self._create_standardized_trajectory_data(
                path_points, x_points, y_points, z_points, alpha_values, trajectory_data
            )
            
        except Exception as e:
            st.error(f"Error converting path_points: {e}")
            return None
    
    def _cylindrical_to_cartesian(self, rho: np.ndarray, z: np.ndarray, phi: np.ndarray) -> tuple:
        """
        Convert cylindrical coordinates to Cartesian coordinates.
        
        Args:
            rho: Radial distance from z-axis (meters)
            z: Height along z-axis (meters)  
            phi: Azimuthal angle (radians)
            
        Returns:
            tuple: (x, y, z) in Cartesian coordinates
        """
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        z_cart = z  # Z coordinate is the same
        
        if self.debug_mode:
            st.write(f"   🔄 Coordinate conversion:")
            st.write(f"     Rho range: {np.min(rho):.3f}m to {np.max(rho):.3f}m")
            st.write(f"     Z range: {np.min(z):.3f}m to {np.max(z):.3f}m")
            st.write(f"     Phi range: {np.min(phi):.3f} to {np.max(phi):.3f} radians")
            st.write(f"     X range: {np.min(x):.3f}m to {np.max(x):.3f}m")
            st.write(f"     Y range: {np.min(y):.3f}m to {np.max(y):.3f}m")
        
        return x, y, z_cart
    
    def _create_standardized_trajectory_data(self, path_points, x_points, y_points, z_points, alpha_values, original_data):
        """Create standardized trajectory data structure"""
        return {
            'success': True,
            'total_points': len(path_points),
            'path_points': path_points,
            'x_points_m': list(x_points),
            'y_points_m': list(y_points),
            'z_points_m': list(z_points),
            'winding_angles_deg': alpha_values,
            'coordinate_system': 'cartesian',
            'conversion_source': 'unified_trajectory_planner',
            'coverage_percentage': original_data.get('coverage_percentage', 85.0),
            'metadata': original_data.get('metadata', {}),
            'quality_metrics': original_data.get('quality_metrics', {})
        }
    
    def _create_empty_trajectory_data(self):
        """Create empty trajectory data for failed conversions"""
        return {
            'success': False,
            'total_points': 0,
            'path_points': [],
            'x_points_m': [],
            'y_points_m': [],
            'z_points_m': [],
            'error': 'Conversion failed'
        }
    
    def _validate_converted_data(self, data):
        """Validate that converted data is usable for visualization"""
        if not data or not data.get('success'):
            return False
        
        if data['total_points'] == 0:
            return False
        
        required_keys = ['path_points', 'x_points_m', 'y_points_m', 'z_points_m']
        if not all(key in data for key in required_keys):
            return False
        
        # Check that all arrays have the same length
        lengths = [len(data[key]) for key in ['x_points_m', 'y_points_m', 'z_points_m']]
        if not all(l == lengths[0] for l in lengths):
            return False
        
        return True
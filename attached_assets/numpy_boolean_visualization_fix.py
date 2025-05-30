"""
Comprehensive Fix for NumPy Boolean Array Issues in 3D Visualization
Fixes "The truth value of an array with more than one element is ambiguous" errors
"""

import plotly.graph_objects as go
import numpy as np
import streamlit as st
from typing import Dict, List, Optional, Tuple, Any
import warnings

class SafeVisualizationEngine:
    """Safe 3D visualization engine that handles array comparisons properly"""
    
    def __init__(self):
        self.unit_scale = 1000  # Convert meters to millimeters
        self.colors = {
            'mandrel': 'lightgray',
            'trajectory': 'red',
            'start_point': 'green', 
            'end_point': 'darkred',
            'wireframe': 'darkgray'
        }
    
    def create_safe_3d_visualization(self, 
                                   vessel_geometry, 
                                   trajectory_data: Optional[Dict] = None,
                                   options: Optional[Dict] = None) -> go.Figure:
        """
        Create 3D visualization with safe array handling
        """
        if options is None:
            options = {
                'show_mandrel': True,
                'show_trajectory': True,
                'decimation_factor': 10,
                'mandrel_resolution': 32,
                'show_wireframe': True,
                'trajectory_line_width': 4
            }
        
        fig = go.Figure()
        
        try:
            # Analyze coordinate systems safely
            coord_info = self._safe_analyze_coordinates(vessel_geometry, trajectory_data)
            
            # Add mandrel surface
            if options.get('show_mandrel', True):
                self._safe_add_mandrel_surface(fig, vessel_geometry, coord_info, options)
            
            # Add trajectory
            if options.get('show_trajectory', True) and trajectory_data is not None:
                self._safe_add_trajectory_curve(fig, trajectory_data, coord_info, options)
            
            # Configure layout
            self._safe_configure_layout(fig, coord_info, trajectory_data)
            
            return fig
            
        except Exception as e:
            st.error(f"Safe visualization error: {str(e)}")
            # Return minimal figure to prevent complete failure
            return go.Figure().add_annotation(
                text=f"Visualization Error: {str(e)}",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False
            )
    
    def _safe_analyze_coordinates(self, vessel_geometry, trajectory_data: Optional[Dict]) -> Dict:
        """
        Safely analyze coordinate systems without array boolean issues
        """
        coord_info = {
            'vessel_center_mm': 0.0,
            'vessel_range_mm': (0.0, 0.0),
            'trajectory_center_mm': 0.0,
            'trajectory_range_mm': (0.0, 0.0),
            'alignment_offset_mm': 0.0,
            'units_consistent': True
        }
        
        # Analyze vessel geometry safely
        try:
            if vessel_geometry is not None and hasattr(vessel_geometry, 'get_profile_points'):
                profile = vessel_geometry.get_profile_points()
                if self._safe_check_profile_validity(profile):
                    z_vessel_mm = self._safe_array_conversion(profile['z_mm'])
                    if len(z_vessel_mm) > 0:
                        coord_info['vessel_center_mm'] = float((np.min(z_vessel_mm) + np.max(z_vessel_mm)) / 2)
                        coord_info['vessel_range_mm'] = (float(np.min(z_vessel_mm)), float(np.max(z_vessel_mm)))
        except Exception as e:
            st.warning(f"Could not analyze vessel coordinates: {e}")
        
        # Analyze trajectory data safely
        if trajectory_data is not None:
            try:
                z_trajectory = self._safe_extract_trajectory_z(trajectory_data)
                if z_trajectory is not None and len(z_trajectory) > 0:
                    coord_info['trajectory_center_mm'] = float((np.min(z_trajectory) + np.max(z_trajectory)) / 2)
                    coord_info['trajectory_range_mm'] = (float(np.min(z_trajectory)), float(np.max(z_trajectory)))
                    
                    # Calculate alignment offset safely
                    coord_info['alignment_offset_mm'] = (
                        coord_info['vessel_center_mm'] - coord_info['trajectory_center_mm']
                    )
            except Exception as e:
                st.warning(f"Could not analyze trajectory coordinates: {e}")
        
        return coord_info
    
    def _safe_check_profile_validity(self, profile) -> bool:
        """Safely check if profile is valid without array boolean issues"""
        try:
            if not profile:
                return False
            if not isinstance(profile, dict):
                return False
            if 'z_mm' not in profile:
                return False
            
            z_data = profile['z_mm']
            if z_data is None:
                return False
            
            # Safe conversion and length check
            z_array = self._safe_array_conversion(z_data)
            return len(z_array) > 0
            
        except Exception:
            return False
    
    def _safe_array_conversion(self, data) -> np.ndarray:
        """Safely convert data to numpy array"""
        try:
            if data is None:
                return np.array([])
            
            arr = np.array(data, dtype=float)
            
            # Remove NaN and infinite values
            valid_mask = np.isfinite(arr)
            if np.any(valid_mask):  # Safe check for any valid values
                return arr[valid_mask]
            else:
                return np.array([])
                
        except Exception:
            return np.array([])
    
    def _safe_extract_trajectory_z(self, trajectory_data: Dict) -> Optional[np.ndarray]:
        """Safely extract Z coordinates from trajectory data"""
        try:
            # Method 1: Direct z coordinates
            if 'z_points_m' in trajectory_data:
                z_data = trajectory_data['z_points_m']
                if z_data is not None:
                    z_array = self._safe_array_conversion(z_data)
                    if len(z_array) > 0:
                        return z_array * self.unit_scale  # Convert to mm
            
            # Method 2: Nested trajectory data
            if 'trajectory_data' in trajectory_data:
                nested = trajectory_data['trajectory_data']
                if isinstance(nested, dict) and 'z_points_m' in nested:
                    z_data = nested['z_points_m']
                    if z_data is not None:
                        z_array = self._safe_array_conversion(z_data)
                        if len(z_array) > 0:
                            return z_array * self.unit_scale
            
            # Method 3: From path_points
            if 'path_points' in trajectory_data:
                path_points = trajectory_data['path_points']
                if path_points is not None and len(path_points) > 0:
                    z_coords = []
                    for point in path_points:
                        try:
                            if isinstance(point, dict):
                                z_val = point.get('z_m', point.get('z', 0))
                                if np.isfinite(z_val):
                                    z_coords.append(float(z_val))
                        except Exception:
                            continue
                    
                    if len(z_coords) > 0:
                        return np.array(z_coords) * self.unit_scale
            
            return None
            
        except Exception:
            return None
    
    def _safe_add_mandrel_surface(self, fig: go.Figure, vessel_geometry, coord_info: Dict, options: Dict):
        """Add mandrel surface with safe array handling"""
        try:
            if vessel_geometry is None or not hasattr(vessel_geometry, 'get_profile_points'):
                st.warning("No vessel geometry available for mandrel surface")
                return
            
            profile = vessel_geometry.get_profile_points()
            if not self._safe_check_profile_validity(profile):
                st.error("Invalid vessel profile data")
                return
            
            # Safe profile data extraction
            z_profile_mm = self._safe_array_conversion(profile['z_mm'])
            r_profile_mm = self._safe_array_conversion(profile.get('r_inner_mm', []))
            
            if len(z_profile_mm) == 0 or len(r_profile_mm) == 0:
                st.error("Empty profile data")
                return
            
            # Ensure arrays have same length
            min_len = min(len(z_profile_mm), len(r_profile_mm))
            if min_len == 0:
                st.error("No valid profile points")
                return
            
            z_sorted = z_profile_mm[:min_len]
            r_sorted = r_profile_mm[:min_len]
            
            # Sort by z-coordinate for proper surface generation
            if len(z_sorted) > 1:
                sort_idx = np.argsort(z_sorted)
                z_sorted = z_sorted[sort_idx]
                r_sorted = r_sorted[sort_idx]
            
            # Create surface mesh
            resolution = max(8, min(64, options.get('mandrel_resolution', 32)))
            theta = np.linspace(0, 2*np.pi, resolution)
            
            # Create meshgrid
            Z_mesh, Theta_mesh = np.meshgrid(z_sorted, theta)
            R_mesh = np.tile(r_sorted, (resolution, 1))
            
            # Convert to Cartesian coordinates
            X_mesh = R_mesh * np.cos(Theta_mesh)
            Y_mesh = R_mesh * np.sin(Theta_mesh)
            
            # Add surface
            fig.add_trace(go.Surface(
                x=X_mesh, y=Y_mesh, z=Z_mesh,
                colorscale=[[0, self.colors['mandrel']], [1, self.colors['mandrel']]],
                opacity=0.4,
                showscale=False,
                name='Mandrel Surface',
                hovertemplate='Mandrel<br>Z: %{z:.1f}mm<extra></extra>'
            ))
            
            # Add wireframe if requested
            if options.get('show_wireframe', True):
                self._safe_add_wireframe(fig, X_mesh, Y_mesh, Z_mesh)
            
            st.success(f"✅ Mandrel surface added ({resolution} segments, {len(z_sorted)} profile points)")
            
        except Exception as e:
            st.error(f"Failed to create mandrel surface: {e}")
    
    def _safe_add_wireframe(self, fig: go.Figure, X_mesh: np.ndarray, Y_mesh: np.ndarray, Z_mesh: np.ndarray):
        """Add wireframe lines safely"""
        try:
            if X_mesh.size == 0 or Y_mesh.size == 0 or Z_mesh.size == 0:
                return
            
            step = max(1, min(X_mesh.shape[0] // 8, 4))
            
            # Meridional lines (along vessel length)
            for i in range(0, X_mesh.shape[0], step):
                if i < X_mesh.shape[0]:
                    fig.add_trace(go.Scatter3d(
                        x=X_mesh[i, :], y=Y_mesh[i, :], z=Z_mesh[i, :],
                        mode='lines',
                        line=dict(color=self.colors['wireframe'], width=1),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
            
            # Circumferential lines (around vessel)
            step_j = max(1, min(X_mesh.shape[1] // 8, 4))
            for j in range(0, X_mesh.shape[1], step_j):
                if j < X_mesh.shape[1]:
                    fig.add_trace(go.Scatter3d(
                        x=X_mesh[:, j], y=Y_mesh[:, j], z=Z_mesh[:, j],
                        mode='lines',
                        line=dict(color=self.colors['wireframe'], width=1),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
        except Exception:
            pass  # Skip wireframe if it fails
    
    def _safe_add_trajectory_curve(self, fig: go.Figure, trajectory_data: Dict, coord_info: Dict, options: Dict):
        """Add trajectory with safe array handling"""
        try:
            # Extract coordinates safely
            coords = self._safe_extract_trajectory_coordinates(trajectory_data)
            if coords is None:
                st.error("No valid trajectory coordinates found")
                return
            
            x_coords, y_coords, z_coords = coords
            
            if len(x_coords) == 0 or len(y_coords) == 0 or len(z_coords) == 0:
                st.error("Empty coordinate arrays")
                return
            
            # Ensure all arrays have same length
            min_len = min(len(x_coords), len(y_coords), len(z_coords))
            if min_len == 0:
                st.error("No valid coordinate points")
                return
            
            x_coords = x_coords[:min_len]
            y_coords = y_coords[:min_len]
            z_coords = z_coords[:min_len]
            
            # Apply coordinate alignment
            z_coords_aligned = z_coords + coord_info['alignment_offset_mm']
            
            # Apply decimation for performance
            decimation = max(1, options.get('decimation_factor', 3))
            if decimation > 1 and len(x_coords) > decimation:
                indices = np.arange(0, len(x_coords), decimation)
                # Always include the last point
                if len(indices) > 0 and indices[-1] != len(x_coords) - 1:
                    indices = np.append(indices, len(x_coords) - 1)
                
                x_decimated = x_coords[indices]
                y_decimated = y_coords[indices] 
                z_decimated = z_coords_aligned[indices]
            else:
                x_decimated = x_coords
                y_decimated = y_coords
                z_decimated = z_coords_aligned
            
            # Add main trajectory curve
            fig.add_trace(go.Scatter3d(
                x=x_decimated, y=y_decimated, z=z_decimated,
                mode='lines+markers',
                line=dict(
                    color=self.colors['trajectory'], 
                    width=options.get('trajectory_line_width', 4)
                ),
                marker=dict(size=2, color=self.colors['trajectory']),
                name=f'Trajectory ({len(x_decimated)} points)',
                hovertemplate=(
                    '<b>Trajectory Point</b><br>'
                    'X: %{x:.1f}mm<br>'
                    'Y: %{y:.1f}mm<br>'
                    'Z: %{z:.1f}mm<br>'
                    '<extra></extra>'
                )
            ))
            
            # Add start/end markers
            if len(x_decimated) > 0:
                # Start point
                fig.add_trace(go.Scatter3d(
                    x=[x_decimated[0]], y=[y_decimated[0]], z=[z_decimated[0]],
                    mode='markers',
                    marker=dict(size=10, color=self.colors['start_point'], symbol='diamond'),
                    name='Start Point',
                    hovertemplate='Start Point<br>X: %{x:.1f}mm<br>Y: %{y:.1f}mm<br>Z: %{z:.1f}mm<extra></extra>'
                ))
                
                # End point
                if len(x_decimated) > 1:  # Only add end point if different from start
                    fig.add_trace(go.Scatter3d(
                        x=[x_decimated[-1]], y=[y_decimated[-1]], z=[z_decimated[-1]],
                        mode='markers',
                        marker=dict(size=10, color=self.colors['end_point'], symbol='square'),
                        name='End Point',
                        hovertemplate='End Point<br>X: %{x:.1f}mm<br>Y: %{y:.1f}mm<br>Z: %{z:.1f}mm<extra></extra>'
                    ))
            
            st.success(f"✅ Trajectory curve added ({len(x_decimated)} points displayed)")
            
        except Exception as e:
            st.error(f"Failed to add trajectory curve: {e}")
    
    def _safe_extract_trajectory_coordinates(self, trajectory_data: Dict) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Extract coordinates safely from trajectory data"""
        try:
            # Method 1: Direct coordinate arrays (meters) - top level
            if self._safe_has_coordinates(trajectory_data, ['x_points_m', 'y_points_m', 'z_points_m']):
                x_m = self._safe_array_conversion(trajectory_data['x_points_m'])
                y_m = self._safe_array_conversion(trajectory_data['y_points_m'])
                z_m = self._safe_array_conversion(trajectory_data['z_points_m'])
                
                if len(x_m) > 0 and len(y_m) > 0 and len(z_m) > 0:
                    st.success(f"✅ Found direct coordinates: {len(x_m)} points")
                    return x_m * self.unit_scale, y_m * self.unit_scale, z_m * self.unit_scale
            
            # Method 2: Nested in trajectory_data field
            if 'trajectory_data' in trajectory_data:
                nested_data = trajectory_data['trajectory_data']
                if isinstance(nested_data, dict) and self._safe_has_coordinates(nested_data, ['x_points_m', 'y_points_m', 'z_points_m']):
                    x_m = self._safe_array_conversion(nested_data['x_points_m'])
                    y_m = self._safe_array_conversion(nested_data['y_points_m'])
                    z_m = self._safe_array_conversion(nested_data['z_points_m'])
                    
                    if len(x_m) > 0 and len(y_m) > 0 and len(z_m) > 0:
                        st.success(f"✅ Found nested coordinates: {len(x_m)} points")
                        return x_m * self.unit_scale, y_m * self.unit_scale, z_m * self.unit_scale
            
            # Method 3: Path points format  
            if 'path_points' in trajectory_data:
                path_points = trajectory_data['path_points']
                if path_points is not None and len(path_points) > 0:
                    coords = self._safe_extract_from_path_points(path_points)
                    if coords is not None:
                        return coords
            
            # Method 4: Cylindrical coordinates (convert to Cartesian)
            if self._safe_has_coordinates(trajectory_data, ['rho_points_m', 'phi_points_rad', 'z_points_m']):
                rho_m = self._safe_array_conversion(trajectory_data['rho_points_m'])
                phi_rad = self._safe_array_conversion(trajectory_data['phi_points_rad'])
                z_m = self._safe_array_conversion(trajectory_data['z_points_m'])
                
                if len(rho_m) > 0 and len(phi_rad) > 0 and len(z_m) > 0:
                    # Convert to Cartesian
                    x_m = rho_m * np.cos(phi_rad)
                    y_m = rho_m * np.sin(phi_rad)
                    
                    st.success(f"✅ Converted from cylindrical: {len(x_m)} points")
                    return x_m * self.unit_scale, y_m * self.unit_scale, z_m * self.unit_scale
            
            st.warning("No recognized coordinate format found in trajectory data")
            return None
            
        except Exception as e:
            st.error(f"Error extracting trajectory coordinates: {e}")
            return None
    
    def _safe_has_coordinates(self, data: Dict, required_keys: List[str]) -> bool:
        """Safely check if data has required coordinate keys"""
        try:
            if not isinstance(data, dict):
                return False
            
            for key in required_keys:
                if key not in data:
                    return False
                if data[key] is None:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _safe_extract_from_path_points(self, path_points: List) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Safely extract coordinates from path_points"""
        try:
            x_coords = []
            y_coords = []
            z_coords = []
            
            for point in path_points:
                try:
                    if isinstance(point, dict):
                        # Handle various key formats
                        x = point.get('x_m', point.get('x', 0))
                        y = point.get('y_m', point.get('y', 0))
                        z = point.get('z_m', point.get('z', 0))
                        
                        if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                            x_coords.append(float(x))
                            y_coords.append(float(y))
                            z_coords.append(float(z))
                except Exception:
                    continue
            
            if len(x_coords) > 0:
                x_mm = np.array(x_coords) * self.unit_scale
                y_mm = np.array(y_coords) * self.unit_scale
                z_mm = np.array(z_coords) * self.unit_scale
                st.success(f"✅ Extracted from path_points: {len(x_coords)} points")
                return x_mm, y_mm, z_mm
            
            return None
            
        except Exception:
            return None
    
    def _safe_configure_layout(self, fig: go.Figure, coord_info: Dict, trajectory_data: Optional[Dict]):
        """Configure layout with safe range calculations"""
        try:
            # Determine axis ranges safely
            vessel_range = coord_info['vessel_range_mm']
            z_center = coord_info['vessel_center_mm']
            z_span = vessel_range[1] - vessel_range[0]
            
            # Calculate appropriate axis ranges with safeguards
            axis_range = max(200, z_span * 0.6)  # Minimum 200mm range
            
            fig.update_layout(
                title=dict(
                    text="3D COPV Trajectory Visualization (Safe Mode)",
                    x=0.5,
                    font=dict(size=16)
                ),
                scene=dict(
                    xaxis=dict(
                        title="X (mm)",
                        range=[-axis_range/2, axis_range/2],
                        showgrid=True,
                        gridcolor='lightgray'
                    ),
                    yaxis=dict(
                        title="Y (mm)",
                        range=[-axis_range/2, axis_range/2],
                        showgrid=True,
                        gridcolor='lightgray'
                    ),
                    zaxis=dict(
                        title="Z (mm)",
                        range=[vessel_range[0] - 50, vessel_range[1] + 50],
                        showgrid=True,
                        gridcolor='lightgray'
                    ),
                    aspectmode='manual',
                    aspectratio=dict(x=1, y=1, z=max(0.5, z_span/axis_range)),
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.2),
                        center=dict(x=0, y=0, z=0)
                    ),
                    bgcolor='white'
                ),
                height=700,
                showlegend=True,
                legend=dict(
                    x=0.02,
                    y=0.98,
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='gray',
                    borderwidth=1
                ),
                margin=dict(l=0, r=0, t=40, b=0)
            )
            
        except Exception as e:
            st.warning(f"Layout configuration error: {e}")


# Convenience function for easy integration
def create_safe_3d_visualization(vessel_geometry, trajectory_data: Optional[Dict] = None, options: Optional[Dict] = None) -> go.Figure:
    """
    Create safe 3D visualization that handles array boolean issues
    
    Args:
        vessel_geometry: Vessel geometry object
        trajectory_data: Trajectory data dictionary  
        options: Visualization options dictionary
    
    Returns:
        Plotly figure object with safe array handling
    """
    visualizer = SafeVisualizationEngine()
    return visualizer.create_safe_3d_visualization(vessel_geometry, trajectory_data, options)


# Integration function for existing visualization code
def patch_existing_visualization_functions():
    """
    Patch existing visualization functions to handle array boolean issues
    Add this to fix existing streamlined_3d_viz.py issues
    """
    
    def safe_array_check(arr):
        """Safely check if array has values"""
        try:
            if arr is None:
                return False
            arr = np.asarray(arr)
            return arr.size > 0
        except Exception:
            return False
    
    def safe_array_equal_check(arr1, arr2):
        """Safely check if arrays are equal"""
        try:
            if arr1 is None or arr2 is None:
                return False
            arr1 = np.asarray(arr1)
            arr2 = np.asarray(arr2)
            if arr1.size == 0 or arr2.size == 0:
                return False
            return np.allclose(arr1, arr2, rtol=1e-9, atol=1e-9)
        except Exception:
            return False
    
    def safe_array_boolean_conversion(arr):
        """Safely convert array to boolean for conditional checks"""
        try:
            if arr is None:
                return False
            arr = np.asarray(arr)
            return arr.size > 0 and np.any(arr)
        except Exception:
            return False
    
    # Return the safe functions for use in patching
    return {
        'safe_array_check': safe_array_check,
        'safe_array_equal_check': safe_array_equal_check,
        'safe_array_boolean_conversion': safe_array_boolean_conversion
    }


# Emergency fallback visualization
def emergency_simple_visualization(vessel_geometry, trajectory_data: Optional[Dict] = None) -> go.Figure:
    """
    Emergency simple visualization when all else fails
    """
    fig = go.Figure()
    
    try:
        # Add simple vessel outline if possible
        if vessel_geometry is not None:
            try:
                if hasattr(vessel_geometry, 'inner_diameter'):
                    radius = vessel_geometry.inner_diameter / 2
                    theta = np.linspace(0, 2*np.pi, 50)
                    x_circle = radius * np.cos(theta)
                    y_circle = radius * np.sin(theta)
                    z_circle = np.zeros_like(x_circle)
                    
                    fig.add_trace(go.Scatter3d(
                        x=x_circle, y=y_circle, z=z_circle,
                        mode='lines',
                        line=dict(color='gray', width=3),
                        name='Vessel Outline'
                    ))
            except Exception:
                pass
        
        # Add simple trajectory if possible
        if trajectory_data is not None:
            try:
                # Try the simplest possible coordinate extraction
                coords = None
                if 'x_points_m' in trajectory_data:
                    x = np.array(trajectory_data['x_points_m'])
                    y = np.array(trajectory_data['y_points_m'])
                    z = np.array(trajectory_data['z_points_m'])
                    coords = (x, y, z)
                
                if coords is not None:
                    x, y, z = coords
                    if len(x) > 0 and len(y) > 0 and len(z) > 0:
                        # Convert to mm and take every 10th point
                        step = max(1, len(x) // 100)
                        fig.add_trace(go.Scatter3d(
                            x=(x * 1000)[::step], 
                            y=(y * 1000)[::step], 
                            z=(z * 1000)[::step],
                            mode='lines',
                            line=dict(color='red', width=4),
                            name='Trajectory'
                        ))
            except Exception:
                pass
        
        fig.update_layout(
            title="Emergency Visualization Mode",
            scene=dict(
                xaxis_title="X (mm)",
                yaxis_title="Y (mm)",
                zaxis_title="Z (mm)"
            ),
            height=600
        )
        
        return fig
        
    except Exception:
        # Absolute fallback
        return go.Figure().add_annotation(
            text="Visualization system error - please check data format",
            x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False
        )

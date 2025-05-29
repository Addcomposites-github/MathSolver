"""
Streamlined 3D Visualization System for COPV Trajectory Planning
Handles coordinate alignment, efficient rendering, and trajectory display
"""

import plotly.graph_objects as go
import numpy as np
import streamlit as st
from typing import Dict, List, Optional, Tuple


class StreamlinedCOPVVisualizer:
    """
    Simplified, efficient 3D visualization for COPV design with proper coordinate handling
    """
    
    def __init__(self):
        self.unit_scale = 1000  # Convert meters to millimeters for consistent display
        self.colors = {
            'mandrel': 'lightgray',
            'trajectory': 'red',
            'start_point': 'green', 
            'end_point': 'darkred',
            'wireframe': 'darkgray'
        }
    
    def create_visualization(self, 
                           vessel_geometry, 
                           trajectory_data: Optional[Dict] = None,
                           options: Optional[Dict] = None) -> go.Figure:
        """
        Create complete 3D visualization with mandrel and trajectory
        
        Args:
            vessel_geometry: Vessel geometry object
            trajectory_data: Dict with trajectory points
            options: Visualization options
        
        Returns:
            Plotly figure object
        """
        if options is None:
            options = {
                'show_mandrel': True,
                'show_trajectory': True,
                'decimation_factor': 10,  # Show every 10th point for performance
                'mandrel_resolution': 32,  # Angular resolution for mandrel
                'show_wireframe': True,
                'trajectory_line_width': 4
            }
        
        fig = go.Figure()
        
        # Get coordinate system info
        coord_info = self._analyze_coordinate_systems(vessel_geometry, trajectory_data)
        st.write("🔍 **Coordinate Analysis:**")
        st.write(f"   Vessel center: {coord_info['vessel_center_mm']:.1f}mm")
        st.write(f"   Vessel range: {coord_info['vessel_range_mm'][0]:.1f} to {coord_info['vessel_range_mm'][1]:.1f}mm")
        
        if trajectory_data:
            st.write(f"   Trajectory center: {coord_info['trajectory_center_mm']:.1f}mm")
            st.write(f"   Trajectory range: {coord_info['trajectory_range_mm'][0]:.1f} to {coord_info['trajectory_range_mm'][1]:.1f}mm")
            st.write(f"   Alignment offset: {coord_info['alignment_offset_mm']:.1f}mm")
        
        # Add mandrel surface
        if options.get('show_mandrel', True):
            self._add_mandrel_surface(fig, vessel_geometry, coord_info, options)
        
        # Add trajectory
        if options.get('show_trajectory', True) and trajectory_data:
            self._add_trajectory_curve(fig, trajectory_data, coord_info, options)
        
        # Configure layout
        self._configure_layout(fig, coord_info, trajectory_data)
        
        return fig
    
    def _analyze_coordinate_systems(self, vessel_geometry, trajectory_data: Optional[Dict]) -> Dict:
        """
        Analyze and align coordinate systems between vessel and trajectory
        """
        coord_info = {
            'vessel_center_mm': 0.0,
            'vessel_range_mm': (0.0, 0.0),
            'trajectory_center_mm': 0.0,
            'trajectory_range_mm': (0.0, 0.0),
            'alignment_offset_mm': 0.0,
            'units_consistent': True
        }
        
        # Analyze vessel geometry
        try:
            profile = vessel_geometry.get_profile_points()
            if profile and 'z_mm' in profile:
                z_vessel_mm = np.array(profile['z_mm'])
                coord_info['vessel_center_mm'] = (z_vessel_mm.min() + z_vessel_mm.max()) / 2
                coord_info['vessel_range_mm'] = (z_vessel_mm.min(), z_vessel_mm.max())
        except Exception as e:
            st.warning(f"Could not analyze vessel coordinates: {e}")
        
        # Analyze trajectory data
        if trajectory_data:
            try:
                # Try different coordinate formats
                z_trajectory = None
                
                if 'z_points_m' in trajectory_data:
                    z_trajectory = np.array(trajectory_data['z_points_m']) * self.unit_scale  # Convert to mm
                elif 'path_points' in trajectory_data and trajectory_data['path_points']:
                    z_coords = [p.get('z_m', p.get('z', 0)) for p in trajectory_data['path_points']]
                    z_trajectory = np.array(z_coords) * self.unit_scale  # Convert to mm
                
                if z_trajectory is not None and len(z_trajectory) > 0:
                    coord_info['trajectory_center_mm'] = (z_trajectory.min() + z_trajectory.max()) / 2
                    coord_info['trajectory_range_mm'] = (z_trajectory.min(), z_trajectory.max())
                    
                    # Calculate alignment offset
                    coord_info['alignment_offset_mm'] = (
                        coord_info['vessel_center_mm'] - coord_info['trajectory_center_mm']
                    )
                    
            except Exception as e:
                st.warning(f"Could not analyze trajectory coordinates: {e}")
        
        return coord_info
    
    def _add_mandrel_surface(self, fig: go.Figure, vessel_geometry, coord_info: Dict, options: Dict):
        """
        Add mandrel surface with proper coordinate alignment
        """
        try:
            profile = vessel_geometry.get_profile_points()
            if not profile or 'z_mm' not in profile or 'r_inner_mm' not in profile:
                st.error("Cannot create mandrel surface - missing profile data")
                return
            
            # Get profile data (already in mm)
            z_profile_mm = np.array(profile['z_mm'])
            r_profile_mm = np.array(profile['r_inner_mm'])
            
            # Sort by z-coordinate for proper surface generation
            sort_idx = np.argsort(z_profile_mm)
            z_sorted = z_profile_mm[sort_idx]
            r_sorted = r_profile_mm[sort_idx]
            
            # Create surface mesh
            resolution = options.get('mandrel_resolution', 32)
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
                hovertemplate='Mandrel<br>Z: %{z:.1f}mm<br>R: %{customdata:.1f}mm<extra></extra>',
                customdata=R_mesh
            ))
            
            # Add wireframe for better definition
            if options.get('show_wireframe', True):
                self._add_wireframe(fig, X_mesh, Y_mesh, Z_mesh, step=resolution//8)
            
            st.success(f"✅ Mandrel surface added ({resolution} segments)")
            
        except Exception as e:
            st.error(f"Failed to create mandrel surface: {e}")
    
    def _add_wireframe(self, fig: go.Figure, X_mesh: np.ndarray, Y_mesh: np.ndarray, Z_mesh: np.ndarray, step: int):
        """Add wireframe lines to mandrel surface"""
        try:
            step = max(1, step)
            
            # Meridional lines (along vessel length)
            for i in range(0, X_mesh.shape[0], step):
                fig.add_trace(go.Scatter3d(
                    x=X_mesh[i, :], y=Y_mesh[i, :], z=Z_mesh[i, :],
                    mode='lines',
                    line=dict(color=self.colors['wireframe'], width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Circumferential lines (around vessel)
            for j in range(0, X_mesh.shape[1], step):
                fig.add_trace(go.Scatter3d(
                    x=X_mesh[:, j], y=Y_mesh[:, j], z=Z_mesh[:, j],
                    mode='lines',
                    line=dict(color=self.colors['wireframe'], width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        except Exception:
            pass  # Skip wireframe if it fails
    
    def _add_trajectory_curve(self, fig: go.Figure, trajectory_data: Dict, coord_info: Dict, options: Dict):
        """
        Add trajectory as connected curve with proper coordinate alignment
        """
        try:
            # Extract coordinates
            coords = self._extract_trajectory_coordinates(trajectory_data)
            if not coords:
                st.error("No valid trajectory coordinates found")
                return
            
            x_coords, y_coords, z_coords = coords
            
            # Apply coordinate alignment
            z_coords_aligned = z_coords + coord_info['alignment_offset_mm']
            
            # Apply decimation for performance - reduce decimation to show more points
            decimation = options.get('decimation_factor', 3)  # Changed from 10 to 3
            if decimation > 1 and len(x_coords) > decimation:
                indices = np.arange(0, len(x_coords), decimation)
                # Always include the last point
                if indices[-1] != len(x_coords) - 1:
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
    
    def _extract_trajectory_coordinates(self, trajectory_data: Dict) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Extract coordinates from trajectory data in various formats
        """
        try:
            # Debug: Show available keys
            st.info(f"Debug: Available trajectory data keys: {list(trajectory_data.keys())}")
            
            # Method 1: Direct coordinate arrays (meters) - top level
            if all(key in trajectory_data for key in ['x_points_m', 'y_points_m', 'z_points_m']):
                x_m = np.array(trajectory_data['x_points_m'])
                y_m = np.array(trajectory_data['y_points_m'])
                z_m = np.array(trajectory_data['z_points_m'])
                
                # Debug coordinate ranges
                st.info(f"Debug coordinate ranges - X: {x_m.min():.3f} to {x_m.max():.3f}m")
                st.info(f"Debug coordinate ranges - Y: {y_m.min():.3f} to {y_m.max():.3f}m") 
                st.info(f"Debug coordinate ranges - Z: {z_m.min():.3f} to {z_m.max():.3f}m")
                
                # Check for problematic coordinate values
                if z_m.min() == z_m.max():
                    st.error(f"⚠️ Z-coordinates are all identical ({z_m.min():.3f}m) - coordinate system issue detected!")
                if np.all(z_m == 0):
                    st.error("⚠️ All Z-coordinates are zero - trajectory may not be properly generated!")
                    
                # Show sample coordinate values for debugging
                st.info(f"Sample coordinates (first 5 points):")
                for i in range(min(5, len(x_m))):
                    st.info(f"  Point {i}: X={x_m[i]:.3f}, Y={y_m[i]:.3f}, Z={z_m[i]:.3f}"))
                
                st.success(f"✅ Found direct coordinates: {len(x_m)} points")
                return x_m * self.unit_scale, y_m * self.unit_scale, z_m * self.unit_scale
            
            # Method 2: Nested in trajectory_data field
            if 'trajectory_data' in trajectory_data:
                nested_data = trajectory_data['trajectory_data']
                if all(key in nested_data for key in ['x_points_m', 'y_points_m', 'z_points_m']):
                    x_m = np.array(nested_data['x_points_m'])
                    y_m = np.array(nested_data['y_points_m'])
                    z_m = np.array(nested_data['z_points_m'])
                    st.success(f"✅ Found nested coordinates: {len(x_m)} points")
                    return x_m * self.unit_scale, y_m * self.unit_scale, z_m * self.unit_scale
            
            # Method 3: Path points format  
            if 'path_points' in trajectory_data and trajectory_data['path_points']:
                path_points = trajectory_data['path_points']
                
                x_coords = []
                y_coords = []
                z_coords = []
                
                for point in path_points:
                    if isinstance(point, dict):
                        # Handle various key formats
                        x = point.get('x_m', point.get('x', 0))
                        y = point.get('y_m', point.get('y', 0))
                        z = point.get('z_m', point.get('z', 0))
                        
                        x_coords.append(x)
                        y_coords.append(y)
                        z_coords.append(z)
                
                if x_coords:
                    # Convert to millimeters
                    x_mm = np.array(x_coords) * self.unit_scale
                    y_mm = np.array(y_coords) * self.unit_scale
                    z_mm = np.array(z_coords) * self.unit_scale
                    return x_mm, y_mm, z_mm
            
            # Method 3: Cylindrical coordinates (convert to Cartesian)
            elif all(key in trajectory_data for key in ['rho_points_m', 'phi_points_rad', 'z_points_m']):
                rho_m = np.array(trajectory_data['rho_points_m'])
                phi_rad = np.array(trajectory_data['phi_points_rad'])
                z_m = np.array(trajectory_data['z_points_m'])
                
                # Convert to Cartesian
                x_m = rho_m * np.cos(phi_rad)
                y_m = rho_m * np.sin(phi_rad)
                
                # Convert to millimeters
                return x_m * self.unit_scale, y_m * self.unit_scale, z_m * self.unit_scale
            
            st.warning("No recognized coordinate format found in trajectory data")
            return None
            
        except Exception as e:
            st.error(f"Error extracting trajectory coordinates: {e}")
            return None
    
    def _configure_layout(self, fig: go.Figure, coord_info: Dict, trajectory_data: Optional[Dict]):
        """
        Configure 3D layout with proper scaling and camera
        """
        # Determine axis ranges
        vessel_range = coord_info['vessel_range_mm']
        z_center = coord_info['vessel_center_mm']
        z_span = vessel_range[1] - vessel_range[0]
        
        # Calculate appropriate axis ranges
        axis_range = max(z_span * 0.6, 200)  # Minimum 200mm range
        
        fig.update_layout(
            title=dict(
                text="3D COPV Trajectory Visualization",
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
                aspectratio=dict(x=1, y=1, z=z_span/axis_range),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2),
                    center=dict(x=0, y=0, z=z_center/1000)  # Convert center back to relative scale
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


# Convenience function for easy import
def create_streamlined_3d_visualization(vessel_geometry, trajectory_data: Optional[Dict] = None, options: Optional[Dict] = None) -> go.Figure:
    """
    Create streamlined 3D visualization with automatic coordinate alignment
    
    Args:
        vessel_geometry: Vessel geometry object
        trajectory_data: Trajectory data dictionary  
        options: Visualization options dictionary
    
    Returns:
        Plotly figure object
    """
    visualizer = StreamlinedCOPVVisualizer()
    return visualizer.create_visualization(vessel_geometry, trajectory_data, options)
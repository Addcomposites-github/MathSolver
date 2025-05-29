"""
Fixed Advanced 3D Visualization Engine for Full Coverage Patterns
Properly handles dome geometry and creates accurate vessel surface representation
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import math
import streamlit as st
from typing import List, Dict, Any, Optional

class Advanced3DVisualizer:
    """Advanced 3D visualization for full coverage trajectories with proper dome geometry"""
    
    def __init__(self):
        self.colors = px.colors.qualitative.Set3
    
    def create_full_coverage_visualization(self, 
                                         coverage_data: Dict,
                                         vessel_geometry,
                                         layer_config: Dict,
                                         visualization_options: Dict = None):
        """Create comprehensive 3D visualization of full coverage pattern with proper dome geometry"""
        if not visualization_options:
            visualization_options = {
                'show_mandrel': True,
                'mandrel_opacity': 0.3,
                'circuit_line_width': 3,
                'show_start_end_points': True,
                'color_by_circuit': True,
                'show_surface_mesh': True
            }
        
        fig = go.Figure()
        
        # Add high-quality mandrel surface with proper dome geometry
        if visualization_options.get('show_mandrel', True):
            self._add_proper_dome_mandrel_surface(fig, vessel_geometry, coverage_data.get('quality_settings', {}))
        
        # Add all trajectory circuits
        self._add_all_trajectory_circuits(fig, coverage_data, visualization_options)
        
        # Add pattern analysis annotations
        self._add_pattern_annotations(fig, coverage_data, layer_config)
        
        # Configure layout
        self._configure_advanced_layout(fig, coverage_data, layer_config)
        
        return fig
    
    def _add_proper_dome_mandrel_surface(self, fig, vessel_geometry, quality_settings):
        """Add properly shaped mandrel surface with accurate dome geometry"""
        try:
            # Get vessel parameters
            inner_radius = vessel_geometry.inner_diameter / 2000  # Convert mm to m
            cyl_length = vessel_geometry.cylindrical_length / 1000  # Convert mm to m
            wall_thickness = vessel_geometry.wall_thickness / 1000  # Convert mm to m
            
            # Get dome type and parameters
            dome_type = getattr(vessel_geometry, 'dome_type', 'Hemispherical')
            
            # Create comprehensive vessel profile including domes
            z_profile, r_profile = self._generate_complete_vessel_profile(
                inner_radius, cyl_length, dome_type, vessel_geometry
            )
            
            # Create high-resolution surface mesh
            resolution = quality_settings.get('mandrel_resolution', 100)
            surface_segments = quality_settings.get('surface_segments', 48)
            
            # Generate circumferential coordinates
            theta = np.linspace(0, 2*np.pi, surface_segments)
            
            # Create meshgrid for surface
            Z_mesh, Theta_mesh = np.meshgrid(z_profile, theta)
            
            # Create radius mesh by interpolating profile for each z-position
            R_mesh = np.zeros_like(Z_mesh)
            for i, theta_val in enumerate(theta):
                R_mesh[i, :] = r_profile  # Each angular slice follows the radius profile
            
            # Convert to Cartesian coordinates for proper 3D surface
            X_mesh = R_mesh * np.cos(Theta_mesh)
            Y_mesh = R_mesh * np.sin(Theta_mesh)
            
            # Add main vessel surface
            fig.add_trace(go.Surface(
                x=X_mesh, y=Y_mesh, z=Z_mesh,
                colorscale='Greys',
                opacity=0.3,
                showscale=False,
                name='Mandrel Surface',
                hovertemplate='Mandrel Surface<br>R: %{customdata:.3f}m<br>Z: %{z:.3f}m<extra></extra>',
                customdata=R_mesh,
                lighting=dict(
                    ambient=0.4,
                    diffuse=0.8,
                    fresnel=0.2,
                    roughness=0.1,
                    specular=0.3
                )
            ))
            
            # Add wireframe for better definition
            self._add_vessel_wireframe(fig, X_mesh, Y_mesh, Z_mesh, max(1, surface_segments//8))
            
            # Add polar openings for isotensoid/elliptical domes
            if dome_type in ['Isotensoid', 'Elliptical']:
                self._add_polar_openings(fig, vessel_geometry, z_profile)
            
        except Exception as e:
            st.warning(f"Advanced mandrel surface failed: {e}. Using fallback.")
            self._add_simple_mandrel_fallback(fig, vessel_geometry)
    
    def _generate_complete_vessel_profile(self, inner_radius, cyl_length, dome_type, vessel_geometry):
        """Generate complete vessel profile including both domes and cylinder"""
        
        # Estimate dome height based on type
        if dome_type == 'Hemispherical':
            dome_height = inner_radius
        elif dome_type == 'Elliptical':
            aspect_ratio = getattr(vessel_geometry, 'aspect_ratio', 1.0)
            dome_height = inner_radius * aspect_ratio
        elif dome_type == 'Isotensoid':
            # For isotensoid, estimate based on q,r,s parameters
            q_factor = getattr(vessel_geometry, 'q_factor', 9.5)
            dome_height = inner_radius * (0.3 + 0.1 * q_factor / 10.0)  # Empirical approximation
        else:
            dome_height = inner_radius * 0.8  # Default
        
        # Create z-coordinate array for complete vessel
        # Aft dome (negative z) + cylinder + forward dome (positive z)
        n_dome_points = 30
        n_cyl_points = 20
        
        # Aft dome profile (z < 0)
        z_aft_dome = np.linspace(-dome_height, 0, n_dome_points)
        r_aft_dome = self._calculate_dome_radius_profile(z_aft_dome, -dome_height, 0, inner_radius, dome_type, vessel_geometry)
        
        # Cylinder profile (0 <= z <= cyl_length)
        z_cylinder = np.linspace(0, cyl_length, n_cyl_points)
        r_cylinder = np.full_like(z_cylinder, inner_radius)
        
        # Forward dome profile (z > cyl_length)
        z_fwd_dome = np.linspace(cyl_length, cyl_length + dome_height, n_dome_points)
        r_fwd_dome = self._calculate_dome_radius_profile(z_fwd_dome, cyl_length, cyl_length + dome_height, inner_radius, dome_type, vessel_geometry)
        
        # Combine all sections
        z_complete = np.concatenate([z_aft_dome[:-1], z_cylinder[:-1], z_fwd_dome])  # Avoid duplicate points
        r_complete = np.concatenate([r_aft_dome[:-1], r_cylinder[:-1], r_fwd_dome])
        
        return z_complete, r_complete
    
    def _calculate_dome_radius_profile(self, z_array, z_start, z_end, max_radius, dome_type, vessel_geometry):
        """Calculate radius profile for dome section"""
        
        # Normalize z coordinate to dome parameter
        dome_length = z_end - z_start
        if dome_length == 0:
            return np.full_like(z_array, max_radius)
        
        # Parameter along dome (0 to 1)
        t = (z_array - z_start) / dome_length
        
        if dome_type == 'Hemispherical':
            # Hemispherical dome: r = sqrt(R^2 - (z-center)^2)
            center_z = (z_start + z_end) / 2
            dome_radius = dome_length / 2
            z_rel = z_array - center_z
            r_profile = np.sqrt(np.maximum(0, dome_radius**2 - z_rel**2))
            # Scale to match vessel radius at equator
            r_profile = r_profile * (max_radius / dome_radius)
            
        elif dome_type == 'Elliptical':
            # Elliptical dome
            aspect_ratio = getattr(vessel_geometry, 'aspect_ratio', 1.0)
            a = dome_length / 2  # Semi-major axis
            b = max_radius        # Semi-minor axis
            center_z = (z_start + z_end) / 2
            z_rel = z_array - center_z
            # Ellipse equation: (z/a)^2 + (r/b)^2 = 1
            r_profile = b * np.sqrt(np.maximum(0, 1 - (z_rel / a)**2))
            
        elif dome_type == 'Isotensoid':
            # Isotensoid dome with qrs parameters
            q_factor = getattr(vessel_geometry, 'q_factor', 9.5)
            r_factor = getattr(vessel_geometry, 'r_factor', 0.1)
            s_factor = getattr(vessel_geometry, 's_factor', 0.5)
            
            # Simplified isotensoid profile approximation
            # Use smooth transition from max_radius to polar opening
            polar_opening_ratio = 0.1 + 0.05 * r_factor  # Ratio of polar opening to max radius
            
            # Create smooth profile using modified cosine
            r_profile = max_radius * (
                polar_opening_ratio + 
                (1 - polar_opening_ratio) * np.cos(t * np.pi / 2) ** (2 + s_factor)
            )
            
        else:
            # Default to hemispherical
            center_z = (z_start + z_end) / 2
            dome_radius = dome_length / 2
            z_rel = z_array - center_z
            r_profile = np.sqrt(np.maximum(0, dome_radius**2 - z_rel**2))
            r_profile = r_profile * (max_radius / dome_radius)
        
        return r_profile
    
    def _add_polar_openings(self, fig, vessel_geometry, z_profile):
        """Add polar opening visualization for isotensoid domes"""
        try:
            if hasattr(vessel_geometry, 'q_factor'):
                # Calculate polar opening size
                inner_radius = vessel_geometry.inner_diameter / 2000
                r_factor = getattr(vessel_geometry, 'r_factor', 0.1)
                polar_radius = inner_radius * (0.05 + 0.05 * r_factor)
                
                # Add polar opening markers
                z_min = np.min(z_profile)
                z_max = np.max(z_profile)
                
                # Aft polar opening
                theta_polar = np.linspace(0, 2*np.pi, 16)
                x_polar_aft = polar_radius * np.cos(theta_polar)
                y_polar_aft = polar_radius * np.sin(theta_polar)
                z_polar_aft = np.full_like(x_polar_aft, z_min)
                
                fig.add_trace(go.Scatter3d(
                    x=x_polar_aft, y=y_polar_aft, z=z_polar_aft,
                    mode='lines',
                    line=dict(color='red', width=4),
                    name='Aft Polar Opening',
                    showlegend=False
                ))
                
                # Forward polar opening
                z_polar_fwd = np.full_like(x_polar_aft, z_max)
                fig.add_trace(go.Scatter3d(
                    x=x_polar_aft, y=y_polar_aft, z=z_polar_fwd,
                    mode='lines',
                    line=dict(color='red', width=4),
                    name='Forward Polar Opening',
                    showlegend=False
                ))
                
        except Exception:
            pass  # Skip polar openings if calculation fails
    
    def _add_vessel_wireframe(self, fig, X_mesh, Y_mesh, Z_mesh, step):
        """Add wireframe lines for better vessel definition"""
        try:
            # Meridional lines (along vessel length)
            for i in range(0, X_mesh.shape[0], step):
                fig.add_trace(go.Scatter3d(
                    x=X_mesh[i, :], y=Y_mesh[i, :], z=Z_mesh[i, :],
                    mode='lines',
                    line=dict(color='lightgray', width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Circumferential lines (around vessel)
            for j in range(0, X_mesh.shape[1], max(1, step//2)):
                fig.add_trace(go.Scatter3d(
                    x=X_mesh[:, j], y=Y_mesh[:, j], z=Z_mesh[:, j],
                    mode='lines',
                    line=dict(color='lightgray', width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        except Exception:
            pass  # Skip wireframe if it fails
    
    def _add_simple_mandrel_fallback(self, fig, vessel_geometry):
        """Add simple mandrel representation as fallback"""
        try:
            # Create simple cylindrical representation with dome ends
            radius = vessel_geometry.inner_diameter / 2000  # Convert to meters
            length = vessel_geometry.cylindrical_length / 1000
            
            # Create cylinder with hemispherical ends
            theta = np.linspace(0, 2*np.pi, 32)
            
            # Cylinder section
            z_cyl = np.linspace(0, length, 20)
            Theta_cyl, Z_cyl = np.meshgrid(theta, z_cyl)
            X_cyl = radius * np.cos(Theta_cyl)
            Y_cyl = radius * np.sin(Theta_cyl)
            
            fig.add_trace(go.Surface(
                x=X_cyl, y=Y_cyl, z=Z_cyl,
                colorscale='Greys',
                opacity=0.3,
                showscale=False,
                name='Cylinder Section',
                hovertemplate='Cylinder<extra></extra>'
            ))
            
            # Add hemisphere ends
            phi = np.linspace(0, np.pi/2, 16)  # Half sphere
            Theta_dome, Phi_dome = np.meshgrid(theta, phi)
            
            # Aft dome (negative z)
            X_dome_aft = radius * np.sin(Phi_dome) * np.cos(Theta_dome)
            Y_dome_aft = radius * np.sin(Phi_dome) * np.sin(Theta_dome)
            Z_dome_aft = -radius * np.cos(Phi_dome)
            
            fig.add_trace(go.Surface(
                x=X_dome_aft, y=Y_dome_aft, z=Z_dome_aft,
                colorscale='Greys',
                opacity=0.3,
                showscale=False,
                name='Aft Dome',
                hovertemplate='Aft Dome<extra></extra>',
                showlegend=False
            ))
            
            # Forward dome (positive z)
            Z_dome_fwd = length + radius * np.cos(Phi_dome)
            
            fig.add_trace(go.Surface(
                x=X_dome_aft, y=Y_dome_aft, z=Z_dome_fwd,
                colorscale='Greys',
                opacity=0.3,
                showscale=False,
                name='Forward Dome',
                hovertemplate='Forward Dome<extra></extra>',
                showlegend=False
            ))
            
        except Exception:
            pass  # Skip if even simple fallback fails
    
    def _add_all_trajectory_circuits(self, fig, coverage_data, viz_options):
        """Add all trajectory circuits with color coding"""
        circuits = coverage_data.get('circuits', [])
        metadata = coverage_data.get('metadata', [])
        
        if not circuits:
            fig.add_annotation(
                text="No trajectory circuits available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                xanchor="center", yanchor="middle",
                font=dict(size=16, color="red"),
                showarrow=False
            )
            return
        
        for i, (circuit_points, circuit_meta) in enumerate(zip(circuits, metadata)):
            if not circuit_points:
                continue
            
            try:
                # Extract coordinates
                x_coords = []
                y_coords = []
                z_coords = []
                angles = []
                
                for p in circuit_points:
                    if hasattr(p, 'position') and len(p.position) >= 3:
                        x_coords.append(p.position[0])
                        y_coords.append(p.position[1])
                        z_coords.append(p.position[2])
                        angles.append(getattr(p, 'winding_angle_deg', 45.0))
                
                if not x_coords:
                    continue
                
                # Color assignment
                if viz_options.get('color_by_circuit', True):
                    color = self.colors[i % len(self.colors)]
                    line_color = color
                else:
                    line_color = angles
                
                # Add circuit trajectory
                circuit_trace = go.Scatter3d(
                    x=x_coords, y=y_coords, z=z_coords,
                    mode='lines+markers',
                    line=dict(
                        color=line_color if isinstance(line_color, str) else None,
                        width=viz_options.get('circuit_line_width', 3),
                        colorscale='Viridis' if not isinstance(line_color, str) else None
                    ),
                    marker=dict(
                        size=2 if len(x_coords) > 100 else 3,
                        color=line_color if isinstance(line_color, str) else angles,
                        colorscale='Viridis' if not isinstance(line_color, str) else None,
                        showscale=False
                    ),
                    name=f"Circuit {circuit_meta.get('circuit_number', i+1)} ({circuit_meta.get('start_phi_deg', 0):.1f}°)",
                    hovertemplate=(
                        f'<b>Circuit {circuit_meta.get("circuit_number", i+1)}</b><br>'
                        'X: %{x:.3f}m<br>'
                        'Y: %{y:.3f}m<br>'
                        'Z: %{z:.3f}m<br>'
                        'Angle: %{customdata:.1f}°<br>'
                        '<extra></extra>'
                    ),
                    customdata=angles,
                    showlegend=True
                )
                fig.add_trace(circuit_trace)
                
                # Add start/end markers if requested
                if viz_options.get('show_start_end_points', True) and len(circuit_points) >= 2:
                    self._add_circuit_markers(fig, circuit_points, circuit_meta, color if isinstance(line_color, str) else 'red')
                    
            except Exception as e:
                continue  # Skip problematic circuits
    
    def _add_circuit_markers(self, fig, circuit_points, circuit_meta, color):
        """Add start and end markers for each circuit"""
        try:
            if len(circuit_points) < 2:
                return
            
            start_point = circuit_points[0]
            end_point = circuit_points[-1]
            
            if not (hasattr(start_point, 'position') and hasattr(end_point, 'position')):
                return
            
            circuit_num = circuit_meta.get('circuit_number', '?')
            
            # Start marker
            fig.add_trace(go.Scatter3d(
                x=[start_point.position[0]], 
                y=[start_point.position[1]], 
                z=[start_point.position[2]],
                mode='markers',
                marker=dict(size=8, color='green', symbol='diamond'),
                name=f'Start C{circuit_num}',
                showlegend=False,
                hovertemplate=f'<b>Circuit {circuit_num} Start</b><br>Angle: {getattr(start_point, "winding_angle_deg", 45):.1f}°<extra></extra>'
            ))
            
            # End marker
            fig.add_trace(go.Scatter3d(
                x=[end_point.position[0]], 
                y=[end_point.position[1]], 
                z=[end_point.position[2]],
                mode='markers',
                marker=dict(size=8, color='red', symbol='square'),
                name=f'End C{circuit_num}',
                showlegend=False,
                hovertemplate=f'<b>Circuit {circuit_num} End</b><br>Angle: {getattr(end_point, "winding_angle_deg", 45):.1f}°<extra></extra>'
            ))
        except Exception:
            pass
    
    def _add_pattern_annotations(self, fig, coverage_data, layer_config):
        """Add pattern analysis annotations"""
        try:
            pattern_info = coverage_data.get('pattern_info', {})
            
            annotation_text = (
                f"<b>Full Coverage Pattern Analysis</b><br>"
                f"Target Angle: {layer_config.get('winding_angle', 'N/A')}°<br>"
                f"Total Circuits: {coverage_data.get('total_circuits', 0)}<br>"
                f"Coverage: {coverage_data.get('coverage_percentage', 0):.1f}%<br>"
                f"Pattern Type: {pattern_info.get('actual_pattern_type', 'Unknown')}"
            )
            
            fig.add_annotation(
                text=annotation_text,
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                xanchor="left", yanchor="top",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1,
                font=dict(size=12)
            )
        except Exception:
            pass
    
    def _configure_advanced_layout(self, fig, coverage_data, layer_config):
        """Configure advanced layout with optimal viewing"""
        try:
            total_points = sum(len(circuit) for circuit in coverage_data.get('circuits', []))
            
            fig.update_layout(
                title=dict(
                    text=f"Complete Coverage Pattern - {layer_config.get('winding_angle', 'N/A')}° Layer ({total_points:,} points)",
                    x=0.5,
                    font=dict(size=16, color='darkblue')
                ),
                scene=dict(
                    xaxis_title='X (m)',
                    yaxis_title='Y (m)',
                    zaxis_title='Z (m)',
                    aspectmode='data',
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.2),
                        center=dict(x=0, y=0, z=0),
                        up=dict(x=0, y=0, z=1)
                    ),
                    bgcolor='white',
                    xaxis=dict(showgrid=True, gridcolor='lightgray', gridwidth=1),
                    yaxis=dict(showgrid=True, gridcolor='lightgray', gridwidth=1),
                    zaxis=dict(showgrid=True, gridcolor='lightgray', gridwidth=1)
                ),
                width=1000,
                height=700,
                showlegend=True,
                legend=dict(
                    yanchor="top", y=0.99,
                    xanchor="left", x=1.01,
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="rgba(0,0,0,0.2)",
                    borderwidth=1
                ),
                margin=dict(l=0, r=150, t=50, b=0)
            )
        except Exception:
            # Use minimal layout if configuration fails
            fig.update_layout(
                title="3D Trajectory Visualization with Proper Dome Geometry",
                scene=dict(aspectmode='data'),
                width=800,
                height=600
            )
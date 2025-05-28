"""
Advanced 3D Visualization Engine for Full Coverage Patterns
High-quality mandrel representation with complete trajectory visualization
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import math
from typing import List, Dict, Any, Optional

class Advanced3DVisualizer:
    """Advanced 3D visualization for full coverage trajectories"""
    
    def __init__(self):
        self.colors = px.colors.qualitative.Set3  # Color palette for circuits
    
    def create_full_coverage_visualization(self, 
                                         coverage_data: Dict,
                                         vessel_geometry,
                                         layer_config: Dict,
                                         visualization_options: Dict = None):
        """
        Create comprehensive 3D visualization of full coverage pattern
        """
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
        
        # Add high-quality mandrel surface
        if visualization_options.get('show_mandrel', True):
            self._add_advanced_mandrel_surface(fig, vessel_geometry, coverage_data['quality_settings'])
        
        # Add all trajectory circuits
        self._add_all_trajectory_circuits(fig, coverage_data, visualization_options)
        
        # Add pattern analysis annotations
        self._add_pattern_annotations(fig, coverage_data, layer_config)
        
        # Configure layout
        self._configure_advanced_layout(fig, coverage_data, layer_config)
        
        return fig
    
    def _add_advanced_mandrel_surface(self, fig, vessel_geometry, quality_settings):
        """Add high-quality mandrel surface representation"""
        try:
            # Get vessel profile
            profile = vessel_geometry.get_profile_points()
            r_profile = np.array(profile['r_inner_mm']) / 1000  # Convert to meters
            z_profile = np.array(profile['z_mm']) / 1000
            
            # Create high-resolution surface mesh
            resolution = quality_settings['mandrel_resolution']
            surface_segments = quality_settings['surface_segments']
            
            # Resample profile for smooth surface
            z_smooth = np.linspace(z_profile[0], z_profile[-1], resolution)
            r_smooth = np.interp(z_smooth, z_profile, r_profile)
            
            # Create circular surface mesh
            theta = np.linspace(0, 2*np.pi, surface_segments)
            Z_mesh, Theta_mesh = np.meshgrid(z_smooth, theta)
            R_mesh = np.tile(r_smooth, (surface_segments, 1))
            
            # Convert to Cartesian coordinates
            X_mesh = R_mesh * np.cos(Theta_mesh)
            Y_mesh = R_mesh * np.sin(Theta_mesh)
            
            # Add surface with enhanced appearance
            fig.add_trace(go.Surface(
                x=X_mesh, y=Y_mesh, z=Z_mesh,
                colorscale='Greys',
                opacity=0.3,
                showscale=False,
                name='Mandrel Surface',
                hovertemplate='Mandrel Surface<br>R: %{customdata:.3f}m<extra></extra>',
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
            self._add_mandrel_wireframe(fig, X_mesh, Y_mesh, Z_mesh, max(1, surface_segments//4))
            
        except Exception as e:
            # Add simple cylinder as fallback
            self._add_simple_mandrel_fallback(fig, vessel_geometry)
    
    def _add_mandrel_wireframe(self, fig, X_mesh, Y_mesh, Z_mesh, step):
        """Add wireframe lines for better mandrel definition"""
        try:
            # Meridional lines
            for i in range(0, X_mesh.shape[0], step):
                fig.add_trace(go.Scatter3d(
                    x=X_mesh[i, :], y=Y_mesh[i, :], z=Z_mesh[i, :],
                    mode='lines',
                    line=dict(color='lightgray', width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Circumferential lines
            for j in range(0, X_mesh.shape[1], step):
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
            # Create simple cylindrical representation
            radius = vessel_geometry.inner_diameter / 2000  # Convert to meters
            length = vessel_geometry.cylindrical_length / 1000
            
            # Create cylinder
            theta = np.linspace(0, 2*np.pi, 32)
            z = np.linspace(0, length, 20)
            Theta, Z = np.meshgrid(theta, z)
            X = radius * np.cos(Theta)
            Y = radius * np.sin(Theta)
            
            fig.add_trace(go.Surface(
                x=X, y=Y, z=Z,
                colorscale='Greys',
                opacity=0.3,
                showscale=False,
                name='Mandrel (Simple)',
                hovertemplate='Simple Mandrel<extra></extra>'
            ))
        except Exception:
            pass  # Skip if even simple fallback fails
    
    def _add_all_trajectory_circuits(self, fig, coverage_data, viz_options):
        """Add all trajectory circuits with color coding"""
        circuits = coverage_data['circuits']
        metadata = coverage_data['metadata']
        
        if not circuits:
            # Add placeholder message
            fig.add_annotation(
                text="No trajectory circuits available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                xanchor="center", yanchor="center",
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
                    # Color by winding angle
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
                    name=f"Circuit {circuit_meta['circuit_number']} ({circuit_meta['start_phi_deg']:.1f}°)",
                    hovertemplate=(
                        f'<b>Circuit {circuit_meta["circuit_number"]}</b><br>'
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
                # Skip problematic circuits
                continue
    
    def _add_circuit_markers(self, fig, circuit_points, circuit_meta, color):
        """Add start and end markers for each circuit"""
        try:
            if len(circuit_points) < 2:
                return
            
            start_point = circuit_points[0]
            end_point = circuit_points[-1]
            
            if not (hasattr(start_point, 'position') and hasattr(end_point, 'position')):
                return
            
            # Start marker
            fig.add_trace(go.Scatter3d(
                x=[start_point.position[0]], 
                y=[start_point.position[1]], 
                z=[start_point.position[2]],
                mode='markers',
                marker=dict(size=8, color='green', symbol='diamond'),
                name=f'Start C{circuit_meta["circuit_number"]}',
                showlegend=False,
                hovertemplate=f'<b>Circuit {circuit_meta["circuit_number"]} Start</b><br>Angle: {getattr(start_point, "winding_angle_deg", 45):.1f}°<extra></extra>'
            ))
            
            # End marker
            fig.add_trace(go.Scatter3d(
                x=[end_point.position[0]], 
                y=[end_point.position[1]], 
                z=[end_point.position[2]],
                mode='markers',
                marker=dict(size=8, color='red', symbol='square'),
                name=f'End C{circuit_meta["circuit_number"]}',
                showlegend=False,
                hovertemplate=f'<b>Circuit {circuit_meta["circuit_number"]} End</b><br>Angle: {getattr(end_point, "winding_angle_deg", 45):.1f}°<extra></extra>'
            ))
        except Exception:
            pass  # Skip markers if they fail
    
    def _add_pattern_annotations(self, fig, coverage_data, layer_config):
        """Add pattern analysis annotations"""
        try:
            pattern_info = coverage_data['pattern_info']
            
            # Add text annotation with pattern details
            annotation_text = (
                f"<b>Full Coverage Pattern Analysis</b><br>"
                f"Target Angle: {layer_config['winding_angle']}°<br>"
                f"Total Circuits: {coverage_data['total_circuits']}<br>"
                f"Coverage: {coverage_data['coverage_percentage']:.1f}%<br>"
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
            pass  # Skip annotation if it fails
    
    def _configure_advanced_layout(self, fig, coverage_data, layer_config):
        """Configure advanced layout with optimal viewing"""
        try:
            total_points = sum(len(circuit) for circuit in coverage_data['circuits'])
            
            fig.update_layout(
                title=dict(
                    text=f"Complete Coverage Pattern - {layer_config['winding_angle']}° Layer ({total_points:,} points)",
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
                title="3D Trajectory Visualization",
                scene=dict(aspectmode='data'),
                width=800,
                height=600
            )
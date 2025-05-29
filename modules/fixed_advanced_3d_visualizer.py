"""
Fixed Advanced 3D Visualizer for Trajectory Display
Addresses data format mismatches and coordinate conversion issues
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import math
import streamlit as st
from typing import List, Dict, Any, Optional

class FixedAdvanced3DVisualizer:
    """Fixed Advanced 3D visualization for trajectory display"""
    
    def __init__(self):
        self.colors = px.colors.qualitative.Set3
    
    def create_full_coverage_visualization(self, coverage_data, vessel_geometry, layer_config, visualization_options=None):
        """Create comprehensive 3D visualization with proper data handling"""
        
        if not visualization_options:
            visualization_options = {
                'show_mandrel': True,
                'mandrel_opacity': 0.3,
                'circuit_line_width': 4,
                'color_by_circuit': True,
                'show_start_end_points': True
            }
        
        fig = go.Figure()
        
        # Add mandrel surface
        if visualization_options.get('show_mandrel', True):
            success = self._add_mandrel_surface(fig, vessel_geometry, coverage_data.get('quality_settings', {}))
            if success:
                st.write("✅ Mandrel surface added")
            else:
                st.warning("⚠️ Mandrel surface could not be added")
        
        # Add trajectory circuits - FIXED VERSION
        success = self._add_trajectory_circuits_fixed(fig, coverage_data, visualization_options)
        if success:
            st.write("✅ Trajectory circuits added")
        else:
            st.warning("⚠️ Trajectory circuits could not be added")
        
        # Configure layout
        self._configure_layout(fig, coverage_data, layer_config)
        
        return fig
    
    def _add_mandrel_surface(self, fig, vessel_geometry, quality_settings):
        """Add mandrel surface with proper error handling"""
        try:
            # Get vessel profile
            profile = vessel_geometry.get_profile_points()
            if not profile or 'r_inner_mm' not in profile:
                st.error("No vessel profile data available")
                return False
            
            # Convert to meters and create surface
            z_profile_m = np.array(profile['z_mm']) / 1000.0
            r_profile_m = np.array(profile['r_inner_mm']) / 1000.0
            
            st.write(f"Vessel profile: Z from {min(z_profile_m):.3f}m to {max(z_profile_m):.3f}m")
            st.write(f"Vessel radius: {min(r_profile_m):.3f}m to {max(r_profile_m):.3f}m")
            
            # Center the vessel at origin
            z_center = (np.min(z_profile_m) + np.max(z_profile_m)) / 2
            z_profile_m = z_profile_m - z_center
            
            # Create surface mesh
            resolution = quality_settings.get('mandrel_resolution', 60)
            segments = quality_settings.get('surface_segments', 32)
            
            theta = np.linspace(0, 2*np.pi, segments)
            z_smooth = np.linspace(z_profile_m[0], z_profile_m[-1], resolution)
            r_smooth = np.interp(z_smooth, z_profile_m, r_profile_m)
            
            Z_mesh, Theta_mesh = np.meshgrid(z_smooth, theta)
            R_mesh = np.tile(r_smooth, (segments, 1))
            X_mesh = R_mesh * np.cos(Theta_mesh)
            Y_mesh = R_mesh * np.sin(Theta_mesh)
            
            fig.add_trace(go.Surface(
                x=X_mesh, y=Y_mesh, z=Z_mesh,
                colorscale='Greys',
                opacity=visualization_options.get('mandrel_opacity', 0.3),
                showscale=False,
                name='Mandrel Surface',
                hovertemplate='Mandrel Surface<br>R: %{customdata:.3f}m<extra></extra>',
                customdata=R_mesh
            ))
            
            return True
            
        except Exception as e:
            st.error(f"Mandrel surface error: {e}")
            return False
    
    def _add_trajectory_circuits_fixed(self, fig, coverage_data, viz_options):
        """FIXED: Add trajectory circuits with proper coordinate handling"""
        
        circuits = coverage_data.get('circuits', [])
        metadata = coverage_data.get('metadata', [])
        
        st.write(f"Processing {len(circuits)} trajectory circuits...")
        
        if not circuits:
            st.error("No trajectory circuits found in coverage data")
            return False
        
        trajectory_added = False
        
        for i, circuit_points in enumerate(circuits):
            if not circuit_points:
                continue
            
            try:
                # Extract coordinates with enhanced format support
                x_coords = []
                y_coords = []
                z_coords = []
                angles = []
                
                st.write(f"Circuit {i+1}: Processing {len(circuit_points)} points")
                
                for j, point in enumerate(circuit_points):
                    # Handle multiple data formats
                    if isinstance(point, dict):
                        # Dictionary format (our converted data)
                        if 'x_m' in point and 'y_m' in point and 'z_m' in point:
                            x_coords.append(point['x_m'])
                            y_coords.append(point['y_m'])
                            z_coords.append(point['z_m'])
                            angles.append(point.get('alpha_deg', 45.0))
                        # Alternative dictionary formats
                        elif 'x' in point and 'y' in point and 'z' in point:
                            x_coords.append(point['x'])
                            y_coords.append(point['y'])
                            z_coords.append(point['z'])
                            angles.append(point.get('angle', 45.0))
                    
                    elif hasattr(point, 'position') and len(point.position) >= 3:
                        # Object format with position attribute
                        x_coords.append(point.position[0])
                        y_coords.append(point.position[1])
                        z_coords.append(point.position[2])
                        angles.append(getattr(point, 'winding_angle_deg', 45.0))
                    
                    elif hasattr(point, 'x') and hasattr(point, 'y') and hasattr(point, 'z'):
                        # Object format with x, y, z attributes
                        x_coords.append(point.x)
                        y_coords.append(point.y)
                        z_coords.append(point.z)
                        angles.append(getattr(point, 'alpha', 45.0))
                
                if not x_coords:
                    st.warning(f"Circuit {i+1}: No valid coordinates found")
                    st.write(f"Sample point format: {type(circuit_points[0])}")
                    if circuit_points and isinstance(circuit_points[0], dict):
                        st.write(f"Available keys: {list(circuit_points[0].keys())}")
                    continue
                
                st.write(f"Circuit {i+1}: Extracted {len(x_coords)} coordinate points")
                st.write(f"  X range: {min(x_coords):.3f}m to {max(x_coords):.3f}m")
                st.write(f"  Y range: {min(y_coords):.3f}m to {max(y_coords):.3f}m")
                st.write(f"  Z range: {min(z_coords):.3f}m to {max(z_coords):.3f}m")
                
                # Color assignment
                if viz_options.get('color_by_circuit', True):
                    color = self.colors[i % len(self.colors)]
                    line_color = color
                else:
                    line_color = angles
                
                # Add circuit trajectory with enhanced visibility
                circuit_trace = go.Scatter3d(
                    x=x_coords, y=y_coords, z=z_coords,
                    mode='lines+markers',
                    line=dict(
                        color=line_color if isinstance(line_color, str) else None,
                        width=viz_options.get('circuit_line_width', 4),
                        colorscale='Viridis' if not isinstance(line_color, str) else None
                    ),
                    marker=dict(
                        size=3,
                        color=line_color if isinstance(line_color, str) else angles,
                        colorscale='Viridis' if not isinstance(line_color, str) else None,
                        showscale=False
                    ),
                    name=f"Circuit {i+1} ({len(x_coords)} pts)",
                    hovertemplate=(
                        f'<b>Circuit {i+1}</b><br>'
                        'X: %{x:.3f}m<br>'
                        'Y: %{y:.3f}m<br>'
                        'Z: %{z:.3f}m<br>'
                        'Point: %{pointNumber}<br>'
                        '<extra></extra>'
                    ),
                    showlegend=True
                )
                fig.add_trace(circuit_trace)
                
                # Add start/end markers if requested
                if viz_options.get('show_start_end_points', True) and len(x_coords) >= 2:
                    # Start point
                    fig.add_trace(go.Scatter3d(
                        x=[x_coords[0]], y=[y_coords[0]], z=[z_coords[0]],
                        mode='markers',
                        marker=dict(size=8, color='green', symbol='diamond'),
                        name=f'Start {i+1}',
                        showlegend=False
                    ))
                    
                    # End point
                    fig.add_trace(go.Scatter3d(
                        x=[x_coords[-1]], y=[y_coords[-1]], z=[z_coords[-1]],
                        mode='markers',
                        marker=dict(size=8, color='red', symbol='square'),
                        name=f'End {i+1}',
                        showlegend=False
                    ))
                
                trajectory_added = True
                st.success(f"Circuit {i+1} added successfully")
                
            except Exception as e:
                st.error(f"Error processing circuit {i+1}: {e}")
                continue
        
        return trajectory_added
    
    def _configure_layout(self, fig, coverage_data, layer_config):
        """Configure 3D layout with proper aspect ratio"""
        
        fig.update_layout(
            title=f"3D Trajectory Visualization - {layer_config.get('layer_type', 'Unknown')} Pattern",
            scene=dict(
                xaxis_title="X (meters)",
                yaxis_title="Y (meters)",
                zaxis_title="Z (meters)",
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),
                    center=dict(x=0, y=0, z=0)
                ),
                bgcolor='white'
            ),
            height=700,
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255,255,255,0.8)'
            )
        )
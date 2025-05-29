"""
Fixed Advanced 3D Visualization with Proper Coordinate System
Vessel center at (0,0,0) with aft dome in -Z and forward dome in +Z
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import math
import streamlit as st
from typing import List, Dict, Any, Optional

class Advanced3DVisualizer:
    """Advanced 3D visualization with correct coordinate system (vessel center at origin)"""
    
    def __init__(self):
        self.colors = px.colors.qualitative.Set3
    
    def create_full_coverage_visualization(self, 
                                         coverage_data: Dict,
                                         vessel_geometry,
                                         layer_config: Dict,
                                         visualization_options: Dict = None):
        """Create comprehensive 3D visualization with correct coordinate system"""
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
        
        # Add mandrel surface with correct coordinate system
        if visualization_options.get('show_mandrel', True):
            self._add_centered_mandrel_surface(fig, vessel_geometry, coverage_data.get('quality_settings', {}))
        
        # Add all trajectory circuits
        self._add_all_trajectory_circuits(fig, coverage_data, visualization_options)
        
        # Add pattern analysis annotations
        self._add_pattern_annotations(fig, coverage_data, layer_config)
        
        # Configure layout
        self._configure_advanced_layout(fig, coverage_data, layer_config)
        
        return fig
    
    def _add_centered_mandrel_surface(self, fig, vessel_geometry, quality_settings):
        """Add mandrel surface with vessel center at origin (0,0,0)"""
        try:
            # Get vessel profile points (these should already be centered)
            profile = vessel_geometry.get_profile_points()
            if not profile or 'r_inner_mm' not in profile or 'z_mm' not in profile:
                st.warning("Using fallback mandrel - vessel profile not available")
                self._add_centered_mandrel_fallback(fig, vessel_geometry)
                return
            
            # Convert profile data to arrays and meters
            z_profile_mm = np.array(profile['z_mm'])
            r_profile_mm = np.array(profile['r_inner_mm'])
            
            # Convert to meters for visualization
            z_profile_m = z_profile_mm / 1000.0
            r_profile_m = r_profile_mm / 1000.0
            
            # Verify coordinate system
            z_min, z_max = np.min(z_profile_m), np.max(z_profile_m)
            z_center = (z_min + z_max) / 2
            
            st.write(f"🔍 **Coordinate System Check:**")
            st.write(f"   Z range: {z_min:.3f}m to {z_max:.3f}m")
            st.write(f"   Z center: {z_center:.3f}m")
            st.write(f"   Vessel center offset: {abs(z_center):.3f}m")
            
            # If the vessel isn't centered, adjust it
            if abs(z_center) > 0.01:  # More than 1cm off center
                st.warning(f"⚠️ Vessel not centered - adjusting by {-z_center:.3f}m")
                z_profile_m = z_profile_m - z_center
                z_min, z_max = np.min(z_profile_m), np.max(z_profile_m)
                st.write(f"   Adjusted Z range: {z_min:.3f}m to {z_max:.3f}m")
            
            # Sort profile for proper interpolation
            sort_indices = np.argsort(z_profile_m)
            z_sorted = z_profile_m[sort_indices]
            r_sorted = r_profile_m[sort_indices]
            
            # Create high-resolution surface mesh
            resolution = quality_settings.get('mandrel_resolution', 100)
            surface_segments = quality_settings.get('surface_segments', 48)
            
            # Generate circumferential coordinates
            theta = np.linspace(0, 2*np.pi, surface_segments)
            
            # Create smooth profile interpolation
            z_smooth = np.linspace(z_sorted[0], z_sorted[-1], resolution)
            r_smooth = np.interp(z_smooth, z_sorted, r_sorted)
            
            # Create surface mesh
            Z_mesh, Theta_mesh = np.meshgrid(z_smooth, theta)
            R_mesh = np.tile(r_smooth, (surface_segments, 1))
            
            # Convert to Cartesian coordinates
            X_mesh = R_mesh * np.cos(Theta_mesh)
            Y_mesh = R_mesh * np.sin(Theta_mesh)
            
            # Add main vessel surface
            fig.add_trace(go.Surface(
                x=X_mesh, y=Y_mesh, z=Z_mesh,
                colorscale='Greys',
                opacity=0.3,
                showscale=False,
                name='Mandrel Surface',
                hovertemplate=(
                    'Mandrel Surface<br>'
                    'X: %{x:.3f}m<br>'
                    'Y: %{y:.3f}m<br>'
                    'Z: %{z:.3f}m<br>'
                    'R: %{customdata:.3f}m<br>'
                    '<extra></extra>'
                ),
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
            self._add_centered_wireframe(fig, X_mesh, Y_mesh, Z_mesh, max(1, surface_segments//8))
            
            # Add coordinate system reference
            self._add_coordinate_reference(fig, z_min, z_max, np.max(r_smooth))
            
            # Add dome section highlights
            self._add_dome_section_highlights(fig, z_smooth, r_smooth, vessel_geometry)
            
            st.success(f"✅ Mandrel surface generated with centered coordinate system")
            st.write(f"   Profile points: {len(z_profile_m)}, Surface mesh: {resolution}×{surface_segments}")
            
        except Exception as e:
            st.error(f"Error generating centered mandrel surface: {e}")
            self._add_centered_mandrel_fallback(fig, vessel_geometry)
    
    def _add_coordinate_reference(self, fig, z_min, z_max, max_radius):
        """Add coordinate system reference markers"""
        try:
            # Add origin marker
            fig.add_trace(go.Scatter3d(
                x=[0], y=[0], z=[0],
                mode='markers',
                marker=dict(size=8, color='black', symbol='cross'),
                name='Origin (0,0,0)',
                hovertemplate='Origin (0,0,0)<extra></extra>'
            ))
            
            # Add Z-axis line
            fig.add_trace(go.Scatter3d(
                x=[0, 0], y=[0, 0], z=[z_min, z_max],
                mode='lines',
                line=dict(color='black', width=2, dash='dash'),
                name='Z-Axis',
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Add vessel center marker
            z_center = (z_min + z_max) / 2
            fig.add_trace(go.Scatter3d(
                x=[0], y=[0], z=[z_center],
                mode='markers',
                marker=dict(size=6, color='blue', symbol='circle'),
                name='Vessel Center',
                hovertemplate=f'Vessel Center<br>Z: {z_center:.3f}m<extra></extra>'
            ))
            
        except Exception:
            pass  # Skip reference if it fails
    
    def _add_dome_section_highlights(self, fig, z_profile, r_profile, vessel_geometry):
        """Add visual highlights for dome sections"""
        try:
            # Identify dome regions based on radius changes
            max_radius = np.max(r_profile)
            radius_threshold = max_radius * 0.95  # 95% of max radius
            
            # Find cylinder region (where radius is near maximum)
            cylinder_mask = r_profile >= radius_threshold
            if np.any(cylinder_mask):
                cylinder_indices = np.where(cylinder_mask)[0]
                z_cyl_start = z_profile[cylinder_indices[0]]
                z_cyl_end = z_profile[cylinder_indices[-1]]
                
                # Add cylinder region marker
                fig.add_trace(go.Scatter3d(
                    x=[max_radius, -max_radius], 
                    y=[0, 0], 
                    z=[z_cyl_start, z_cyl_start],
                    mode='lines',
                    line=dict(color='green', width=4),
                    name='Cylinder Start',
                    showlegend=False,
                    hovertemplate=f'Cylinder Region Start<br>Z: {z_cyl_start:.3f}m<extra></extra>'
                ))
                
                fig.add_trace(go.Scatter3d(
                    x=[max_radius, -max_radius], 
                    y=[0, 0], 
                    z=[z_cyl_end, z_cyl_end],
                    mode='lines',
                    line=dict(color='green', width=4),
                    name='Cylinder End',
                    showlegend=False,
                    hovertemplate=f'Cylinder Region End<br>Z: {z_cyl_end:.3f}m<extra></extra>'
                ))
                
                # Add dome region annotations
                z_min, z_max = np.min(z_profile), np.max(z_profile)
                
                if z_cyl_start > z_min:  # Aft dome exists
                    z_aft_dome_center = (z_min + z_cyl_start) / 2
                    fig.add_annotation(
                        x=0, y=max_radius*0.8, z=z_aft_dome_center,
                        text="Aft Dome",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor="red",
                        font=dict(size=12, color="red")
                    )
                
                if z_cyl_end < z_max:  # Forward dome exists
                    z_fwd_dome_center = (z_cyl_end + z_max) / 2
                    fig.add_annotation(
                        x=0, y=max_radius*0.8, z=z_fwd_dome_center,
                        text="Forward Dome",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor="red",
                        font=dict(size=12, color="red")
                    )
        except Exception:
            pass  # Skip highlights if they fail
    
    def _add_centered_wireframe(self, fig, X_mesh, Y_mesh, Z_mesh, step):
        """Add wireframe with centered coordinate system"""
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
            pass
    
    def _add_centered_mandrel_fallback(self, fig, vessel_geometry):
        """Add simple centered mandrel as fallback"""
        try:
            # Get vessel parameters
            inner_radius = vessel_geometry.inner_diameter / 2000  # Convert to meters
            cyl_length = vessel_geometry.cylindrical_length / 1000  # Convert to meters
            
            # Calculate dome height
            if hasattr(vessel_geometry, 'dome_type'):
                if vessel_geometry.dome_type == 'Hemispherical':
                    dome_height = inner_radius
                elif vessel_geometry.dome_type == 'Elliptical':
                    dome_height = inner_radius * getattr(vessel_geometry, 'aspect_ratio', 1.0)
                else:
                    dome_height = inner_radius * 0.8
            else:
                dome_height = inner_radius
            
            # **KEY FIX: CENTER THE VESSEL AT ORIGIN**
            # Cylinder section centered around Z=0
            z_cyl_start = -cyl_length / 2
            z_cyl_end = cyl_length / 2
            
            # Aft dome extends from z_cyl_start backwards
            z_aft_dome_start = z_cyl_start - dome_height
            
            # Forward dome extends from z_cyl_end forward  
            z_fwd_dome_end = z_cyl_end + dome_height
            
            st.write(f"🔧 **Fallback Coordinate System:**")
            st.write(f"   Aft dome: {z_aft_dome_start:.3f}m to {z_cyl_start:.3f}m")
            st.write(f"   Cylinder: {z_cyl_start:.3f}m to {z_cyl_end:.3f}m") 
            st.write(f"   Fwd dome: {z_cyl_end:.3f}m to {z_fwd_dome_end:.3f}m")
            
            # Generate surface coordinates
            theta = np.linspace(0, 2*np.pi, 32)
            
            # Cylinder section
            z_cyl = np.linspace(z_cyl_start, z_cyl_end, 20)
            Theta_cyl, Z_cyl = np.meshgrid(theta, z_cyl)
            X_cyl = inner_radius * np.cos(Theta_cyl)
            Y_cyl = inner_radius * np.sin(Theta_cyl)
            
            fig.add_trace(go.Surface(
                x=X_cyl, y=Y_cyl, z=Z_cyl,
                colorscale='Greys',
                opacity=0.3,
                showscale=False,
                name='Cylinder Section',
                hovertemplate='Cylinder<br>Z: %{z:.3f}m<extra></extra>'
            ))
            
            # Aft hemispherical dome
            phi = np.linspace(0, np.pi/2, 16)
            Theta_dome, Phi_dome = np.meshgrid(theta, phi)
            
            X_dome_aft = inner_radius * np.sin(Phi_dome) * np.cos(Theta_dome)
            Y_dome_aft = inner_radius * np.sin(Phi_dome) * np.sin(Theta_dome)
            Z_dome_aft = z_cyl_start - inner_radius * np.cos(Phi_dome)  # Extends backward from cylinder
            
            fig.add_trace(go.Surface(
                x=X_dome_aft, y=Y_dome_aft, z=Z_dome_aft,
                colorscale='Greys',
                opacity=0.3,
                showscale=False,
                name='Aft Dome',
                hovertemplate='Aft Dome<br>Z: %{z:.3f}m<extra></extra>',
                showlegend=False
            ))
            
            # Forward hemispherical dome
            Z_dome_fwd = z_cyl_end + inner_radius * np.cos(Phi_dome)  # Extends forward from cylinder
            
            fig.add_trace(go.Surface(
                x=X_dome_aft, y=Y_dome_aft, z=Z_dome_fwd,
                colorscale='Greys',
                opacity=0.3,
                showscale=False,
                name='Forward Dome',
                hovertemplate='Forward Dome<br>Z: %{z:.3f}m<extra></extra>',
                showlegend=False
            ))
            
            st.success("✅ Centered fallback mandrel generated")
            
        except Exception as e:
            st.error(f"Error in fallback mandrel: {e}")
    
    def _add_all_trajectory_circuits(self, fig, coverage_data, viz_options):
        """Add trajectory circuits (assumes they're already in correct coordinate system)"""
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
        
        # Check trajectory coordinate system
        all_z_coords = []
        for circuit_points in circuits:
            for p in circuit_points:
                if hasattr(p, 'position') and len(p.position) >= 3:
                    all_z_coords.append(p.position[2])
        
        if all_z_coords:
            z_traj_min, z_traj_max = min(all_z_coords), max(all_z_coords)
            z_traj_center = (z_traj_min + z_traj_max) / 2
            
            st.write(f"🎯 **Trajectory Coordinate Check:**")
            st.write(f"   Trajectory Z range: {z_traj_min:.3f}m to {z_traj_max:.3f}m")
            st.write(f"   Trajectory center: {z_traj_center:.3f}m")
            
            if abs(z_traj_center) > 0.05:  # More than 5cm off center
                st.warning(f"⚠️ Trajectory appears offset from mandrel center by {z_traj_center:.3f}m")
        
        # Add circuits with existing logic
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
                hovertemplate=f'<b>Circuit {circuit_num} Start</b><br>Z: {start_point.position[2]:.3f}m<br>Angle: {getattr(start_point, "winding_angle_deg", 45):.1f}°<extra></extra>'
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
                hovertemplate=f'<b>Circuit {circuit_num} End</b><br>Z: {end_point.position[2]:.3f}m<br>Angle: {getattr(end_point, "winding_angle_deg", 45):.1f}°<extra></extra>'
            ))
        except Exception:
            pass
    
    def _add_pattern_annotations(self, fig, coverage_data, layer_config):
        """Add pattern analysis annotations"""
        try:
            pattern_info = coverage_data.get('pattern_info', {})
            
            annotation_text = (
                f"<b>Full Coverage Pattern Analysis</b><br>"
                f"Coordinate System: Vessel Center at Origin<br>"
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
        """Configure layout with proper centered view"""
        try:
            total_points = sum(len(circuit) for circuit in coverage_data.get('circuits', []))
            
            fig.update_layout(
                title=dict(
                    text=f"Centered Coordinate System - {layer_config.get('winding_angle', 'N/A')}° Layer ({total_points:,} points)",
                    x=0.5,
                    font=dict(size=16, color='darkblue')
                ),
                scene=dict(
                    xaxis_title='X (m)',
                    yaxis_title='Y (m)', 
                    zaxis_title='Z (m) - Vessel Center at Origin',
                    aspectmode='data',
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.2),
                        center=dict(x=0, y=0, z=0),  # Center view on origin
                        up=dict(x=0, y=0, z=1)
                    ),
                    bgcolor='white',
                    xaxis=dict(showgrid=True, gridcolor='lightgray', gridwidth=1, zeroline=True, zerolinecolor='black', zerolinewidth=2),
                    yaxis=dict(showgrid=True, gridcolor='lightgray', gridwidth=1, zeroline=True, zerolinecolor='black', zerolinewidth=2),
                    zaxis=dict(showgrid=True, gridcolor='lightgray', gridwidth=1, zeroline=True, zerolinecolor='black', zerolinewidth=2)
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
            fig.update_layout(
                title="3D Visualization - Centered Coordinate System",
                scene=dict(aspectmode='data'),
                width=800,
                height=600
            )
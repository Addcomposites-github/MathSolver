"""
Advanced 3D Visualization Engine for Full Coverage Patterns
High-quality mandrel representation with complete trajectory visualization
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import math
import streamlit as st
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
        """Add mandrel surface with vessel center at origin (0,0,0)"""
        try:
            st.write("🔧 **Mandrel Surface Generation Debug:**")
            
            # Get vessel profile points (these should already be centered)
            profile = vessel_geometry.get_profile_points()
            if not profile or 'r_inner_mm' not in profile or 'z_mm' not in profile:
                st.error("❌ Vessel profile not available - using fallback mandrel")
                self._add_centered_mandrel_fallback(fig, vessel_geometry)
                return
            
            st.write(f"   ✅ Profile data retrieved: {len(profile['z_mm'])} points")
            
            # Convert profile data to arrays and meters
            z_profile_mm = np.array(profile['z_mm'])
            r_profile_mm = np.array(profile['r_inner_mm'])
            
            # Convert to meters for visualization
            z_profile_m = z_profile_mm / 1000.0
            r_profile_m = r_profile_mm / 1000.0
            
            # Debug the profile data
            st.write(f"   📊 Z range: {np.min(z_profile_m):.3f}m to {np.max(z_profile_m):.3f}m")
            st.write(f"   📊 R range: {np.min(r_profile_m):.3f}m to {np.max(r_profile_m):.3f}m")
            
            # Check for dome geometry
            r_variation = np.max(r_profile_m) - np.min(r_profile_m)
            st.write(f"   🔍 Radius variation: {r_variation*1000:.1f}mm (dome indicator)")
            
            if r_variation < 0.001:  # Less than 1mm variation
                st.warning("   ⚠️ Minimal radius variation detected - may appear cylindrical")
            
            # Verify coordinate system
            z_min, z_max = np.min(z_profile_m), np.max(z_profile_m)
            z_center = (z_min + z_max) / 2
            
            # If the vessel isn't centered, adjust it
            if abs(z_center) > 0.01:  # More than 1cm off center
                st.warning(f"Vessel not centered - adjusting by {-z_center:.3f}m")
                z_profile_m = z_profile_m - z_center
                z_min, z_max = np.min(z_profile_m), np.max(z_profile_m)
            
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
            self._add_mandrel_wireframe(fig, X_mesh, Y_mesh, Z_mesh, max(1, surface_segments//8))
            
            # Add coordinate system reference
            self._add_coordinate_reference(fig, z_min, z_max, np.max(r_smooth))
            
            st.success(f"   ✅ Mandrel surface generated successfully")
            st.write(f"   📊 Surface mesh: {resolution}×{surface_segments} points")
            
        except Exception as e:
            st.error(f"❌ Error generating mandrel surface: {str(e)}")
            self._add_centered_mandrel_fallback(fig, vessel_geometry)
    
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
            dome_height = inner_radius * (0.3 + 0.1 * q_factor / 10.0)
        else:
            dome_height = inner_radius * 0.8  # Default
        
        # Create z-coordinate array for complete vessel
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
        z_complete = np.concatenate([z_aft_dome[:-1], z_cylinder[:-1], z_fwd_dome])
        r_complete = np.concatenate([r_aft_dome[:-1], r_cylinder[:-1], r_fwd_dome])
        
        return z_complete, r_complete
    
    def _calculate_dome_radius_profile(self, z_array, z_start, z_end, max_radius, dome_type, vessel_geometry):
        """Calculate radius profile for dome section"""
        
        dome_length = z_end - z_start
        if dome_length == 0:
            return np.full_like(z_array, max_radius)
        
        # Parameter along dome (0 to 1)
        t = (z_array - z_start) / dome_length
        
        if dome_type == 'Hemispherical':
            # Hemispherical dome
            center_z = (z_start + z_end) / 2
            dome_radius = dome_length / 2
            z_rel = z_array - center_z
            r_profile = np.sqrt(np.maximum(0, dome_radius**2 - z_rel**2))
            r_profile = r_profile * (max_radius / dome_radius)
            
        elif dome_type == 'Elliptical':
            # Elliptical dome
            aspect_ratio = getattr(vessel_geometry, 'aspect_ratio', 1.0)
            a = dome_length / 2
            b = max_radius
            center_z = (z_start + z_end) / 2
            z_rel = z_array - center_z
            r_profile = b * np.sqrt(np.maximum(0, 1 - (z_rel / a)**2))
            
        elif dome_type == 'Isotensoid':
            # Isotensoid dome with qrs parameters
            q_factor = getattr(vessel_geometry, 'q_factor', 9.5)
            r_factor = getattr(vessel_geometry, 'r_factor', 0.1)
            s_factor = getattr(vessel_geometry, 's_factor', 0.5)
            
            # Simplified isotensoid profile
            polar_opening_ratio = 0.1 + 0.05 * r_factor
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
        except Exception:
            pass  # Skip reference if it fails
    
    def _add_centered_mandrel_fallback(self, fig, vessel_geometry):
        """Add simple centered mandrel as fallback"""
        try:
            # Get vessel parameters
            inner_radius = vessel_geometry.inner_diameter / 2000  # Convert to meters
            cyl_length = vessel_geometry.cylindrical_length / 1000  # Convert to meters
            
            # Calculate dome height
            dome_height = inner_radius * 0.8  # Default dome height
            
            # CENTER THE VESSEL AT ORIGIN
            # Cylinder section centered around Z=0
            z_cyl_half = cyl_length / 2.0
            z_cyl_start = -z_cyl_half
            z_cyl_end = +z_cyl_half
            
            # Aft dome: extends backward from cylinder
            z_aft_start = z_cyl_start - dome_height
            z_aft_end = z_cyl_start
            
            # Forward dome: extends forward from cylinder
            z_fwd_start = z_cyl_end
            z_fwd_end = z_cyl_end + dome_height
            
            # Create circumferential coordinates
            theta = np.linspace(0, 2*np.pi, 32)
            
            # Cylinder section
            z_cyl = np.linspace(z_cyl_start, z_cyl_end, 20)
            Theta_cyl, Z_cyl = np.meshgrid(theta, z_cyl)
            X_cyl = inner_radius * np.cos(Theta_cyl)
            Y_cyl = inner_radius * np.sin(Theta_cyl)
            
            # Add cylinder surface
            fig.add_trace(go.Surface(
                x=X_cyl, y=Y_cyl, z=Z_cyl,
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
"""
Fixed Advanced 3D Visualizer - Coordinate System Alignment
Solves trajectory truncation by maintaining consistent coordinate systems
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import streamlit as st

class FixedAdvanced3DVisualizer:
    """Fixed Advanced 3D visualization with proper coordinate handling"""
    
    def __init__(self):
        self.colors = px.colors.qualitative.Set3
    
    def create_full_coverage_visualization(self, coverage_data, vessel_geometry, layer_config, visualization_options=None):
        """Create 3D visualization with consistent coordinate systems"""
        
        if not visualization_options:
            visualization_options = {
                'show_mandrel': True,
                'mandrel_opacity': 0.3,
                'circuit_line_width': 4,
                'color_by_circuit': True,
                'show_start_end_points': True
            }
        
        fig = go.Figure()
        
        # Step 1: Determine if trajectory data is already coordinate-aligned
        circuits = coverage_data.get('circuits', [])
        coordinate_system_info = self._analyze_coordinate_systems(circuits, vessel_geometry)
        
        st.write("🔍 **Coordinate System Analysis:**")
        st.write(f"   - Vessel Z range: {coordinate_system_info['vessel_z_range']}")
        st.write(f"   - Trajectory Z range: {coordinate_system_info['traj_z_range']}")
        st.write(f"   - Alignment needed: {coordinate_system_info['needs_alignment']}")
        
        # Step 2: Add mandrel surface (NO CENTERING if trajectory is pre-aligned)
        if visualization_options.get('show_mandrel', True):
            success = self._add_mandrel_surface_fixed(
                fig, vessel_geometry, 
                apply_centering=coordinate_system_info['needs_alignment'],
                coordinate_info=coordinate_system_info
            )
            if success:
                st.write("✅ Mandrel surface added")
            else:
                st.warning("⚠️ Mandrel surface could not be added")
        
        # Step 3: Add trajectory circuits with consistent coordinates
        success = self._add_trajectory_circuits_fixed(
            fig, coverage_data, visualization_options, coordinate_system_info
        )
        if success:
            st.write("✅ Trajectory circuits added")
        else:
            st.warning("⚠️ Trajectory circuits could not be added")
        
        # Step 4: Configure layout
        self._configure_layout(fig, coverage_data, layer_config)
        
        return fig
    
    def _analyze_coordinate_systems(self, circuits, vessel_geometry):
        """Analyze coordinate systems to determine alignment strategy"""
        
        # Get vessel geometry range
        try:
            profile = vessel_geometry.get_profile_points()
            vessel_z_m = np.array(profile['z_mm']) / 1000.0
            vessel_z_min, vessel_z_max = np.min(vessel_z_m), np.max(vessel_z_m)
            vessel_z_center = (vessel_z_min + vessel_z_max) / 2
            vessel_z_range = f"{vessel_z_min:.3f}m to {vessel_z_max:.3f}m"
        except:
            return {'needs_alignment': False, 'vessel_z_range': 'Unknown', 'traj_z_range': 'Unknown'}
        
        # Get trajectory range
        if circuits and len(circuits) > 0 and len(circuits[0]) > 0:
            try:
                first_circuit = circuits[0]
                z_coords = []
                
                for point in first_circuit:
                    if isinstance(point, dict) and 'z_m' in point:
                        z_coords.append(point['z_m'])
                
                if z_coords:
                    traj_z_min, traj_z_max = np.min(z_coords), np.max(z_coords)
                    traj_z_center = (traj_z_min + traj_z_max) / 2
                    traj_z_range = f"{traj_z_min:.3f}m to {traj_z_max:.3f}m"
                    
                    # Check if coordinates are already aligned (trajectory center near vessel center)
                    center_diff = abs(traj_z_center - vessel_z_center)
                    needs_alignment = center_diff > 0.01  # 1cm tolerance
                    
                    return {
                        'needs_alignment': needs_alignment,
                        'vessel_z_range': vessel_z_range,
                        'traj_z_range': traj_z_range,
                        'vessel_z_center': vessel_z_center,
                        'traj_z_center': traj_z_center,
                        'center_diff': center_diff
                    }
            except Exception as e:
                st.write(f"Error analyzing trajectory coordinates: {e}")
        
        return {'needs_alignment': False, 'vessel_z_range': vessel_z_range, 'traj_z_range': 'No trajectory data'}
    
    def _add_mandrel_surface_fixed(self, fig, vessel_geometry, apply_centering=True, coordinate_info=None):
        """Add mandrel surface with optional centering"""
        try:
            profile = vessel_geometry.get_profile_points()
            if not profile or 'r_inner_mm' not in profile:
                st.error("No vessel profile data available")
                return False
            
            # Convert to meters
            z_profile_m = np.array(profile['z_mm']) / 1000.0
            r_profile_m = np.array(profile['r_inner_mm']) / 1000.0
            
            # Apply centering only if requested
            if apply_centering:
                z_center = (np.min(z_profile_m) + np.max(z_profile_m)) / 2
                z_profile_m = z_profile_m - z_center
                st.write(f"Applied centering: offset = {z_center:.3f}m")
            else:
                st.write("No centering applied - using absolute coordinates")
            
            st.write(f"Final vessel Z range: {np.min(z_profile_m):.3f}m to {np.max(z_profile_m):.3f}m")
            
            # Create surface mesh
            theta = np.linspace(0, 2*np.pi, 32)
            Z_mesh, Theta_mesh = np.meshgrid(z_profile_m, theta)
            R_mesh = np.tile(r_profile_m, (32, 1))
            X_mesh = R_mesh * np.cos(Theta_mesh)
            Y_mesh = R_mesh * np.sin(Theta_mesh)
            
            fig.add_trace(go.Surface(
                x=X_mesh, y=Y_mesh, z=Z_mesh,
                colorscale='Greys',
                opacity=0.3,
                showscale=False,
                name='Mandrel Surface',
                hovertemplate='Mandrel Surface<br>R: %{customdata:.3f}m<extra></extra>',
                customdata=R_mesh
            ))
            
            return True
            
        except Exception as e:
            st.error(f"Mandrel surface error: {e}")
            return False
    
    def _add_trajectory_circuits_fixed(self, fig, coverage_data, viz_options, coordinate_info):
        """Add trajectory circuits with coordinate system consistency"""
        
        circuits = coverage_data.get('circuits', [])
        metadata = coverage_data.get('metadata', [])
        
        st.write(f"Processing {len(circuits)} trajectory circuits...")
        
        if not circuits:
            st.error("No trajectory circuits found")
            return False
        
        trajectory_added = False
        
        for i, circuit_points in enumerate(circuits):
            if not circuit_points:
                continue
            
            try:
                # Extract coordinates
                x_coords = []
                y_coords = []
                z_coords = []
                
                for point in circuit_points:
                    if isinstance(point, dict) and 'x_m' in point and 'y_m' in point and 'z_m' in point:
                        x_coords.append(point['x_m'])
                        y_coords.append(point['y_m'])
                        z_coords.append(point['z_m'])
                
                if not x_coords:
                    st.warning(f"Circuit {i+1}: No valid coordinates found")
                    continue
                
                st.write(f"Circuit {i+1}: {len(x_coords)} points")
                st.write(f"   Z range: {min(z_coords):.3f}m to {max(z_coords):.3f}m")
                
                # Apply same coordinate transformation as vessel if needed
                if coordinate_info.get('needs_alignment', False):
                    # If vessel was centered, center trajectory too
                    vessel_z_center = coordinate_info.get('vessel_z_center', 0)
                    z_coords = [z - vessel_z_center for z in z_coords]
                    st.write(f"   Applied trajectory centering: offset = {vessel_z_center:.3f}m")
                    st.write(f"   Adjusted Z range: {min(z_coords):.3f}m to {max(z_coords):.3f}m")
                
                # Color assignment
                color = self.colors[i % len(self.colors)]
                
                # Add circuit trajectory
                fig.add_trace(go.Scatter3d(
                    x=x_coords, y=y_coords, z=z_coords,
                    mode='lines+markers',
                    line=dict(color=color, width=viz_options.get('circuit_line_width', 4)),
                    marker=dict(size=3, color=color),
                    name=f"Circuit {i+1} ({len(x_coords)} pts)",
                    hovertemplate=(
                        f'<b>Circuit {i+1}</b><br>'
                        'X: %{x:.3f}m<br>'
                        'Y: %{y:.3f}m<br>'
                        'Z: %{z:.3f}m<br>'
                        '<extra></extra>'
                    ),
                    showlegend=True
                ))
                
                # Add start/end markers
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
        """Configure 3D layout"""
        
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
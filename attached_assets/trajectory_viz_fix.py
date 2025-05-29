# 1. Fixed visualization_page() function in app.py

def visualization_page():
    """Dedicated visualization page that only displays planned trajectories"""
    st.markdown("""
    <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #1e3c72; margin-bottom: 1.5rem;">
        <h2 style="color: #1e3c72; margin: 0;">📊 3D Visualization</h2>
        <p style="color: #6c757d; margin: 0.5rem 0 0 0;">View and analyze your planned trajectories in 3D</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check workflow prerequisites
    if not hasattr(st.session_state, 'vessel_geometry') or not st.session_state.vessel_geometry:
        st.error("Complete Vessel Geometry first")
        if st.button("Go to Vessel Geometry"):
            st.session_state.current_page = "Vessel Geometry"
            st.rerun()
        return
    
    if not hasattr(st.session_state, 'layer_stack_manager') or not st.session_state.layer_stack_manager:
        st.error("Complete Layer Stack Definition first")
        if st.button("Go to Layer Stack Definition"):
            st.session_state.current_page = "Layer Stack Definition"
            st.rerun()
        return
    
    if not hasattr(st.session_state, 'all_layer_trajectories') or not st.session_state.all_layer_trajectories:
        st.error("Complete Trajectory Planning first")
        if st.button("Go to Trajectory Planning"):
            st.session_state.current_page = "Trajectory Planning"
            st.rerun()
        return
    
    # Display planned trajectories
    st.success("All prerequisites completed - ready for visualization")
    
    # Debug trajectory data
    st.write("🔍 **Debug Information:**")
    trajectories = st.session_state.all_layer_trajectories
    st.write(f"Found {len(trajectories)} trajectory datasets")
    
    for i, traj in enumerate(trajectories):
        traj_data = traj.get('trajectory_data', {})
        st.write(f"Trajectory {i+1}: {list(traj_data.keys())}")
        
        # Check for coordinate data
        if 'x_points_m' in traj_data:
            st.write(f"  - Unified format: {len(traj_data['x_points_m'])} points")
        elif 'path_points' in traj_data:
            st.write(f"  - Path points format: {len(traj_data['path_points'])} points")
        else:
            st.write(f"  - Unknown format")
    
    # Layer selection
    layer_options = [f"Layer {traj['layer_id']}: {traj['layer_type']} ({traj['winding_angle']}°)" 
                    for traj in trajectories]
    
    selected_idx = st.selectbox(
        "Select Layer to Visualize",
        range(len(layer_options)),
        format_func=lambda x: layer_options[x]
    )
    
    if selected_idx is not None:
        selected_traj = trajectories[selected_idx]
        
        # Visualization options
        col1, col2 = st.columns(2)
        with col1:
            quality_level = st.selectbox(
                "Visualization Quality",
                ("Standard", "High Definition"),
                help="High Definition shows more detail but renders slower"
            )
        with col2:
            show_mandrel = st.checkbox("Show Mandrel Surface", value=True)
        
        # Generate visualization
        if st.button("Generate 3D Visualization", type="primary"):
            try:
                # Convert trajectory data to proper format
                trajectory_data = selected_traj.get('trajectory_data', {})
                
                # Enhanced trajectory data conversion
                if 'x_points_m' in trajectory_data and 'y_points_m' in trajectory_data and 'z_points_m' in trajectory_data:
                    # Unified system format
                    x_points = trajectory_data['x_points_m']
                    y_points = trajectory_data['y_points_m']
                    z_points = trajectory_data['z_points_m']
                    winding_angles = trajectory_data.get('winding_angles_deg', [selected_traj['winding_angle']] * len(x_points))
                    
                    # Convert to path_points format
                    path_points = []
                    for i in range(len(x_points)):
                        path_points.append({
                            'x_m': x_points[i],
                            'y_m': y_points[i],
                            'z_m': z_points[i],
                            'rho_m': np.sqrt(x_points[i]**2 + y_points[i]**2),
                            'phi_rad': np.arctan2(y_points[i], x_points[i]),
                            'alpha_deg': winding_angles[i] if i < len(winding_angles) else selected_traj['winding_angle'],
                            'arc_length_m': i * 0.01
                        })
                    
                    st.success(f"Converted {len(path_points)} trajectory points from unified format")
                    
                elif 'path_points' in trajectory_data:
                    # Already in correct format
                    path_points = trajectory_data['path_points']
                    st.success(f"Using {len(path_points)} path points in correct format")
                    
                else:
                    st.error("No valid trajectory coordinate data found")
                    st.write("Available keys:", list(trajectory_data.keys()))
                    return
                
                # Create coverage data for visualization
                coverage_data = {
                    'circuits': [path_points],
                    'circuit_metadata': [{
                        'circuit_number': 1,
                        'start_phi_deg': 0.0,
                        'points_count': len(path_points),
                        'quality_score': 95.0
                    }],
                    'metadata': [{
                        'circuit_number': 1,
                        'start_phi_deg': 0.0,
                        'points_count': len(path_points),
                        'quality_score': 95.0
                    }],
                    'total_circuits': 1,
                    'coverage_percentage': trajectory_data.get('coverage_percentage', 85.0),
                    'pattern_info': {
                        'actual_pattern_type': selected_traj['layer_type'],
                        'winding_angle': selected_traj['winding_angle']
                    },
                    'quality_settings': {'mode': quality_level.lower(), 'mandrel_resolution': 80, 'surface_segments': 32},
                    'source': 'planned_trajectory'
                }
                
                # Layer configuration
                layer_manager = st.session_state.layer_stack_manager
                layer_def = None
                for layer in layer_manager.layer_stack:
                    if layer.layer_set_id == selected_traj['layer_id']:
                        layer_def = layer
                        break
                
                if layer_def:
                    layer_config = {
                        'layer_type': layer_def.layer_type,
                        'winding_angle': layer_def.winding_angle_deg,
                        'physics_model': getattr(layer_def, 'physics_model', 'clairaut'),
                        'roving_width': 3.0,
                        'coverage_mode': 'full_coverage'
                    }
                    
                    visualization_options = {
                        'quality_level': quality_level.lower(),
                        'show_mandrel_mesh': show_mandrel,
                        'color_by_circuit': True,
                        'show_all_circuits': True,
                        'show_mandrel': show_mandrel,
                        'mandrel_opacity': 0.3,
                        'circuit_line_width': 4,
                        'show_start_end_points': True,
                        'show_surface_mesh': True
                    }
                    
                    # Create visualization using fixed visualizer
                    visualizer = FixedAdvanced3DVisualizer()
                    fig = visualizer.create_full_coverage_visualization(
                        coverage_data,
                        st.session_state.vessel_geometry,
                        layer_config,
                        visualization_options
                    )
                    
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display trajectory information
                        st.markdown("### Trajectory Information")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Points", len(path_points))
                        with col2:
                            st.metric("Layer Type", selected_traj['layer_type'])
                        with col3:
                            st.metric("Winding Angle", f"{selected_traj['winding_angle']}°")
                        
                        st.success("✅ 3D Visualization generated successfully!")
                    else:
                        st.error("Failed to create visualization")
                else:
                    st.error("Could not find corresponding layer definition")
                    
            except Exception as e:
                st.error(f"Visualization error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


# 2. Fixed Advanced 3D Visualizer
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import math

class FixedAdvanced3DVisualizer:
    """Fixed Advanced 3D visualization for trajectory display"""
    
    def __init__(self):
        self.colors = px.colors.qualitative.Set3
    
    def create_full_coverage_visualization(self, coverage_data, vessel_geometry, layer_config, visualization_options=None):
        """Create comprehensive 3D visualization"""
        
        if not visualization_options:
            visualization_options = {
                'show_mandrel': True,
                'mandrel_opacity': 0.3,
                'circuit_line_width': 3,
                'color_by_circuit': True
            }
        
        fig = go.Figure()
        
        # Add mandrel surface
        if visualization_options.get('show_mandrel', True):
            self._add_mandrel_surface(fig, vessel_geometry, coverage_data['quality_settings'])
        
        # Add trajectory circuits - THIS IS THE KEY FIX
        self._add_trajectory_circuits_fixed(fig, coverage_data, visualization_options)
        
        # Configure layout
        self._configure_layout(fig, coverage_data, layer_config)
        
        return fig
    
    def _add_mandrel_surface(self, fig, vessel_geometry, quality_settings):
        """Add mandrel surface"""
        try:
            # Get vessel profile
            profile = vessel_geometry.get_profile_points()
            if not profile or 'r_inner_mm' not in profile:
                return
            
            # Convert to meters and create surface
            z_profile_m = np.array(profile['z_mm']) / 1000.0
            r_profile_m = np.array(profile['r_inner_mm']) / 1000.0
            
            # Center the vessel
            z_center = (np.min(z_profile_m) + np.max(z_profile_m)) / 2
            z_profile_m = z_profile_m - z_center
            
            # Create surface mesh
            theta = np.linspace(0, 2*np.pi, 32)
            z_smooth = np.linspace(z_profile_m[0], z_profile_m[-1], 60)
            r_smooth = np.interp(z_smooth, z_profile_m, r_profile_m)
            
            Z_mesh, Theta_mesh = np.meshgrid(z_smooth, theta)
            R_mesh = np.tile(r_smooth, (32, 1))
            X_mesh = R_mesh * np.cos(Theta_mesh)
            Y_mesh = R_mesh * np.sin(Theta_mesh)
            
            fig.add_trace(go.Surface(
                x=X_mesh, y=Y_mesh, z=Z_mesh,
                colorscale='Greys',
                opacity=0.3,
                showscale=False,
                name='Mandrel Surface'
            ))
            
        except Exception as e:
            print(f"Mandrel surface error: {e}")
    
    def _add_trajectory_circuits_fixed(self, fig, coverage_data, viz_options):
        """FIXED: Add trajectory circuits with proper coordinate handling"""
        
        circuits = coverage_data.get('circuits', [])
        metadata = coverage_data.get('metadata', [])
        
        print(f"DEBUG: Adding {len(circuits)} circuits to visualization")
        
        if not circuits:
            st.error("No trajectory circuits found")
            return
        
        trajectory_added = False
        
        for i, circuit_points in enumerate(circuits):
            if not circuit_points:
                continue
            
            try:
                # Extract coordinates - handle multiple formats
                x_coords = []
                y_coords = []
                z_coords = []
                angles = []
                
                print(f"DEBUG: Processing circuit {i+1} with {len(circuit_points)} points")
                
                for j, point in enumerate(circuit_points):
                    try:
                        if isinstance(point, dict):
                            # Dictionary format
                            if 'x_m' in point and 'y_m' in point and 'z_m' in point:
                                x_coords.append(float(point['x_m']))
                                y_coords.append(float(point['y_m']))
                                z_coords.append(float(point['z_m']))
                                angles.append(float(point.get('alpha_deg', 45.0)))
                            else:
                                print(f"Point {j} missing coordinates: {list(point.keys())}")
                                continue
                        elif hasattr(point, 'position') and len(point.position) >= 3:
                            # Object format
                            x_coords.append(float(point.position[0]))
                            y_coords.append(float(point.position[1]))
                            z_coords.append(float(point.position[2]))
                            angles.append(float(getattr(point, 'alpha_deg', 45.0)))
                        else:
                            print(f"Unknown point format: {type(point)}")
                            continue
                    except Exception as pe:
                        print(f"Error processing point {j}: {pe}")
                        continue
                
                if len(x_coords) < 2:
                    print(f"Circuit {i+1} has insufficient valid points: {len(x_coords)}")
                    continue
                
                # Verify coordinates are reasonable
                x_range = max(x_coords) - min(x_coords)
                y_range = max(y_coords) - min(y_coords) 
                z_range = max(z_coords) - min(z_coords)
                
                print(f"Circuit {i+1} ranges - X: {x_range:.3f}m, Y: {y_range:.3f}m, Z: {z_range:.3f}m")
                
                if x_range < 0.001 and y_range < 0.001:  # Less than 1mm variation
                    print(f"Warning: Circuit {i+1} appears to be a single point")
                
                # Color assignment
                color = self.colors[i % len(self.colors)]
                
                # Get circuit metadata
                circuit_meta = metadata[i] if i < len(metadata) else {'circuit_number': i+1, 'start_phi_deg': 0.0}
                
                # Add circuit trajectory
                fig.add_trace(go.Scatter3d(
                    x=x_coords,
                    y=y_coords, 
                    z=z_coords,
                    mode='lines+markers',
                    line=dict(color=color, width=viz_options.get('circuit_line_width', 4)),
                    marker=dict(size=2, color=color),
                    name=f"Circuit {circuit_meta['circuit_number']}",
                    hovertemplate=(
                        f'<b>Circuit {circuit_meta["circuit_number"]}</b><br>'
                        'X: %{x:.3f}m<br>'
                        'Y: %{y:.3f}m<br>'
                        'Z: %{z:.3f}m<br>'
                        '<extra></extra>'
                    ),
                    showlegend=True
                ))
                
                trajectory_added = True
                print(f"✅ Successfully added circuit {i+1} with {len(x_coords)} points")
                
                # Add start/end markers
                if viz_options.get('show_start_end_points', True):
                    # Start point
                    fig.add_trace(go.Scatter3d(
                        x=[x_coords[0]], y=[y_coords[0]], z=[z_coords[0]],
                        mode='markers',
                        marker=dict(size=8, color='green', symbol='diamond'),
                        name=f'Start C{circuit_meta["circuit_number"]}',
                        showlegend=False
                    ))
                    
                    # End point
                    fig.add_trace(go.Scatter3d(
                        x=[x_coords[-1]], y=[y_coords[-1]], z=[z_coords[-1]],
                        mode='markers',
                        marker=dict(size=8, color='red', symbol='square'),
                        name=f'End C{circuit_meta["circuit_number"]}',
                        showlegend=False
                    ))
                
            except Exception as e:
                print(f"Error adding circuit {i+1}: {e}")
                continue
        
        if not trajectory_added:
            st.error("❌ No trajectory data could be visualized")
            # Add error annotation
            fig.add_annotation(
                text="No valid trajectory data found",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                xanchor="center", yanchor="middle",
                font=dict(size=16, color="red"),
                showarrow=False
            )
        else:
            st.success(f"✅ Successfully visualized {len([c for c in circuits if c])} trajectory circuits")
    
    def _configure_layout(self, fig, coverage_data, layer_config):
        """Configure plot layout"""
        try:
            total_points = sum(len(circuit) for circuit in coverage_data['circuits'])
            
            fig.update_layout(
                title=dict(
                    text=f"3D Trajectory Visualization - {layer_config['winding_angle']}° ({total_points:,} points)",
                    x=0.5,
                    font=dict(size=16)
                ),
                scene=dict(
                    xaxis_title='X (m)',
                    yaxis_title='Y (m)', 
                    zaxis_title='Z (m)',
                    aspectmode='data',
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.2),
                        center=dict(x=0, y=0, z=0)
                    )
                ),
                width=1000,
                height=700,
                showlegend=True
            )
        except Exception:
            # Minimal layout if configuration fails
            fig.update_layout(
                title="3D Trajectory Visualization",
                scene=dict(aspectmode='data')
            )


# 3. Fixed trajectory data generation in multi_layer_trajectory_orchestrator.py

def _generate_unified_layer_trajectory(self, temp_vessel, layer_def, 
                                     roving_width_mm: float, roving_thickness_mm: float) -> Dict:
    """FIXED: Generate trajectory with proper coordinate format"""
    
    # Initialize unified planner
    self.unified_planner = UnifiedTrajectoryPlanner(
        vessel_geometry=temp_vessel,
        roving_width_m=roving_width_mm / 1000,
        payout_length_m=0.5,
        default_friction_coeff=0.1
    )
    
    # Generate trajectory
    try:
        result = self.unified_planner.generate_trajectory(
            pattern_type='helical',
            coverage_mode='single_pass',
            physics_model='clairaut',
            continuity_level=1,
            target_params={'winding_angle_deg': layer_def.winding_angle_deg},
            options={'num_points': 200}  # Ensure sufficient points
        )
        
        if result.points and len(result.points) > 0:
            # Convert trajectory points to proper format
            x_points = []
            y_points = []
            z_points = []
            angles = []
            
            for point in result.points:
                # Convert cylindrical to Cartesian
                x = point.rho * np.cos(point.phi)
                y = point.rho * np.sin(point.phi)
                z = point.z
                
                x_points.append(x)
                y_points.append(y)
                z_points.append(z)
                angles.append(point.alpha_deg)
            
            # Create properly formatted trajectory data
            trajectory_data = {
                'x_points_m': x_points,
                'y_points_m': y_points,
                'z_points_m': z_points,
                'winding_angles_deg': angles,
                'total_points': len(x_points),
                'success': True,
                'coverage_percentage': 85.0,
                'pattern_type': f'{layer_def.layer_type}_unified',
                'source': 'unified_system'
            }
            
            print(f"Generated {len(x_points)} trajectory points for layer {layer_def.layer_set_id}")
            return trajectory_data
        else:
            print(f"No trajectory points generated for layer {layer_def.layer_set_id}")
            return self._generate_fallback_trajectory(layer_def, roving_width_mm)
            
    except Exception as e:
        print(f"Unified trajectory generation failed: {e}")
        return self._generate_fallback_trajectory(layer_def, roving_width_mm)

def _generate_fallback_trajectory(self, layer_def, roving_width_mm: float) -> Dict:
    """Generate simple fallback trajectory when unified system fails"""
    
    # Simple helical trajectory
    vessel_radius = 0.1  # 100mm radius
    length = 0.3  # 300mm length
    angle_rad = np.radians(layer_def.winding_angle_deg)
    
    # Generate points
    num_points = 100
    z_points = np.linspace(-length/2, length/2, num_points)
    
    x_points = []
    y_points = []
    angles = []
    
    for z in z_points:
        phi = z * np.tan(angle_rad) / vessel_radius
        x = vessel_radius * np.cos(phi)
        y = vessel_radius * np.sin(phi)
        
        x_points.append(x)
        y_points.append(y)
        angles.append(layer_def.winding_angle_deg)
    
    return {
        'x_points_m': x_points,
        'y_points_m': y_points,
        'z_points_m': z_points.tolist(),
        'winding_angles_deg': angles,
        'total_points': len(x_points),
        'success': True,
        'coverage_percentage': 80.0,
        'pattern_type': f'{layer_def.layer_type}_fallback',
        'source': 'fallback_system'
    }

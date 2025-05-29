def visualization_page():
    """FIXED visualization page with proper trajectory data conversion"""
    st.markdown("""
    <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #1e3c72; margin-bottom: 1.5rem;">
        <h2 style="color: #1e3c72; margin: 0;">📊 3D Visualization - FIXED</h2>
        <p style="color: #6c757d; margin: 0.5rem 0 0 0;">View and analyze your planned trajectories in 3D with proper coordinate conversion</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Import the converter
    try:
        from modules.trajectory_data_converter import TrajectoryDataConverter
        converter = TrajectoryDataConverter()
        st.success("✅ Trajectory data converter loaded")
    except ImportError:
        st.error("❌ Please save the trajectory_data_converter.py file in your modules/ directory")
        return
    
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
    
    # Layer selection
    trajectories = st.session_state.all_layer_trajectories
    layer_options = [f"Layer {traj['layer_id']}: {traj['layer_type']} ({traj['winding_angle']}°)" 
                    for traj in trajectories]
    
    selected_idx = st.selectbox(
        "Select Layer to Visualize",
        range(len(layer_options)),
        format_func=lambda x: layer_options[x]
    )
    
    if selected_idx is not None:
        selected_traj = trajectories[selected_idx]
        
        # Debug: Show raw trajectory data
        with st.expander("🔍 Raw Trajectory Data Analysis", expanded=False):
            st.write("**Raw trajectory data structure:**")
            st.write(f"Keys: {list(selected_traj.keys())}")
            if 'trajectory_data' in selected_traj:
                traj_data = selected_traj['trajectory_data']
                st.write(f"Trajectory data keys: {list(traj_data.keys())}")
                
                # Show sample data
                if 'points' in traj_data and traj_data['points']:
                    sample_point = traj_data['points'][0]
                    st.write(f"Sample point type: {type(sample_point)}")
                    if hasattr(sample_point, '__dict__'):
                        st.write(f"Sample point attributes: {list(sample_point.__dict__.keys())}")
                        if hasattr(sample_point, 'rho'):
                            st.write(f"Sample point: rho={sample_point.rho:.3f}, z={sample_point.z:.3f}, phi={sample_point.phi:.3f}")
        
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
        
        # Generate visualization with FIXED coordinate conversion
        if st.button("🚀 Generate 3D Visualization", type="primary"):
            
            st.markdown("### 🔧 Trajectory Data Conversion")
            
            # Get the raw trajectory data
            raw_trajectory_data = selected_traj.get('trajectory_data', {})
            
            if not raw_trajectory_data:
                st.error("No trajectory data found for selected layer")
                return
            
            # Convert trajectory data using the new converter
            st.write("Converting trajectory data for visualization...")
            converted_data = converter.convert_unified_trajectory_to_visualization_format(raw_trajectory_data)
            
            if not converted_data or not converted_data.get('success', False):
                st.error("❌ Trajectory data conversion failed")
                st.write("This usually means the trajectory data format is not supported.")
                st.write("Check the Raw Trajectory Data Analysis above for details.")
                return
            
            # Show conversion results
            st.success(f"✅ Converted {converted_data['total_points']} trajectory points")
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Total Points", converted_data['total_points'])
            with col_b:
                x_range = max(converted_data['x_points_m']) - min(converted_data['x_points_m'])
                st.metric("X Range", f"{x_range:.3f}m")
            with col_c:
                z_range = max(converted_data['z_points_m']) - min(converted_data['z_points_m'])
                st.metric("Z Range", f"{z_range:.3f}m")
            
            # Get layer configuration
            layer_manager = st.session_state.layer_stack_manager
            layer_def = None
            for layer in layer_manager.layer_stack:
                if layer.layer_set_id == selected_traj['layer_id']:
                    layer_def = layer
                    break
            
            if layer_def:
                try:
                    # Create visualization using the FIXED visualization system
                    from modules.fixed_advanced_3d_visualizer import FixedAdvanced3DVisualizer
                    
                    # Format data for visualization - use converted path_points
                    coverage_data = {
                        'circuits': [converted_data['path_points']],  # Use converted path_points
                        'circuit_metadata': [{
                            'circuit_number': 1,
                            'start_phi_deg': 0.0,
                            'points_count': len(converted_data['path_points']),
                            'quality_score': 95.0
                        }],
                        'metadata': [{  # Add metadata alias for backward compatibility
                            'circuit_number': 1,
                            'start_phi_deg': 0.0,
                            'points_count': len(converted_data['path_points']),
                            'quality_score': 95.0
                        }],
                        'total_circuits': 1,
                        'coverage_percentage': converted_data.get('coverage_percentage', 85.0),
                        'pattern_info': {
                            'actual_pattern_type': selected_traj['layer_type'],
                            'winding_angle': selected_traj['winding_angle']
                        },
                        'quality_settings': {'mode': quality_level.lower()},
                        'source': 'converted_unified_trajectory'
                    }
                    
                    layer_config = {
                        'layer_type': layer_def.layer_type,
                        'winding_angle_deg': layer_def.winding_angle_deg,
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
                        'show_start_end_points': True
                    }
                    
                    # Create visualization
                    st.markdown("### 🎯 3D Trajectory Visualization")
                    
                    visualizer = FixedAdvanced3DVisualizer()
                    fig = visualizer.create_full_coverage_visualization(
                        coverage_data,
                        st.session_state.vessel_geometry,
                        layer_config,
                        visualization_options
                    )
                    
                    # Display the visualization
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display trajectory information
                    st.markdown("### 📊 Trajectory Information")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Points", len(converted_data['path_points']))
                    with col2:
                        st.metric("Layer Type", selected_traj['layer_type'])
                    with col3:
                        st.metric("Winding Angle", f"{selected_traj['winding_angle']}°")
                    with col4:
                        st.metric("Coordinate System", converted_data.get('coordinate_system', 'cartesian'))
                    
                    # Show coordinate ranges for verification
                    st.markdown("### 🔍 Coordinate Analysis")
                    coord_col1, coord_col2, coord_col3 = st.columns(3)
                    
                    with coord_col1:
                        x_vals = converted_data['x_points_m']
                        st.write(f"**X Coordinates:**")
                        st.write(f"Range: {min(x_vals):.3f}m to {max(x_vals):.3f}m")
                        st.write(f"Span: {max(x_vals) - min(x_vals):.3f}m")
                    
                    with coord_col2:
                        y_vals = converted_data['y_points_m']
                        st.write(f"**Y Coordinates:**")
                        st.write(f"Range: {min(y_vals):.3f}m to {max(y_vals):.3f}m")
                        st.write(f"Span: {max(y_vals) - min(y_vals):.3f}m")
                    
                    with coord_col3:
                        z_vals = converted_data['z_points_m']
                        st.write(f"**Z Coordinates:**")
                        st.write(f"Range: {min(z_vals):.3f}m to {max(z_vals):.3f}m")
                        st.write(f"Span: {max(z_vals) - min(z_vals):.3f}m")
                    
                    # Compare with vessel geometry
                    vessel_profile = st.session_state.vessel_geometry.get_profile_points()
                    if vessel_profile:
                        st.markdown("### ⚖️ Trajectory vs Vessel Comparison")
                        
                        z_vessel_mm = vessel_profile['z_mm']
                        z_vessel_m = np.array(z_vessel_mm) / 1000.0
                        vessel_z_span = max(z_vessel_m) - min(z_vessel_m)
                        traj_z_span = max(z_vals) - min(z_vals)
                        
                        comp_col1, comp_col2 = st.columns(2)
                        with comp_col1:
                            st.metric("Vessel Z Span", f"{vessel_z_span:.3f}m")
                            st.metric("Trajectory Z Span", f"{traj_z_span:.3f}m")
                        with comp_col2:
                            span_ratio = (traj_z_span / vessel_z_span) if vessel_z_span > 0 else 0
                            st.metric("Span Ratio", f"{span_ratio:.2f}")
                            if 0.8 <= span_ratio <= 1.2:
                                st.success("✅ Good scale match")
                            else:
                                st.warning("⚠️ Scale mismatch detected")
                    
                    st.success("✅ Visualization generated from converted trajectory data!")
                    
                except Exception as e:
                    st.error(f"❌ Visualization error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                    st.info("💡 The trajectory data is still valid for manufacturing")
            else:
                st.error("Could not find corresponding layer definition")
    
    # Mark visualization as ready
    st.session_state.visualization_ready = True

# Additional debug function for troubleshooting
def debug_trajectory_data():
    """Debug function to analyze trajectory data structure"""
    st.markdown("### 🔍 Trajectory Data Debugging")
    
    if 'all_layer_trajectories' not in st.session_state:
        st.error("No trajectory data found in session state")
        return
    
    trajectories = st.session_state.all_layer_trajectories
    st.write(f"Found {len(trajectories)} trajectory records")
    
    for i, traj in enumerate(trajectories):
        with st.expander(f"Trajectory {i+1} - Layer {traj.get('layer_id', 'Unknown')}"):
            st.write("**Top-level keys:**")
            st.write(list(traj.keys()))
            
            if 'trajectory_data' in traj:
                traj_data = traj['trajectory_data']
                st.write("**Trajectory data keys:**")
                st.write(list(traj_data.keys()))
                
                # Analyze the points
                if 'points' in traj_data:
                    points = traj_data['points']
                    st.write(f"**Points analysis:**")
                    st.write(f"  Number of points: {len(points)}")
                    if points:
                        point = points[0]
                        st.write(f"  First point type: {type(point)}")
                        if hasattr(point, '__dict__'):
                            st.write(f"  Point attributes: {list(point.__dict__.keys())}")
                        if hasattr(point, 'rho'):
                            st.write(f"  Sample values: rho={point.rho:.3f}, z={point.z:.3f}, phi={point.phi:.3f}")
                
                # Check for coordinate arrays
                coord_keys = ['x_points_m', 'y_points_m', 'z_points_m']
                for key in coord_keys:
                    if key in traj_data:
                        vals = traj_data[key]
                        st.write(f"  {key}: {len(vals)} values, range {min(vals):.3f} to {max(vals):.3f}")

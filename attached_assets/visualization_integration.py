"""
Integration Guide: Replace Complex Visualization with Streamlined System
This shows how to integrate the new streamlined visualizer into your existing app.py
"""

# 1. REPLACE THE EXISTING visualization_page() FUNCTION IN app.py
def visualization_page():
    """Updated visualization page using streamlined system"""
    st.markdown("""
    <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #1e3c72; margin-bottom: 1.5rem;">
        <h2 style="color: #1e3c72; margin: 0;">📊 3D Visualization</h2>
        <p style="color: #6c757d; margin: 0.5rem 0 0 0;">Streamlined visualization of planned trajectories</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check workflow prerequisites
    if not hasattr(st.session_state, 'vessel_geometry') or not st.session_state.vessel_geometry:
        st.error("Complete Vessel Geometry first")
        if st.button("Go to Vessel Geometry"):
            st.session_state.current_page = "Vessel Geometry"
            st.rerun()
        return
    
    # Import the new streamlined visualizer
    from streamlined_3d_viz import create_streamlined_3d_visualization
    
    # Configuration options
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎛️ Visualization Settings")
        show_mandrel = st.checkbox("Show Mandrel Surface", value=True)
        show_wireframe = st.checkbox("Show Wireframe", value=True)
        mandrel_resolution = st.slider("Mandrel Resolution", 16, 64, 32, step=8)
        
    with col2:
        st.markdown("### ⚡ Performance Settings")
        decimation_factor = st.selectbox(
            "Point Decimation (for performance)",
            [1, 5, 10, 20, 50],
            index=2,
            help="Show every Nth point. Higher = faster rendering"
        )
        trajectory_line_width = st.slider("Trajectory Line Width", 1, 8, 4)
    
    # Visualization options
    viz_options = {
        'show_mandrel': show_mandrel,
        'show_trajectory': True,
        'decimation_factor': decimation_factor,
        'mandrel_resolution': mandrel_resolution,
        'show_wireframe': show_wireframe,
        'trajectory_line_width': trajectory_line_width
    }
    
    # Check for trajectory data
    trajectory_data = None
    if hasattr(st.session_state, 'all_layer_trajectories') and st.session_state.all_layer_trajectories:
        # Multiple layers available
        st.markdown("### 📋 Layer Selection")
        layer_trajectories = st.session_state.all_layer_trajectories
        
        layer_options = []
        for i, traj in enumerate(layer_trajectories):
            layer_info = f"Layer {traj.get('layer_id', i+1)}: {traj.get('layer_type', 'Unknown')} ({traj.get('winding_angle', 0)}°)"
            layer_options.append(layer_info)
        
        if layer_options:
            selected_idx = st.selectbox(
                "Select Layer to Visualize",
                range(len(layer_options)),
                format_func=lambda x: layer_options[x]
            )
            
            if selected_idx is not None:
                selected_trajectory = layer_trajectories[selected_idx]
                trajectory_data = selected_trajectory.get('trajectory_data', {})
                
                # Add layer info for better visualization
                if trajectory_data:
                    trajectory_data.update({
                        'pattern_type': selected_trajectory.get('layer_type', 'Unknown'),
                        'layer_id': selected_trajectory.get('layer_id', selected_idx + 1),
                        'winding_angle': selected_trajectory.get('winding_angle', 0)
                    })
    
    elif hasattr(st.session_state, 'trajectory_data') and st.session_state.trajectory_data:
        # Single trajectory available
        trajectory_data = st.session_state.trajectory_data
        st.info("Using single trajectory data")
    
    # Generate visualization
    if st.button("🚀 Generate 3D Visualization", type="primary"):
        try:
            fig = create_streamlined_3d_visualization(
                st.session_state.vessel_geometry,
                trajectory_data,
                viz_options
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show trajectory statistics if available
            if trajectory_data:
                st.markdown("### 📊 Trajectory Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_points = trajectory_data.get('total_points', 0)
                    if not total_points and 'path_points' in trajectory_data:
                        total_points = len(trajectory_data['path_points'])
                    st.metric("Total Points", f"{total_points:,}")
                
                with col2:
                    pattern_type = trajectory_data.get('pattern_type', 'Unknown')
                    st.metric("Pattern Type", pattern_type)
                
                with col3:
                    layer_id = trajectory_data.get('layer_id', 'N/A')
                    st.metric("Layer ID", layer_id)
                
                with col4:
                    winding_angle = trajectory_data.get('winding_angle', 0)
                    st.metric("Winding Angle", f"{winding_angle}°")
            
        except Exception as e:
            st.error(f"Visualization failed: {str(e)}")
            st.info("💡 Check that vessel geometry and trajectory data are properly formatted")
    
    # Vessel-only visualization option
    st.markdown("---")
    if st.button("🏗️ Show Vessel Only (No Trajectory)", type="secondary"):
        try:
            vessel_only_options = viz_options.copy()
            vessel_only_options['show_trajectory'] = False
            
            fig = create_streamlined_3d_visualization(
                st.session_state.vessel_geometry,
                None,  # No trajectory data
                vessel_only_options
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.success("✅ Vessel geometry displayed")
            
        except Exception as e:
            st.error(f"Vessel visualization failed: {str(e)}")


# 2. ADD THIS TO YOUR MODULES DIRECTORY
# Save the streamlined_3d_viz.py file from the previous artifact in your modules/ directory

# 3. UPDATE IMPORTS IN app.py (add this near the top of your app.py file)
"""
Replace the complex visualization imports with this simple import:

# REMOVE these complex imports:
# from modules.advanced_3d_visualization import Advanced3DVisualizer
# from modules.fixed_coordinate_visualizer import FixedAdvanced3DVisualizer  
# from modules.trajectory_data_converter import TrajectoryDataConverter
# from modules.coordinate_diagnostic import diagnose_coordinate_systems

# ADD this simple import:
from modules.streamlined_3d_viz import create_streamlined_3d_visualization
"""

# 4. OPTIONAL: Update other visualization functions to use streamlined system
def enhanced_layer_stack_visualization():
    """Enhanced layer stack visualization using streamlined system"""
    
    if 'layer_stack_manager' not in st.session_state:
        st.warning("No layer stack defined")
        return
    
    manager = st.session_state.layer_stack_manager
    stack_summary = manager.get_layer_stack_summary()
    
    if not hasattr(st.session_state, 'all_layer_trajectories'):
        st.info("Generate trajectories first to see layer visualizations")
        return
    
    st.markdown("### 🎨 Multi-Layer Visualization")
    
    # Layer selection with checkboxes for multiple layers
    st.markdown("**Select layers to display:**")
    
    selected_layers = []
    for i, traj in enumerate(st.session_state.all_layer_trajectories):
        layer_name = f"Layer {traj.get('layer_id', i+1)}: {traj.get('layer_type', 'Unknown')} ({traj.get('winding_angle', 0)}°)"
        if st.checkbox(layer_name, key=f"layer_viz_{i}"):
            selected_layers.append(traj)
    
    if selected_layers and st.button("🚀 Visualize Selected Layers"):
        # For multiple layers, we can cycle through them or show them separately
        if len(selected_layers) == 1:
            # Single layer
            trajectory_data = selected_layers[0].get('trajectory_data', {})
            trajectory_data.update({
                'pattern_type': selected_layers[0].get('layer_type', 'Unknown'),
                'layer_id': selected_layers[0].get('layer_id', 1)
            })
            
            fig = create_streamlined_3d_visualization(
                st.session_state.vessel_geometry,
                trajectory_data,
                {'decimation_factor': 10, 'trajectory_line_width': 6}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            # Multiple layers - show in tabs
            tabs = st.tabs([f"Layer {traj.get('layer_id', i+1)}" for i, traj in enumerate(selected_layers)])
            
            for tab, traj in zip(tabs, selected_layers):
                with tab:
                    trajectory_data = traj.get('trajectory_data', {})
                    trajectory_data.update({
                        'pattern_type': traj.get('layer_type', 'Unknown'),
                        'layer_id': traj.get('layer_id', 1)
                    })
                    
                    fig = create_streamlined_3d_visualization(
                        st.session_state.vessel_geometry,
                        trajectory_data,
                        {'decimation_factor': 15, 'trajectory_line_width': 5}
                    )
                    st.plotly_chart(fig, use_container_width=True)


# 5. PERFORMANCE COMPARISON
def performance_comparison():
    """Show performance improvements of streamlined system"""
    
    st.markdown("### ⚡ Performance Improvements")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**❌ Old System Issues:**")
        st.write("• Multiple coordinate converters")
        st.write("• Complex adapter layers") 
        st.write("• Coordinate misalignment")
        st.write("• Poor performance with 3000+ points")
        st.write("• Trajectories not displaying")
        st.write("• Multiple format conversions")
    
    with col2:
        st.markdown("**✅ New System Benefits:**")
        st.write("• Single coordinate system")
        st.write("• Direct data handling")
        st.write("• Automatic alignment correction")
        st.write("• Efficient decimation")
        st.write("• Guaranteed trajectory display")
        st.write("• Minimal memory usage")
    
    st.markdown("### 🎯 Key Features")
    
    features = [
        ("**Coordinate Alignment**", "Automatically detects and corrects coordinate system mismatches"),
        ("**Performance Decimation**", "Intelligent point reduction for smooth rendering"),
        ("**Unified Units**", "Handles mm/m conversion automatically"),
        ("**Error Recovery**", "Graceful fallbacks when data is malformed"),
        ("**Memory Efficient**", "Processes data in-place without copies"),
        ("**Clean Interface**", "Single function call replaces complex pipeline")
    ]
    
    for title, description in features:
        st.write(f"{title}: {description}")


# 6. TROUBLESHOOTING GUIDE
def troubleshooting_guide():
    """Troubleshooting guide for common visualization issues"""
    
    st.markdown("### 🔧 Troubleshooting Guide")
    
    issues = [
        {
            "issue": "Trajectory not displaying",
            "causes": ["Missing coordinate data", "Coordinate format mismatch", "Empty trajectory data"],
            "solutions": ["Check trajectory_data has x_points_m, y_points_m, z_points_m", "Verify path_points format", "Ensure trajectory generation succeeded"]
        },
        {
            "issue": "Coordinate offset/misalignment", 
            "causes": ["Unit mismatch (mm vs m)", "Different coordinate centers", "Profile data inconsistency"],
            "solutions": ["System automatically detects and corrects", "Check vessel geometry is properly generated", "Verify coordinate analysis output"]
        },
        {
            "issue": "Poor performance",
            "causes": ["Too many trajectory points", "High mandrel resolution", "Complex wireframe"],
            "solutions": ["Increase decimation_factor", "Reduce mandrel_resolution", "Disable wireframe"]
        },
        {
            "issue": "Empty or missing visualization",
            "causes": ["Missing vessel geometry", "Invalid profile data", "Import errors"],
            "solutions": ["Generate vessel geometry first", "Check profile_points has z_mm and r_inner_mm", "Verify streamlined_3d_viz import"]
        }
    ]
    
    for issue_data in issues:
        with st.expander(f"🚨 {issue_data['issue']}"):
            st.markdown("**Possible Causes:**")
            for cause in issue_data['causes']:
                st.write(f"• {cause}")
            
            st.markdown("**Solutions:**")
            for solution in issue_data['solutions']:
                st.write(f"✅ {solution}")


if __name__ == "__main__":
    st.title("Streamlined 3D Visualization Integration Guide")
    
    tab1, tab2, tab3 = st.tabs(["Integration Steps", "Performance", "Troubleshooting"])
    
    with tab1:
        st.markdown("Follow the numbered steps in the code above to integrate the streamlined visualization system")
    
    with tab2:
        performance_comparison()
    
    with tab3:
        troubleshooting_guide()

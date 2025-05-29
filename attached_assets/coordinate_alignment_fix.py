"""
Coordinate Alignment Fix for Trajectory Visualization
Aligns trajectory coordinates with vessel geometry coordinate system
"""

import numpy as np
import streamlit as st

def align_trajectory_with_vessel(trajectory_data: dict, vessel_geometry) -> dict:
    """
    Align trajectory coordinates with vessel geometry coordinate system.
    
    Args:
        trajectory_data: Converted trajectory data
        vessel_geometry: Vessel geometry object
        
    Returns:
        dict: Trajectory data with aligned coordinates
    """
    st.write("🔧 **Coordinate Alignment Debug:**")
    
    # Get vessel coordinate system
    vessel_profile = vessel_geometry.get_profile_points()
    vessel_z_mm = np.array(vessel_profile['z_mm'])
    vessel_z_m = vessel_z_mm / 1000.0
    
    vessel_z_min = np.min(vessel_z_m)
    vessel_z_max = np.max(vessel_z_m)
    vessel_z_center = (vessel_z_min + vessel_z_max) / 2
    vessel_z_span = vessel_z_max - vessel_z_min
    
    st.write(f"   📊 Vessel Z system: {vessel_z_min:.3f}m to {vessel_z_max:.3f}m")
    st.write(f"   📊 Vessel Z center: {vessel_z_center:.3f}m, span: {vessel_z_span:.3f}m")
    
    # Get trajectory coordinate system
    traj_z = np.array(trajectory_data['z_points_m'])
    traj_z_min = np.min(traj_z)
    traj_z_max = np.max(traj_z)
    traj_z_center = (traj_z_min + traj_z_max) / 2
    traj_z_span = traj_z_max - traj_z_min
    
    st.write(f"   📊 Trajectory Z system: {traj_z_min:.3f}m to {traj_z_max:.3f}m")
    st.write(f"   📊 Trajectory Z center: {traj_z_center:.3f}m, span: {traj_z_span:.3f}m")
    
    # Calculate alignment correction
    z_offset = vessel_z_center - traj_z_center
    
    st.write(f"   🎯 Required Z offset: {z_offset:.3f}m")
    
    if abs(z_offset) > 0.01:  # More than 1cm misalignment
        st.warning(f"⚠️ **Coordinate Misalignment Detected!** Applying {z_offset:.3f}m Z-offset")
        
        # Apply correction to all Z coordinates
        corrected_z = traj_z + z_offset
        
        # Update trajectory data
        aligned_data = trajectory_data.copy()
        aligned_data['z_points_m'] = corrected_z.tolist()
        
        # Update path_points
        aligned_path_points = []
        for i, point in enumerate(trajectory_data['path_points']):
            aligned_point = point.copy()
            aligned_point['z_m'] = corrected_z[i]
            aligned_path_points.append(aligned_point)
        
        aligned_data['path_points'] = aligned_path_points
        
        # Verify alignment
        new_z_min = np.min(corrected_z)
        new_z_max = np.max(corrected_z)
        st.success(f"✅ **Coordinates Aligned!** New trajectory Z: {new_z_min:.3f}m to {new_z_max:.3f}m")
        
        # Add alignment metadata
        aligned_data['coordinate_alignment'] = {
            'z_offset_applied': z_offset,
            'original_z_range': [traj_z_min, traj_z_max],
            'aligned_z_range': [new_z_min, new_z_max],
            'vessel_z_range': [vessel_z_min, vessel_z_max]
        }
        
        return aligned_data
    else:
        st.success("✅ Coordinates already properly aligned")
        return trajectory_data

def check_trajectory_vessel_alignment(trajectory_data: dict, vessel_geometry) -> dict:
    """
    Check and report on trajectory-vessel alignment quality.
    
    Returns alignment analysis for debugging.
    """
    vessel_profile = vessel_geometry.get_profile_points()
    vessel_z_m = np.array(vessel_profile['z_mm']) / 1000.0
    vessel_r_m = np.array(vessel_profile['r_inner_mm']) / 1000.0
    
    traj_x = np.array(trajectory_data['x_points_m'])
    traj_y = np.array(trajectory_data['y_points_m'])
    traj_z = np.array(trajectory_data['z_points_m'])
    traj_r = np.sqrt(traj_x**2 + traj_y**2)
    
    analysis = {
        'vessel_z_range': [np.min(vessel_z_m), np.max(vessel_z_m)],
        'vessel_r_range': [np.min(vessel_r_m), np.max(vessel_r_m)],
        'trajectory_z_range': [np.min(traj_z), np.max(traj_z)],
        'trajectory_r_range': [np.min(traj_r), np.max(traj_r)],
        'z_overlap': False,
        'r_overlap': False,
        'alignment_quality': 'poor'
    }
    
    # Check Z overlap
    z_overlap = not (np.max(traj_z) < np.min(vessel_z_m) or np.min(traj_z) > np.max(vessel_z_m))
    analysis['z_overlap'] = z_overlap
    
    # Check R overlap  
    r_overlap = not (np.max(traj_r) < np.min(vessel_r_m) or np.min(traj_r) > np.max(vessel_r_m))
    analysis['r_overlap'] = r_overlap
    
    # Overall alignment quality
    if z_overlap and r_overlap:
        analysis['alignment_quality'] = 'good'
    elif z_overlap or r_overlap:
        analysis['alignment_quality'] = 'partial'
    else:
        analysis['alignment_quality'] = 'poor'
    
    return analysis

# Updated visualization function with alignment fix
def visualization_page_with_alignment():
    """Visualization page with automatic coordinate alignment"""
    st.markdown("""
    <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #1e3c72; margin-bottom: 1.5rem;">
        <h2 style="color: #1e3c72; margin: 0;">📊 3D Visualization - COORDINATE ALIGNED</h2>
        <p style="color: #6c757d; margin: 0.5rem 0 0 0;">View trajectories with automatic coordinate system alignment</p>
    </div>
    """, unsafe_allow_html=True)
    
    # [Previous prerequisite checks remain the same...]
    
    # Import the converter
    try:
        from modules.trajectory_data_converter import TrajectoryDataConverter
        converter = TrajectoryDataConverter()
        st.success("✅ Trajectory data converter loaded")
    except ImportError:
        st.error("❌ Please save the trajectory_data_converter.py file in your modules/ directory")
        return
    
    # [Previous checks for vessel_geometry, layer_stack_manager, trajectories...]
    if not hasattr(st.session_state, 'vessel_geometry') or not st.session_state.vessel_geometry:
        st.error("Complete Vessel Geometry first")
        return
    
    if not hasattr(st.session_state, 'all_layer_trajectories') or not st.session_state.all_layer_trajectories:
        st.error("Complete Trajectory Planning first")
        return
    
    trajectories = st.session_state.all_layer_trajectories
    layer_options = [f"Layer {traj['layer_id']}: {traj['layer_type']} ({traj['winding_angle']}°)" 
                    for traj in trajectories]
    
    selected_idx = st.selectbox(
        "Select Layer to Visualize",
        range(len(layer_options)),
        format_func=lambda x: layer_options[x]
    )
    
    if selected_idx is not None and st.button("🚀 Generate Aligned 3D Visualization", type="primary"):
        selected_traj = trajectories[selected_idx]
        raw_trajectory_data = selected_traj.get('trajectory_data', {})
        
        if not raw_trajectory_data:
            st.error("No trajectory data found")
            return
        
        # Step 1: Convert trajectory data
        st.markdown("### 🔧 Step 1: Trajectory Data Conversion")
        converted_data = converter.convert_unified_trajectory_to_visualization_format(raw_trajectory_data)
        
        if not converted_data or not converted_data.get('success', False):
            st.error("❌ Trajectory data conversion failed")
            return
        
        # Step 2: Align coordinates with vessel geometry
        st.markdown("### 🎯 Step 2: Coordinate System Alignment")
        aligned_data = align_trajectory_with_vessel(converted_data, st.session_state.vessel_geometry)
        
        # Step 3: Verify alignment quality
        st.markdown("### ✅ Step 3: Alignment Verification")
        alignment_analysis = check_trajectory_vessel_alignment(aligned_data, st.session_state.vessel_geometry)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Z Overlap", "✅ Yes" if alignment_analysis['z_overlap'] else "❌ No")
        with col2:
            st.metric("R Overlap", "✅ Yes" if alignment_analysis['r_overlap'] else "❌ No")  
        with col3:
            quality = alignment_analysis['alignment_quality'].title()
            st.metric("Alignment Quality", quality)
        
        if alignment_analysis['alignment_quality'] == 'poor':
            st.error("⚠️ **Poor alignment detected** - trajectory may not be visible")
        elif alignment_analysis['alignment_quality'] == 'partial':
            st.warning("⚠️ **Partial alignment** - some trajectory parts may be off-scale")
        else:
            st.success("✅ **Good alignment** - trajectory should be properly visible")
        
        # Step 4: Create visualization with aligned data
        st.markdown("### 🎯 Step 4: 3D Visualization")
        
        try:
            from modules.fixed_advanced_3d_visualizer import FixedAdvanced3DVisualizer
            
            # Use aligned data for visualization
            coverage_data = {
                'circuits': [aligned_data['path_points']],
                'circuit_metadata': [{
                    'circuit_number': 1,
                    'start_phi_deg': 0.0,
                    'points_count': len(aligned_data['path_points']),
                    'quality_score': 95.0
                }],
                'metadata': [{
                    'circuit_number': 1,
                    'start_phi_deg': 0.0,
                    'points_count': len(aligned_data['path_points']),
                    'quality_score': 95.0
                }],
                'total_circuits': 1,
                'coverage_percentage': aligned_data.get('coverage_percentage', 85.0),
                'pattern_info': {
                    'actual_pattern_type': selected_traj['layer_type'],
                    'winding_angle': selected_traj['winding_angle']
                },
                'quality_settings': {'mode': 'standard'},
                'source': 'aligned_trajectory'
            }
            
            # Get layer configuration
            layer_manager = st.session_state.layer_stack_manager
            layer_def = None
            for layer in layer_manager.layer_stack:
                if layer.layer_set_id == selected_traj['layer_id']:
                    layer_def = layer
                    break
            
            if layer_def:
                layer_config = {
                    'layer_type': layer_def.layer_type,
                    'winding_angle_deg': layer_def.winding_angle_deg,
                    'physics_model': getattr(layer_def, 'physics_model', 'clairaut'),
                    'roving_width': 3.0,
                    'coverage_mode': 'full_coverage'
                }
                
                visualization_options = {
                    'quality_level': 'standard',
                    'show_mandrel_mesh': True,
                    'color_by_circuit': True,
                    'show_all_circuits': True,
                    'show_mandrel': True,
                    'mandrel_opacity': 0.3,
                    'circuit_line_width': 4,
                    'show_start_end_points': True
                }
                
                # Create visualization
                visualizer = FixedAdvanced3DVisualizer()
                fig = visualizer.create_full_coverage_visualization(
                    coverage_data,
                    st.session_state.vessel_geometry,
                    layer_config,
                    visualization_options
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show final coordinate comparison
                st.markdown("### 📊 Final Coordinate Comparison")
                comp_col1, comp_col2 = st.columns(2)
                
                with comp_col1:
                    st.write("**Vessel Geometry:**")
                    vessel_analysis = alignment_analysis
                    st.write(f"Z: {vessel_analysis['vessel_z_range'][0]:.3f}m to {vessel_analysis['vessel_z_range'][1]:.3f}m")
                    st.write(f"R: {vessel_analysis['vessel_r_range'][0]:.3f}m to {vessel_analysis['vessel_r_range'][1]:.3f}m")
                
                with comp_col2:
                    st.write("**Aligned Trajectory:**")
                    st.write(f"Z: {vessel_analysis['trajectory_z_range'][0]:.3f}m to {vessel_analysis['trajectory_z_range'][1]:.3f}m")
                    st.write(f"R: {vessel_analysis['trajectory_r_range'][0]:.3f}m to {vessel_analysis['trajectory_r_range'][1]:.3f}m")
                
                if 'coordinate_alignment' in aligned_data:
                    st.info(f"🎯 **Applied Z-offset:** {aligned_data['coordinate_alignment']['z_offset_applied']:.3f}m")
                
                st.success("✅ Visualization with coordinate alignment complete!")
                
        except Exception as e:
            st.error(f"❌ Visualization error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# Add this to replace your current visualization_page function

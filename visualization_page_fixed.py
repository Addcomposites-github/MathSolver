"""
Fixed Visualization Page with Streamlined 3D System
Clean implementation without corrupted code fragments
"""

import streamlit as st
import numpy as np
from modules.streamlined_3d_viz import create_streamlined_3d_visualization

def visualization_page():
    """Streamlined 3D visualization page using the new coordinate-aligned system"""
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">
        <h2 style="color: white; margin: 0; text-align: center;">
            🎯 3D Trajectory Visualization
        </h2>
        <p style="color: white; margin: 0.5rem 0 0 0; text-align: center; opacity: 0.9;">
            Advanced 3D visualization with coordinate alignment
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check for required data
    if not hasattr(st.session_state, 'vessel_geometry') or st.session_state.vessel_geometry is None:
        st.error("Please define vessel geometry first in the Vessel Geometry page")
        return
    
    if not hasattr(st.session_state, 'trajectories') or not st.session_state.trajectories:
        st.warning("No trajectory data available. Please generate trajectories in the Trajectory Planning page first.")
        
        # Show vessel-only visualization
        st.markdown("### Vessel Geometry Preview")
        
        try:
            # Create vessel visualization without trajectories
            vessel_viz_options = {
                'show_mandrel': True,
                'show_trajectory': False,
                'mandrel_resolution': 48,
                'show_wireframe': True,
                'wireframe_color': '#888888',
                'mandrel_color': '#4CAF50'
            }
            
            fig = create_streamlined_3d_visualization(
                st.session_state.vessel_geometry,
                None,  # No trajectory data
                vessel_viz_options
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Visualization error: {str(e)}")
        
        return
    
    # Main visualization with trajectories
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown("### Visualization Options")
        
        # Trajectory selection
        trajectory_names = list(st.session_state.trajectories.keys())
        selected_trajectory = st.selectbox("Select Trajectory", trajectory_names)
        
        # Visualization options
        st.markdown("#### Display Options")
        show_mandrel = st.checkbox("Show Mandrel Surface", value=True)
        show_wireframe = st.checkbox("Show Wireframe", value=True)
        show_trajectory = st.checkbox("Show Trajectory", value=True)
        
        # Quality settings
        st.markdown("#### Quality Settings")
        mandrel_resolution = st.slider("Mandrel Resolution", 16, 64, 32)
        
        # Color options
        st.markdown("#### Colors")
        mandrel_color = st.color_picker("Mandrel Color", "#4CAF50")
        trajectory_color = st.color_picker("Trajectory Color", "#FF6B6B")
    
    with col1:
        st.markdown("### 3D Visualization")
        
        if selected_trajectory and selected_trajectory in st.session_state.trajectories:
            try:
                # Get trajectory data
                trajectory_data = st.session_state.trajectories[selected_trajectory]
                
                # Create visualization options
                viz_options = {
                    'show_mandrel': show_mandrel,
                    'show_trajectory': show_trajectory,
                    'show_wireframe': show_wireframe,
                    'mandrel_resolution': mandrel_resolution,
                    'mandrel_color': mandrel_color,
                    'trajectory_color': trajectory_color,
                    'wireframe_color': '#888888'
                }
                
                # Create the visualization
                fig = create_streamlined_3d_visualization(
                    st.session_state.vessel_geometry,
                    trajectory_data,
                    viz_options
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display trajectory information
                if 'trajectory_data' in trajectory_data:
                    traj_info = trajectory_data['trajectory_data']
                    
                    st.markdown("### Trajectory Information")
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        if 'x_points_m' in traj_info:
                            st.metric("Total Points", len(traj_info['x_points_m']))
                    
                    with col_b:
                        if 'winding_angles_deg' in traj_info and traj_info['winding_angles_deg']:
                            avg_angle = np.mean(traj_info['winding_angles_deg'])
                            st.metric("Avg Winding Angle", f"{avg_angle:.1f}°")
                    
                    with col_c:
                        if 'coverage_percentage' in traj_info:
                            st.metric("Coverage", f"{traj_info['coverage_percentage']:.1f}%")
                
            except Exception as e:
                st.error(f"Visualization error: {str(e)}")
                st.write("Debug info:", str(e))
        else:
            st.info("Select a trajectory to visualize")
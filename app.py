import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from modules.geometry import VesselGeometry
from modules.trajectories import TrajectoryPlanner
from modules.materials import MaterialDatabase
from modules.calculations import VesselCalculations
from modules.visualizations import VesselVisualizer
from data.material_database import FIBER_MATERIALS, RESIN_MATERIALS

# Configure page
st.set_page_config(
    page_title="COPV Design Tool",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'vessel_geometry' not in st.session_state:
    st.session_state.vessel_geometry = None
if 'trajectory_data' not in st.session_state:
    st.session_state.trajectory_data = None

def main():
    st.title("🏗️ Composite Pressure Vessel Design Tool")
    st.markdown("Engineering application for COPV 2D profile generation and filament winding trajectory planning")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Analysis Module",
        ["Vessel Geometry", "Material Properties", "Trajectory Planning", "Performance Analysis", "Export Results"]
    )
    
    if page == "Vessel Geometry":
        vessel_geometry_page()
    elif page == "Material Properties":
        material_properties_page()
    elif page == "Trajectory Planning":
        trajectory_planning_page()
    elif page == "Performance Analysis":
        performance_analysis_page()
    elif page == "Export Results":
        export_results_page()

def vessel_geometry_page():
    st.header("Vessel Geometry Design")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Input Parameters")
        
        # Basic vessel parameters
        st.markdown("### Basic Dimensions")
        inner_diameter = st.number_input("Inner Diameter (mm)", min_value=10.0, value=200.0, step=1.0)
        cylindrical_length = st.number_input("Cylindrical Length (mm)", min_value=0.0, value=300.0, step=1.0)
        wall_thickness = st.number_input("Wall Thickness (mm)", min_value=0.1, value=5.0, step=0.1)
        
        # Dome parameters
        st.markdown("### Dome Configuration")
        dome_type = st.selectbox("Dome Type", ["Isotensoid", "Geodesic", "Elliptical", "Hemispherical"])
        
        if dome_type == "Isotensoid":
            st.markdown("#### qrs-Parameterization (Koussios)")
            q_factor = st.slider("q-factor", min_value=0.1, max_value=20.0, value=2.0, step=0.1)
            r_factor = st.slider("r-factor", min_value=-1.0, max_value=2.0, value=0.1, step=0.01)
            s_factor = st.slider("s-factor", min_value=0.0, max_value=2.0, value=0.5, step=0.01)
        elif dome_type == "Elliptical":
            aspect_ratio = st.slider("Dome Aspect Ratio (height/radius)", min_value=0.1, max_value=2.0, value=1.0, step=0.01)
        
        # Operating conditions
        st.markdown("### Operating Conditions")
        operating_pressure = st.number_input("Operating Pressure (MPa)", min_value=0.1, value=30.0, step=0.1)
        safety_factor = st.number_input("Safety Factor", min_value=1.0, value=2.0, step=0.1)
        operating_temp = st.number_input("Operating Temperature (°C)", value=20.0, step=1.0)
        
        # Generate geometry button
        if st.button("Generate Vessel Geometry", type="primary"):
            try:
                geometry = VesselGeometry(
                    inner_diameter=inner_diameter,
                    wall_thickness=wall_thickness,
                    cylindrical_length=cylindrical_length,
                    dome_type=dome_type
                )
                
                if dome_type == "Isotensoid":
                    geometry.set_qrs_parameters(q_factor, r_factor, s_factor)
                elif dome_type == "Elliptical":
                    geometry.set_elliptical_parameters(aspect_ratio)
                
                geometry.generate_profile()
                st.session_state.vessel_geometry = geometry
                st.success("Vessel geometry generated successfully!")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error generating geometry: {str(e)}")
    
    with col2:
        st.subheader("Vessel Profile Visualization")
        
        if st.session_state.vessel_geometry is not None:
            visualizer = VesselVisualizer()
            fig = visualizer.plot_vessel_profile(st.session_state.vessel_geometry)
            st.pyplot(fig)
            
            # Display geometric properties
            st.subheader("Geometric Properties")
            props = st.session_state.vessel_geometry.get_geometric_properties()
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Total Volume", f"{props['total_volume']:.2f} L")
                st.metric("Surface Area", f"{props['surface_area']:.2f} m²")
                st.metric("Dome Height", f"{props['dome_height']:.2f} mm")
            
            with col_b:
                st.metric("Overall Length", f"{props['overall_length']:.2f} mm")
                st.metric("Weight (Est.)", f"{props['estimated_weight']:.2f} kg")
                st.metric("Aspect Ratio", f"{props['aspect_ratio']:.2f}")
        else:
            st.info("Please configure and generate vessel geometry to see visualization.")

def material_properties_page():
    st.header("Material Properties")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Fiber Selection")
        fiber_type = st.selectbox("Fiber Type", list(FIBER_MATERIALS.keys()))
        
        if fiber_type:
            fiber_props = FIBER_MATERIALS[fiber_type]
            st.write("**Fiber Properties:**")
            for prop, value in fiber_props.items():
                st.write(f"- {prop}: {value}")
    
    with col2:
        st.subheader("Resin Selection")
        resin_type = st.selectbox("Resin Type", list(RESIN_MATERIALS.keys()))
        
        if resin_type:
            resin_props = RESIN_MATERIALS[resin_type]
            st.write("**Resin Properties:**")
            for prop, value in resin_props.items():
                st.write(f"- {prop}: {value}")
    
    # Composite properties calculation
    st.subheader("Composite Laminate Properties")
    
    col3, col4 = st.columns(2)
    with col3:
        fiber_volume_fraction = st.slider("Fiber Volume Fraction", min_value=0.1, max_value=0.9, value=0.6, step=0.01)
        void_content = st.slider("Void Content (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
        ply_thickness = st.number_input("Ply Thickness (mm)", min_value=0.01, value=0.125, step=0.001, format="%.3f")
    
    with col4:
        if st.button("Calculate Composite Properties"):
            if fiber_type and resin_type:
                material_db = MaterialDatabase()
                composite_props = material_db.calculate_composite_properties(
                    fiber_type, resin_type, fiber_volume_fraction, void_content
                )
                
                st.write("**Calculated Composite Properties:**")
                for prop, value in composite_props.items():
                    if isinstance(value, (int, float)):
                        st.write(f"- {prop}: {value:.2f}")
                    else:
                        st.write(f"- {prop}: {value}")

def trajectory_planning_page():
    st.header("Filament Winding Trajectory Planning")
    
    if st.session_state.vessel_geometry is None:
        st.warning("Please generate vessel geometry first in the 'Vessel Geometry' section.")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Winding Parameters")
        
        # Winding pattern type
        pattern_type = st.selectbox("Winding Pattern", ["Geodesic", "Helical", "Hoop", "Polar", "Transitional"])
        
        if pattern_type in ["Helical", "Transitional"]:
            winding_angle = st.slider("Winding Angle (degrees)", min_value=5.0, max_value=85.0, value=55.0, step=1.0)
        
        # Band properties
        st.markdown("### Band Properties")
        band_width = st.number_input("Band Width (mm)", min_value=1.0, value=6.0, step=0.1)
        num_tows = st.number_input("Number of Tows per Band", min_value=1, value=1, step=1)
        
        # Machine parameters
        st.markdown("### Machine Parameters")
        mandrel_speed = st.number_input("Mandrel Speed (rpm)", min_value=1.0, value=10.0, step=0.1)
        carriage_speed = st.number_input("Carriage Speed (mm/min)", min_value=1.0, value=100.0, step=1.0)
        
        # Pattern parameters
        if pattern_type == "Geodesic":
            st.markdown("### Geodesic Parameters")
            roving_width = st.number_input("Roving Width (mm)", min_value=0.1, value=3.0, step=0.1)
            roving_thickness = st.number_input("Roving Thickness (mm)", min_value=0.01, value=0.2, step=0.01)
            polar_eccentricity = st.number_input("Polar Eccentricity (mm)", min_value=0.0, value=0.0, step=0.1)
            
            st.markdown("### 🎯 Target Winding Angle")
            use_target_angle = st.checkbox("Specify Target Cylinder Angle", value=True,
                                         help="Define desired winding angle instead of using geometric limit")
            
            target_angle = None
            if use_target_angle:
                target_angle = st.slider("Target Cylinder Angle (degrees)", 
                                        min_value=10.0, max_value=80.0, value=62.0, step=1.0,
                                        help="Desired winding angle on cylinder section")
                st.info(f"🎯 **Target**: {target_angle}° winding angle on cylinder")
            else:
                st.info("🔧 **Mode**: Using geometric limit (minimum physically possible angle)")
            
            st.markdown("### 🔄 Continuous Winding Configuration")
            st.info("🔧 **Continuous Winding Mode**: Single continuous filament path with multiple passes for complete coverage")
            num_circuits = 1  # Always single circuit with multiple passes

            st.markdown("### 🚀 Adaptive Point Distribution")
            st.info("**Smart Optimization**: Use more points in dome regions (high curvature) and fewer in cylinder (constant curvature)")
            col_dome, col_cyl = st.columns(2)
            with col_dome:
                dome_points = st.slider("Dome Density", 50, 300, 150, 10,
                                      help="Higher density for dome regions where curvature changes rapidly")
            with col_cyl:
                cylinder_points = st.slider("Cylinder Density", 5, 100, 20, 5,
                                          help="Lower density for cylinder where curvature is constant")
        elif pattern_type == "Helical":
            st.markdown("### Pattern Parameters")
            circuits_to_close = st.number_input("Circuits to Close Pattern", min_value=1, value=8, step=1)
            overlap_allowance = st.slider("Overlap Allowance (%)", min_value=-10.0, max_value=50.0, value=10.0, step=1.0)
        
        if st.button("Calculate Trajectory", type="primary"):
            try:
                if pattern_type == "Geodesic":
                    planner = TrajectoryPlanner(
                        st.session_state.vessel_geometry,
                        dry_roving_width_m=roving_width/1000,
                        dry_roving_thickness_m=roving_thickness/1000,
                        roving_eccentricity_at_pole_m=polar_eccentricity/1000,
                        target_cylinder_angle_deg=target_angle
                    )
                    
                    # Show validation results
                    validation = planner.get_validation_results()
                    if validation and not validation.get('is_valid', True):
                        if validation['error_type'] == 'too_shallow':
                            st.error(f"❌ **Target angle {target_angle}° is too shallow!**")
                            st.info(f"🔧 **Minimum achievable**: {validation['min_achievable_angle']:.1f}°")
                            st.info(f"💡 **Why**: Requires turning radius smaller than physical limit")
                        elif validation['error_type'] == 'too_steep':
                            st.error(f"❌ **Target angle {target_angle}° is too steep!**")
                            st.info(f"🔧 **Maximum practical**: {validation['max_practical_angle']}°")
                        else:
                            st.error(f"❌ **Invalid target angle**: {validation['message']}")
                        return
                    elif validation and validation.get('is_valid'):
                        st.success(f"✅ **Target angle {target_angle}° is achievable!**")
                        st.info(f"🎯 **Clairaut's constant**: {validation['clairaut_constant_mm']:.1f}mm")
                        st.info(f"🛡️ **Safety margin**: {validation['validation_details']['safety_margin_mm']:.1f}mm")
                    
                    # Generate single circuit trajectory for continuous winding
                    trajectory_data = planner.generate_geodesic_trajectory(dome_points, cylinder_points)
                    
                    # Ensure trajectory data is properly stored
                    if trajectory_data and len(trajectory_data.get('path_points', [])) > 0:
                        st.session_state.trajectory_data = trajectory_data
                        st.success(f"🎯 Single circuit trajectory calculated successfully! Generated {len(trajectory_data['path_points'])} points")
                        st.rerun()
                    else:
                        st.error("❌ No trajectory points were generated. Check the debug logs for details.")
                else:
                    planner = TrajectoryPlanner(st.session_state.vessel_geometry)
                    trajectory_params = {
                        'pattern_type': pattern_type,
                        'band_width': band_width,
                        'num_tows': num_tows,
                        'mandrel_speed': mandrel_speed,
                        'carriage_speed': carriage_speed
                    }
                
                if pattern_type in ["Helical", "Transitional"]:
                    trajectory_params['winding_angle'] = winding_angle
                
                if pattern_type == "Helical":
                    trajectory_params['circuits_to_close'] = circuits_to_close
                    trajectory_params['overlap_allowance'] = overlap_allowance
                
                trajectory_data = planner.calculate_trajectory(trajectory_params)
                st.session_state.trajectory_data = trajectory_data
                st.success("Trajectory calculated successfully!")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error calculating trajectory: {str(e)}")
    
    with col2:
        st.subheader("Trajectory Visualization")
        
        if st.session_state.trajectory_data is not None:
            st.write(f"✅ Trajectory data loaded: {st.session_state.trajectory_data.get('pattern_type', 'Unknown')}")
            st.write(f"📊 Data keys: {list(st.session_state.trajectory_data.keys())}")
            
            visualizer = VesselVisualizer()
            
            # Plot trajectory - 2D view
            fig = visualizer.plot_winding_trajectory(
                st.session_state.vessel_geometry,
                st.session_state.trajectory_data
            )
            st.pyplot(fig)
            
            # Add 3D visualization for all trajectory types
            if st.session_state.trajectory_data.get('pattern_type') in ['Geodesic', 'Multi-Circuit Geodesic', 'Geodesic_MultiPass']:
                st.subheader("3D Trajectory Visualization")
                
                # Check if we have the required 3D coordinate data
                has_3d_data = ('x_points_m' in st.session_state.trajectory_data and 
                              'y_points_m' in st.session_state.trajectory_data and 
                              'z_points_m' in st.session_state.trajectory_data)
                
                if has_3d_data:
                    # Prepare vessel profile data for 3D visualization
                    vessel_profile_for_plot = {
                        'r_m': st.session_state.vessel_geometry.profile_points['r_inner'] * 1e-3, # Convert to meters
                        'z_m': st.session_state.vessel_geometry.profile_points['z'] * 1e-3    # Convert to meters
                    }
                    
                    # Create 3D trajectory plot using the enhanced visualization function
                    fig_3d = visualizer.plot_3d_trajectory(
                        trajectory_data=st.session_state.trajectory_data,
                        vessel_profile_data=vessel_profile_for_plot,
                        title=f"3D {st.session_state.trajectory_data.get('pattern_type', 'Geodesic')} Trajectory"
                    )
                    
                    if fig_3d:
                        st.pyplot(fig_3d)
                        
                        # Add trajectory statistics
                        st.write("**Trajectory Statistics:**")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Points", st.session_state.trajectory_data.get('total_points', 'N/A'))
                        with col2:
                            st.metric("Total Circuits", st.session_state.trajectory_data.get('total_circuits_legs', 'N/A'))
                        with col3:
                            final_angle = st.session_state.trajectory_data.get('final_turn_around_angle_deg', 0)
                            st.metric("Final Turn Angle", f"{final_angle:.1f}°" if final_angle else 'N/A')
                    
                    # Create interactive 3D visualization with Plotly
                    st.subheader("Interactive 3D Trajectory")
                    
                    import plotly.graph_objects as go
                    import plotly.express as px
                    
                    # Get trajectory data
                    x_m = st.session_state.trajectory_data['x_points_m']
                    y_m = st.session_state.trajectory_data['y_points_m'] 
                    z_m = st.session_state.trajectory_data['z_points_m']
                    
                    # Create interactive plot
                    fig_plotly = go.Figure()
                    
                    # Add fiber trajectory with color gradient
                    fig_plotly.add_trace(go.Scatter3d(
                        x=x_m, y=y_m, z=z_m,
                        mode='lines+markers',
                        marker=dict(
                            size=3,
                            color=range(len(x_m)),  # Color by point sequence
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Path Progress")
                        ),
                        line=dict(width=4, color='orange'),
                        name='Fiber Path',
                        text=[f'Point {i}<br>X: {x:.3f}m<br>Y: {y:.3f}m<br>Z: {z:.3f}m' 
                              for i, (x, y, z) in enumerate(zip(x_m, y_m, z_m))],
                        hovertemplate='%{text}<extra></extra>'
                    ))
                    
                    # Add vessel surface
                    profile_r = vessel_profile_for_plot['r_m']
                    profile_z = vessel_profile_for_plot['z_m']
                    
                    # Create surface mesh
                    import numpy as np
                    phi_surf = np.linspace(0, 2 * np.pi, 50)
                    R_mesh, PHI_mesh = np.meshgrid(profile_r, phi_surf)
                    Z_mesh, _ = np.meshgrid(profile_z, phi_surf)
                    X_mesh = R_mesh * np.cos(PHI_mesh)
                    Y_mesh = R_mesh * np.sin(PHI_mesh)
                    
                    fig_plotly.add_trace(go.Surface(
                        x=X_mesh, y=Y_mesh, z=Z_mesh,
                        opacity=0.3,
                        colorscale='Greys',
                        showscale=False,
                        name='Vessel Surface'
                    ))
                    
                    # Configure interactive plot layout
                    fig_plotly.update_layout(
                        title=f"Interactive 3D {st.session_state.trajectory_data.get('pattern_type', 'Geodesic')} Trajectory",
                        scene=dict(
                            xaxis_title="X (m)",
                            yaxis_title="Y (m)", 
                            zaxis_title="Z (m)",
                            aspectmode='data',
                            camera=dict(
                                eye=dict(x=1.5, y=1.5, z=1.2)
                            )
                        ),
                        width=800,
                        height=600,
                        margin=dict(r=0, b=0, l=0, t=40)
                    )
                    
                    # Display interactive plot
                    st.plotly_chart(fig_plotly, use_container_width=True)
                    
                    # Add viewing controls
                    st.write("**Interactive Controls:**")
                    st.write("🔄 **Rotate:** Click and drag")
                    st.write("🔍 **Zoom:** Mouse wheel or pinch")
                    st.write("📍 **Pan:** Shift + click and drag")
                    st.write("🎯 **Reset view:** Double-click")
                    
                else:
                    st.info("3D coordinates not available for this trajectory type.")
        else:
            st.info("No trajectory data available. Please generate a trajectory first.")
                            # Enhanced pass generation with smooth turnaround transitions
                            if (i_pass % 2) == 0:  # Forward pass (south to north)
                                pass_rho = np.array(base_rho_m)
                                pass_z = np.array(base_z_m)
                                # For forward pass, use original phi progression
                                pass_phi_relative = np.array(base_phi_rad) - base_phi_rad[0]
                            else:  # Backward pass (north to south)
                                pass_rho = np.array(base_rho_m[::-1])
                                pass_z = np.array(base_z_m[::-1])
                                # For backward pass, maintain smooth phi continuity
                                # Use pattern advancement from enhanced polar turnaround
                                pass_phi_relative = np.array(base_phi_rad[::-1]) - base_phi_rad[-1]
                                pass_phi_relative = -pass_phi_relative  # Flip direction for return
                            
                            # Apply smooth tangent vector continuity at turnarounds
                            if i_pass > 0:
                                # Add pattern advancement angle for proper spacing
                                pattern_advance = (2 * math.pi / num_passes) * 0.5  # Smooth distribution
                                cumulative_phi += pattern_advance
                            
                            # Calculate absolute phi with enhanced continuity
                            pass_phi_absolute = pass_phi_relative + cumulative_phi
                            
                            # Convert to Cartesian coordinates with smooth transitions
                            pass_x = pass_rho * np.cos(pass_phi_absolute)
                            pass_y = pass_rho * np.sin(pass_phi_absolute)
                            
                            # Enhanced joining point handling with C¹ tangent continuity
                            if i_pass == 0:
                                # First pass: include all points
                                start_idx = 0
                            else:
                                # Subsequent passes: ensure tangent vector continuity
                                start_idx = 1  # Skip duplicate point but maintain smoothness
                                
                                # Enhanced tangent vector continuity verification
                                if len(all_x_coords) > 1 and len(pass_x) > 1:
                                    # Calculate incoming tangent vector (end of previous pass)
                                    prev_dx = all_x_coords[-1] - all_x_coords[-2]
                                    prev_dy = all_y_coords[-1] - all_y_coords[-2]
                                    prev_dz = all_z_coords[-1] - all_z_coords[-2]
                                    
                                    # Calculate outgoing tangent vector (start of new pass)
                                    new_dx = pass_x[1] - pass_x[0]
                                    new_dy = pass_y[1] - pass_y[0]
                                    new_dz = pass_z[1] - pass_z[0]
                                    
                                    # Normalize tangent vectors for comparison
                                    prev_mag = math.sqrt(prev_dx**2 + prev_dy**2 + prev_dz**2)
                                    new_mag = math.sqrt(new_dx**2 + new_dy**2 + new_dz**2)
                                    
                                    if prev_mag > 1e-8 and new_mag > 1e-8:
                                        # Calculate 3D tangent vector alignment
                                        dot_product = (prev_dx * new_dx + prev_dy * new_dy + prev_dz * new_dz) / (prev_mag * new_mag)
                                        tangent_angle_deg = math.degrees(math.acos(np.clip(dot_product, -1, 1)))
                                        
                                        # Store tangent continuity metric for analysis
                                        if 'tangent_continuity_angles' not in locals():
                                            tangent_continuity_angles = []
                                        tangent_continuity_angles.append(tangent_angle_deg)
                            
                            # Append trajectory points with enhanced continuity
                            all_x_coords.extend(pass_x[start_idx:])
                            all_y_coords.extend(pass_y[start_idx:])
                            all_z_coords.extend(pass_z[start_idx:])
                            all_phi_continuous.extend(pass_phi_absolute[start_idx:])
                            
                            # Update cumulative phi for perfect continuity
                            cumulative_phi = pass_phi_absolute[-1]
                        
                        # Validate continuity at joining points
                        phi_jumps = []
                        for i in range(1, len(all_phi_continuous)):
                            phi_diff = abs(all_phi_continuous[i] - all_phi_continuous[i-1])
                            if phi_diff > 0.1:  # Threshold for detecting jumps
                                phi_jumps.append((i, phi_diff))
                        
                        # Display continuity diagnostic
                        if phi_jumps:
                            st.warning(f"⚠️ Found {len(phi_jumps)} potential discontinuities at joining points")
                            for idx, jump in phi_jumps[:3]:  # Show first 3
                                st.info(f"Point {idx}: φ jump = {math.degrees(jump):.1f}°")
                        else:
                            st.success("✅ Continuous trajectory validated - no discontinuities detected")
                        
                        # Plot the continuous trajectory
                        fig_plotly.add_trace(go.Scatter3d(
                            x=all_x_coords, y=all_y_coords, z=all_z_coords,
                            mode='lines+markers',
                            line=dict(color='red', width=4),
                            marker=dict(size=2, color='darkred'),
                            name=f'Continuous Path ({num_passes} passes)',
                            hovertemplate='<b>Continuous Winding Path</b><br>' +
                                          'X: %{x:.4f} m<br>' +
                                          'Y: %{y:.4f} m<br>' +
                                          'Z: %{z:.4f} m<extra></extra>'
                        ))
                        
                        # Add start and end markers
                        fig_plotly.add_trace(go.Scatter3d(
                            x=[all_x_coords[0]], y=[all_y_coords[0]], z=[all_z_coords[0]],
                            mode='markers',
                            marker=dict(size=10, color='green', symbol='diamond'),
                            name='Start Point'
                        ))
                        
                        fig_plotly.add_trace(go.Scatter3d(
                            x=[all_x_coords[-1]], y=[all_y_coords[-1]], z=[all_z_coords[-1]],
                            mode='markers',
                            marker=dict(size=10, color='blue', symbol='square'),
                            name='End Point'
                        ))
                        
                        # Enhanced trajectory analysis and statistics
                        total_points = len(all_x_coords)
                        total_phi_rotation = all_phi_continuous[-1] - all_phi_continuous[0]
                        
                        # Calculate path lengths between consecutive points
                        path_lengths = []
                        for i in range(1, len(all_x_coords)):
                            dx = all_x_coords[i] - all_x_coords[i-1]
                            dy = all_y_coords[i] - all_y_coords[i-1] 
                            dz = all_z_coords[i] - all_z_coords[i-1]
                            path_lengths.append(math.sqrt(dx*dx + dy*dy + dz*dz))
                        
                        total_path_length = sum(path_lengths) if path_lengths else 0
                        
                        # Identify joining points (transitions between passes)
                        joining_points = []
                        points_per_pass = len(base_rho_m)
                        for i in range(1, num_passes):
                            join_index = i * points_per_pass - (i-1)  # Account for skipped duplicate points
                            if join_index < len(all_x_coords):
                                joining_points.append(join_index)
                        
                        # Enhanced summary with continuity metrics
                        st.info(f"🎯 **Continuous Winding Analysis:**\n"
                               f"• {num_passes} passes with {total_points} trajectory points\n"
                               f"• Total φ rotation: {math.degrees(total_phi_rotation):.1f}°\n"
                               f"• Total path length: {total_path_length:.3f} m\n"
                               f"• Joining points: {len(joining_points)} transitions")
                        
                        # Enhanced joining point analysis with C¹ continuity metrics
                        if joining_points:
                            st.subheader("🔍 Enhanced Joining Point Analysis")
                            join_col1, join_col2 = st.columns(2)
                            
                            with join_col1:
                                st.write("**Transition Points:**")
                                for i, join_idx in enumerate(joining_points):
                                    if join_idx < len(all_phi_continuous):
                                        phi_at_join = all_phi_continuous[join_idx]
                                        st.write(f"Pass {i+1}→{i+2}: φ = {math.degrees(phi_at_join):.1f}°")
                            
                            with join_col2:
                                st.write("**C¹ Continuity Analysis:**")
                                max_gap = 0
                                smooth_transitions = 0
                                
                                for join_idx in joining_points:
                                    if join_idx > 0 and join_idx < len(path_lengths):
                                        gap = path_lengths[join_idx-1]
                                        max_gap = max(max_gap, gap)
                                        if gap <= 0.001:  # 1mm threshold
                                            smooth_transitions += 1
                                        else:
                                            st.warning(f"Gap at point {join_idx}: {gap*1000:.1f}mm")
                                
                                # Display tangent continuity results
                                if 'tangent_continuity_angles' in locals():
                                    avg_tangent_angle = sum(tangent_continuity_angles) / len(tangent_continuity_angles)
                                    max_tangent_angle = max(tangent_continuity_angles)
                                    
                                    if max_tangent_angle < 5.0:  # Less than 5° deviation
                                        st.success(f"✅ Tangent continuity: {avg_tangent_angle:.1f}° avg deviation")
                                    elif max_tangent_angle < 15.0:
                                        st.info(f"⚠️ Moderate tangent deviation: {max_tangent_angle:.1f}° max")
                                    else:
                                        st.warning(f"🔧 Tangent discontinuity: {max_tangent_angle:.1f}° max")
                                
                                if smooth_transitions == len(joining_points):
                                    st.success("✅ All transitions C⁰ continuous")
                                else:
                                    st.info(f"Position continuity: {smooth_transitions}/{len(joining_points)} smooth")
                    
                    else:
                        # Fallback to single trajectory if calculations fail
                        fig_plotly.add_trace(go.Scatter3d(
                                x=x_m, y=y_m, z=z_m,
                                mode='lines+markers',
                                line=dict(color='red', width=6),
                                marker=dict(size=3, color='darkred'),
                                name='Single Geodesic Path',
                                hovertemplate='<b>Geodesic Path</b><br>' +
                                              'X: %{x:.4f} m<br>' +
                                              'Y: %{y:.4f} m<br>' +
                                              'Z: %{z:.4f} m<extra></extra>'
                            ))
                    
                    # Plot vessel outline in 3D (with error handling)
                    try:
                        vessel_profile = st.session_state.vessel_geometry.get_profile_points()
                        
                        # Check which keys are available and use the correct ones
                        if 'r_inner' in vessel_profile:
                            r_vessel = np.array(vessel_profile['r_inner']) / 1000  # Already in mm, convert to meters
                            z_vessel = np.array(vessel_profile['z']) / 1000
                        elif 'r_inner_mm' in vessel_profile:
                            r_vessel = np.array(vessel_profile['r_inner_mm']) / 1000  # Convert to meters
                            z_vessel = np.array(vessel_profile['z_mm']) / 1000
                        else:
                            # Generate simple cylinder outline as fallback
                            z_range = np.linspace(-0.2, 0.2, 10)
                            r_vessel = np.full_like(z_range, 0.1)  # 100mm radius
                            z_vessel = z_range
                    except Exception as e:
                        # Simple fallback outline
                        z_vessel = np.linspace(-0.2, 0.2, 10) 
                        r_vessel = np.full_like(z_vessel, 0.1)
                    
                    # Create circular cross-sections at various z-positions
                    theta_circle = np.linspace(0, 2*np.pi, 30)
                    outline_colors = ['lightgray', 'silver', 'darkgray']
                    
                    for i, step in enumerate(range(0, len(z_vessel), 15)):  # Every 15th point
                        if step < len(z_vessel):
                            z_level = z_vessel[step]
                            r_level = r_vessel[step]
                            x_circle = r_level * np.cos(theta_circle)
                            y_circle = r_level * np.sin(theta_circle)
                            z_circle = np.full_like(x_circle, z_level)
                            
                            fig_plotly.add_trace(go.Scatter3d(
                                x=x_circle, y=y_circle, z=z_circle,
                                mode='lines',
                                line=dict(color=outline_colors[i % len(outline_colors)], 
                                        width=2, dash='dash'),
                                name='Vessel Outline',
                                showlegend=(i == 0),
                                hoverinfo='skip'
                            ))
                    
                    # Enhanced layout with better controls
                    fig_plotly.update_layout(
                        title=dict(
                            text="🚀 Interactive 3D Geodesic Trajectory",
                            x=0.5,
                            font=dict(size=18)
                        ),
                        scene=dict(
                            xaxis_title="X Coordinate (m)",
                            yaxis_title="Y Coordinate (m)", 
                            zaxis_title="Z Coordinate (m)",
                            aspectmode='cube',
                            camera=dict(
                                eye=dict(x=1.8, y=1.8, z=1.2),
                                center=dict(x=0, y=0, z=0)
                            ),
                            bgcolor='white'
                        ),
                        width=900,
                        height=700,
                        margin=dict(l=0, r=0, b=0, t=50),
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=0.01
                        )
                    )
                    
                    # Display the interactive plot
                    st.plotly_chart(fig_plotly, use_container_width=True)
                    
                    # Add helpful interaction instructions
                    st.info("🎯 **Interactive Controls:** "
                           "• **Rotate:** Click and drag to rotate the view "
                           "• **Zoom:** Scroll wheel to zoom in/out "
                           "• **Pan:** Hold Shift + drag to pan "
                           "• **Reset:** Double-click to reset view "
                           "• **Hover:** Mouse over points for coordinates")
                    
                    # Add trajectory debugging section
                    st.subheader("🔍 Trajectory Analysis & Debugging")
                    
                    # Coordinate system tabs
                    tab1, tab2, tab3 = st.tabs(["📊 Data Summary", "🌐 Cylindrical Coordinates", "📐 Cartesian Coordinates"])
                    
                    with tab1:
                        col_info1, col_info2 = st.columns(2)
                        with col_info1:
                            st.metric("Total Path Points", f"{len(x_points)}")
                            st.metric("X Range", f"{min(x_points):.1f} to {max(x_points):.1f} mm")
                            st.metric("Y Range", f"{min(y_points):.1f} to {max(y_points):.1f} mm")
                        with col_info2:
                            st.metric("Z Range", f"{min(z_points):.1f} to {max(z_points):.1f} mm")
                            
                            # Handle different data structures for single vs multi-circuit
                            if 'rho_points' in st.session_state.trajectory_data:
                                st.metric("Max Radius", f"{max(st.session_state.trajectory_data['rho_points']):.1f} mm")
                                st.metric("Phi Range", f"{min(st.session_state.trajectory_data['phi_rad']):.2f} to {max(st.session_state.trajectory_data['phi_rad']):.2f} rad")
                            else:
                                # Calculate rho from x,y coordinates for multi-circuit data
                                rho_values = [math.sqrt(x**2 + y**2) for x, y in zip(x_points, y_points)]
                                st.metric("Max Radius", f"{max(rho_values):.1f} mm")
                                if 'phi_rad_continuous' in st.session_state.trajectory_data:
                                    phi_values = st.session_state.trajectory_data['phi_rad_continuous']
                                    st.metric("Phi Range", f"{min(phi_values):.2f} to {max(phi_values):.2f} rad")
                    
                    with tab2:
                        st.write("**Cylindrical Coordinate Analysis (ρ, z, φ)**")
                        
                        # Plot cylindrical coordinates
                        fig_cyl, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
                        
                        # Handle different data structures for single vs multi-circuit
                        if 'rho_points' in st.session_state.trajectory_data:
                            # Single circuit data
                            rho_data = st.session_state.trajectory_data['rho_points']
                            phi_data = st.session_state.trajectory_data['phi_rad']
                            alpha_data = st.session_state.trajectory_data['alpha_deg']
                        else:
                            # Multi-circuit data - calculate from x,y coordinates
                            rho_data = [math.sqrt(x**2 + y**2) for x, y in zip(x_points, y_points)]
                            phi_data = st.session_state.trajectory_data.get('phi_rad_continuous', [])
                            # Calculate alpha for multi-circuit
                            c_eff_mm = st.session_state.trajectory_data.get('c_eff_m', 0.04) * 1000
                            alpha_data = [math.degrees(math.asin(min(c_eff_mm / rho, 1.0))) if rho > 0 else 90.0 for rho in rho_data]
                        
                        # ρ vs point index
                        ax1.plot(rho_data, 'b-', linewidth=2)
                        ax1.set_xlabel('Point Index')
                        ax1.set_ylabel('ρ (mm)')
                        ax1.set_title('Radial Coordinate vs Index')
                        ax1.grid(True, alpha=0.3)
                        
                        # z vs point index  
                        ax2.plot(z_points, 'g-', linewidth=2)
                        ax2.set_xlabel('Point Index')
                        ax2.set_ylabel('z (mm)')
                        ax2.set_title('Axial Coordinate vs Index')
                        ax2.grid(True, alpha=0.3)
                        
                        # φ vs point index
                        if len(phi_data) > 0:
                            ax3.plot(phi_data, 'r-', linewidth=2)
                        ax3.set_xlabel('Point Index')
                        ax3.set_ylabel('φ (radians)')
                        ax3.set_title('Parallel Angle vs Index')
                        ax3.grid(True, alpha=0.3)
                        
                        # α vs ρ
                        ax4.plot(rho_data, alpha_data, 'm-', linewidth=2)
                        ax4.set_xlabel('ρ (mm)')
                        ax4.set_ylabel('α (degrees)')
                        ax4.set_title('Winding Angle vs Radius')
                        ax4.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig_cyl)
                        
                        # Show all trajectory points for complete debugging
                        traj_data = st.session_state.trajectory_data
                        
                        # Check if this is multi-circuit data
                        if 'num_circuits' in traj_data and traj_data['num_circuits'] > 1:
                            st.write(f"**All Trajectory Points - {traj_data['num_circuits']} Circuits ({traj_data['total_points']} total points)**")
                            debug_data_cyl = []
                            
                            # Use multi-circuit data - check if arrays exist
                            if 'x_points' in traj_data and len(traj_data['x_points']) > 0:
                                x_multi = traj_data['x_points']
                                y_multi = traj_data['y_points'] 
                                z_multi = traj_data['z_coords']
                                phi_multi = traj_data['phi_rad_continuous']
                                
                                # Calculate rho and alpha for each point
                                for i in range(len(x_multi)):
                                    # Convert from meters to millimeters
                                    rho_i = math.sqrt(x_multi[i]**2 + y_multi[i]**2) * 1000  # Convert to mm
                                    z_i = z_multi[i] * 1000  # Convert to mm
                                    c_eff_mm = traj_data['c_eff_m'] * 1000  # Convert to mm
                                    
                                    alpha_i = math.degrees(math.asin(min(c_eff_mm / rho_i, 1.0))) if rho_i > 0 else 90.0
                                    circuit_num = (i // traj_data['points_per_circuit']) + 1
                                    
                                    debug_data_cyl.append({
                                        "Point": i,
                                        "Circuit": circuit_num,
                                        "ρ (mm)": f"{rho_i:.3f}",
                                        "z (mm)": f"{z_i:.3f}",
                                        "φ (rad)": f"{phi_multi[i]:.4f}",
                                        "φ (deg)": f"{math.degrees(phi_multi[i]):.2f}",
                                        "α (deg)": f"{alpha_i:.2f}"
                                    })
                            else:
                                st.error("Multi-circuit data structure not found. Please regenerate trajectory.")
                                # Fall back to available data keys
                                st.write("Available data keys:", list(traj_data.keys()))
                        else:
                            st.write("**All Trajectory Points (Single Circuit - Cylindrical Coordinates)**")
                            debug_data_cyl = []
                            for i in range(len(x_points)):
                                debug_data_cyl.append({
                                    "Point": i,
                                    "ρ (mm)": f"{st.session_state.trajectory_data['rho_points'][i]:.3f}",
                                    "z (mm)": f"{z_points[i]:.3f}",
                                    "φ (rad)": f"{st.session_state.trajectory_data['phi_rad'][i]:.4f}",
                                    "φ (deg)": f"{math.degrees(st.session_state.trajectory_data['phi_rad'][i]):.2f}",
                                    "α (deg)": f"{st.session_state.trajectory_data['alpha_deg'][i]:.2f}"
                                })
                        
                        st.dataframe(debug_data_cyl, height=400)
                    
                    with tab3:
                        st.write("**Cartesian Coordinate Analysis (x, y, z)**")
                        
                        # Plot Cartesian coordinates
                        fig_cart, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
                        
                        # x vs point index
                        ax1.plot(x_points, 'b-', linewidth=2)
                        ax1.set_xlabel('Point Index')
                        ax1.set_ylabel('x (mm)')
                        ax1.set_title('X Coordinate vs Index')
                        ax1.grid(True, alpha=0.3)
                        
                        # y vs point index
                        ax2.plot(y_points, 'g-', linewidth=2)
                        ax2.set_xlabel('Point Index')
                        ax2.set_ylabel('y (mm)')
                        ax2.set_title('Y Coordinate vs Index')
                        ax2.grid(True, alpha=0.3)
                        
                        # x vs y (top view)
                        ax3.plot(x_points, y_points, 'r-', linewidth=1, alpha=0.7)
                        ax3.scatter(x_points[0], y_points[0], color='green', s=100, label='Start')
                        ax3.scatter(x_points[-1], y_points[-1], color='red', s=100, label='End')
                        ax3.set_xlabel('x (mm)')
                        ax3.set_ylabel('y (mm)')
                        ax3.set_title('Top View (x-y plane)')
                        ax3.legend()
                        ax3.grid(True, alpha=0.3)
                        ax3.axis('equal')
                        
                        # Distance from origin
                        distances = [math.sqrt(x**2 + y**2) for x, y in zip(x_points, y_points)]
                        ax4.plot(distances, 'purple', linewidth=2)
                        ax4.set_xlabel('Point Index')
                        ax4.set_ylabel('Distance from Z-axis (mm)')
                        ax4.set_title('Radial Distance vs Index')
                        ax4.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig_cart)
                        
                        # Show all trajectory points for complete debugging
                        st.write("**All Trajectory Points (Cartesian Coordinates)**")
                        debug_data_cart = []
                        for i in range(len(x_points)):
                            distance_from_axis = math.sqrt(x_points[i]**2 + y_points[i]**2)
                            debug_data_cart.append({
                                "Point": i,
                                "x (mm)": f"{x_points[i]:.3f}",
                                "y (mm)": f"{y_points[i]:.3f}",
                                "z (mm)": f"{z_points[i]:.3f}",
                                "R_dist (mm)": f"{distance_from_axis:.3f}",
                                "Angle (deg)": f"{math.degrees(math.atan2(y_points[i], x_points[i])):.1f}"
                            })
                        st.dataframe(debug_data_cart, height=400)
                    
                    # Display geodesic parameters
                    st.subheader("Geodesic Parameters")
                    col_geo1, col_geo2 = st.columns(2)
                    with col_geo1:
                        if 'c_eff_m' in st.session_state.trajectory_data:
                            st.metric("Effective Polar Opening", f"{st.session_state.trajectory_data['c_eff_m']*1000:.2f} mm")
                        if 'alpha_equator_deg' in st.session_state.trajectory_data:
                            st.metric("Equatorial Winding Angle", f"{st.session_state.trajectory_data['alpha_equator_deg']:.1f}°")
                    
                    with col_geo2:
                        if 'turn_around_angle_deg' in st.session_state.trajectory_data:
                            st.metric("Turn-around Angle", f"{st.session_state.trajectory_data['turn_around_angle_deg']:.1f}°")
                        st.metric("Path Points", f"{len(x_points)}")
                else:
                    st.info("3D coordinate data not available for this trajectory type.")
            
            # Display trajectory properties
            st.subheader("Trajectory Properties")
            traj_props = st.session_state.trajectory_data
            
            col_a, col_b = st.columns(2)
            with col_a:
                if 'total_fiber_length' in traj_props:
                    st.metric("Total Fiber Length", f"{traj_props['total_fiber_length']:.2f} m")
                if 'winding_time' in traj_props:
                    st.metric("Estimated Winding Time", f"{traj_props['winding_time']:.2f} min")
                if 'number_of_circuits' in traj_props:
                    st.metric("Number of Circuits", f"{traj_props['number_of_circuits']}")
            
            with col_b:
                if 'coverage_efficiency' in traj_props:
                    st.metric("Coverage Efficiency", f"{traj_props['coverage_efficiency']:.1f}%")
                if 'mandrel_rotations' in traj_props:
                    st.metric("Total Mandrel Rotations", f"{traj_props['mandrel_rotations']:.1f}")
                if 'fiber_utilization' in traj_props:
                    st.metric("Fiber Utilization", f"{traj_props['fiber_utilization']:.1f}%")
        else:
            st.write("❌ No trajectory data available in session state")

def performance_analysis_page():
    st.header("Performance Analysis")
    
    if st.session_state.vessel_geometry is None:
        st.warning("Please generate vessel geometry first.")
        return
    
    # Stress analysis
    st.subheader("Stress Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Loading Conditions")
        analysis_pressure = st.number_input("Analysis Pressure (MPa)", min_value=0.1, value=30.0, step=0.1)
        axial_load = st.number_input("Axial Load (N)", value=0.0, step=100.0)
        
        # Material properties for analysis
        if 'fiber_type' in locals() and 'resin_type' in locals():
            st.markdown("### Material Properties")
            st.write(f"Fiber: {fiber_type if 'fiber_type' in locals() else 'Not selected'}")
            st.write(f"Resin: {resin_type if 'resin_type' in locals() else 'Not selected'}")
        
        if st.button("Perform Stress Analysis"):
            try:
                calculator = VesselCalculations()
                stress_results = calculator.calculate_vessel_stresses(
                    st.session_state.vessel_geometry,
                    analysis_pressure,
                    axial_load
                )
                
                st.subheader("Stress Results")
                
                # Hoop stress
                st.write(f"**Hoop Stress (Cylindrical):** {stress_results['hoop_stress_cyl']:.2f} MPa")
                st.write(f"**Axial Stress (Cylindrical):** {stress_results['axial_stress_cyl']:.2f} MPa")
                
                # Dome stresses
                if 'dome_stress_max' in stress_results:
                    st.write(f"**Maximum Dome Stress:** {stress_results['dome_stress_max']:.2f} MPa")
                
                # Safety factors
                if 'safety_factor_hoop' in stress_results:
                    st.write(f"**Safety Factor (Hoop):** {stress_results['safety_factor_hoop']:.2f}")
                if 'safety_factor_axial' in stress_results:
                    st.write(f"**Safety Factor (Axial):** {stress_results['safety_factor_axial']:.2f}")
                
            except Exception as e:
                st.error(f"Error in stress analysis: {str(e)}")
    
    with col2:
        st.subheader("Performance Metrics")
        
        # Calculate basic performance metrics
        try:
            calculator = VesselCalculations()
            performance = calculator.calculate_performance_metrics(st.session_state.vessel_geometry)
            
            st.metric("Weight Efficiency (Volume/Weight)", f"{performance.get('volume_weight_ratio', 0):.2f} L/kg")
            st.metric("Burst Pressure (Est.)", f"{performance.get('estimated_burst_pressure', 0):.1f} MPa")
            st.metric("PV/W Ratio", f"{performance.get('pv_w_ratio', 0):.2f} kJ/kg")
            
        except Exception as e:
            st.error(f"Error calculating performance metrics: {str(e)}")

def export_results_page():
    st.header("Export Results")
    
    if st.session_state.vessel_geometry is None:
        st.warning("No vessel geometry data to export. Please generate geometry first.")
        return
    
    st.subheader("Available Data for Export")
    
    # Geometry data
    if st.session_state.vessel_geometry is not None:
        st.write("✅ Vessel Geometry Data")
        
        if st.button("Export Geometry as CSV"):
            try:
                # Get profile points
                profile_data = st.session_state.vessel_geometry.get_profile_points()
                df = pd.DataFrame(profile_data)
                
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Geometry CSV",
                    data=csv,
                    file_name="vessel_geometry.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error exporting geometry: {str(e)}")
    
    # Trajectory data
    if st.session_state.trajectory_data is not None:
        st.write("✅ Trajectory Data")
        
        if st.button("Export Trajectory as CSV"):
            try:
                # Convert trajectory data to DataFrame
                if 'path_points' in st.session_state.trajectory_data:
                    df = pd.DataFrame(st.session_state.trajectory_data['path_points'])
                    
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Trajectory CSV",
                        data=csv,
                        file_name="winding_trajectory.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"Error exporting trajectory: {str(e)}")
    
    # Combined report
    st.subheader("Design Report")
    if st.button("Generate Design Report"):
        try:
            report_content = generate_design_report()
            st.download_button(
                label="Download Design Report",
                data=report_content,
                file_name="copv_design_report.txt",
                mime="text/plain"
            )
        except Exception as e:
            st.error(f"Error generating report: {str(e)}")

def generate_design_report():
    """Generate a comprehensive design report"""
    report = []
    report.append("COMPOSITE PRESSURE VESSEL DESIGN REPORT")
    report.append("=" * 50)
    report.append("")
    
    if st.session_state.vessel_geometry is not None:
        props = st.session_state.vessel_geometry.get_geometric_properties()
        report.append("VESSEL GEOMETRY:")
        for key, value in props.items():
            report.append(f"  {key}: {value}")
        report.append("")
    
    if st.session_state.trajectory_data is not None:
        report.append("TRAJECTORY PARAMETERS:")
        for key, value in st.session_state.trajectory_data.items():
            if key != 'path_points':  # Skip the large array
                report.append(f"  {key}: {value}")
        report.append("")
    
    report.append(f"Report generated: {pd.Timestamp.now()}")
    
    return "\n".join(report)

if __name__ == "__main__":
    main()

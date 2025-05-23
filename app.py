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
            num_points = st.number_input("Number of Path Points", min_value=50, value=100, step=10)
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
                        roving_eccentricity_at_pole_m=polar_eccentricity/1000
                    )
                    trajectory_params = {
                        'pattern_type': pattern_type,
                        'num_points': int(num_points)
                    }
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
            visualizer = VesselVisualizer()
            
            # Plot trajectory - 2D view
            fig = visualizer.plot_winding_trajectory(
                st.session_state.vessel_geometry,
                st.session_state.trajectory_data
            )
            st.pyplot(fig)
            
            # Add 3D visualization for geodesic trajectories
            if st.session_state.trajectory_data.get('pattern_type') == 'Geodesic':
                st.subheader("3D Trajectory Visualization")
                
                # Check if we have 3D coordinate data
                if 'x_points' in st.session_state.trajectory_data and 'y_points' in st.session_state.trajectory_data:
                    import matplotlib.pyplot as plt
                    from mpl_toolkits.mplot3d import Axes3D
                    
                    # Create 3D plot
                    fig_3d = plt.figure(figsize=(12, 10))
                    ax = fig_3d.add_subplot(111, projection='3d')
                    
                    # Get trajectory data
                    x_points = st.session_state.trajectory_data['x_points']
                    y_points = st.session_state.trajectory_data['y_points'] 
                    z_points = st.session_state.trajectory_data['z_coords']
                    
                    # Create interactive 3D visualization with Plotly
                    import plotly.graph_objects as go
                    
                    # Points are already in meters from trajectory calculation
                    x_m = x_points
                    y_m = y_points  
                    z_m = z_points
                    
                    # Create interactive plot
                    fig_plotly = go.Figure()
                    
                    # Generate continuous trajectory path (realistic winding simulation)
                    import math
                    
                    # Get base trajectory data for one pass
                    base_rho_m = st.session_state.trajectory_data.get('rho_points', [])
                    base_z_m = st.session_state.trajectory_data.get('z_coords', [])
                    base_phi_rad = st.session_state.trajectory_data.get('phi_rad', [])
                    
                    # Number of passes to simulate (back-and-forth motion)
                    st.subheader("Continuous Winding Parameters")
                    num_passes = st.slider("Number of Passes", min_value=1, max_value=8, value=4, 
                                         help="Number of pole-to-pole passes to simulate")
                    
                    if len(base_rho_m) > 0 and len(base_phi_rad) > 0:
                        # Calculate delta_phi for one complete pass
                        delta_phi_one_pass = base_phi_rad[-1] - base_phi_rad[0]
                        
                        # Generate continuous trajectory
                        all_x_coords, all_y_coords, all_z_coords = [], [], []
                        current_phi_offset = 0.0
                        
                        for i_pass in range(num_passes):
                            # Determine direction: forward (even) or backward (odd)
                            if (i_pass % 2) == 0:  # Forward pass
                                pass_rho = np.array(base_rho_m)
                                pass_z = np.array(base_z_m)
                                pass_phi_relative = np.array(base_phi_rad)
                            else:  # Backward pass (reverse direction)
                                pass_rho = np.array(base_rho_m[::-1])
                                pass_z = np.array(base_z_m[::-1])
                                pass_phi_relative = np.array(base_phi_rad)
                            
                            # Calculate absolute phi for this pass
                            pass_phi_absolute = pass_phi_relative + current_phi_offset
                            
                            # Convert to Cartesian coordinates
                            pass_x = pass_rho * np.cos(pass_phi_absolute)
                            pass_y = pass_rho * np.sin(pass_phi_absolute)
                            
                            # Append points (skip first point for subsequent passes to avoid duplication)
                            start_idx = 1 if i_pass > 0 else 0
                            all_x_coords.extend(pass_x[start_idx:])
                            all_y_coords.extend(pass_y[start_idx:])
                            all_z_coords.extend(pass_z[start_idx:])
                            
                            # Update phi offset for next pass
                            current_phi_offset += delta_phi_one_pass
                        
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
                        
                        # Calculate total path statistics
                        total_phi_rotation = current_phi_offset
                        total_points = len(all_x_coords)
                        
                        # Add summary
                        st.info(f"🎯 **Continuous Winding Simulation:** {num_passes} passes • "
                               f"{total_points} trajectory points • "
                               f"Total φ rotation: {math.degrees(total_phi_rotation):.1f}°")
                    
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
                            st.metric("Max Radius", f"{max(st.session_state.trajectory_data['rho_points']):.1f} mm")
                            st.metric("Phi Range", f"{min(st.session_state.trajectory_data['phi_rad']):.2f} to {max(st.session_state.trajectory_data['phi_rad']):.2f} rad")
                    
                    with tab2:
                        st.write("**Cylindrical Coordinate Analysis (ρ, z, φ)**")
                        
                        # Plot cylindrical coordinates
                        fig_cyl, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
                        
                        # ρ vs point index
                        ax1.plot(st.session_state.trajectory_data['rho_points'], 'b-', linewidth=2)
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
                        ax3.plot(st.session_state.trajectory_data['phi_rad'], 'r-', linewidth=2)
                        ax3.set_xlabel('Point Index')
                        ax3.set_ylabel('φ (radians)')
                        ax3.set_title('Parallel Angle vs Index')
                        ax3.grid(True, alpha=0.3)
                        
                        # α vs ρ
                        ax4.plot(st.session_state.trajectory_data['rho_points'], st.session_state.trajectory_data['alpha_deg'], 'm-', linewidth=2)
                        ax4.set_xlabel('ρ (mm)')
                        ax4.set_ylabel('α (degrees)')
                        ax4.set_title('Winding Angle vs Radius')
                        ax4.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig_cyl)
                        
                        # Show all trajectory points for complete debugging
                        st.write("**All Trajectory Points (Cylindrical Coordinates)**")
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
            st.info("Please calculate trajectory to see visualization and properties.")

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

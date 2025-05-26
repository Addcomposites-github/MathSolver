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
        pattern_type = st.selectbox("Winding Pattern", ["Geodesic", "Non-Geodesic", "Multi-Circuit Pattern", "Helical", "Hoop", "Polar", "Transitional"])
        
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
            
            st.markdown("### ⚙️ Advanced Physics")
            friction_coefficient = st.slider("Friction Coefficient (μ)", min_value=0.0, max_value=1.0, value=0.0, step=0.05,
                                            help="Coefficient of friction between fiber and mandrel. 0.0 = Pure geodesic paths, >0.0 = Non-geodesic with realistic physics")
            if friction_coefficient > 0:
                st.info(f"🔬 **Non-Geodesic Mode**: μ = {friction_coefficient:.2f} - Fibers can deviate from pure geodesic paths")
            else:
                st.info("🎯 **Pure Geodesic Mode**: Fibers follow shortest paths on surface")
            
            st.markdown("### 🔄 Continuous Winding Configuration")
            st.info("🔧 **Continuous Winding Mode**: Single continuous filament path with multiple passes for complete coverage")
            num_circuits = 1  # Always single circuit with multiple passes
            
            # Calculation parameters
            dome_points = st.number_input("Points per Dome Segment", min_value=20, max_value=300, value=150, step=10)
            cylinder_points = st.number_input("Points per Cylinder Segment", min_value=5, max_value=100, value=20, step=5)
            
        elif pattern_type == "Non-Geodesic":
            st.markdown("### 🔬 Non-Geodesic Parameters")
            st.info("🚀 **Advanced Mode**: No geometric limitations - explore extreme angles with friction physics!")
            
            roving_width = st.number_input("Roving Width (mm)", min_value=0.1, value=3.0, step=0.1)
            roving_thickness = st.number_input("Roving Thickness (mm)", min_value=0.01, value=0.2, step=0.01)
            polar_eccentricity = st.number_input("Polar Eccentricity (mm)", min_value=0.0, value=0.0, step=0.1)
            
            st.markdown("### 🎯 Extreme Winding Angles")
            st.warning("⚠️ **No Limits Mode**: Can specify any angle - friction physics will handle feasibility")
            target_angle = st.slider("Target Cylinder Angle (degrees)", 
                                    min_value=5.0, max_value=89.0, value=18.0, step=1.0,
                                    help="Any angle allowed! Friction coefficient will determine path feasibility")
            st.info(f"🎯 **Extreme Target**: {target_angle}° - Testing physics limits!")
            
            st.markdown("### ⚙️ Friction Physics Engine")
            friction_coefficient = st.slider("Friction Coefficient (μ)", min_value=0.05, max_value=2.0, value=0.3, step=0.05,
                                            help="Higher friction enables more extreme angles and non-geodesic paths")
            st.success(f"🔬 **Friction Physics**: μ = {friction_coefficient:.2f} - Solving Koussios Eq. 5.62")
            
            if friction_coefficient < 0.1:
                st.warning("⚠️ Low friction - may struggle with extreme angles")
            elif friction_coefficient > 1.0:
                st.info("🚀 High friction - enables very aggressive winding patterns")
            
            st.markdown("### 🔄 Advanced Configuration")
            st.info("🔧 **Non-Geodesic Mode**: Advanced differential equation solving with surface curvature")
            
            # Calculation parameters  
            dome_points = st.number_input("Points per Dome Segment", min_value=20, max_value=300, value=150, step=10)
            cylinder_points = st.number_input("Points per Cylinder Segment", min_value=5, max_value=100, value=20, step=5)
            

        elif pattern_type == "Multi-Circuit Pattern":
            st.markdown("### 🔄 Multi-Circuit Pattern Configuration")
            st.info("🎯 **Full Coverage System**: Generate systematic winding patterns for complete pole-to-pole vessel coverage")
            
            # Coverage level selection
            coverage_level = st.selectbox(
                "Coverage Level",
                ["Quick Preview (3-5 circuits)", "Moderate Coverage (8-12 circuits)", "Full Coverage (15-20 circuits)", "Custom"],
                help="Choose predefined coverage levels or customize your own"
            )
            
            # Set default values based on coverage level
            if coverage_level == "Quick Preview (3-5 circuits)":
                default_total = 8
                default_generate = 4
            elif coverage_level == "Moderate Coverage (8-12 circuits)":
                default_total = 12
                default_generate = 8
            elif coverage_level == "Full Coverage (15-20 circuits)":
                default_total = 18
                default_generate = 12
            else:  # Custom
                default_total = 10
                default_generate = 6
            
            # Roving & Band Properties (mathematically connected)
            st.markdown("### 🧵 Roving & Band Properties")
            col_roving1, col_roving2 = st.columns(2)
            with col_roving1:
                roving_width = st.number_input("Roving Width (mm)", min_value=1.0, max_value=20.0, value=3.0, step=0.1,
                                             help="Dry roving width in millimeters")
                roving_thickness = st.number_input("Roving Thickness (μm)", min_value=50, max_value=1000, value=200, step=10,
                                                 help="Dry roving thickness in micrometers")
            with col_roving2:
                num_rovings = st.number_input("Rovings per Band", min_value=1, max_value=10, value=1, step=1,
                                            help="Number of rovings laid side by side in each band")
                band_width = roving_width * num_rovings
                st.metric("Calculated Band Width", f"{band_width:.1f}mm", 
                         help="Band width = Roving width × Number of rovings")
                polar_eccentricity = st.number_input("Polar Eccentricity (mm)", min_value=0.0, max_value=10.0, value=0.0, step=0.1,
                                                    help="Polar opening eccentricity for asymmetric domes")

            # Pattern configuration
            col_pattern1, col_pattern2 = st.columns(2)
            with col_pattern1:
                # Get real Koussios calculations using the effective band width
                if st.session_state.vessel_geometry is not None:
                    try:
                        # Create planner with current roving parameters (using effective band width)
                        planner = TrajectoryPlanner(st.session_state.vessel_geometry, 
                                                  dry_roving_width_m=band_width*1e-3,  # Use calculated band width
                                                  dry_roving_thickness_m=roving_thickness*1e-6)  # Convert μm to m
                        pattern_info = planner.calculate_koussios_pattern_parameters()
                        max_theoretical = max(100, int(pattern_info['n_bands_target'] * 2))  # Allow 2x theoretical for flexibility
                        recommended_circuits = pattern_info['recommended_solution']['p_circuits']
                        
                        # Show real-time update message
                        st.info(f"🔄 **Live Calculation**: {band_width:.1f}mm band → {pattern_info['n_bands_target']} optimal bands → {recommended_circuits} circuits needed")
                    except Exception as e:
                        st.warning(f"Calculation error: {str(e)}")
                        max_theoretical = 100
                        recommended_circuits = default_total
                else:
                    max_theoretical = 100
                    recommended_circuits = default_total
                
                circuits_to_close = st.number_input("Target Circuits for Full Pattern", 
                                                   min_value=4, max_value=max_theoretical, value=min(recommended_circuits, max_theoretical), step=1,
                                                   help=f"Total circuits needed for complete coverage (Theoretical optimal: {recommended_circuits})")
                num_circuits_for_vis = st.number_input("Circuits to Generate for Visualization", 
                                                     min_value=1, max_value=min(50, circuits_to_close), value=min(default_generate, circuits_to_close), step=1,
                                                     help="Number of circuits to actually calculate and display")
            with col_pattern2:
                pattern_skip_factor = st.selectbox("Pattern Advancement", 
                                                 options=[1, 2, 3], 
                                                 format_func=lambda x: {1: "Side-by-side (Dense)", 2: "Skip 1 band (Medium)", 3: "Skip 2 bands (Sparse)"}[x],
                                                 help="Controls spacing between adjacent circuits")
                
                # Show estimated coverage and pattern insights
                estimated_coverage = min(100, (num_circuits_for_vis / circuits_to_close) * 100)
                st.metric("Estimated Coverage", f"{estimated_coverage:.0f}%", 
                         help="Percentage of vessel surface covered by generated circuits")
                
                # Show Koussios pattern insights when vessel geometry is available
                if st.session_state.vessel_geometry is not None:
                    try:
                        # Use current band parameters for live calculations
                        planner = TrajectoryPlanner(st.session_state.vessel_geometry,
                                                  dry_roving_width_m=band_width*1e-3,  # Use calculated band width
                                                  dry_roving_thickness_m=roving_thickness*1e-6)  # Convert μm to m
                        pattern_info = planner.calculate_koussios_pattern_parameters()
                        
                        with st.expander("📊 Koussios Pattern Analysis", expanded=False):
                            st.markdown("**Mathematical Analysis Based on Vessel Geometry & Roving Properties**")
                            
                            col_theory1, col_theory2, col_theory3 = st.columns(3)
                            with col_theory1:
                                st.metric("Equatorial Radius", f"{pattern_info['equatorial_radius_m']*1000:.0f}mm")
                                st.metric("Winding Angle α", f"{pattern_info['alpha_equator_deg']:.1f}°")
                                st.metric("Roving Width", f"{roving_width:.1f}mm")
                                st.metric("Rovings per Band", f"{num_rovings}")
                            
                            with col_theory2:
                                st.metric("Effective Band Width", f"{pattern_info['B_eff_equator_m']:.2f}mm")
                                st.metric("Band Subtended Angle", f"{pattern_info['delta_phi_band_deg']:.2f}°")
                                st.metric("Theoretical Bands Needed", f"{pattern_info['n_bands_theoretical']:.1f}")
                            
                            with col_theory3:
                                st.metric("Optimal Target Bands", f"{pattern_info['n_bands_target']}")
                                circumference = 2 * 3.14159 * pattern_info['equatorial_radius_m'] * 1000
                                st.metric("Equatorial Circumference", f"{circumference:.0f}mm")
                                coverage_per_band = (pattern_info['B_eff_equator_m'] / circumference) * 100
                                st.metric("Coverage per Band", f"{coverage_per_band:.1f}%")
                                
                            st.markdown("**Pattern Solutions (Mathematical)**")
                            
                            # Create a more detailed pattern solutions table
                            pattern_data = []
                            for solution in pattern_info['pattern_solutions']:
                                pattern_data.append({
                                    "Pattern Type": solution['type'],
                                    "Required Circuits": solution['p_circuits'],
                                    "Advancement per Circuit": f"{solution['delta_phi_deg']:.1f}°",
                                    "Coverage Efficiency": f"{solution['coverage_efficiency']:.0%}",
                                    "Skip Factor": solution['pattern_skip_factor']
                                })
                            
                            pattern_df = pd.DataFrame(pattern_data)
                            st.dataframe(pattern_df, use_container_width=True, hide_index=True)
                            
                            st.info(f"🎯 **Recommended**: {pattern_info['recommended_solution']['type']} pattern with "
                                   f"{pattern_info['recommended_solution']['p_circuits']} circuits for optimal coverage")
                    except:
                        pass  # Don't show if planner can't be created yet
            

            
            # Target angle configuration
            st.markdown("### 🎯 Target Winding Angle")
            use_target_angle = st.checkbox("Specify Target Cylinder Angle", value=True,
                                         help="Define desired winding angle instead of using geometric limit")
            
            target_angle = None
            if use_target_angle:
                target_angle = st.slider("Target Cylinder Angle (degrees)", 
                                        min_value=10.0, max_value=80.0, value=45.0, step=1.0,
                                        help="Desired winding angle on cylinder section")
                st.info(f"🎯 **Target**: {target_angle}° winding angle on cylinder")
            else:
                st.info("🔧 **Mode**: Using geometric limit (minimum physically possible angle)")
                
            # Point distribution
            st.markdown("### 🎯 Point Distribution")
            col_dome2, col_cyl2 = st.columns(2)
            with col_dome2:
                dome_points = st.slider("Dome Density", 25, 150, 50, 5,
                                      help="Points per dome segment (optimized for multi-circuit)")
            with col_cyl2:
                cylinder_points = st.slider("Cylinder Density", 5, 50, 10, 5,
                                          help="Points per cylinder segment")
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
                        target_cylinder_angle_deg=target_angle,
                        mu_friction_coefficient=friction_coefficient
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
                        
                        # Check for physics warnings in non-geodesic mode
                        if hasattr(planner, '_kink_warnings') and planner._kink_warnings:
                            dome_kinks = [w for w in planner._kink_warnings if w['location'] == 'dome_opening']
                            if dome_kinks:
                                st.error(f"⚠️ **Physics Warning: {len(dome_kinks)} kinks detected near dome openings!**")
                                st.markdown("**These represent mathematical limitations where the physics becomes impossible:**")
                                
                                with st.expander("📊 View Kink Analysis", expanded=True):
                                    for i, kink in enumerate(dome_kinks[:3]):  # Show first 3
                                        st.markdown(f"**Kink #{i+1}:** {kink['rho_mm']:.1f}mm from axis, Δsin(α)={kink['delta_sin_alpha']:.3f}")
                                
                                st.info("💡 **Try:** Lower target angle, increase friction coefficient, or use geodesic patterns")
                            planner._kink_warnings = []  # Clear for next calculation
                
                elif pattern_type == "Non-Geodesic":
                    st.info(f"🚀 **Non-Geodesic Mode**: Generating extreme angle trajectory with μ = {friction_coefficient:.2f}")
                    
                    # Create planner with NO validation - allows any angle
                    planner = TrajectoryPlanner(
                        st.session_state.vessel_geometry,
                        dry_roving_width_m=roving_width/1000,
                        dry_roving_thickness_m=roving_thickness/1000,
                        roving_eccentricity_at_pole_m=polar_eccentricity/1000,
                        target_cylinder_angle_deg=target_angle,  # ANY angle allowed!
                        mu_friction_coefficient=friction_coefficient
                    )
                    
                    # Skip geodesic validation completely for non-geodesic mode
                    st.success(f"🔬 **Non-Geodesic Physics**: Target {target_angle}° with friction μ = {friction_coefficient:.2f}")
                    st.info("✨ **No geometric limits** - Using advanced differential equation solving")
                    
                    # Generate trajectory using TRUE non-geodesic differential equations
                    trajectory_data = planner.generate_non_geodesic_trajectory(dome_points, cylinder_points)
                    
                    if trajectory_data and len(trajectory_data.get('path_points', [])) > 0:
                        st.session_state.trajectory_data = trajectory_data
                        st.success(f"🎯 **Non-geodesic trajectory generated!** {len(trajectory_data['path_points'])} points with advanced physics")
                        
                        # Show friction physics insights
                        st.markdown("### 🔬 Non-Geodesic Physics Analysis")
                        st.info(f"**Koussios Eq. 5.62 solved** with surface curvatures and friction effects")
                        st.info(f"**Friction coefficient**: μ = {friction_coefficient:.2f} enabled extreme {target_angle}° winding")
                    else:
                        st.error("❌ Non-geodesic trajectory generation failed - try adjusting friction coefficient")

                elif pattern_type == "Multi-Circuit Pattern":
                    # Check if target angle is physically possible with current band width
                    effective_roving_width = band_width / 1000  # Convert to meters
                    
                    planner = TrajectoryPlanner(
                        st.session_state.vessel_geometry,
                        dry_roving_width_m=effective_roving_width,
                        dry_roving_thickness_m=roving_thickness/1e6,  # Convert μm to m
                        roving_eccentricity_at_pole_m=polar_eccentricity/1000,
                        target_cylinder_angle_deg=target_angle,
                        mu_friction_coefficient=friction_coefficient
                    )
                    
                    # Smart validation - automatically fall back to geometric limit if target is impossible
                    validation = planner.get_validation_results()
                    if validation and not validation.get('is_valid', True):
                        st.warning(f"⚠️ **Target angle {target_angle}° is impossible with {band_width:.1f}mm band width**")
                        if validation['error_type'] == 'too_shallow':
                            st.info(f"🔧 **Auto-fallback**: Using geometric limit ({validation['min_achievable_angle']:.1f}°) instead")
                            # Create new planner without target angle to use geometric limit
                            planner = TrajectoryPlanner(
                                st.session_state.vessel_geometry,
                                dry_roving_width_m=effective_roving_width,
                                dry_roving_thickness_m=roving_thickness/1e6,
                                roving_eccentricity_at_pole_m=polar_eccentricity/1000,
                                target_cylinder_angle_deg=None,  # Use geometric limit
                                mu_friction_coefficient=friction_coefficient
                            )
                        else:
                            st.info(f"🔧 **Try**: Reduce band width or increase target angle")
                    elif validation and validation.get('is_valid'):
                        st.success(f"✅ **Target angle {target_angle}° is achievable with {band_width:.1f}mm bands!**")
                        st.info(f"🎯 **Clairaut's constant**: {validation['clairaut_constant_mm']:.1f}mm")
                    
                    # Generate multi-circuit pattern with systematic advancement
                    trajectory_data = planner.generate_multi_circuit_trajectory(
                        num_target_circuits_for_pattern=circuits_to_close,
                        num_circuits_to_generate_for_vis=num_circuits_for_vis,
                        num_points_dome=dome_points,
                        num_points_cylinder=cylinder_points,
                        pattern_skip_factor=pattern_skip_factor
                    )
                    
                    # Ensure trajectory data is properly stored
                    if trajectory_data and trajectory_data.get('total_points', 0) > 0:
                        st.session_state.trajectory_data = trajectory_data
                        st.success(f"🎯 Multi-circuit pattern calculated successfully!")
                        st.info(f"Generated {trajectory_data['num_circuits_generated']} circuits with {trajectory_data['total_points']} total points")
                        st.info(f"Pattern advancement: {trajectory_data['advancement_angle_per_circuit_deg']:.1f}° per circuit")
                        st.info(f"Coverage efficiency: {trajectory_data['coverage_efficiency']:.1%}")
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
            if st.session_state.trajectory_data.get('pattern_type') in ['Geodesic', 'Multi-Circuit Geodesic', 'Geodesic_MultiPass', 'Multi-Circuit Pattern']:
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
                    
                else:
                    st.info("3D visualization requires x_points_m, y_points_m, and z_points_m data from the trajectory generation.")
                    
            else:
                st.info("Select 'Geodesic' trajectory pattern to enable 3D visualization.")
        else:
            st.info("Generate a trajectory first to view visualizations.")
    
    # Add tabbed interface for detailed analysis
    if st.session_state.trajectory_data is not None:
        tab1, tab2, tab3 = st.tabs(["🎯 3D Analysis", "📊 2D Analysis", "📋 Export & Reports"])
        
        with tab1:
            st.header("3D Trajectory Analysis")
            st.info("3D trajectory analysis will be displayed here.")
        
        with tab2:
            st.header("2D Trajectory Analysis")
            
            if 'trajectory_data' in st.session_state and st.session_state.trajectory_data:
                # Show 2D trajectory plots and analysis
                st.info("2D trajectory analysis and plots will be displayed here.")
            else:
                st.info("Generate a trajectory first to view 2D analysis.")
        
        with tab3:
            st.header("Export & Reports")
            
            if 'trajectory_data' in st.session_state and st.session_state.trajectory_data:
                # Export functionality will be implemented here
                st.info("Export and reporting functionality will be available here.")
            else:
                st.info("Generate a trajectory first to access export features.")

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

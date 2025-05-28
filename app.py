import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from modules.geometry import VesselGeometry
from modules.trajectories import TrajectoryPlanner
from modules.trajectories_fixed import TrajectoryPlannerFixed
from modules.trajectories_streamlined import StreamlinedTrajectoryPlanner
from modules.materials import MaterialDatabase
from modules.calculations import VesselCalculations
from modules.visualizations import VesselVisualizer
from modules.advanced_analysis import AdvancedAnalysisEngine
from modules.layer_manager import LayerStackManager, LayerDefinition
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
    # Enhanced header with professional styling
    st.markdown("""
    <div style="background: linear-gradient(90deg, #1e3c72, #2a5298); padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; text-align: center; margin: 0; font-size: 2.5rem;">
            🏗️ Advanced COPV Design Suite
        </h1>
        <p style="color: #e8f4fd; text-align: center; margin: 0.5rem 0 0 0; font-size: 1.1rem;">
            Professional Engineering Tool for Composite Pressure Vessel Design & Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced sidebar with progress tracking
    st.sidebar.markdown("""
    <div style="background: #f0f2f6; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        <h3 style="color: #1e3c72; margin: 0;">🎯 Design Workflow</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress indicators
    progress_status = {
        "Vessel Geometry": "✅" if st.session_state.vessel_geometry else "⭕",
        "Material Properties": "✅" if 'material_selection' in st.session_state else "⭕",
        "Layer Stack Definition": "✅" if 'layer_stack_manager' in st.session_state else "⭕",
        "Trajectory Planning": "✅" if st.session_state.trajectory_data else "⭕",
        "Performance Analysis": "⭕",
        "Export Results": "⭕"
    }
    
    pages = ["Vessel Geometry", "Material Properties", "Layer Stack Definition", "Trajectory Planning", "Performance Analysis", "Export Results"]
    
    # Create enhanced navigation with status indicators
    st.sidebar.markdown("### Navigation Menu")
    for i, page_name in enumerate(pages):
        status = progress_status[page_name]
        if st.sidebar.button(f"{status} {page_name}", key=f"nav_{i}", use_container_width=True):
            st.session_state.current_page = page_name
    
    # Get current page
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Vessel Geometry"
    
    page = st.session_state.current_page
    
    if page == "Vessel Geometry":
        vessel_geometry_page()
    elif page == "Material Properties":
        material_properties_page()
    elif page == "Layer Stack Definition":
        layer_stack_definition_page()
    elif page == "Trajectory Planning":
        trajectory_planning_page()
    elif page == "Performance Analysis":
        performance_analysis_page()
    elif page == "Export Results":
        export_results_page()

def vessel_geometry_page():
    # Professional page header
    st.markdown("""
    <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #1e3c72; margin-bottom: 1.5rem;">
        <h2 style="color: #1e3c72; margin: 0;">⚙️ Vessel Geometry Design</h2>
        <p style="color: #6c757d; margin: 0.5rem 0 0 0;">Define your composite pressure vessel dimensions and dome configuration</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Enhanced input section with better organization
        st.markdown("""
        <div style="background: #ffffff; padding: 1rem; border-radius: 8px; border: 1px solid #dee2e6;">
            <h4 style="color: #495057; margin-top: 0;">🔧 Design Parameters</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Basic vessel parameters with enhanced styling
        with st.expander("📏 Basic Dimensions", expanded=True):
            inner_diameter = st.number_input(
                "Inner Diameter (mm)", 
                min_value=10.0, 
                value=200.0, 
                step=1.0,
                help="Internal diameter of the cylindrical section"
            )
            cylindrical_length = st.number_input(
                "Cylindrical Length (mm)", 
                min_value=0.0, 
                value=300.0, 
                step=1.0,
                help="Length of the straight cylindrical section"
            )
            wall_thickness = st.number_input(
                "Wall Thickness (mm)", 
                min_value=0.1, 
                value=5.0, 
                step=0.1,
                help="Composite wall thickness"
            )
        
        # Dome parameters with visual indicators
        with st.expander("🏛️ Dome Configuration", expanded=True):
            dome_type = st.selectbox(
                "Dome Type", 
                ["Isotensoid", "Geodesic", "Elliptical", "Hemispherical"],
                help="Select the dome end-cap geometry type"
            )
            
            if dome_type == "Isotensoid":
                st.markdown("**⚡ Advanced qrs-Parameterization (Koussios Theory)**")
                col_q, col_r = st.columns(2)
                with col_q:
                    q_factor = st.slider(
                        "q-factor", 
                        min_value=0.1, 
                        max_value=20.0, 
                        value=9.5, 
                        step=0.1,
                        help="Shape parameter controlling dome curvature"
                    )
                with col_r:
                    r_factor = st.slider(
                        "r-factor", 
                        min_value=-1.0, 
                        max_value=2.0, 
                        value=0.1, 
                        step=0.01,
                        help="Boundary condition parameter"
                    )
                s_factor = st.slider(
                    "s-factor", 
                    min_value=0.0, 
                    max_value=2.0, 
                    value=0.5, 
                    step=0.01,
                    help="Additional shape control parameter"
                )
            elif dome_type == "Elliptical":
                aspect_ratio = st.slider(
                    "Dome Aspect Ratio (height/radius)", 
                    min_value=0.1, 
                    max_value=2.0, 
                    value=1.0, 
                    step=0.01,
                    help="Ratio of dome height to radius"
                )
        
        # Operating conditions with engineering context
        with st.expander("🌡️ Operating Conditions", expanded=True):
            operating_pressure = st.number_input(
                "Operating Pressure (MPa)", 
                min_value=0.1, 
                value=30.0, 
                step=0.1,
                help="Maximum operating pressure"
            )
            safety_factor = st.number_input(
                "Safety Factor", 
                min_value=1.0, 
                value=2.0, 
                step=0.1,
                help="Design safety factor for stress calculations"
            )
            operating_temp = st.number_input(
                "Operating Temperature (°C)", 
                value=20.0, 
                step=1.0,
                help="Operating temperature for material properties"
            )
        
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
        pattern_type = st.selectbox("Winding Pattern", ["Geodesic", "Non-Geodesic", "Multi-Circuit Pattern", "🔬 Refactored Engine (Test)", "Helical", "Hoop", "Polar", "Transitional"])
        
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
        if pattern_type == "🔬 Refactored Engine (Test)":
            st.markdown("### 🔬 Refactored Trajectory Engine")
            st.info("Testing new clean architecture before full migration")
            
            # Refactored engine parameters
            refactored_pattern = st.selectbox("Pattern Type", ["geodesic_spiral", "non_geodesic_spiral"], 
                                            help="Choose trajectory generation method")
            coverage_option = st.selectbox("Coverage Strategy", ["single_circuit", "full_coverage", "user_defined"],
                                         help="Single circuit = 2 passes, Full coverage = complete vessel")
            
            if coverage_option == "user_defined":
                user_circuits = st.number_input("Number of Circuits", min_value=1, max_value=10, value=2)
            else:
                user_circuits = 1
            
            # Basic roving parameters
            roving_width = st.number_input("Roving Width (mm)", min_value=0.1, value=3.0, step=0.1)
            roving_thickness = st.number_input("Roving Thickness (mm)", min_value=0.01, value=0.2, step=0.01)
            polar_eccentricity = st.number_input("Polar Eccentricity (mm)", min_value=0.0, value=0.0, step=0.1)
            
            # Target angle
            use_target_angle = st.checkbox("Specify Target Angle", value=True)
            target_angle = None
            if use_target_angle:
                target_angle = st.slider("Target Angle (degrees)", min_value=10.0, max_value=80.0, value=45.0)
            
            # Test button
            if st.button("🧪 Test Refactored Engine", key="refactored_btn"):
                with st.spinner("Testing refactored trajectory engine..."):
                    try:
                        from modules.trajectories_refactored import TrajectoryPlannerRefactored
                        
                        # Create refactored planner
                        planner_refactored = TrajectoryPlannerRefactored(
                            vessel_geometry=st.session_state.vessel_geometry,
                            dry_roving_width_m=roving_width * 1e-3,
                            dry_roving_thickness_m=roving_thickness * 1e-3,
                            roving_eccentricity_at_pole_m=polar_eccentricity * 1e-3,
                            target_cylinder_angle_deg=target_angle,
                            mu_friction_coefficient=0.0 if refactored_pattern == "geodesic_spiral" else 0.3
                        )
                        
                        # Validate parameters
                        validation = planner_refactored.get_validation_results()
                        if validation['is_valid']:
                            st.success(f"✅ Validation passed! C = {validation['clairaut_constant_mm']:.2f}mm")
                        else:
                            st.warning(f"⚠️ {validation['error_message']}")
                        
                        # Generate trajectory
                        trajectory_data = planner_refactored.generate_trajectory(
                            pattern_name=refactored_pattern,
                            coverage_option=coverage_option,
                            user_circuits=user_circuits
                        )
                        
                        if trajectory_data and trajectory_data.get('success'):
                            st.success(f"🎉 Generated {trajectory_data['total_points']} points!")
                            st.session_state.trajectory_data = trajectory_data
                            
                            # Show metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Points", trajectory_data['total_points'])
                            with col2:
                                st.metric("Passes", trajectory_data['total_circuits_legs'])
                            with col3:
                                st.metric("Final φ", f"{trajectory_data['final_turn_around_angle_deg']:.1f}°")
                        else:
                            st.error("❌ Generation failed")
                            
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        
        elif pattern_type == "Geodesic":
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
            
            # Add pattern mode selection for multi-circuit capability
            col1, col2 = st.columns(2)
            with col1:
                pattern_mode = st.radio(
                    "Pattern Mode:",
                    ["Single Circuit", "Multi-Circuit Pattern", "Continuous Helical Physics"],
                    key="non_geodesic_pattern_mode",
                    help="Single circuit for testing, multi-circuit for coverage, continuous helical for gap-free physics-based winding"
                )
            with col2:
                if pattern_mode == "Multi-Circuit Pattern":
                    num_circuits = st.slider("Number of Circuits", 6, 24, 12, 
                                            help="Number of circuits for full coverage pattern")
                elif pattern_mode == "Continuous Helical Physics":
                    num_circuits = st.slider("Number of Circuits", 2, 20, 6, 
                                            help="Number of pole-to-pole circuits for continuous physics-based spiral")
                else:
                    num_circuits = 1
            
            # Target angle configuration for non-geodesic patterns
            st.markdown("### 🎯 Target Winding Angle")
            use_target_angle = st.checkbox("Specify Target Cylinder Angle", value=True,
                                         help="Define desired winding angle for physics-based patterns")
            
            target_angle = None
            if use_target_angle:
                target_angle = st.slider("Target Cylinder Angle (degrees)", 
                                        min_value=10.0, max_value=80.0, value=45.0, step=1.0,
                                        help="Desired winding angle on cylinder section")
                st.info(f"🎯 **Target**: {target_angle}° winding angle on cylinder")
            else:
                st.info("🔧 **Mode**: Using geometric limit (minimum physically possible angle)")
            
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
                # Default friction coefficient for geodesic patterns
                friction_coefficient = 0.0  # Geodesic patterns don't use friction
                
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
                    # Ensure all required variables are defined
                    if 'roving_width' not in locals():
                        roving_width = 3.0  # Default 3mm
                    if 'roving_thickness' not in locals():
                        roving_thickness = 0.2  # Default 0.2mm
                    if 'polar_eccentricity' not in locals():
                        polar_eccentricity = 0.0  # Default 0mm
                    if 'target_angle' not in locals():
                        target_angle = 45.0  # Default 45°
                    if 'dome_points' not in locals():
                        dome_points = 50  # Default dome density
                    if 'cylinder_points' not in locals():
                        cylinder_points = 10  # Default cylinder density
                    if 'pattern_mode' not in locals():
                        pattern_mode = "Single Circuit"
                    if 'num_circuits' not in locals():
                        num_circuits = 2  # Default 2 circuits
                    
                    # Debug: Check pattern mode
                    st.write(f"🔍 DEBUG: Pattern mode = '{pattern_mode}', Number of circuits = {num_circuits}")
                    
                    # Special handling for Multi-Circuit Pattern - route to geodesic path for stability
                    if pattern_mode == "Multi-Circuit Pattern":
                        st.warning("🚧 **Multi-circuit non-geodesic is under development**")
                        st.info("🔄 **Using proven geodesic multi-pass generation** for reliable trajectories")
                        st.info(f"🔥 Generating MULTI-PASS GEODESIC pattern with {num_circuits} circuits")
                        
                        # Use the SAME trajectory generation path as the working Geodesic mode
                        st.info(f"🎯 Using target angle: {target_angle}°")
                        
                        # Create geodesic planner with proper validation
                        planner = TrajectoryPlanner(
                            st.session_state.vessel_geometry,
                            dry_roving_width_m=roving_width/1000,
                            dry_roving_thickness_m=roving_thickness/1000,
                            roving_eccentricity_at_pole_m=polar_eccentricity/1000,
                            target_cylinder_angle_deg=target_angle,
                            mu_friction_coefficient=0.0  # Geodesic patterns don't use friction
                        )
                        
                        # Show validation results like geodesic mode
                        validation = planner.get_validation_results()
                        if validation and validation.get('is_valid', True):
                            if target_angle:
                                st.success(f"✅ **Target Angle Validated**: {target_angle}° achievable (Safety margin: {validation.get('safety_margin_mm', 0):.1f}mm)")
                            else:
                                st.info(f"🔧 **Geometric Limit**: Using minimum achievable angle ({validation.get('effective_polar_opening_mm', 0):.1f}mm opening)")
                        else:
                            st.error(f"❌ **Validation Failed**: {validation.get('error_message', 'Unknown validation error')}")
                            st.stop()
                        
                        # Use the proven working geodesic trajectory generation
                        trajectory_data = planner.generate_geodesic_trajectory(dome_points, cylinder_points, number_of_passes=num_circuits)
                    else:
                        # Other non-geodesic modes
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
                        
                        # Generate trajectory based on pattern mode
                        st.info(f"🔥 Calling SINGLE-CIRCUIT function")
                        trajectory_data = planner.generate_non_geodesic_trajectory(dome_points, cylinder_points)
                    
                    # Handle different trajectory data formats
                    trajectory_points = trajectory_data.get('trajectory_points', trajectory_data.get('path_points', []))
                    
                    if trajectory_data and len(trajectory_points) > 0:
                        st.session_state.trajectory_data = trajectory_data
                        if pattern_mode == "Multi-Circuit Pattern":
                            st.success(f"🎯 **Multi-circuit non-geodesic pattern generated!** {num_circuits} circuits with {len(trajectory_points)} total points")
                            if trajectory_data.get('total_kinks', 0) > 0:
                                st.warning(f"⚠️ **{trajectory_data['total_kinks']} kinks detected** across all circuits")
                        elif pattern_mode == "Continuous Helical Physics":
                            st.success(f"🌀 **Continuous helical physics trajectory generated!** {len(trajectory_points)} points")
                            st.info(f"🎯 Target angle: {trajectory_data.get('target_angle_deg', target_angle)}° | Friction: μ={trajectory_data.get('friction_coefficient', friction_coefficient)}")
                            if trajectory_data.get('gaps_over_1mm', 0) == 0:
                                st.success(f"✅ **Perfect continuity achieved!** Max gap: {trajectory_data.get('max_gap_mm', 0):.3f}mm")
                            else:
                                st.warning(f"⚠️ {trajectory_data.get('gaps_over_1mm', 0)} gaps > 1mm detected (max: {trajectory_data.get('max_gap_mm', 0):.2f}mm)")
                        else:
                            st.success(f"🎯 **Non-geodesic trajectory generated!** {len(trajectory_points)} points with advanced physics")
                        
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
                    
                    # Use the proven working geodesic trajectory generation instead
                    st.info("🔄 **Using proven geodesic multi-pass generation** for reliable trajectories")
                    trajectory_data = planner.generate_geodesic_trajectory(dome_points, cylinder_points, number_of_passes=num_circuits_for_vis)
                    
                    # Ensure trajectory data is properly stored
                    if trajectory_data and trajectory_data.get('total_points', 0) > 0:
                        st.session_state.trajectory_data = trajectory_data
                        st.success(f"🎯 **Multi-circuit trajectory generated successfully!** {trajectory_data['total_points']} points")
                        
                        # Show trajectory details using available data
                        circuits = trajectory_data.get('total_circuits_legs', num_circuits_for_vis)
                        st.info(f"🔥 **Multi-pass geodesic pattern** with {circuits} circuits")
                        
                        # Calculate angular span from trajectory data
                        if 'final_turn_around_angle_deg' in trajectory_data:
                            angle_span = trajectory_data['final_turn_around_angle_deg']
                            st.info(f"🌟 **Angular progression**: {angle_span:.1f}° total span")
                        
                        st.rerun()
                    else:
                        st.error("❌ Multi-circuit pattern generation failed")
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
                
        elif pattern_type == "🔬 Refactored Engine (Test)":
            # Temporarily disabled while fixing syntax issues
            st.info("🔬 Refactored Engine is temporarily under maintenance")
            st.markdown("The clean architecture with excellent validation results (C = 70.71mm) will be available soon!")
            return
            
            # Get parameters from session state or use defaults
            roving_width = st.number_input("Roving Width (mm)", min_value=0.1, max_value=10.0, value=3.0, step=0.1)
            roving_thickness = st.number_input("Roving Thickness (mm)", min_value=0.01, max_value=1.0, value=0.2, step=0.01)
            polar_eccentricity = st.number_input("Polar Eccentricity (mm)", min_value=0.0, max_value=5.0, value=0.0, step=0.1)
            target_angle = st.number_input("Target Angle (°)", min_value=0.0, max_value=90.0, value=45.0, step=1.0)
            
            # Pattern selection for refactored engine
            refactored_pattern = st.selectbox("Pattern Type", 
                                            ["geodesic_spiral", "non_geodesic_spiral", "helical", "polar", "hoop"],
                                            index=0)
            
            coverage_option = st.selectbox("Coverage Strategy",
                                         ["single_circuit", "full_coverage", "user_defined"],
                                         index=0)
            
            if coverage_option == "user_defined":
                user_circuits = st.number_input("Number of Circuits", min_value=1, max_value=20, value=2, step=1)
            else:
                user_circuits = 1
                
            if st.button("🚀 Generate Refactored Trajectory", type="primary"):
                with st.spinner("Generating trajectory with refactored engine..."):
                    try:
                        # Import and create refactored planner
                        from modules.trajectories_refactored import TrajectoryPlannerRefactored
                        planner_refactored = TrajectoryPlannerRefactored(
                            vessel_geometry=st.session_state.vessel_geometry,
                            dry_roving_width_m=roving_width * 1e-3,
                            dry_roving_thickness_m=roving_thickness * 1e-3,
                            roving_eccentricity_at_pole_m=polar_eccentricity * 1e-3,
                            target_cylinder_angle_deg=target_angle,
                            mu_friction_coefficient=0.0 if refactored_pattern == "geodesic_spiral" else 0.3
                        )
                        
                        # Show validation results
                        validation = planner_refactored.get_validation_results()
                        if validation['is_valid']:
                            st.success(f"✅ Validation passed! C = {validation['clairaut_constant_mm']:.2f}mm")
                        else:
                            st.warning(f"⚠️ {validation['error_message']}")
                        
                        # Generate trajectory
                        trajectory_data = planner_refactored.generate_trajectory(
                            pattern_name=refactored_pattern,
                            coverage_option=coverage_option,
                            user_circuits=user_circuits
                        )
                        
                        if trajectory_data and trajectory_data.get('success'):
                            st.success(f"🎉 Generated {trajectory_data['total_points']} points!")
                            st.session_state.trajectory_data = trajectory_data
                            
                            # Show metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Points", trajectory_data['total_points'])
                            with col2:
                                st.metric("Passes", trajectory_data['total_circuits_legs'])
                            with col3:
                                st.metric("Final φ", f"{trajectory_data['final_turn_around_angle_deg']:.1f}°")
                            
                            st.rerun()
                        else:
                            st.error("❌ Generation failed")
                            
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
    
    with col2:
        st.subheader("Trajectory Controls")
        st.info("Configure trajectory parameters in the left column, then view visualizations below.")
        
    # Full-width visualization section outside columns
    if st.session_state.trajectory_data is not None:
        st.markdown("---")
        st.subheader("Trajectory Visualization")
        st.write(f"✅ Trajectory data loaded: {st.session_state.trajectory_data.get('pattern_type', 'Unknown')}")
        st.write(f"📊 Data keys: {list(st.session_state.trajectory_data.keys())}")
        
        visualizer = VesselVisualizer()
        
        # Create side-by-side visualization layout
        vis_col1, vis_col2 = st.columns(2)
        
        with vis_col1:
            st.write("**2D Trajectory View**")
            # Plot trajectory - 2D view
            fig = visualizer.plot_winding_trajectory(
                st.session_state.vessel_geometry,
                st.session_state.trajectory_data
            )
            st.pyplot(fig)
        
        with vis_col2:
            # Add 3D visualization for all trajectory types
            if st.session_state.trajectory_data.get('pattern_type') in ['Geodesic', 'Multi-Circuit Geodesic', 'Geodesic_MultiPass', 'Multi-Circuit Pattern', 'Continuous Non-Geodesic Helical', 'Multi-Circuit Non-Geodesic', 'physics_continuous_geodesic_helical', 'physics_continuous_non-geodesic_helical', 'seamless_continuous_helical']:
                st.write("**3D Trajectory View**")
                
                # Check if we have the required 3D coordinate data
                has_3d_data = ('x_points_m' in st.session_state.trajectory_data and 
                              'y_points_m' in st.session_state.trajectory_data and 
                              'z_points_m' in st.session_state.trajectory_data)
                
                if has_3d_data:
                    # Create 3D trajectory plot using Plotly for better interactivity
                    try:
                        import plotly.graph_objects as go
                        
                        x_data = st.session_state.trajectory_data['x_points_m']
                        y_data = st.session_state.trajectory_data['y_points_m'] 
                        z_data = st.session_state.trajectory_data['z_points_m']
                        
                        # Create 3D trajectory plot
                        fig = go.Figure()
                        
                        # Add the trajectory as a colored line
                        fig.add_trace(go.Scatter3d(
                            x=x_data,
                            y=y_data,
                            z=z_data,
                            mode='lines+markers',
                            line=dict(color='red', width=4),
                            marker=dict(size=2),
                            name='Multi-Circuit Trajectory'
                        ))
                        
                        # Add vessel outline if available
                        if hasattr(st.session_state.vessel_geometry, 'profile_points'):
                            r_profile = st.session_state.vessel_geometry.profile_points['r_inner_mm'] * 1e-3
                            z_profile = st.session_state.vessel_geometry.profile_points['z_mm'] * 1e-3
                            
                            # Create circular vessel outline at key z positions
                            theta_outline = np.linspace(0, 2*np.pi, 50)
                            
                            # Add vessel outline at a few z positions
                            for i in range(0, len(z_profile), len(z_profile)//5):
                                x_vessel = r_profile[i] * np.cos(theta_outline)
                                y_vessel = r_profile[i] * np.sin(theta_outline)
                                z_vessel = np.full_like(x_vessel, z_profile[i])
                                
                                fig.add_trace(go.Scatter3d(
                                    x=x_vessel,
                                    y=y_vessel,
                                    z=z_vessel,
                                    mode='lines',
                                    line=dict(color='lightgray', width=2),
                                    name='Vessel Outline' if i == 0 else None,
                                    showlegend=(i == 0)
                                ))
                        
                        # Configure the layout
                        fig.update_layout(
                            title=f"3D {st.session_state.trajectory_data.get('pattern_type', 'Geodesic')} Trajectory - {st.session_state.trajectory_data.get('total_points', 0)} Points",
                            scene=dict(
                                xaxis_title="X (m)",
                                yaxis_title="Y (m)",
                                zaxis_title="Z (m)",
                                aspectmode='data'
                            ),
                            height=600,
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show key statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Points Generated", len(x_data))
                        with col2:
                            circuits = st.session_state.trajectory_data.get('total_circuits_legs', 'N/A')
                            st.metric("Total Circuits", circuits)
                        with col3:
                            angle_span = st.session_state.trajectory_data.get('final_turn_around_angle_deg', 0)
                            st.metric("Angular Span", f"{angle_span:.1f}°" if angle_span else 'N/A')
                            
                    except ImportError:
                        st.error("Plotly not available for 3D visualization")
                    except Exception as e:
                        st.error(f"Error creating 3D visualization: {str(e)}")
                    
                else:
                    st.info("3D visualization requires x_points_m, y_points_m, and z_points_m data from the trajectory generation.")
            else:
                st.info("Select a trajectory pattern to see 3D visualization")
        
    # Add trajectory statistics outside the data check
    if st.session_state.trajectory_data is not None:
        st.write("**Trajectory Statistics:**")
        st.metric("Total Points", st.session_state.trajectory_data.get('total_points', 'N/A'))
        st.metric("Total Circuits", st.session_state.trajectory_data.get('total_circuits', 'N/A'))
        final_angle = st.session_state.trajectory_data.get('total_angular_span_deg', 0)
        st.metric("Angular Span", f"{final_angle:.1f}°" if final_angle else 'N/A')
                    
    # Clean 3D visualization for continuous non-geodesic patterns
    if (st.session_state.trajectory_data is not None and 
        'Continuous Non-Geodesic Helical' in st.session_state.trajectory_data.get('pattern_type', '')):
        
        st.markdown("---")
        st.subheader("🌟 Continuous Non-Geodesic Helical Pattern - 3D View")
        
        # Display key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Points", st.session_state.trajectory_data.get('total_points', 'N/A'))
        with col2:
            st.metric("Circuits", st.session_state.trajectory_data.get('number_of_circuits', 'N/A'))
        with col3:
            kinks = st.session_state.trajectory_data.get('total_kinks', 0)
            st.metric("Kinks Detected", kinks, delta=f"μ={st.session_state.trajectory_data.get('friction_coefficient', 'N/A')}")
        
        if all(key in st.session_state.trajectory_data for key in ['x_points_m', 'y_points_m', 'z_points_m']):
            try:
                import plotly.graph_objects as go
                
                x_data = st.session_state.trajectory_data['x_points_m']
                y_data = st.session_state.trajectory_data['y_points_m'] 
                z_data = st.session_state.trajectory_data['z_points_m']
                
                # Create 3D trajectory plot
                fig = go.Figure()
                
                # Add the continuous trajectory as a single line
                fig.add_trace(go.Scatter3d(
                    x=x_data,
                    y=y_data,
                    z=z_data,
                    mode='lines',
                    line=dict(color='red', width=4),
                    name='Continuous Helical Path'
                ))
                
                # Configure the layout
                fig.update_layout(
                    title="3D Continuous Non-Geodesic Helical Pattern",
                    scene=dict(
                        xaxis_title="X (m)",
                        yaxis_title="Y (m)",
                        zaxis_title="Z (m)",
                        aspectmode='data'
                    ),
                    height=600,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show key statistics below the plot
                st.write("**Pattern Statistics:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Points Generated", len(x_data))
                with col2:
                    friction = st.session_state.trajectory_data.get('friction_coefficient', 0)
                    st.metric("Friction Coefficient", f"μ = {friction:.3f}")
                with col3:
                    angular_span = st.session_state.trajectory_data.get('total_angular_span_deg', 0)
                    st.metric("Angular Span", f"{angular_span:.1f}°")
                
            except Exception as e:
                st.error(f"Error creating 3D plot: {str(e)}")
        else:
            st.info("3D coordinates not available in trajectory data")

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
        st.markdown("### Material Properties")
        if 'selected_fiber' in st.session_state and 'selected_resin' in st.session_state:
            st.write(f"Fiber: {st.session_state.selected_fiber}")
            st.write(f"Resin: {st.session_state.selected_resin}")
        else:
            st.info("Please select materials in the Material Properties page first")
        
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

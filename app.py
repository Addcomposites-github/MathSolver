import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
            q_factor = st.slider("q-factor", min_value=0.1, max_value=2.0, value=1.0, step=0.01)
            r_factor = st.slider("r-factor", min_value=0.1, max_value=2.0, value=1.0, step=0.01)
            s_factor = st.slider("s-factor", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
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
        pattern_type = st.selectbox("Winding Pattern", ["Helical", "Hoop", "Polar", "Transitional"])
        
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
        if pattern_type == "Helical":
            st.markdown("### Pattern Parameters")
            circuits_to_close = st.number_input("Circuits to Close Pattern", min_value=1, value=8, step=1)
            overlap_allowance = st.slider("Overlap Allowance (%)", min_value=-10.0, max_value=50.0, value=10.0, step=1.0)
        
        if st.button("Calculate Trajectory", type="primary"):
            try:
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
            
            # Plot trajectory
            fig = visualizer.plot_winding_trajectory(
                st.session_state.vessel_geometry,
                st.session_state.trajectory_data
            )
            st.pyplot(fig)
            
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

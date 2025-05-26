import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import io
from typing import Dict, List, Optional

# Import custom modules
from modules.geometry import VesselGeometry
from modules.trajectories import TrajectoryPlanner
from modules.visualizations import VesselVisualizer
from modules.calculations import VesselCalculations
from data.material_database import FIBER_MATERIALS, RESIN_MATERIALS, get_recommended_combinations

# Configure page
st.set_page_config(
    page_title="COPV Trajectory Planner",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🚀 Composite Pressure Vessel Trajectory Planner")
    st.markdown("Advanced geodesic winding pattern design and optimization")
    
    # Initialize session state
    if 'vessel_geometry' not in st.session_state:
        st.session_state.vessel_geometry = None
    if 'trajectory_data' not in st.session_state:
        st.session_state.trajectory_data = None
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["Vessel Geometry", "Material Properties", "Trajectory Planning", 
         "Performance Analysis", "Export Results"]
    )
    
    # Page routing
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
    st.header("Vessel Geometry Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Basic Dimensions")
        inner_radius = st.number_input("Inner Radius (m)", min_value=0.1, max_value=5.0, value=0.5, step=0.01)
        wall_thickness = st.number_input("Wall Thickness (m)", min_value=0.001, max_value=0.1, value=0.01, step=0.001)
        cylinder_length = st.number_input("Cylinder Length (m)", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
        
        st.subheader("Dome Configuration")
        dome_type = st.selectbox("Dome Type", ["hemispherical", "elliptical", "isotensoid"])
        
        if dome_type == "elliptical":
            aspect_ratio = st.slider("Aspect Ratio (height/radius)", 0.3, 1.5, 1.0, 0.1)
        elif dome_type == "isotensoid":
            dome_subtype = st.selectbox("Isotensoid Type", ["general_orthotropic", "koussios_qrs"])
            if dome_subtype == "koussios_qrs":
                q_factor = st.slider("Q Factor", 0.1, 2.0, 1.0, 0.1)
                r_factor = st.slider("R Factor", 0.1, 2.0, 1.0, 0.1)
                s_factor = st.slider("S Factor", 0.1, 2.0, 1.0, 0.1)
        
    with col2:
        st.subheader("Mesh Parameters")
        dome_points = st.slider("Dome Points", 50, 500, 200, 50)
        cylinder_points = st.slider("Cylinder Points", 20, 200, 100, 20)
        
        if st.button("Generate Geometry", type="primary"):
            try:
                # Create vessel geometry
                geometry = VesselGeometry(inner_radius, wall_thickness, cylinder_length)
                
                # Configure dome
                if dome_type == "hemispherical":
                    geometry.add_hemispherical_dome()
                elif dome_type == "elliptical":
                    geometry.add_elliptical_dome(aspect_ratio=aspect_ratio)
                elif dome_type == "isotensoid":
                    if dome_subtype == "general_orthotropic":
                        geometry.add_isotensoid_dome_general_orthotropic()
                    else:
                        geometry.add_isotensoid_dome_koussios_qrs(q_factor, r_factor, s_factor)
                
                # Generate mesh
                geometry.generate_mesh(dome_points=dome_points, cylinder_points=cylinder_points)
                
                # Store in session state
                st.session_state.vessel_geometry = geometry
                st.success("Vessel geometry generated successfully!")
                
                # Display geometry info
                st.subheader("Geometry Summary")
                st.write(f"**Total Volume:** {geometry.total_volume:.4f} m³")
                st.write(f"**Total Mass:** {geometry.total_mass:.2f} kg")
                st.write(f"**Surface Area:** {geometry.surface_area:.2f} m²")
                
            except Exception as e:
                st.error(f"Error generating geometry: {str(e)}")

def material_properties_page():
    st.header("Material Properties")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Fiber Selection")
        fiber_type = st.selectbox("Fiber Type", list(FIBER_MATERIALS.keys()))
        
        if fiber_type in FIBER_MATERIALS:
            fiber_props = FIBER_MATERIALS[fiber_type]
            st.write(f"**Tensile Strength:** {fiber_props['tensile_strength']} MPa")
            st.write(f"**Elastic Modulus:** {fiber_props['elastic_modulus']} GPa")
            st.write(f"**Density:** {fiber_props['density']} kg/m³")
    
    with col2:
        st.subheader("Resin Selection")
        resin_type = st.selectbox("Resin Type", list(RESIN_MATERIALS.keys()))
        
        if resin_type in RESIN_MATERIALS:
            resin_props = RESIN_MATERIALS[resin_type]
            st.write(f"**Tensile Strength:** {resin_props['tensile_strength']} MPa")
            st.write(f"**Elastic Modulus:** {resin_props['elastic_modulus']} GPa")
            st.write(f"**Density:** {resin_props['density']} kg/m³")
    
    # Store material selection
    st.session_state.fiber_type = fiber_type
    st.session_state.resin_type = resin_type

def trajectory_planning_page():
    st.header("Trajectory Planning")
    
    if st.session_state.vessel_geometry is None:
        st.warning("Please configure vessel geometry first.")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Winding Parameters")
        
        pattern_type = st.selectbox("Pattern Type", ["geodesic", "planar"])
        winding_angle = st.slider("Winding Angle (degrees)", 10, 80, 45, 5)
        
        # Advanced parameters
        with st.expander("Advanced Parameters"):
            roving_width = st.number_input("Roving Width (mm)", 1.0, 20.0, 6.0, 0.5)
            roving_thickness = st.number_input("Roving Thickness (mm)", 0.1, 5.0, 0.5, 0.1)
            polar_eccentricity = st.slider("Polar Eccentricity", 0.0, 0.2, 0.05, 0.01)
            
            # Target angle option
            use_target_angle = st.checkbox("Use Target Angle")
            if use_target_angle:
                target_angle = st.number_input("Target Angle (degrees)", 0.0, 90.0, 45.0, 1.0)
            else:
                target_angle = None
        
        if st.button("Generate Trajectory", type="primary"):
            try:
                planner = TrajectoryPlanner(st.session_state.vessel_geometry)
                
                # Set trajectory parameters
                trajectory_params = {
                    'pattern_type': pattern_type,
                    'winding_angle': math.radians(winding_angle),
                    'roving_width': roving_width / 1000,  # Convert to meters
                    'roving_thickness': roving_thickness / 1000,
                    'polar_eccentricity': polar_eccentricity
                }
                
                if use_target_angle:
                    trajectory_params['target_angle'] = math.radians(target_angle)
                
                # Additional parameters for multi-circuit planning
                trajectory_params['circuits_to_close'] = 2  # Default circuits
                trajectory_params['overlap_allowance'] = 0.1  # 10% overlap
                
                result = planner.generate_trajectory(**trajectory_params)
                
                if result and 'success' in result and result['success']:
                    st.session_state.trajectory_data = result
                    st.success(f"Trajectory generated successfully! {len(result.get('rho_m', []))} points generated.")
                else:
                    st.error("Failed to generate trajectory. Please check parameters.")
                    
            except Exception as e:
                st.error(f"Error generating trajectory: {str(e)}")
    
    with col2:
        st.subheader("Trajectory Visualization")
        
        if st.session_state.trajectory_data:
            result = st.session_state.trajectory_data
            
            # Check if we have 3D coordinates
            if 'x_m' in result and 'y_m' in result and 'z_m' in result:
                # Create interactive 3D plot
                fig_plotly = go.Figure()
                
                # Add trajectory path
                fig_plotly.add_trace(go.Scatter3d(
                    x=result['x_m'],
                    y=result['y_m'],
                    z=result['z_m'],
                    mode='lines+markers',
                    line=dict(width=4, color=np.arange(len(result['x_m'])), colorscale='Viridis'),
                    marker=dict(size=2, color=np.arange(len(result['x_m'])), colorscale='Viridis'),
                    name=f"{result.get('pattern_type', 'Geodesic')} Trajectory",
                    showlegend=True
                ))
                
                # Add vessel surface
                geometry = st.session_state.vessel_geometry
                if hasattr(geometry, 'profile_points') and geometry.profile_points:
                    # Create vessel surface mesh
                    phi_surface = np.linspace(0, 2*np.pi, 50)
                    rho_profile = np.array([p[0] for p in geometry.profile_points])
                    z_profile = np.array([p[1] for p in geometry.profile_points])
                    
                    Phi_surf, Rho_surf = np.meshgrid(phi_surface, rho_profile)
                    Z_surf = np.tile(z_profile.reshape(-1, 1), (1, len(phi_surface)))
                    X_surf = Rho_surf * np.cos(Phi_surf)
                    Y_surf = Rho_surf * np.sin(Phi_surf)
                    
                    fig_plotly.add_trace(go.Surface(
                        x=X_surf, y=Y_surf, z=Z_surf,
                        opacity=0.3,
                        colorscale='Greys',
                        showscale=False,
                        name='Vessel Surface'
                    ))
                
                # Configure layout
                fig_plotly.update_layout(
                    title=f"Interactive 3D {result.get('pattern_type', 'Geodesic')} Trajectory",
                    scene=dict(
                        xaxis_title="X (m)",
                        yaxis_title="Y (m)", 
                        zaxis_title="Z (m)",
                        aspectmode='data',
                        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
                    ),
                    width=800,
                    height=600,
                    margin=dict(r=0, b=0, l=0, t=40)
                )
                
                # Display interactive plot
                st.plotly_chart(fig_plotly, use_container_width=True)
                
                # Add controls info
                st.write("**Interactive Controls:**")
                st.write("🔄 **Rotate:** Click and drag")
                st.write("🔍 **Zoom:** Mouse wheel or pinch")
                st.write("📍 **Pan:** Shift + click and drag")
                st.write("🎯 **Reset view:** Double-click")
                
            else:
                st.info("3D coordinates not available for this trajectory type.")
        else:
            st.info("No trajectory data available. Please generate a trajectory first.")

def performance_analysis_page():
    st.header("Performance Analysis")
    
    if st.session_state.vessel_geometry is None or st.session_state.trajectory_data is None:
        st.warning("Please configure vessel geometry and generate trajectory first.")
        return
    
    st.subheader("Coming Soon")
    st.info("Performance analysis features will be available in the next update.")

def export_results_page():
    st.header("Export Results")
    
    if st.session_state.trajectory_data is None:
        st.warning("No trajectory data available for export.")
        return
    
    st.subheader("Trajectory Data Export")
    
    try:
        result = st.session_state.trajectory_data
        
        # Create DataFrame
        export_data = {}
        if 'rho_m' in result:
            export_data['rho_m'] = result['rho_m']
        if 'phi_rad' in result:
            export_data['phi_rad'] = result['phi_rad']
        if 'z_m' in result:
            export_data['z_m'] = result['z_m']
        if 'x_m' in result:
            export_data['x_m'] = result['x_m']
        if 'y_m' in result:
            export_data['y_m'] = result['y_m']
        
        if export_data:
            df = pd.DataFrame(export_data)
            
            # Display preview
            st.subheader("Data Preview")
            st.dataframe(df.head(10))
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Trajectory CSV",
                data=csv,
                file_name="winding_trajectory.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Error exporting trajectory: {str(e)}")

if __name__ == "__main__":
    main()
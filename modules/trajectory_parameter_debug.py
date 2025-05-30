"""
Trajectory Parameter Debugging Module
Shows exactly what parameters are being used during trajectory generation
"""

import streamlit as st
import numpy as np
from typing import Dict, Any, List

def debug_trajectory_parameters(layer_manager, roving_width, roving_thickness, dome_points, cylinder_points):
    """Debug and show all trajectory generation parameters"""
    
    st.markdown("### 🔍 Trajectory Parameter Debug Report")
    
    debug_data = []
    
    # Check vessel geometry parameters
    if hasattr(st.session_state, 'vessel_geometry') and st.session_state.vessel_geometry:
        vessel = st.session_state.vessel_geometry
        vessel_props = vessel.get_geometric_properties()
        
        debug_data.append({
            "Category": "Vessel Geometry",
            "Parameter": "Inner Diameter",
            "Value": f"{vessel.inner_diameter:.1f} mm",
            "Source": "vessel_geometry.inner_diameter"
        })
        
        debug_data.append({
            "Category": "Vessel Geometry", 
            "Parameter": "Wall Thickness",
            "Value": f"{vessel.wall_thickness:.1f} mm",
            "Source": "vessel_geometry.wall_thickness"
        })
        
        debug_data.append({
            "Category": "Vessel Geometry",
            "Parameter": "Cylindrical Length", 
            "Value": f"{vessel.cylindrical_length:.1f} mm",
            "Source": "vessel_geometry.cylindrical_length"
        })
        
        debug_data.append({
            "Category": "Vessel Geometry",
            "Parameter": "Dome Type",
            "Value": vessel.dome_type,
            "Source": "vessel_geometry.dome_type"
        })
    
    # Check layer parameters
    for i, layer in enumerate(layer_manager.layer_stack):
        debug_data.append({
            "Category": f"Layer {layer.layer_set_id}",
            "Parameter": "Layer Type",
            "Value": layer.layer_type,
            "Source": f"layer_stack[{i}].layer_type"
        })
        
        debug_data.append({
            "Category": f"Layer {layer.layer_set_id}",
            "Parameter": "Winding Angle",
            "Value": f"{layer.winding_angle_deg}°",
            "Source": f"layer_stack[{i}].winding_angle_deg"
        })
        
        debug_data.append({
            "Category": f"Layer {layer.layer_set_id}",
            "Parameter": "Physics Model",
            "Value": getattr(layer, 'physics_model', 'Not set'),
            "Source": f"layer_stack[{i}].physics_model"
        })
        
        debug_data.append({
            "Category": f"Layer {layer.layer_set_id}",
            "Parameter": "Number of Plies",
            "Value": str(layer.num_plies),
            "Source": f"layer_stack[{i}].num_plies"
        })
        
        debug_data.append({
            "Category": f"Layer {layer.layer_set_id}",
            "Parameter": "Ply Thickness",
            "Value": f"{layer.single_ply_thickness_mm:.3f} mm",
            "Source": f"layer_stack[{i}].single_ply_thickness_mm"
        })
    
    # Check roving parameters
    debug_data.append({
        "Category": "Roving Parameters",
        "Parameter": "Roving Width",
        "Value": f"{roving_width:.1f} mm",
        "Source": "user_input"
    })
    
    debug_data.append({
        "Category": "Roving Parameters", 
        "Parameter": "Roving Thickness",
        "Value": f"{roving_thickness:.3f} mm",
        "Source": "user_input"
    })
    
    # Check discretization parameters
    debug_data.append({
        "Category": "Discretization",
        "Parameter": "Dome Points",
        "Value": str(dome_points),
        "Source": "user_input"
    })
    
    debug_data.append({
        "Category": "Discretization",
        "Parameter": "Cylinder Points", 
        "Value": str(cylinder_points),
        "Source": "user_input"
    })
    
    # Show the debug table
    import pandas as pd
    df = pd.DataFrame(debug_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    return debug_data

def debug_unified_planner_call(layer_def, vessel_geometry, roving_width_mm, roving_thickness_mm):
    """Debug the exact parameters passed to UnifiedTrajectoryPlanner"""
    
    st.markdown("### 🎯 UnifiedTrajectoryPlanner Call Debug")
    
    # Show what should be passed to the planner
    planner_params = {
        "vessel_geometry": "VesselGeometry object",
        "roving_width_m": roving_width_mm / 1000.0,
        "roving_thickness_m": roving_thickness_mm / 1000.0,
        "pattern_type": "helical",
        "coverage_mode": "full_coverage", 
        "physics_model": "constant_angle",
        "continuity_level": "C2",
        "target_angle_deg": layer_def.winding_angle_deg,
        "num_layers": layer_def.num_plies
    }
    
    st.write("**Parameters that should be passed to UnifiedTrajectoryPlanner:**")
    for key, value in planner_params.items():
        st.write(f"- {key}: {value}")
    
    # Check if the values match what's actually being used
    st.markdown("### ⚠️ Known Issues")
    st.warning("The trajectory planner appears to be using hardcoded values instead of these parameters")
    st.info("This explains why you always get 2800 points regardless of your inputs")
    
    return planner_params

def show_trajectory_generation_flow():
    """Show the actual trajectory generation code flow"""
    
    st.markdown("### 🔄 Trajectory Generation Flow Analysis")
    
    flow_steps = [
        "1. User sets parameters (angle, roving width, etc.)",
        "2. Layer stack manager stores layer definitions", 
        "3. Generate All Layer Trajectories button clicked",
        "4. trajectory_generation_with_fallbacks() called",
        "5. For each layer: UnifiedTrajectoryPlanner created",
        "6. generate_trajectory() called with parameters",
        "7. ❌ ISSUE: Planner ignores parameters, uses hardcoded values",
        "8. Result: Always 2800 points, same pattern"
    ]
    
    for step in flow_steps:
        if "❌ ISSUE" in step:
            st.error(step)
        elif step.startswith(("1.", "2.", "3.", "4.")):
            st.success(step)
        else:
            st.info(step)
    
    st.markdown("### 🛠️ Fix Required")
    st.error("The UnifiedTrajectoryPlanner needs to be updated to actually use the input parameters instead of hardcoded values")

def analyze_trajectory_output(trajectory_data):
    """Analyze the actual trajectory output to confirm the issue"""
    
    if not trajectory_data or not trajectory_data.get('success'):
        st.error("No valid trajectory data to analyze")
        return
    
    st.markdown("### 📊 Trajectory Output Analysis")
    
    # Check if winding angles vary
    winding_angles = trajectory_data.get('winding_angles_deg', [])
    if winding_angles:
        angles_array = np.array(winding_angles)
        angle_min = np.min(angles_array)
        angle_max = np.max(angles_array)
        angle_std = np.std(angles_array)
        
        st.write(f"**Winding Angle Analysis:**")
        st.write(f"- Min angle: {angle_min:.1f}°")
        st.write(f"- Max angle: {angle_max:.1f}°")
        st.write(f"- Standard deviation: {angle_std:.3f}°")
        
        if angle_std < 0.1:
            st.warning("Winding angle is nearly constant - confirms hardcoded behavior")
        else:
            st.success("Winding angle varies as expected")
    
    # Check total points
    total_points = trajectory_data.get('total_points', 0)
    st.write(f"**Total Points:** {total_points}")
    
    if total_points == 2800:
        st.error("Always getting 2800 points confirms hardcoded behavior")
    else:
        st.success("Point count varies based on parameters")
    
    # Check coordinate ranges
    x_points = trajectory_data.get('x_points_m', [])
    y_points = trajectory_data.get('y_points_m', [])
    z_points = trajectory_data.get('z_points_m', [])
    
    if x_points and y_points and z_points:
        x_range = max(x_points) - min(x_points)
        y_range = max(y_points) - min(y_points) 
        z_range = max(z_points) - min(z_points)
        
        st.write(f"**Coordinate Ranges:**")
        st.write(f"- X range: {x_range:.3f} m")
        st.write(f"- Y range: {y_range:.3f} m") 
        st.write(f"- Z range: {z_range:.3f} m")
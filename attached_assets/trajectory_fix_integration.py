"""
Integration code for adding trajectory array fixes to your Streamlit app
Add this to your app.py or create as a separate module
"""

import streamlit as st
import numpy as np
import pandas as pd
from trajectory_array_fix import (
    fix_trajectory_array_mismatches, 
    diagnose_trajectory_issues,
    apply_trajectory_array_fix_to_session
)

def add_trajectory_debug_section():
    """
    Add this to your trajectory planning page to provide debugging tools
    """
    st.markdown("---")
    st.markdown("### 🔧 Trajectory Array Debugging Tools")
    
    with st.expander("🔍 Array Mismatch Diagnostics & Fixes", expanded=False):
        st.markdown("""
        **Common Issues:**
        - Constant radius trajectories (all points at same distance from axis)
        - Missing trajectory points
        - Array length mismatches between x, y, z coordinates
        - Invalid coordinate values (NaN, infinite)
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔍 Diagnose Trajectory Issues", type="secondary"):
                run_trajectory_diagnostics()
        
        with col2:
            if st.button("🔧 Apply Array Fixes", type="primary"):
                if apply_trajectory_array_fix_to_session():
                    st.rerun()
        
        with col3:
            if st.button("📊 Show Coordinate Analysis", type="secondary"):
                show_detailed_coordinate_analysis()

def run_trajectory_diagnostics():
    """Run comprehensive trajectory diagnostics"""
    st.markdown("#### 🔍 Trajectory Diagnostics Results")
    
    # Check single trajectory
    if 'trajectory_data' in st.session_state and st.session_state.trajectory_data:
        st.markdown("**Single Trajectory Analysis:**")
        diagnosis = diagnose_trajectory_issues(st.session_state.trajectory_data)
        display_diagnosis_results(diagnosis, "Single Trajectory")
    
    # Check layer trajectories
    if 'all_layer_trajectories' in st.session_state and st.session_state.all_layer_trajectories:
        st.markdown("**Layer Trajectories Analysis:**")
        for i, layer_traj in enumerate(st.session_state.all_layer_trajectories):
            if 'trajectory_data' in layer_traj:
                diagnosis = diagnose_trajectory_issues(layer_traj['trajectory_data'])
                display_diagnosis_results(diagnosis, f"Layer {layer_traj.get('layer_id', i+1)}")

def display_diagnosis_results(diagnosis: dict, trajectory_name: str):
    """Display diagnosis results in a clear format"""
    if diagnosis['status'] == 'error':
        st.error(f"❌ {trajectory_name}: {diagnosis['message']}")
        if 'available_keys' in diagnosis:
            st.write(f"Available keys: {diagnosis['available_keys']}")
        return
    
    st.success(f"✅ {trajectory_name}: Found {diagnosis['coordinate_sets_found']} coordinate sets")
    
    # Create summary table
    summary_data = []
    for analysis in diagnosis['sets_analysis']:
        summary_data.append({
            'Source': analysis['source'],
            'Points': analysis['points_count'],
            'Quality': f"{analysis['quality_score']:.1f}",
            'Radius Var %': f"{analysis['radius_variation_pct']:.3f}",
            'Z Range (m)': f"{analysis['z_range_m']:.3f}",
            'Constant R Issue': "❌ Yes" if analysis['constant_radius_issue'] else "✅ No"
        })
    
    if summary_data:
        df = pd.DataFrame(summary_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Highlight issues
        issues = [analysis for analysis in diagnosis['sets_analysis'] if analysis['constant_radius_issue']]
        if issues:
            st.error(f"⚠️ {len(issues)} coordinate set(s) have constant radius issues!")
        else:
            st.success("✅ No constant radius issues detected")

def show_detailed_coordinate_analysis():
    """Show detailed coordinate analysis and statistics"""
    st.markdown("#### 📊 Detailed Coordinate Analysis")
    
    trajectory_data = None
    if 'trajectory_data' in st.session_state and st.session_state.trajectory_data:
        trajectory_data = st.session_state.trajectory_data
    elif ('all_layer_trajectories' in st.session_state and 
          st.session_state.all_layer_trajectories and
          st.session_state.all_layer_trajectories[0].get('trajectory_data')):
        trajectory_data = st.session_state.all_layer_trajectories[0]['trajectory_data']
    
    if not trajectory_data:
        st.warning("No trajectory data available for analysis")
        return
    
    # Try to extract coordinates
    coords_extracted = extract_coordinates_for_analysis(trajectory_data)
    
    if not coords_extracted:
        st.error("Could not extract coordinates from trajectory data")
        st.write("**Available keys:**", list(trajectory_data.keys()))
        return
    
    x, y, z = coords_extracted
    
    # Calculate analysis metrics
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    
    # Create analysis dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Points", len(x))
        st.metric("X Range (m)", f"{np.max(x) - np.min(x):.3f}")
    
    with col2:
        st.metric("Y Range (m)", f"{np.max(y) - np.min(y):.3f}")
        st.metric("Z Range (m)", f"{np.max(z) - np.min(z):.3f}")
    
    with col3:
        st.metric("Mean Radius (m)", f"{np.mean(rho):.3f}")
        st.metric("Radius Std (m)", f"{np.std(rho):.6f}")
    
    with col4:
        radius_var_pct = (np.std(rho) / np.mean(rho) * 100) if np.mean(rho) > 0 else 0
        st.metric("Radius Variation %", f"{radius_var_pct:.3f}")
        
        # Critical issue indicator
        if radius_var_pct < 0.1:
            st.metric("Status", "❌ Constant R")
        elif radius_var_pct < 1.0:
            st.metric("Status", "⚠️ Low Var")
        else:
            st.metric("Status", "✅ Good")
    
    # Plot coordinate analysis
    if st.checkbox("Show Coordinate Plots"):
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot 1: 3D trajectory
        ax = axes[0, 0]
        ax.plot(x, y, 'b-', alpha=0.7)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Trajectory XY View')
        ax.set_aspect('equal')
        ax.grid(True)
        
        # Plot 2: Radius vs index
        ax = axes[0, 1]
        ax.plot(rho, 'r-', alpha=0.7)
        ax.set_xlabel('Point Index')
        ax.set_ylabel('Radius (m)')
        ax.set_title('Radius Variation')
        ax.grid(True)
        
        # Plot 3: Z vs index
        ax = axes[1, 0]
        ax.plot(z, 'g-', alpha=0.7)
        ax.set_xlabel('Point Index')
        ax.set_ylabel('Z (m)')
        ax.set_title('Z Coordinate')
        ax.grid(True)
        
        # Plot 4: Phi vs index
        ax = axes[1, 1]
        ax.plot(np.degrees(phi), 'm-', alpha=0.7)
        ax.set_xlabel('Point Index')
        ax.set_ylabel('Phi (degrees)')
        ax.set_title('Angular Position')
        ax.grid(True)
        
        plt.tight_layout()
        st.pyplot(fig)

def extract_coordinates_for_analysis(trajectory_data: dict):
    """Extract coordinates for analysis from various possible formats"""
    try:
        # Method 1: Direct arrays
        if all(key in trajectory_data for key in ['x_points_m', 'y_points_m', 'z_points_m']):
            x = np.array(trajectory_data['x_points_m'])
            y = np.array(trajectory_data['y_points_m'])
            z = np.array(trajectory_data['z_points_m'])
            if len(x) > 0 and len(y) > 0 and len(z) > 0:
                return x, y, z
        
        # Method 2: Nested data
        if 'trajectory_data' in trajectory_data:
            nested = trajectory_data['trajectory_data']
            if all(key in nested for key in ['x_points_m', 'y_points_m', 'z_points_m']):
                x = np.array(nested['x_points_m'])
                y = np.array(nested['y_points_m'])
                z = np.array(nested['z_points_m'])
                if len(x) > 0 and len(y) > 0 and len(z) > 0:
                    return x, y, z
        
        # Method 3: From path_points
        if 'path_points' in trajectory_data and trajectory_data['path_points']:
            points = trajectory_data['path_points']
            x_coords, y_coords, z_coords = [], [], []
            
            for point in points:
                if isinstance(point, dict):
                    x_val = point.get('x_m', point.get('x', 0))
                    y_val = point.get('y_m', point.get('y', 0))
                    z_val = point.get('z_m', point.get('z', 0))
                    x_coords.append(x_val)
                    y_coords.append(y_val)
                    z_coords.append(z_val)
            
            if len(x_coords) > 0:
                return np.array(x_coords), np.array(y_coords), np.array(z_coords)
        
        return None
        
    except Exception as e:
        st.error(f"Error extracting coordinates: {e}")
        return None

def add_quick_fix_button():
    """Add a quick fix button to the main interface"""
    st.markdown("---")
    
    # Quick status check
    has_trajectory_issues = check_for_trajectory_issues()
    
    if has_trajectory_issues:
        st.error("⚠️ **Trajectory Array Issues Detected**")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write("Common issues: constant radius, missing points, array mismatches")
        with col2:
            if st.button("🔧 Quick Fix", type="primary", key="quick_trajectory_fix"):
                with st.spinner("Applying trajectory fixes..."):
                    if apply_trajectory_array_fix_to_session():
                        st.success("✅ Fixes applied successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Could not apply fixes")
    else:
        st.success("✅ No obvious trajectory issues detected")

def check_for_trajectory_issues() -> bool:
    """Quick check for obvious trajectory issues"""
    try:
        # Check single trajectory
        if 'trajectory_data' in st.session_state and st.session_state.trajectory_data:
            coords = extract_coordinates_for_analysis(st.session_state.trajectory_data)
            if coords:
                x, y, z = coords
                rho = np.sqrt(x**2 + y**2)
                if len(rho) > 1:
                    radius_var_pct = (np.std(rho) / np.mean(rho) * 100) if np.mean(rho) > 0 else 0
                    if radius_var_pct < 0.1:  # Less than 0.1% variation indicates constant radius
                        return True
        
        # Check layer trajectories
        if 'all_layer_trajectories' in st.session_state and st.session_state.all_layer_trajectories:
            for layer_traj in st.session_state.all_layer_trajectories:
                if 'trajectory_data' in layer_traj:
                    coords = extract_coordinates_for_analysis(layer_traj['trajectory_data'])
                    if coords:
                        x, y, z = coords
                        rho = np.sqrt(x**2 + y**2)
                        if len(rho) > 1:
                            radius_var_pct = (np.std(rho) / np.mean(rho) * 100) if np.mean(rho) > 0 else 0
                            if radius_var_pct < 0.1:
                                return True
        
        return False
        
    except Exception:
        return True  # Assume issues if we can't check properly

# Add this to your visualization page
def enhanced_visualization_with_array_fixes():
    """Enhanced visualization that automatically applies array fixes"""
    
    # Auto-fix check
    if check_for_trajectory_issues():
        st.warning("🔧 Trajectory array issues detected - applying automatic fixes...")
        if apply_trajectory_array_fix_to_session():
            st.success("✅ Array fixes applied automatically")
        else:
            st.error("❌ Could not apply automatic fixes")
            return
    
    # Continue with normal visualization...
    # (Your existing visualization code here)

# Integration example for your main app.py
def integrate_into_trajectory_planning_page():
    """
    Example of how to integrate into your trajectory planning page
    Add this at the end of your trajectory_planning_page() function
    """
    
    # Add debugging section
    add_trajectory_debug_section()
    
    # Add quick fix button
    add_quick_fix_button()

def integrate_into_visualization_page():
    """
    Example of how to integrate into your visualization page
    Add this at the beginning of your visualization_page() function
    """
    
    # Auto-check and fix at start of visualization
    if check_for_trajectory_issues():
        with st.expander("🔧 Array Issues Detected - Auto-Fix Available", expanded=True):
            st.warning("Trajectory has array mismatch issues that may cause visualization problems")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔧 Apply Auto-Fix", type="primary"):
                    if apply_trajectory_array_fix_to_session():
                        st.success("✅ Fixes applied - page will reload")
                        st.rerun()
            with col2:
                if st.button("📊 Show Details"):
                    run_trajectory_diagnostics()

# Example usage in your app.py:
"""
# Add this to the end of your trajectory_planning_page() function:
integrate_into_trajectory_planning_page()

# Add this to the beginning of your visualization_page() function:
integrate_into_visualization_page()

# Or add a dedicated debugging page:
def trajectory_debugging_page():
    st.title("🔧 Trajectory Debugging")
    st.markdown("Advanced tools for diagnosing and fixing trajectory issues")
    
    add_trajectory_debug_section()
    show_detailed_coordinate_analysis()
"""
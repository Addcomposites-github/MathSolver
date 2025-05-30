"""
Quick integration fixes for the numpy boolean array error in visualization
Add these fixes to your existing code to resolve the error immediately
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Optional, Any

# QUICK FIX 1: Add this function to your app.py or visualization module
def fix_numpy_boolean_errors():
    """
    Apply numpy boolean error fixes to session state data
    Call this before any visualization
    """
    
    def safe_array_conversion(data):
        """Convert data to numpy array safely"""
        try:
            if data is None:
                return np.array([])
            arr = np.array(data, dtype=float)
            # Remove NaN and inf values
            valid_mask = np.isfinite(arr)
            if np.any(valid_mask):
                return arr[valid_mask]
            return np.array([])
        except Exception:
            return np.array([])
    
    def safe_has_data(data):
        """Safely check if data exists and has content"""
        try:
            if data is None:
                return False
            arr = np.asarray(data)
            return arr.size > 0
        except Exception:
            return False
    
    # Fix trajectory data in session state
    if 'trajectory_data' in st.session_state and st.session_state.trajectory_data:
        traj_data = st.session_state.trajectory_data
        
        # Fix direct coordinate arrays
        for coord_key in ['x_points_m', 'y_points_m', 'z_points_m']:
            if coord_key in traj_data:
                traj_data[coord_key] = safe_array_conversion(traj_data[coord_key]).tolist()
        
        # Fix nested trajectory data
        if 'trajectory_data' in traj_data and isinstance(traj_data['trajectory_data'], dict):
            nested_data = traj_data['trajectory_data']
            for coord_key in ['x_points_m', 'y_points_m', 'z_points_m']:
                if coord_key in nested_data:
                    nested_data[coord_key] = safe_array_conversion(nested_data[coord_key]).tolist()
    
    # Fix layer trajectories
    if 'all_layer_trajectories' in st.session_state and st.session_state.all_layer_trajectories:
        for layer_traj in st.session_state.all_layer_trajectories:
            if 'trajectory_data' in layer_traj and layer_traj['trajectory_data']:
                traj_data = layer_traj['trajectory_data']
                for coord_key in ['x_points_m', 'y_points_m', 'z_points_m']:
                    if coord_key in traj_data:
                        traj_data[coord_key] = safe_array_conversion(traj_data[coord_key]).tolist()


# QUICK FIX 2: Replace problematic visualization calls
def create_error_safe_visualization(vessel_geometry, trajectory_data: Optional[Dict] = None, options: Optional[Dict] = None) -> go.Figure:
    """
    Error-safe visualization that prevents numpy boolean issues
    Use this instead of your current visualization function
    """
    
    # Apply fixes first
    fix_numpy_boolean_errors()
    
    fig = go.Figure()
    
    try:
        # Safe vessel geometry handling
        if vessel_geometry is not None:
            add_safe_vessel_geometry(fig, vessel_geometry)
        
        # Safe trajectory handling
        if trajectory_data is not None:
            add_safe_trajectory_data(fig, trajectory_data, options or {})
        
        # Safe layout configuration
        configure_safe_layout(fig)
        
        return fig
        
    except Exception as e:
        st.error(f"Visualization error caught and handled: {str(e)}")
        return create_fallback_visualization()


def add_safe_vessel_geometry(fig: go.Figure, vessel_geometry):
    """Add vessel geometry without array boolean errors"""
    try:
        if not hasattr(vessel_geometry, 'get_profile_points'):
            return
        
        profile = vessel_geometry.get_profile_points()
        if not profile or 'z_mm' not in profile or 'r_inner_mm' not in profile:
            return
        
        # Safe array extraction
        z_data = profile['z_mm']
        r_data = profile['r_inner_mm']
        
        if z_data is None or r_data is None:
            return
        
        # Convert to numpy arrays safely
        z_arr = np.asarray(z_data, dtype=float)
        r_arr = np.asarray(r_data, dtype=float)
        
        # Check arrays have content (avoid boolean array error)
        if z_arr.size == 0 or r_arr.size == 0:
            return
        
        # Remove invalid values
        valid_mask = np.isfinite(z_arr) & np.isfinite(r_arr)
        if not np.any(valid_mask):  # Safe check
            return
        
        z_clean = z_arr[valid_mask]
        r_clean = r_arr[valid_mask]
        
        if len(z_clean) < 2:  # Need at least 2 points
            return
        
        # Create simple surface
        theta = np.linspace(0, 2*np.pi, 32)
        Z_mesh, Theta_mesh = np.meshgrid(z_clean, theta)
        R_mesh = np.tile(r_clean, (32, 1))
        
        X_mesh = R_mesh * np.cos(Theta_mesh)
        Y_mesh = R_mesh * np.sin(Theta_mesh)
        
        fig.add_trace(go.Surface(
            x=X_mesh, y=Y_mesh, z=Z_mesh,
            colorscale='Greys',
            opacity=0.3,
            name='Vessel',
            showscale=False
        ))
        
        st.success(f"✅ Vessel geometry added ({len(z_clean)} points)")
        
    except Exception as e:
        st.warning(f"Could not add vessel geometry: {e}")


def add_safe_trajectory_data(fig: go.Figure, trajectory_data: Dict, options: Dict):
    """Add trajectory data without array boolean errors"""
    try:
        # Extract coordinates safely
        x_coords, y_coords, z_coords = extract_safe_coordinates(trajectory_data)
        
        if x_coords is None or len(x_coords) == 0:
            st.warning("No valid trajectory coordinates found")
            return
        
        # Convert to millimeters
        x_mm = x_coords * 1000
        y_mm = y_coords * 1000
        z_mm = z_coords * 1000
        
        # Apply decimation for performance
        decimation = options.get('decimation_factor', 5)
        if decimation > 1 and len(x_mm) > decimation:
            indices = np.arange(0, len(x_mm), decimation)
            x_mm = x_mm[indices]
            y_mm = y_mm[indices]
            z_mm = z_mm[indices]
        
        # Add trajectory line
        fig.add_trace(go.Scatter3d(
            x=x_mm, y=y_mm, z=z_mm,
            mode='lines+markers',
            line=dict(color='red', width=4),
            marker=dict(size=2),
            name=f'Trajectory ({len(x_mm)} points)',
            hovertemplate='X: %{x:.1f}mm<br>Y: %{y:.1f}mm<br>Z: %{z:.1f}mm<extra></extra>'
        ))
        
        # Add start/end markers
        if len(x_mm) > 0:
            fig.add_trace(go.Scatter3d(
                x=[x_mm[0]], y=[y_mm[0]], z=[z_mm[0]],
                mode='markers',
                marker=dict(size=10, color='green'),
                name='Start'
            ))
            
            if len(x_mm) > 1:
                fig.add_trace(go.Scatter3d(
                    x=[x_mm[-1]], y=[y_mm[-1]], z=[z_mm[-1]],
                    mode='markers',
                    marker=dict(size=10, color='darkred'),
                    name='End'
                ))
        
        st.success(f"✅ Trajectory added ({len(x_mm)} points)")
        
    except Exception as e:
        st.warning(f"Could not add trajectory: {e}")


def extract_safe_coordinates(trajectory_data: Dict):
    """Extract coordinates safely without array boolean errors"""
    try:
        # Method 1: Direct arrays
        if all(key in trajectory_data for key in ['x_points_m', 'y_points_m', 'z_points_m']):
            x_data = trajectory_data['x_points_m']
            y_data = trajectory_data['y_points_m']
            z_data = trajectory_data['z_points_m']
            
            if x_data is not None and y_data is not None and z_data is not None:
                x_arr = np.asarray(x_data, dtype=float)
                y_arr = np.asarray(y_data, dtype=float)
                z_arr = np.asarray(z_data, dtype=float)
                
                # Check if arrays have content
                if x_arr.size > 0 and y_arr.size > 0 and z_arr.size > 0:
                    # Ensure same length
                    min_len = min(len(x_arr), len(y_arr), len(z_arr))
                    return x_arr[:min_len], y_arr[:min_len], z_arr[:min_len]
        
        # Method 2: Nested data
        if 'trajectory_data' in trajectory_data:
            nested = trajectory_data['trajectory_data']
            if isinstance(nested, dict):
                return extract_safe_coordinates(nested)
        
        # Method 3: Path points
        if 'path_points' in trajectory_data:
            path_points = trajectory_data['path_points']
            if path_points is not None and len(path_points) > 0:
                x_coords, y_coords, z_coords = [], [], []
                
                for point in path_points:
                    try:
                        if isinstance(point, dict):
                            x = point.get('x_m', point.get('x', 0))
                            y = point.get('y_m', point.get('y', 0))
                            z = point.get('z_m', point.get('z', 0))
                            
                            if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                                x_coords.append(float(x))
                                y_coords.append(float(y))
                                z_coords.append(float(z))
                    except Exception:
                        continue
                
                if len(x_coords) > 0:
                    return np.array(x_coords), np.array(y_coords), np.array(z_coords)
        
        return None, None, None
        
    except Exception:
        return None, None, None


def configure_safe_layout(fig: go.Figure):
    """Configure layout safely"""
    try:
        fig.update_layout(
            title="3D COPV Visualization (Error-Safe Mode)",
            scene=dict(
                xaxis_title="X (mm)",
                yaxis_title="Y (mm)",
                zaxis_title="Z (mm)",
                aspectmode='data'
            ),
            height=700,
            showlegend=True
        )
    except Exception as e:
        st.warning(f"Layout configuration error: {e}")


def create_fallback_visualization() -> go.Figure:
    """Create minimal fallback visualization when everything fails"""
    fig = go.Figure()
    fig.add_annotation(
        text="Visualization Error - Using Fallback Mode<br>Please check trajectory data format",
        x=0.5, y=0.5,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=16, color="red")
    )
    fig.update_layout(
        title="Fallback Visualization",
        height=400
    )
    return fig


# QUICK FIX 3: Add this to your visualization page function
def apply_visualization_quick_fix():
    """
    Apply this at the start of your visualization_page() function
    """
    
    # Check for the specific numpy boolean error
    try:
        # Test for potential array boolean issues in session state
        if 'trajectory_data' in st.session_state and st.session_state.trajectory_data:
            traj_data = st.session_state.trajectory_data
            
            # Test coordinate extraction without triggering boolean error
            for coord_key in ['x_points_m', 'y_points_m', 'z_points_m']:
                if coord_key in traj_data:
                    coord_data = traj_data[coord_key]
                    if coord_data is not None:
                        # This would normally cause the boolean error
                        # if coord_data:  # DON'T DO THIS WITH ARRAYS
                        
                        # Instead do this:
                        if isinstance(coord_data, (list, tuple, np.ndarray)) and len(coord_data) > 0:
                            # Safe check
                            arr = np.asarray(coord_data)
                            if arr.size > 0:  # Safe size check
                                st.info(f"✅ Found {coord_key}: {arr.size} values")
        
        return True
        
    except Exception as e:
        if "truth value of an array" in str(e).lower():
            st.error("❌ Numpy boolean array error detected!")
            
            # Apply automatic fix
            st.info("🔧 Applying automatic fix...")
            fix_numpy_boolean_errors()
            st.success("✅ Fix applied - try visualization again")
            
            return False
        else:
            st.warning(f"Other visualization error: {e}")
            return False


# INTEGRATION EXAMPLE: How to use in your existing code
def integrate_visualization_fixes():
    """
    Example of how to integrate these fixes into your existing visualization_page()
    """
    
    # Add this at the very beginning of your visualization_page() function:
    
    # STEP 1: Apply quick fix check
    if not apply_visualization_quick_fix():
        st.warning("Applied fixes - please try visualization again")
        return
    
    # STEP 2: Replace your existing visualization call
    # OLD CODE (causes error):
    # fig = create_streamlined_3d_visualization(vessel_geometry, trajectory_data, options)
    
    # NEW CODE (error-safe):
    # fig = create_error_safe_visualization(vessel_geometry, trajectory_data, options)
    
    # STEP 3: Add error handling wrapper
    # try:
    #     fig = create_error_safe_visualization(vessel_geometry, trajectory_data, options)
    #     st.plotly_chart(fig, use_container_width=True)
    # except Exception as e:
    #     st.error(f"Visualization failed: {e}")
    #     # Show fallback
    #     fig = create_fallback_visualization()
    #     st.plotly_chart(fig, use_container_width=True)
    
    pass


# QUICK TEST FUNCTION: Add this button to test fixes
def add_visualization_test_button():
    """
    Add this button to your visualization page to test the fixes
    """
    if st.button("🔧 Test Visualization Fixes", type="secondary"):
        st.markdown("### Testing Visualization System...")
        
        # Test 1: Check for trajectory data
        if 'trajectory_data' in st.session_state:
            st.success("✅ Trajectory data found")
            
            # Test 2: Safe coordinate extraction
            coords = extract_safe_coordinates(st.session_state.trajectory_data)
            if coords[0] is not None:
                st.success(f"✅ Coordinates extracted: {len(coords[0])} points")
                
                # Test 3: Create test visualization
                try:
                    fig = create_error_safe_visualization(
                        st.session_state.vessel_geometry,
                        st.session_state.trajectory_data
                    )
                    st.success("✅ Visualization created successfully")
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Visualization test failed: {e}")
            else:
                st.error("❌ Could not extract coordinates")
        else:
            st.warning("⚠️ No trajectory data in session state")


# IMMEDIATE FIX: Add this single line to your visualization page
def immediate_fix_for_visualization_page():
    """
    IMMEDIATE FIX: Add this single line at the start of your visualization_page()
    """
    
    # Add this one line to fix the numpy boolean error immediately:
    fix_numpy_boolean_errors()
    
    # Then use your existing visualization code, but wrap it in try/except:
    # try:
    #     # Your existing visualization code here
    #     pass
    # except Exception as e:
    #     if "truth value of an array" in str(e).lower():
    #         st.error("Array boolean error - applying fix...")
    #         fix_numpy_boolean_errors()
    #         st.info("Fix applied - please refresh page")
    #     else:
    #         st.error(f"Other error: {e}")

"""
Quick fix for numpy boolean array errors in visualization
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Optional, Tuple

def fix_numpy_boolean_errors():
    """Apply numpy boolean error fixes to session state data"""
    
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

def create_error_safe_visualization(vessel_geometry, trajectory_data: Optional[Dict] = None) -> go.Figure:
    """Error-safe visualization that prevents numpy boolean issues"""
    
    # Apply fixes first
    fix_numpy_boolean_errors()
    
    fig = go.Figure()
    
    try:
        # Safe vessel geometry handling
        if vessel_geometry is not None:
            add_safe_vessel_geometry(fig, vessel_geometry)
        
        # Safe trajectory handling
        if trajectory_data is not None:
            add_safe_trajectory_data(fig, trajectory_data)
        
        # Safe layout configuration
        fig.update_layout(
            title="3D COPV Visualization (Error-Safe)",
            scene=dict(
                xaxis_title="X (mm)",
                yaxis_title="Y (mm)",
                zaxis_title="Z (mm)",
                aspectmode='data'
            ),
            height=700,
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Visualization error: {str(e)}")
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
        
        # Safe checks - avoid numpy boolean issues
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
        
        # Create surface
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
        
    except Exception as e:
        st.warning(f"Could not add vessel geometry: {e}")

def add_safe_trajectory_data(fig: go.Figure, trajectory_data: Dict):
    """Add trajectory data without array boolean errors"""
    try:
        # Extract coordinates safely
        x_coords, y_coords, z_coords = extract_safe_coordinates(trajectory_data)
        
        # Safe checks - avoid boolean array issues
        if x_coords is None or len(x_coords) == 0:
            st.warning("No valid trajectory coordinates found")
            return
        
        # Convert to millimeters
        x_mm = x_coords * 1000
        y_mm = y_coords * 1000
        z_mm = z_coords * 1000
        
        # Apply decimation for performance
        decimation = 5
        if decimation > 1 and len(x_mm) > decimation:
            indices = np.arange(0, len(x_mm), decimation)
            x_mm = x_mm[indices]
            y_mm = y_mm[indices]
            z_mm = z_mm[indices]
        
        # Add trajectory line
        fig.add_trace(go.Scatter3d(
            x=x_mm, y=y_mm, z=z_mm,
            mode='lines',
            line=dict(color='red', width=4),
            name=f'Trajectory ({len(x_mm)} points)'
        ))
        
        # Calculate quality metrics
        rho = np.sqrt(x_coords**2 + y_coords**2)
        radius_var_pct = (np.std(rho) / np.mean(rho) * 100) if np.mean(rho) > 0 else 0
        
        st.info(f"""
        **Trajectory Quality:**
        - Total points: {len(x_coords)}
        - Z span: {np.max(z_mm) - np.min(z_mm):.1f} mm
        - Radius variation: {radius_var_pct:.3f}%
        - Status: {'✅ Good' if radius_var_pct > 1.0 else '⚠️ Low variation' if radius_var_pct > 0.1 else '❌ Constant radius'}
        """)
        
    except Exception as e:
        st.warning(f"Could not add trajectory: {e}")

def extract_safe_coordinates(trajectory_data: Dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract coordinates safely without array boolean errors"""
    try:
        # Method 1: Direct arrays
        if all(key in trajectory_data for key in ['x_points_m', 'y_points_m', 'z_points_m']):
            x_data = trajectory_data['x_points_m']
            y_data = trajectory_data['y_points_m']
            z_data = trajectory_data['z_points_m']
            
            # Safe checks - avoid boolean array issues
            if x_data is not None and y_data is not None and z_data is not None:
                x_arr = np.asarray(x_data, dtype=float)
                y_arr = np.asarray(y_data, dtype=float)
                z_arr = np.asarray(z_data, dtype=float)
                
                # Check if arrays have content using .size instead of boolean
                if x_arr.size > 0 and y_arr.size > 0 and z_arr.size > 0:
                    # Ensure same length
                    min_len = min(len(x_arr), len(y_arr), len(z_arr))
                    return x_arr[:min_len], y_arr[:min_len], z_arr[:min_len]
        
        # Method 2: Nested data
        if 'trajectory_data' in trajectory_data:
            nested = trajectory_data['trajectory_data']
            if isinstance(nested, dict):
                return extract_safe_coordinates(nested)
        
        return None, None, None
        
    except Exception:
        return None, None, None

def create_fallback_visualization() -> go.Figure:
    """Create minimal fallback visualization when everything fails"""
    fig = go.Figure()
    fig.add_annotation(
        text="Visualization Error - Please check trajectory data",
        x=0.5, y=0.5,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=16, color="red")
    )
    fig.update_layout(title="Fallback Visualization", height=400)
    return fig
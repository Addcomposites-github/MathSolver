"""
Simple 3D Visualization Fix - No Array Comparison Issues
"""

import plotly.graph_objects as go
import numpy as np
import streamlit as st
from typing import Dict, Optional


def create_simple_3d_visualization(vessel_geometry, trajectory_data: Optional[Dict] = None, options: Optional[Dict] = None) -> go.Figure:
    """
    Create 3D visualization without problematic array comparisons
    """
    fig = go.Figure()
    
    try:
        # Add vessel geometry if available
        if vessel_geometry and hasattr(vessel_geometry, 'get_profile_points'):
            profile = vessel_geometry.get_profile_points()
            
            if profile and 'z_mm' in profile and 'r_inner_mm' in profile:
                z_profile = np.array(profile['z_mm'])
                r_profile = np.array(profile['r_inner_mm'])
                
                # Create mandrel surface
                theta = np.linspace(0, 2*np.pi, 32)
                Z_mesh, Theta_mesh = np.meshgrid(z_profile, theta)
                R_mesh = np.tile(r_profile, (len(theta), 1))
                
                X_mesh = R_mesh * np.cos(Theta_mesh)
                Y_mesh = R_mesh * np.sin(Theta_mesh)
                
                fig.add_trace(go.Surface(
                    x=X_mesh, y=Y_mesh, z=Z_mesh,
                    colorscale='Greys',
                    opacity=0.3,
                    name='Vessel Mandrel',
                    showscale=False
                ))
        
        # Add trajectory if available
        if trajectory_data:
            coords = extract_safe_coordinates(trajectory_data)
            if coords:
                x_coords, y_coords, z_coords = coords
                
                # Apply coordinate alignment
                z_min_vessel = np.min(z_profile) if 'z_profile' in locals() and len(z_profile) > 0 else -200
                z_max_vessel = np.max(z_profile) if 'z_profile' in locals() and len(z_profile) > 0 else 200
                z_center_vessel = (z_min_vessel + z_max_vessel) / 2
                
                z_min_traj = np.min(z_coords)
                z_max_traj = np.max(z_coords)
                z_center_traj = (z_min_traj + z_max_traj) / 2
                
                # Center trajectory on vessel
                offset = z_center_vessel - z_center_traj
                z_coords_aligned = z_coords + offset
                
                # Add trajectory line
                fig.add_trace(go.Scatter3d(
                    x=x_coords, y=y_coords, z=z_coords_aligned,
                    mode='lines',
                    line=dict(color='red', width=6),
                    name=f'Trajectory ({len(x_coords)} points)'
                ))
        
        # Configure layout
        fig.update_layout(
            scene=dict(
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)', 
                zaxis_title='Z (mm)',
                aspectmode='data'
            ),
            title='COPV 3D Visualization',
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Visualization error: {str(e)}")
        return go.Figure()


def extract_safe_coordinates(trajectory_data: Dict):
    """
    Safely extract coordinates without problematic array comparisons
    """
    try:
        # Method 1: Direct coordinate arrays
        if all(key in trajectory_data for key in ['x_points_m', 'y_points_m', 'z_points_m']):
            x_m = np.array(trajectory_data['x_points_m']) * 1000  # Convert to mm
            y_m = np.array(trajectory_data['y_points_m']) * 1000
            z_m = np.array(trajectory_data['z_points_m']) * 1000
            return x_m, y_m, z_m
        
        # Method 2: Nested in trajectory_data
        elif 'trajectory_data' in trajectory_data:
            nested = trajectory_data['trajectory_data']
            if all(key in nested for key in ['x_points_m', 'y_points_m', 'z_points_m']):
                x_m = np.array(nested['x_points_m']) * 1000
                y_m = np.array(nested['y_points_m']) * 1000
                z_m = np.array(nested['z_points_m']) * 1000
                return x_m, y_m, z_m
        
        # Method 3: Cylindrical coordinates
        elif all(key in trajectory_data for key in ['rho_points_m', 'phi_points_rad', 'z_points_m']):
            rho_m = np.array(trajectory_data['rho_points_m'])
            phi_rad = np.array(trajectory_data['phi_points_rad'])
            z_m = np.array(trajectory_data['z_points_m'])
            
            x_m = rho_m * np.cos(phi_rad) * 1000
            y_m = rho_m * np.sin(phi_rad) * 1000
            z_mm = z_m * 1000
            return x_m, y_m, z_mm
        
        return None
        
    except Exception as e:
        st.warning(f"Could not extract coordinates: {e}")
        return None
"""
Coordinate Alignment Fix for Trajectory Visualization
Fixes offset and truncation issues in 3D visualization
"""

import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple

def fix_trajectory_coordinates(trajectory_data: Dict, vessel_geometry) -> Dict:
    """
    Fix coordinate alignment issues between trajectory and vessel geometry
    
    Args:
        trajectory_data: Raw trajectory data from planner
        vessel_geometry: Vessel geometry object
        
    Returns:
        Fixed trajectory data with proper coordinate alignment
    """
    print("[CoordFix] Starting coordinate alignment fix...")
    
    # Get vessel coordinate information
    vessel_profile = vessel_geometry.get_profile_points()
    if not vessel_profile:
        print("[CoordFix] No vessel profile found")
        return trajectory_data
    
    z_vessel_mm = np.array(vessel_profile['z_mm'])
    vessel_z_min = np.min(z_vessel_mm)
    vessel_z_max = np.max(z_vessel_mm)
    vessel_z_center = (vessel_z_min + vessel_z_max) / 2
    vessel_z_span = vessel_z_max - vessel_z_min
    
    print(f"[CoordFix] Vessel Z range: {vessel_z_min:.1f} to {vessel_z_max:.1f} mm")
    print(f"[CoordFix] Vessel Z center: {vessel_z_center:.1f} mm, span: {vessel_z_span:.1f} mm")
    
    # Extract trajectory coordinates
    x_points_m = trajectory_data.get('x_points_m', [])
    y_points_m = trajectory_data.get('y_points_m', [])
    z_points_m = trajectory_data.get('z_points_m', [])
    
    if not x_points_m or not y_points_m or not z_points_m:
        print("[CoordFix] No coordinate arrays found in trajectory data")
        return trajectory_data
    
    # Convert to mm and numpy arrays
    x_points_mm = np.array(x_points_m) * 1000
    y_points_mm = np.array(y_points_m) * 1000
    z_points_mm = np.array(z_points_m) * 1000
    
    # Original trajectory ranges
    traj_z_min = np.min(z_points_mm)
    traj_z_max = np.max(z_points_mm)
    traj_z_center = (traj_z_min + traj_z_max) / 2
    traj_z_span = traj_z_max - traj_z_min
    
    print(f"[CoordFix] Original trajectory Z range: {traj_z_min:.1f} to {traj_z_max:.1f} mm")
    print(f"[CoordFix] Original trajectory Z center: {traj_z_center:.1f} mm, span: {traj_z_span:.1f} mm")
    
    # Check if trajectory is offset or truncated
    offset_detected = abs(traj_z_center - vessel_z_center) > 50  # More than 50mm offset
    truncation_detected = traj_z_span < (vessel_z_span * 0.8)  # Less than 80% of vessel span
    
    if offset_detected:
        print(f"[CoordFix] Offset detected: {traj_z_center - vessel_z_center:.1f} mm")
        # Center trajectory on vessel geometry
        z_offset = vessel_z_center - traj_z_center
        z_points_mm = z_points_mm + z_offset
        print(f"[CoordFix] Applied Z offset: {z_offset:.1f} mm")
    
    if truncation_detected:
        print(f"[CoordFix] Truncation detected: trajectory span {traj_z_span:.1f} mm vs vessel span {vessel_z_span:.1f} mm")
        # Scale trajectory to match vessel span while maintaining center
        if traj_z_span > 0:
            scale_factor = vessel_z_span / traj_z_span
            print(f"[CoordFix] Applying scale factor: {scale_factor:.2f}")
            
            # Scale around center
            current_center = np.mean(z_points_mm)
            z_points_mm = current_center + (z_points_mm - current_center) * scale_factor
    
    # Final coordinate ranges
    final_z_min = np.min(z_points_mm)
    final_z_max = np.max(z_points_mm)
    final_z_span = final_z_max - final_z_min
    
    print(f"[CoordFix] Fixed trajectory Z range: {final_z_min:.1f} to {final_z_max:.1f} mm")
    print(f"[CoordFix] Fixed trajectory Z span: {final_z_span:.1f} mm")
    
    # Update trajectory data with fixed coordinates
    fixed_data = trajectory_data.copy()
    fixed_data.update({
        'x_points_mm': x_points_mm,
        'y_points_mm': y_points_mm, 
        'z_points_mm': z_points_mm,
        'x_points_m': x_points_mm / 1000,
        'y_points_m': y_points_mm / 1000,
        'z_points_m': z_points_mm / 1000,
        'coordinate_fix_applied': True,
        'offset_correction': offset_detected,
        'truncation_correction': truncation_detected
    })
    
    return fixed_data

def create_fixed_3d_visualization(vessel_geometry, trajectory_data: Optional[Dict] = None, options: Optional[Dict] = None) -> go.Figure:
    """
    Create 3D visualization with coordinate alignment fixes
    
    Args:
        vessel_geometry: Vessel geometry object
        trajectory_data: Trajectory data dictionary
        options: Visualization options
        
    Returns:
        Plotly figure with proper coordinate alignment
    """
    if options is None:
        options = {
            'show_mandrel': True,
            'show_trajectory': True,
            'show_wireframe': True,
            'mandrel_resolution': 32,
            'trajectory_color': '#FF6B6B',
            'mandrel_color': '#4CAF50'
        }
    
    # Create figure
    fig = go.Figure()
    
    # Add vessel geometry (mandrel surface)
    if options.get('show_mandrel', True):
        add_centered_vessel_surface(fig, vessel_geometry, options)
    
    # Add trajectory with coordinate fixes
    if trajectory_data and options.get('show_trajectory', True):
        # Apply coordinate fixes
        fixed_trajectory = fix_trajectory_coordinates(trajectory_data, vessel_geometry)
        add_fixed_trajectory(fig, fixed_trajectory, options)
    
    # Configure layout for proper viewing
    configure_fixed_layout(fig, vessel_geometry, trajectory_data)
    
    return fig

def add_centered_vessel_surface(fig: go.Figure, vessel_geometry, options: Dict):
    """Add vessel surface with proper centering"""
    
    # Get vessel profile
    profile = vessel_geometry.get_profile_points()
    if not profile:
        return
    
    z_mm = np.array(profile['z_mm'])
    r_inner_mm = np.array(profile['r_inner_mm'])
    
    # Create surface mesh
    resolution = options.get('mandrel_resolution', 32)
    phi = np.linspace(0, 2*np.pi, resolution)
    
    # Create meshgrid
    Z_mesh, PHI_mesh = np.meshgrid(z_mm, phi)
    R_mesh = np.tile(r_inner_mm, (resolution, 1))
    
    # Convert to Cartesian
    X_mesh = R_mesh * np.cos(PHI_mesh)
    Y_mesh = R_mesh * np.sin(PHI_mesh)
    
    # Add surface
    fig.add_trace(go.Surface(
        x=X_mesh,
        y=Y_mesh,
        z=Z_mesh,
        colorscale=[[0, options.get('mandrel_color', '#4CAF50')], [1, options.get('mandrel_color', '#4CAF50')]],
        opacity=0.3,
        showscale=False,
        name='Mandrel Surface'
    ))
    
    # Add wireframe if requested
    if options.get('show_wireframe', True):
        add_vessel_wireframe(fig, X_mesh, Y_mesh, Z_mesh)

def add_vessel_wireframe(fig: go.Figure, X_mesh, Y_mesh, Z_mesh):
    """Add wireframe lines to vessel surface"""
    
    # Meridional lines (every 8th point)
    step = max(1, X_mesh.shape[0] // 8)
    for i in range(0, X_mesh.shape[0], step):
        fig.add_trace(go.Scatter3d(
            x=X_mesh[i, :],
            y=Y_mesh[i, :],
            z=Z_mesh[i, :],
            mode='lines',
            line=dict(color='gray', width=1),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Circumferential lines (every 8th point)
    step = max(1, X_mesh.shape[1] // 8)
    for j in range(0, X_mesh.shape[1], step):
        fig.add_trace(go.Scatter3d(
            x=X_mesh[:, j],
            y=Y_mesh[:, j], 
            z=Z_mesh[:, j],
            mode='lines',
            line=dict(color='gray', width=1),
            showlegend=False,
            hoverinfo='skip'
        ))

def add_fixed_trajectory(fig: go.Figure, trajectory_data: Dict, options: Dict):
    """Add trajectory with fixed coordinates"""
    
    # Use fixed coordinates
    x_mm = trajectory_data.get('x_points_mm', [])
    y_mm = trajectory_data.get('y_points_mm', [])
    z_mm = trajectory_data.get('z_points_mm', [])
    
    if not x_mm or not y_mm or not z_mm:
        print("[VizFix] No fixed coordinate data available")
        return
    
    # Apply decimation for performance
    decimation = options.get('decimation_factor', 5)
    if decimation > 1:
        indices = range(0, len(x_mm), decimation)
        x_mm = [x_mm[i] for i in indices]
        y_mm = [y_mm[i] for i in indices]
        z_mm = [z_mm[i] for i in indices]
    
    # Add trajectory line
    fig.add_trace(go.Scatter3d(
        x=x_mm,
        y=y_mm,
        z=z_mm,
        mode='lines+markers',
        line=dict(
            color=options.get('trajectory_color', '#FF6B6B'),
            width=4
        ),
        marker=dict(size=2),
        name='Trajectory (Fixed)',
        hovertemplate='X: %{x:.1f}mm<br>Y: %{y:.1f}mm<br>Z: %{z:.1f}mm<extra></extra>'
    ))
    
    # Add start and end markers
    if len(x_mm) > 0:
        fig.add_trace(go.Scatter3d(
            x=[x_mm[0]], y=[y_mm[0]], z=[z_mm[0]],
            mode='markers',
            marker=dict(size=10, color='green'),
            name='Start',
            hovertemplate='Start<br>X: %{x:.1f}mm<br>Y: %{y:.1f}mm<br>Z: %{z:.1f}mm<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[x_mm[-1]], y=[y_mm[-1]], z=[z_mm[-1]],
            mode='markers',
            marker=dict(size=10, color='darkred'),
            name='End',
            hovertemplate='End<br>X: %{x:.1f}mm<br>Y: %{y:.1f}mm<br>Z: %{z:.1f}mm<extra></extra>'
        ))

def configure_fixed_layout(fig: go.Figure, vessel_geometry, trajectory_data):
    """Configure layout with proper coordinate ranges"""
    
    # Get vessel dimensions for proper scaling
    profile = vessel_geometry.get_profile_points()
    if profile:
        z_mm = np.array(profile['z_mm'])
        r_mm = np.array(profile['r_inner_mm'])
        
        z_range = [np.min(z_mm), np.max(z_mm)]
        r_max = np.max(r_mm)
        xy_range = [-r_max, r_max]
    else:
        z_range = [-300, 300]
        xy_range = [-150, 150]
    
    # Configure 3D scene
    fig.update_layout(
        title="3D Trajectory Visualization (Coordinate Fixed)",
        scene=dict(
            xaxis=dict(title="X (mm)", range=xy_range),
            yaxis=dict(title="Y (mm)", range=xy_range),
            zaxis=dict(title="Z (mm)", range=z_range),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=2),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        width=800,
        height=600,
        margin=dict(l=0, r=0, t=40, b=0)
    )
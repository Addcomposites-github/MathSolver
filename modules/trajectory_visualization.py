"""
3D Trajectory Visualization for Multi-Layer COPV Design
Provides interactive 3D visualization of planned trajectories with layer-specific rendering
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import streamlit as st
from typing import Dict, List, Optional

def create_3d_trajectory_visualization(trajectory_data: Dict, vessel_geometry, layer_info: Dict = None,
                                     decimation_factor: int = 10, surface_segments: int = 30):
    """
    Create interactive 3D visualization of a single layer trajectory with performance optimization
    
    Parameters:
    -----------
    trajectory_data : Dict
        Trajectory data with path_points
    vessel_geometry : VesselGeometry
        Vessel geometry for mandrel surface
    layer_info : Dict
        Layer information (type, angle, etc.)
    decimation_factor : int
        Plot every Nth point for performance (default: 10)
    surface_segments : int
        Number of segments for mandrel surface (default: 30)
    """
    fig = go.Figure()
    
    # Get vessel profile for mandrel surface with performance optimization
    profile = vessel_geometry.get_profile_points()
    z_profile_mm = np.array(profile['z_mm'])
    r_profile_mm = np.array(profile['r_inner_mm'])
    
    # Downsample profile points if very dense
    max_profile_points = 50
    if len(z_profile_mm) > max_profile_points:
        indices = np.linspace(0, len(z_profile_mm) - 1, max_profile_points, dtype=int)
        z_profile_mm = z_profile_mm[indices]
        r_profile_mm = r_profile_mm[indices]
    
    # Convert to meters for plotting
    z_profile = z_profile_mm / 1000.0
    r_profile = r_profile_mm / 1000.0
    
    # Create mandrel surface with optimized segment count
    theta_surface = np.linspace(0, 2*np.pi, surface_segments)
    z_surface = np.tile(z_profile, (len(theta_surface), 1)).T
    r_surface = np.tile(r_profile, (len(theta_surface), 1)).T
    x_surface = r_surface * np.cos(theta_surface)
    y_surface = r_surface * np.sin(theta_surface)
    
    # Add mandrel surface
    fig.add_trace(go.Surface(
        x=x_surface,
        y=y_surface, 
        z=z_surface,
        colorscale='Greys',
        opacity=0.3,
        name='Mandrel Surface',
        showscale=False
    ))
    
    # Add trajectory path if available
    if trajectory_data and trajectory_data.get('success', False):
        # Get coordinate arrays from TrajectoryOutputStandardizer format
        x_points = trajectory_data.get('x_points_m', [])
        y_points = trajectory_data.get('y_points_m', [])
        z_points = trajectory_data.get('z_points_m', [])
        
        if len(x_points) > 0:
            # Apply decimation for performance
            if decimation_factor > 1 and len(x_points) > decimation_factor:
                indices = np.arange(0, len(x_points), decimation_factor)
                # Always include the last point
                if indices[-1] != len(x_points) - 1:
                    indices = np.append(indices, len(x_points) - 1)
                x_traj = np.array(x_points)[indices] * 1000  # Convert m to mm
                y_traj = np.array(y_points)[indices] * 1000
                z_traj = np.array(z_points)[indices] * 1000
            else:
                x_traj = np.array(x_points) * 1000  # Convert m to mm
                y_traj = np.array(y_points) * 1000
                z_traj = np.array(z_points) * 1000
            
            # Determine color based on layer type
            if layer_info:
                layer_type = layer_info.get('layer_type', 'unknown')
                if layer_type == 'hoop':
                    color = 'red'
                    name = f"Hoop Layer ({layer_info.get('winding_angle', 0)}°)"
                elif layer_type == 'helical':
                    color = 'blue' 
                    name = f"Helical Layer ({layer_info.get('winding_angle', 0)}°)"
                elif layer_type == 'polar':
                    color = 'green'
                    name = f"Polar Layer ({layer_info.get('winding_angle', 0)}°)"
                else:
                    color = 'orange'
                    name = f"Layer ({layer_info.get('winding_angle', 0)}°)"
            else:
                color = 'blue'
                name = 'Trajectory Path'
            
            # Add trajectory line
            fig.add_trace(go.Scatter3d(
                x=x_traj,
                y=y_traj,
                z=z_traj,
                mode='lines+markers',
                line=dict(color=color, width=6),
                marker=dict(size=3, color=color),
                name=name,
                hovertemplate='<b>%{fullData.name}</b><br>' +
                            'X: %{x:.1f}mm<br>' +
                            'Y: %{y:.1f}mm<br>' +
                            'Z: %{z:.1f}mm<br>' +
                            '<extra></extra>'
            ))
            
            # Add start and end markers
            if len(x_traj) > 0:
                # Start point
                fig.add_trace(go.Scatter3d(
                    x=[x_traj[0]],
                    y=[y_traj[0]], 
                    z=[z_traj[0]],
                    mode='markers',
                    marker=dict(size=8, color='green', symbol='diamond'),
                    name='Start Point',
                    showlegend=True
                ))
                
                # End point
                fig.add_trace(go.Scatter3d(
                    x=[x_traj[-1]],
                    y=[y_traj[-1]],
                    z=[z_traj[-1]],
                    mode='markers',
                    marker=dict(size=8, color='red', symbol='square'),
                    name='End Point',
                    showlegend=True
                ))
    
    # Update layout for better visualization
    fig.update_layout(
        title=dict(
            text=f"3D Trajectory Visualization - {layer_info.get('layer_type', 'Layer').title()} at {layer_info.get('winding_angle', 0)}°" if layer_info else "3D Trajectory Visualization",
            x=0.5,
            font=dict(size=16)
        ),
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)', 
            zaxis_title='Z (mm)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            ),
            bgcolor='white'
        ),
        width=800,
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left", 
            x=0.01
        )
    )
    
    return fig

def create_multi_layer_comparison(all_layer_trajectories: List[Dict], vessel_geometry):
    """
    Create visualization comparing multiple layer trajectories
    """
    fig = go.Figure()
    
    # Add mandrel surface (simplified)
    profile = vessel_geometry.get_profile_points()
    z_profile = profile['z_mm']
    r_profile = profile['r_inner_mm']
    
    theta_surface = np.linspace(0, 2*np.pi, 30)
    z_surface = np.tile(z_profile, (len(theta_surface), 1)).T
    r_surface = np.tile(r_profile, (len(theta_surface), 1)).T
    x_surface = r_surface * np.cos(theta_surface)
    y_surface = r_surface * np.sin(theta_surface)
    
    fig.add_trace(go.Surface(
        x=x_surface,
        y=y_surface,
        z=z_surface,
        colorscale='Greys',
        opacity=0.2,
        name='Mandrel',
        showscale=False
    ))
    
    # Color scheme for different layers
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan']
    
    for i, traj in enumerate(all_layer_trajectories):
        if 'path_points' in traj['trajectory_data']:
            path_points = traj['trajectory_data']['path_points']
            
            if len(path_points) > 0:
                z_traj = [point.get('z_mm', 0) for point in path_points]
                r_traj = [point.get('r_mm', 100) for point in path_points]
                theta_traj = [np.radians(point.get('theta_deg', 0)) for point in path_points]
                
                x_traj = [r * np.cos(theta) for r, theta in zip(r_traj, theta_traj)]
                y_traj = [r * np.sin(theta) for r, theta in zip(r_traj, theta_traj)]
                
                color = colors[i % len(colors)]
                
                fig.add_trace(go.Scatter3d(
                    x=x_traj,
                    y=y_traj,
                    z=z_traj,
                    mode='lines',
                    line=dict(color=color, width=4),
                    name=f"Layer {traj['layer_id']}: {traj['layer_type']} ({traj['winding_angle']}°)",
                    hovertemplate=f'<b>Layer {traj["layer_id"]}</b><br>' +
                                'X: %{x:.1f}mm<br>' +
                                'Y: %{y:.1f}mm<br>' +
                                'Z: %{z:.1f}mm<br>' +
                                '<extra></extra>'
                ))
    
    fig.update_layout(
        title=dict(
            text="Multi-Layer Trajectory Comparison",
            x=0.5,
            font=dict(size=16)
        ),
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        width=900,
        height=700,
        showlegend=True
    )
    
    return fig

def display_trajectory_metrics(trajectory_data: Dict, layer_info: Dict = None):
    """
    Display key metrics about the trajectory
    """
    if not trajectory_data or 'path_points' not in trajectory_data:
        st.warning("No trajectory data available for metrics")
        return
    
    path_points = trajectory_data['path_points']
    
    if len(path_points) == 0:
        st.warning("No path points in trajectory data")
        return
    
    # Calculate basic metrics
    num_points = len(path_points)
    z_values = [point.get('z_mm', 0) for point in path_points]
    r_values = [point.get('r_mm', 100) for point in path_points]
    theta_values = [point.get('theta_deg', 0) for point in path_points]
    
    z_range = max(z_values) - min(z_values) if z_values else 0
    r_range = max(r_values) - min(r_values) if r_values else 0
    theta_range = max(theta_values) - min(theta_values) if theta_values else 0
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Path Points", f"{num_points:,}")
    
    with col2:
        st.metric("Z Range", f"{z_range:.1f} mm")
    
    with col3:
        st.metric("R Range", f"{r_range:.1f} mm")
    
    with col4:
        st.metric("Angular Span", f"{theta_range:.1f}°")
    
    if layer_info:
        st.info(f"**Layer Type**: {layer_info.get('layer_type', 'Unknown')} | "
                f"**Winding Angle**: {layer_info.get('winding_angle', 0)}° | "
                f"**Status**: {trajectory_data.get('status', 'Unknown')}")
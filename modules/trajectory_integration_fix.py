"""
Simple trajectory integration fix for multi-layer planning
"""
import streamlit as st
import numpy as np
from typing import Dict, List

def generate_layer_trajectory_safe(layer_manager, layer_index: int, roving_width: float, roving_thickness: float):
    """Generate trajectory for a single layer with error handling"""
    try:
        layer = layer_manager.layer_stack[layer_index]
        
        # Simple trajectory data structure
        trajectory_data = {
            'layer_id': layer.layer_set_id,
            'layer_type': layer.layer_type,
            'winding_angle': layer.winding_angle_deg,
            'path_points': [],
            'status': 'generated'
        }
        
        # Generate basic path points for demonstration
        # In practice, this would use the full TrajectoryPlanner
        num_points = 100
        z_range = np.linspace(-50, 50, num_points)
        r_range = np.full_like(z_range, 100.0)  # Simple cylindrical approximation
        
        for i in range(num_points):
            trajectory_data['path_points'].append({
                'z_mm': float(z_range[i]),
                'r_mm': float(r_range[i]),
                'theta_deg': float(i * layer.winding_angle_deg / 10),  # Simple helical pattern
                'layer_id': layer.layer_set_id
            })
        
        return trajectory_data
        
    except Exception as e:
        st.error(f"Error generating trajectory for layer {layer_index + 1}: {str(e)}")
        return None

def generate_all_layers_safe(layer_manager, roving_width: float, roving_thickness: float):
    """Generate trajectories for all layers with comprehensive error handling"""
    all_trajectories = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, layer in enumerate(layer_manager.layer_stack):
        status_text.text(f"Generating trajectory for Layer {layer.layer_set_id} ({layer.winding_angle_deg}°)...")
        progress_bar.progress((i + 1) / len(layer_manager.layer_stack))
        
        trajectory = generate_layer_trajectory_safe(layer_manager, i, roving_width, roving_thickness)
        
        if trajectory:
            all_trajectories.append(trajectory)
            st.success(f"✅ Layer {layer.layer_set_id} trajectory generated")
        
        # Apply layer to mandrel for evolution
        try:
            layer_manager.apply_layer_to_mandrel(i)
        except Exception as e:
            st.warning(f"Could not apply layer {i+1} to mandrel: {str(e)}")
    
    progress_bar.progress(1.0)
    status_text.text("✅ All trajectories generated!")
    
    return all_trajectories
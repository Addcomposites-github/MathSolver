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
        
        # Create trajectory data structure that matches visualization expectations
        trajectory_data = {
            'path_points': [],
            'status': 'generated'
        }
        
        # Generate realistic trajectory based on layer type and mandrel geometry
        current_mandrel = layer_manager.get_current_mandrel_for_trajectory()
        profile = current_mandrel['profile_points']
        
        # Get vessel geometry details
        z_mm = profile['z_mm']
        r_mm = profile['r_inner_mm']
        
        # Find cylinder section and total length
        cylinder_mask = (np.abs(z_mm) <= np.max(np.abs(z_mm)) * 0.6)
        cylinder_length = np.max(z_mm) - np.min(z_mm)
        cylinder_radius = np.mean(r_mm[cylinder_mask]) if np.any(cylinder_mask) else 100.0
        
        trajectory_data['path_points'] = []
        
        if layer.layer_type == 'hoop':
            # Hoop layers: Full cylinder coverage with circumferential windings
            num_wraps = max(8, int(cylinder_length / roving_width))  # Ensure adequate coverage
            z_positions = np.linspace(np.min(z_mm), np.max(z_mm), num_wraps)
            
            for z_pos in z_positions:
                # Interpolate radius at this z position
                r_interp = np.interp(z_pos, z_mm, r_mm)
                
                # Complete circumferential wrap at this z position
                theta_range = np.linspace(0, 360, 72)  # 5-degree increments
                for theta in theta_range:
                    trajectory_data['path_points'].append({
                        'z_mm': float(z_pos),
                        'r_mm': float(r_interp),
                        'theta_deg': float(theta),
                        'layer_id': layer.layer_set_id
                    })
        else:
            # Helical layers: Proper helical winding
            angle_rad = np.radians(layer.winding_angle_deg)
            num_turns = max(3, int(cylinder_length / (roving_width * 2)))
            points_per_turn = 60
            total_points = num_turns * points_per_turn
            
            for i in range(total_points):
                progress = i / max(1, total_points - 1)
                z_pos = np.min(z_mm) + cylinder_length * progress
                theta = progress * num_turns * 360  # Degrees
                
                # Interpolate radius at this z position
                r_interp = np.interp(z_pos, z_mm, r_mm)
                
                trajectory_data['path_points'].append({
                    'z_mm': float(z_pos),
                    'r_mm': float(r_interp),
                    'theta_deg': float(theta % 360),
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
            # Wrap trajectory in the expected format for visualization
            wrapped_trajectory = {
                'layer_id': layer.layer_set_id,
                'layer_type': layer.layer_type,
                'winding_angle': layer.winding_angle_deg,
                'trajectory_data': trajectory,  # This contains the path_points
                'mandrel_state': None
            }
            all_trajectories.append(wrapped_trajectory)
            st.success(f"✅ Layer {layer.layer_set_id} trajectory generated")
        
        # Apply layer to mandrel for evolution
        try:
            layer_manager.apply_layer_to_mandrel(i)
        except Exception as e:
            st.warning(f"Could not apply layer {i+1} to mandrel: {str(e)}")
    
    progress_bar.progress(1.0)
    status_text.text("✅ All trajectories generated!")
    
    return all_trajectories
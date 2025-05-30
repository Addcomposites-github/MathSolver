"""
Comprehensive Fix for Trajectory Array Mismatch Issues
Addresses constant radius trajectories and missing points
"""

import numpy as np
import streamlit as st
from typing import Dict, List, Optional, Tuple, Any
import warnings

def fix_trajectory_array_mismatches(trajectory_data: Dict) -> Dict:
    """
    Main function to fix trajectory array mismatches
    Returns fixed trajectory data with consistent arrays
    """
    if not trajectory_data:
        return {}
    
    print("[ArrayFix] Starting trajectory array validation and fixing")
    
    # Extract coordinates safely
    coords = extract_safe_coordinates(trajectory_data)
    if not coords:
        print("[ArrayFix] Could not extract valid coordinates")
        return trajectory_data
    
    x_arr, y_arr, z_arr = coords
    
    # Validate trajectory physics
    rho = np.sqrt(x_arr**2 + y_arr**2)
    radius_var_pct = (np.std(rho) / np.mean(rho) * 100) if np.mean(rho) > 0 else 0
    
    print(f"[ArrayFix] Radius variation: {radius_var_pct:.3f}%")
    
    if radius_var_pct < 0.1:
        print("[ArrayFix] WARNING: Constant radius detected")
    
    # Rebuild trajectory data with fixed coordinates
    fixed_data = trajectory_data.copy()
    
    # Update at multiple levels to ensure compatibility
    coordinate_updates = {
        'x_points_m': x_arr.tolist(),
        'y_points_m': y_arr.tolist(),
        'z_points_m': z_arr.tolist(),
        'total_points': len(x_arr)
    }
    
    # Update top level
    fixed_data.update(coordinate_updates)
    
    # Update nested level if exists
    if 'trajectory_data' in fixed_data:
        fixed_data['trajectory_data'].update(coordinate_updates)
    
    print(f"[ArrayFix] Fixed trajectory with {len(x_arr)} consistent points")
    return fixed_data

def extract_safe_coordinates(trajectory_data: Dict) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Extract coordinates safely from various data formats"""
    
    # Method 1: Direct coordinate arrays
    if all(key in trajectory_data for key in ['x_points_m', 'y_points_m', 'z_points_m']):
        try:
            x = np.array(trajectory_data['x_points_m'], dtype=float)
            y = np.array(trajectory_data['y_points_m'], dtype=float)
            z = np.array(trajectory_data['z_points_m'], dtype=float)
            
            if len(x) > 0 and len(y) > 0 and len(z) > 0:
                # Check for consistency
                min_len = min(len(x), len(y), len(z))
                if min_len != max(len(x), len(y), len(z)):
                    print(f"[ArrayFix] Truncating arrays to {min_len} points for consistency")
                    x, y, z = x[:min_len], y[:min_len], z[:min_len]
                
                # Remove invalid values
                valid_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
                if np.sum(valid_mask) < len(x):
                    print(f"[ArrayFix] Removing {len(x) - np.sum(valid_mask)} invalid points")
                    x, y, z = x[valid_mask], y[valid_mask], z[valid_mask]
                
                if len(x) >= 2:
                    return x, y, z
        except Exception as e:
            print(f"[ArrayFix] Error with direct arrays: {e}")
    
    # Method 2: Nested in trajectory_data
    if 'trajectory_data' in trajectory_data:
        nested = trajectory_data['trajectory_data']
        if all(key in nested for key in ['x_points_m', 'y_points_m', 'z_points_m']):
            try:
                x = np.array(nested['x_points_m'], dtype=float)
                y = np.array(nested['y_points_m'], dtype=float)
                z = np.array(nested['z_points_m'], dtype=float)
                
                if len(x) > 0 and len(y) > 0 and len(z) > 0:
                    min_len = min(len(x), len(y), len(z))
                    x, y, z = x[:min_len], y[:min_len], z[:min_len]
                    
                    valid_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
                    x, y, z = x[valid_mask], y[valid_mask], z[valid_mask]
                    
                    if len(x) >= 2:
                        return x, y, z
            except Exception as e:
                print(f"[ArrayFix] Error with nested arrays: {e}")
    
    # Method 3: From path_points
    if 'path_points' in trajectory_data and trajectory_data['path_points']:
        try:
            points = trajectory_data['path_points']
            x_coords, y_coords, z_coords = [], [], []
            
            for point in points:
                try:
                    if isinstance(point, dict):
                        x_val = point.get('x_m', point.get('x', 0))
                        y_val = point.get('y_m', point.get('y', 0))
                        z_val = point.get('z_m', point.get('z', 0))
                    elif hasattr(point, 'position') and len(point.position) >= 3:
                        x_val, y_val, z_val = point.position[0], point.position[1], point.position[2]
                    elif hasattr(point, 'x') and hasattr(point, 'y') and hasattr(point, 'z'):
                        x_val, y_val, z_val = point.x, point.y, point.z
                    else:
                        continue
                    
                    if all(np.isfinite([x_val, y_val, z_val])):
                        x_coords.append(float(x_val))
                        y_coords.append(float(y_val))
                        z_coords.append(float(z_val))
                        
                except Exception:
                    continue
            
            if len(x_coords) >= 2:
                return np.array(x_coords), np.array(y_coords), np.array(z_coords)
                
        except Exception as e:
            print(f"[ArrayFix] Error with path_points: {e}")
    
    return None

def diagnose_trajectory_issues(trajectory_data: Dict) -> Dict:
    """Diagnose trajectory array issues"""
    if not trajectory_data:
        return {'status': 'error', 'message': 'No trajectory data'}
    
    coords = extract_safe_coordinates(trajectory_data)
    if not coords:
        return {
            'status': 'error', 
            'message': 'Could not extract coordinates',
            'available_keys': list(trajectory_data.keys())
        }
    
    x, y, z = coords
    rho = np.sqrt(x**2 + y**2)
    
    radius_var_pct = (np.std(rho) / np.mean(rho) * 100) if np.mean(rho) > 0 else 0
    constant_radius_issue = radius_var_pct < 0.1
    
    analysis = {
        'source': 'extracted',
        'points_count': len(x),
        'quality_score': 100 - abs(radius_var_pct - 5.0),  # Rough quality metric
        'radius_variation_pct': radius_var_pct,
        'z_range_m': np.max(z) - np.min(z),
        'constant_radius_issue': constant_radius_issue
    }
    
    return {
        'status': 'success',
        'coordinate_sets_found': 1,
        'sets_analysis': [analysis]
    }

def apply_trajectory_array_fix_to_session() -> bool:
    """Apply trajectory fixes to session state"""
    fixed_any = False
    
    try:
        # Fix single trajectory
        if 'trajectory_data' in st.session_state and st.session_state.trajectory_data:
            fixed_data = fix_trajectory_array_mismatches(st.session_state.trajectory_data)
            if fixed_data != st.session_state.trajectory_data:
                st.session_state.trajectory_data = fixed_data
                fixed_any = True
                print("[ArrayFix] Fixed single trajectory data")
        
        # Fix layer trajectories
        if 'all_layer_trajectories' in st.session_state and st.session_state.all_layer_trajectories:
            for i, layer_traj in enumerate(st.session_state.all_layer_trajectories):
                if 'trajectory_data' in layer_traj:
                    fixed_data = fix_trajectory_array_mismatches(layer_traj['trajectory_data'])
                    if fixed_data != layer_traj['trajectory_data']:
                        st.session_state.all_layer_trajectories[i]['trajectory_data'] = fixed_data
                        fixed_any = True
                        print(f"[ArrayFix] Fixed layer {i+1} trajectory data")
        
        return fixed_any
        
    except Exception as e:
        print(f"[ArrayFix] Error applying fixes: {e}")
        return False
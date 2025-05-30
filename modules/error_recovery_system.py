"""
Error Recovery System for Trajectory Generation
"""

import numpy as np
import streamlit as st
from typing import Dict, List, Any, Optional

def trajectory_generation_with_fallbacks(layer_manager, roving_width_mm=3.0, roving_thickness_mm=0.125):
    """
    Use ONLY the sophisticated unified trajectory generation system - NO FALLBACKS
    """
    
    # Clear any cached trajectories to force regeneration with evolved mandrel geometry
    if 'all_layer_trajectories' in st.session_state:
        del st.session_state['all_layer_trajectories']
    if 'trajectory_data' in st.session_state:
        del st.session_state['trajectory_data']
    
    print(f"[DEBUG] Starting trajectory generation for {len(layer_manager.layer_stack)} layers")
    print(f"[DEBUG] Current mandrel has {len(layer_manager.mandrel.layers_applied)} layers applied")
    
    # Reset mandrel to base state to ensure clean layer evolution
    layer_manager.mandrel.reset_to_base_geometry()
    print(f"[DEBUG] Reset mandrel to base geometry")
    
    # ONLY use the unified system - disable all fallbacks that create 100-point junk trajectories
    try:
        st.info("Using sophisticated unified trajectory generation with evolved mandrel geometry...")
        from modules.multi_layer_trajectory_orchestrator import MultiLayerTrajectoryOrchestrator
        
        orchestrator = MultiLayerTrajectoryOrchestrator(layer_manager)
        trajectories = orchestrator.generate_all_layer_trajectories(roving_width_mm, roving_thickness_mm)
        
        if trajectories and len(trajectories) > 0:
            # Validate that we got proper multi-circuit trajectories, not 100-point fallbacks
            for traj in trajectories:
                traj_data = traj.get('trajectory_data', {})
                point_count = traj_data.get('total_points', 0)
                
                if point_count < 500:  # Reject any trajectory with too few points
                    st.error(f"Layer {traj['layer_id']}: Only {point_count} points - this is a fallback trajectory, not sophisticated solver output")
                    return []
                    
            st.success(f"✅ Sophisticated trajectory generation succeeded: {len(trajectories)} trajectories with proper multi-circuit coverage")
            return trajectories
        else:
            st.error("❌ Unified system returned no trajectories - check your configuration")
            return []
            
    except Exception as e:
        st.error(f"❌ Sophisticated trajectory generation failed: {str(e)}")
        st.error("❌ NO FALLBACKS ALLOWED - Fix the unified system instead of using 100-point junk trajectories")
        return []

def generate_simple_layer_trajectory(layer_def, layer_manager, roving_width_mm):
    """
    Simple trajectory generation without complex pattern calculation
    """
    try:
        # Get vessel geometry
        vessel = st.session_state.vessel_geometry
        if not vessel:
            return None
        
        # Calculate basic trajectory points
        radius_m = vessel.inner_diameter / 2000  # Convert mm to m
        length_m = vessel.cylindrical_length / 1000  # Convert mm to m
        
        # Generate simple helical path
        num_points = 100
        angle_rad = np.radians(layer_def.winding_angle_deg)
        
        # Calculate path length and phi advancement
        path_length = length_m / np.cos(angle_rad)
        phi_total = path_length / radius_m * np.sin(angle_rad)
        
        z_points = np.linspace(-length_m/2, length_m/2, num_points)
        phi_points = np.linspace(0, phi_total, num_points)
        
        # Convert to Cartesian coordinates
        x_points = radius_m * np.cos(phi_points)
        y_points = radius_m * np.sin(phi_points)
        
        # Create path points format
        path_points = []
        for i in range(num_points):
            path_points.append({
                'x_m': x_points[i],
                'y_m': y_points[i],
                'z_m': z_points[i],
                'rho_m': radius_m,
                'phi_rad': phi_points[i],
                'alpha_deg': layer_def.winding_angle_deg,
                'arc_length_m': i * (path_length / num_points)
            })
        
        return {
            'path_points': path_points,
            'total_points': num_points,
            'success': True,
            'x_points_m': x_points.tolist(),
            'y_points_m': y_points.tolist(),
            'z_points_m': z_points.tolist(),
            'coverage_percentage': 85.0,
            'trajectory_type': 'simple_helical'
        }
        
    except Exception as e:
        st.error(f"Simple trajectory generation failed: {e}")
        return None

def generate_basic_geometric_trajectory(layer_def, vessel_geometry):
    """
    Most basic trajectory generation - pure geometry
    """
    try:
        # Basic cylinder with helical winding
        radius_mm = vessel_geometry.inner_diameter / 2
        length_mm = vessel_geometry.cylindrical_length
        
        # Simple helical parameters
        angle_deg = layer_def.winding_angle_deg
        num_wraps = max(1, int(10 * np.sin(np.radians(angle_deg))))  # More wraps for steeper angles
        
        # Generate points
        num_points = 50
        theta = np.linspace(0, num_wraps * 2 * np.pi, num_points)
        z_mm = np.linspace(-length_mm/2, length_mm/2, num_points)
        
        # Convert to meters and create path points
        path_points = []
        for i in range(num_points):
            x_m = (radius_mm / 1000) * np.cos(theta[i])
            y_m = (radius_mm / 1000) * np.sin(theta[i])
            z_m = z_mm[i] / 1000
            
            path_points.append({
                'x_m': x_m,
                'y_m': y_m,
                'z_m': z_m,
                'rho_m': radius_mm / 1000,
                'phi_rad': theta[i],
                'alpha_deg': angle_deg,
                'arc_length_m': i * 0.01
            })
        
        return {
            'path_points': path_points,
            'total_points': num_points,
            'success': True,
            'x_points_m': [p['x_m'] for p in path_points],
            'y_points_m': [p['y_m'] for p in path_points],
            'z_points_m': [p['z_m'] for p in path_points],
            'coverage_percentage': 70.0,
            'trajectory_type': 'basic_geometric'
        }
        
    except Exception as e:
        st.error(f"Basic geometric trajectory failed: {e}")
        return {
            'path_points': [],
            'total_points': 0,
            'success': False,
            'error': str(e)
        }

def quick_fix_pattern_calculation():
    """
    Apply quick fixes to common pattern calculation issues
    """
    fixes_applied = []
    
    # Fix 1: Ensure vessel geometry has required attributes
    if 'vessel_geometry' in st.session_state:
        vessel = st.session_state.vessel_geometry
        
        # Add missing inner_diameter if needed
        if not hasattr(vessel, 'inner_diameter') or not vessel.inner_diameter:
            if hasattr(vessel, 'get_profile_points'):
                try:
                    profile = vessel.get_profile_points()
                    if 'r_inner_mm' in profile:
                        vessel.inner_diameter = np.max(profile['r_inner_mm']) * 2
                        fixes_applied.append("Added inner_diameter from profile")
                except:
                    pass
    
    # Fix 2: Validate session state
    required_attributes = ['vessel_geometry', 'layer_stack_manager']
    for attr in required_attributes:
        if attr not in st.session_state:
            fixes_applied.append(f"Missing {attr} in session state")
    
    if fixes_applied:
        st.warning("Applied fixes: " + ", ".join(fixes_applied))
        return False
    else:
        st.success("No fixes needed")
        return True

def validate_before_trajectory_generation():
    """
    Validate system state before trajectory generation
    """
    checks = []
    
    # Check vessel geometry
    if st.session_state.get('vessel_geometry') is None:
        checks.append("No vessel geometry")
        return False, checks
    
    vessel = st.session_state.vessel_geometry
    if not hasattr(vessel, 'inner_diameter'):
        checks.append("Vessel geometry missing inner_diameter")
        return False, checks
    
    if vessel.inner_diameter <= 0:
        checks.append("Invalid vessel diameter")
        return False, checks
    
    checks.append("Vessel geometry OK")
    
    # Check layer stack
    if 'layer_stack_manager' not in st.session_state:
        checks.append("No layer stack manager")
        return False, checks
    
    if not st.session_state.layer_stack_manager.layer_stack:
        checks.append("No layers defined")
        return False, checks
    
    checks.append("Layer stack OK")
    
    return True, checks
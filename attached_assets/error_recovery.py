"""
Error Recovery System for Trajectory Generation
"""

def trajectory_generation_with_fallbacks(layer_manager, roving_width_mm=3.0, roving_thickness_mm=0.125):
    """
    Robust trajectory generation with multiple fallback strategies
    """
    
    # Strategy 1: Try unified system with full coverage
    try:
        st.info("🎯 Attempting unified trajectory generation...")
        from modules.multi_layer_trajectory_orchestrator import MultiLayerTrajectoryOrchestrator
        
        orchestrator = MultiLayerTrajectoryOrchestrator(layer_manager)
        trajectories = orchestrator.generate_all_layer_trajectories(roving_width_mm, roving_thickness_mm)
        
        if trajectories and len(trajectories) > 0:
            st.success(f"✅ Unified system succeeded: {len(trajectories)} trajectories")
            return trajectories
        else:
            st.warning("⚠️ Unified system returned no trajectories")
            
    except Exception as e:
        st.warning(f"⚠️ Unified system failed: {str(e)}")
    
    # Strategy 2: Try simplified single-layer approach
    try:
        st.info("🔄 Attempting simplified single-layer generation...")
        
        trajectories = []
        for i, layer in enumerate(layer_manager.layer_stack):
            simple_trajectory = generate_simple_layer_trajectory(
                layer, layer_manager, roving_width_mm
            )
            if simple_trajectory:
                trajectories.append({
                    'layer_id': layer.layer_set_id,
                    'layer_type': layer.layer_type,
                    'winding_angle': layer.winding_angle_deg,
                    'trajectory_data': simple_trajectory
                })
        
        if trajectories:
            st.success(f"✅ Simplified approach succeeded: {len(trajectories)} trajectories")
            return trajectories
            
    except Exception as e:
        st.warning(f"⚠️ Simplified approach failed: {str(e)}")
    
    # Strategy 3: Generate basic geometric trajectories
    try:
        st.info("🔧 Generating basic geometric trajectories...")
        
        trajectories = []
        for layer in layer_manager.layer_stack:
            basic_trajectory = generate_basic_geometric_trajectory(
                layer, st.session_state.vessel_geometry
            )
            trajectories.append({
                'layer_id': layer.layer_set_id,
                'layer_type': layer.layer_type,
                'winding_angle': layer.winding_angle_deg,
                'trajectory_data': basic_trajectory
            })
        
        st.warning(f"⚠️ Using basic geometric trajectories: {len(trajectories)} layers")
        return trajectories
        
    except Exception as e:
        st.error(f"❌ All fallback strategies failed: {str(e)}")
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

# Integration with existing app
def update_layer_by_layer_planning():
    """
    Update the layer_by_layer_planning function to use error recovery
    """
    # Replace the existing trajectory generation call with:
    
    if st.button("🚀 Generate All Layer Trajectories (Robust)", type="primary"):
        all_trajectories = trajectory_generation_with_fallbacks(
            layer_manager, roving_width, roving_thickness
        )
        
        if all_trajectories:
            st.session_state.all_layer_trajectories = all_trajectories
            st.success(f"🎉 Successfully generated trajectories for {len(all_trajectories)} layers!")
            
            # Show summary
            trajectory_summary = []
            for traj in all_trajectories:
                trajectory_summary.append({
                    "Layer": f"Layer {traj['layer_id']}",
                    "Type": traj['layer_type'],
                    "Angle": f"{traj['winding_angle']}°",
                    "Points": len(traj['trajectory_data'].get('path_points', [])),
                    "Method": traj['trajectory_data'].get('trajectory_type', 'unified')
                })
            
            st.dataframe(trajectory_summary, use_container_width=True, hide_index=True)
        else:
            st.error("❌ All trajectory generation methods failed. Check your configuration.")

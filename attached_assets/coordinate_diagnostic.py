def diagnose_coordinate_systems(vessel_geometry, trajectory_data):
    """
    Diagnostic tool to understand coordinate system mismatches
    Add this to your visualization_page() function before creating the plot
    """
    st.markdown("### 🔍 Coordinate System Diagnostic")
    
    # Get vessel coordinates
    try:
        profile = vessel_geometry.get_profile_points()
        vessel_z_mm = np.array(profile['z_mm'])
        vessel_r_mm = np.array(profile['r_inner_mm'])
        
        vessel_z_m = vessel_z_mm / 1000.0
        vessel_z_min = np.min(vessel_z_m)
        vessel_z_max = np.max(vessel_z_m)
        vessel_z_center = (vessel_z_min + vessel_z_max) / 2
        
        st.write(f"**Vessel Geometry:**")
        st.write(f"  - Z range: {vessel_z_min:.3f}m to {vessel_z_max:.3f}m")
        st.write(f"  - Z center: {vessel_z_center:.3f}m")
        st.write(f"  - R range: {np.min(vessel_r_mm/1000):.3f}m to {np.max(vessel_r_mm/1000):.3f}m")
        
    except Exception as e:
        st.error(f"Error reading vessel geometry: {e}")
        return
    
    # Get trajectory coordinates from coverage_data format
    if 'circuits' in trajectory_data and len(trajectory_data['circuits']) > 0:
        circuit = trajectory_data['circuits'][0]
        if len(circuit) > 0:
            try:
                # Extract Z coordinates from trajectory
                traj_z_coords = []
                traj_x_coords = []
                traj_y_coords = []
                
                for point in circuit:
                    if isinstance(point, dict):
                        if 'z_m' in point:
                            traj_z_coords.append(point['z_m'])
                        if 'x_m' in point:
                            traj_x_coords.append(point['x_m'])
                        if 'y_m' in point:
                            traj_y_coords.append(point['y_m'])
                
                if traj_z_coords:
                    traj_z_min = np.min(traj_z_coords)
                    traj_z_max = np.max(traj_z_coords)
                    traj_z_center = (traj_z_min + traj_z_max) / 2
                    
                    st.write(f"**Trajectory Data:**")
                    st.write(f"  - Z range: {traj_z_min:.3f}m to {traj_z_max:.3f}m")
                    st.write(f"  - Z center: {traj_z_center:.3f}m")
                    
                    # Calculate coordinate differences
                    z_center_diff = abs(vessel_z_center - traj_z_center)
                    z_range_overlap = min(vessel_z_max, traj_z_max) - max(vessel_z_min, traj_z_min)
                    
                    st.write(f"**Alignment Analysis:**")
                    st.write(f"  - Center difference: {z_center_diff:.3f}m")
                    st.write(f"  - Z range overlap: {z_range_overlap:.3f}m")
                    
                    if z_center_diff < 0.01:
                        st.success("✅ Coordinates appear well-aligned")
                    elif z_range_overlap > 0:
                        st.warning("⚠️ Partial overlap - possible centering issue")
                    else:
                        st.error("❌ No overlap - major coordinate mismatch")
                    
                    # Calculate radial coordinates for additional check
                    if traj_x_coords and traj_y_coords:
                        traj_r_coords = [np.sqrt(x**2 + y**2) for x, y in zip(traj_x_coords, traj_y_coords)]
                        traj_r_min = np.min(traj_r_coords)
                        traj_r_max = np.max(traj_r_coords)
                        
                        vessel_r_m = vessel_r_mm / 1000.0
                        vessel_r_min = np.min(vessel_r_m)
                        vessel_r_max = np.max(vessel_r_m)
                        
                        st.write(f"**Radial Check:**")
                        st.write(f"  - Vessel R: {vessel_r_min:.3f}m to {vessel_r_max:.3f}m")
                        st.write(f"  - Trajectory R: {traj_r_min:.3f}m to {traj_r_max:.3f}m")
                        
                        r_diff = abs((vessel_r_max + vessel_r_min)/2 - (traj_r_max + traj_r_min)/2)
                        if r_diff < 0.01:
                            st.success("✅ Radial coordinates match vessel")
                        else:
                            st.warning(f"⚠️ Radial mismatch: {r_diff:.3f}m difference")
                
                else:
                    st.error("No Z coordinates found in trajectory data")
                    
            except Exception as e:
                st.error(f"Error analyzing trajectory coordinates: {e}")
    
    else:
        st.error("No trajectory circuit data found")

# Usage: Add this call in your visualization_page() function before creating the plot:
# diagnose_coordinate_systems(st.session_state.vessel_geometry, coverage_data)
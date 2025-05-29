# INSERT THIS CODE in your visualization_page() function
# RIGHT AFTER: converted_data = converter.convert_unified_trajectory_to_visualization_format(raw_trajectory_data)
# AND BEFORE: coverage_data = {...}

# ===== COORDINATE ALIGNMENT FIX =====
st.markdown("### 🎯 Coordinate Alignment Fix")

# Get vessel coordinate system
vessel_profile = st.session_state.vessel_geometry.get_profile_points()
vessel_z_m = np.array(vessel_profile['z_mm']) / 1000.0
vessel_z_min = np.min(vessel_z_m)
vessel_z_max = np.max(vessel_z_m)
vessel_z_center = (vessel_z_min + vessel_z_max) / 2

# Get trajectory coordinate system
traj_z = np.array(converted_data['z_points_m'])
traj_z_min = np.min(traj_z)
traj_z_max = np.max(traj_z)
traj_z_center = (traj_z_min + traj_z_max) / 2

# Calculate required offset
z_offset = vessel_z_center - traj_z_center

st.write(f"**Coordinate Analysis:**")
st.write(f"🔹 Vessel Z center: {vessel_z_center:.3f}m (range: {vessel_z_min:.3f} to {vessel_z_max:.3f})")
st.write(f"🔹 Trajectory Z center: {traj_z_center:.3f}m (range: {traj_z_min:.3f} to {traj_z_max:.3f})")
st.write(f"🎯 **Required Z offset: {z_offset:.3f}m**")

if abs(z_offset) > 0.01:  # More than 1cm misalignment
    st.warning(f"⚠️ **Applying coordinate alignment: {z_offset:.3f}m Z-offset**")
    
    # Apply correction to all trajectory coordinates
    corrected_z = traj_z + z_offset
    
    # Update the converted_data arrays
    converted_data['z_points_m'] = corrected_z.tolist()
    
    # Update path_points
    for i, point in enumerate(converted_data['path_points']):
        point['z_m'] = corrected_z[i]
    
    # Verify the fix
    new_z_min = np.min(corrected_z)
    new_z_max = np.max(corrected_z)
    st.success(f"✅ **Coordinates aligned!** New trajectory Z: {new_z_min:.3f}m to {new_z_max:.3f}m")
    
    # Show before/after comparison
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Before Alignment", f"{traj_z_min:.3f} to {traj_z_max:.3f}m")
    with col2:
        st.metric("After Alignment", f"{new_z_min:.3f} to {new_z_max:.3f}m")
        
else:
    st.success("✅ Coordinates already properly aligned")

# ===== END COORDINATE ALIGNMENT FIX =====

# Now continue with your existing visualization code...
# coverage_data = { ... }
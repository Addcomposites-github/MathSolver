# 5-MINUTE TRAJECTORY CHECKLIST
# Add this as a button to your trajectory planning page

def five_minute_trajectory_check():
    """Quick 5-minute check to identify trajectory generation issues"""
    st.markdown("## 🚀 5-Minute Trajectory Checklist")
    
    checks_passed = 0
    total_checks = 6
    
    # Check 1: Session State
    st.markdown("### ✅ Check 1: Session State")
    if hasattr(st.session_state, 'vessel_geometry') and st.session_state.vessel_geometry:
        st.success("✅ Vessel geometry exists")
        checks_passed += 1
    else:
        st.error("❌ No vessel geometry")
        return
    
    if hasattr(st.session_state, 'layer_stack_manager') and st.session_state.layer_stack_manager:
        manager = st.session_state.layer_stack_manager
        summary = manager.get_layer_stack_summary()
        if summary['total_layers'] > 0:
            st.success(f"✅ Layer stack has {summary['total_layers']} layers")
            checks_passed += 1
        else:
            st.error("❌ No layers defined")
            return
    else:
        st.error("❌ No layer stack manager")
        return
    
    # Check 2: Layer Application
    st.markdown("### ✅ Check 2: Layer Application Status")
    if summary['layers_applied_to_mandrel'] > 0:
        st.success(f"✅ {summary['layers_applied_to_mandrel']} layers applied to mandrel")
        checks_passed += 1
    else:
        st.error("❌ NO LAYERS APPLIED TO MANDREL - This is likely your issue!")
        st.info("💡 Go to Layer Stack Definition and click 'Apply Layer to Mandrel'")
        return
    
    # Check 3: Trajectory Generation Result
    st.markdown("### ✅ Check 3: Trajectory Generation Results")
    if hasattr(st.session_state, 'all_layer_trajectories') and st.session_state.all_layer_trajectories:
        trajectories = st.session_state.all_layer_trajectories
        st.success(f"✅ {len(trajectories)} trajectories generated")
        checks_passed += 1
        
        # Check trajectory content
        for i, traj in enumerate(trajectories):
            traj_data = traj.get('trajectory_data', {})
            if 'points' in traj_data and len(traj_data['points']) > 0:
                st.success(f"  ✅ Trajectory {i+1}: {len(traj_data['points'])} points")
            else:
                st.error(f"  ❌ Trajectory {i+1}: NO POINTS")
                return
    else:
        st.error("❌ No trajectories generated")
        return
    
    # Check 4: Point Data Quality
    st.markdown("### ✅ Check 4: Point Data Quality")
    first_traj = st.session_state.all_layer_trajectories[0]
    points = first_traj['trajectory_data']['points']
    
    if points:
        sample_point = points[0]
        if hasattr(sample_point, 'rho') and hasattr(sample_point, 'z'):
            st.success(f"✅ Points have correct format: rho={sample_point.rho:.6f}, z={sample_point.z:.6f}")
            
            # Check coordinate ranges
            all_rho = [p.rho for p in points]
            all_z = [p.z for p in points]
            
            rho_range = max(all_rho) - min(all_rho)
            z_range = max(all_z) - min(all_z)
            
            if rho_range > 1e-6 and z_range > 1e-6:
                st.success(f"✅ Coordinate ranges reasonable: rho={rho_range:.6f}m, z={z_range:.6f}m")
                checks_passed += 1
            else:
                st.error(f"❌ Coordinates collapsed: rho_range={rho_range:.9f}, z_range={z_range:.9f}")
                st.error("💥 THIS IS YOUR ISSUE - trajectory generation is collapsing coordinates")
                return
        else:
            st.error(f"❌ Points missing coordinates: {type(sample_point)}")
            return
    
    # Check 5: Vessel Geometry Scale
    st.markdown("### ✅ Check 5: Vessel Geometry Scale")
    vessel = st.session_state.vessel_geometry
    profile = vessel.get_profile_points()
    
    vessel_diameter = vessel.inner_diameter  # mm
    vessel_length = max(profile['z_mm']) - min(profile['z_mm'])  # mm
    
    st.write(f"Vessel dimensions: {vessel_diameter}mm diameter × {vessel_length}mm length")
    
    if 50 <= vessel_diameter <= 2000 and 50 <= vessel_length <= 5000:
        st.success("✅ Vessel dimensions reasonable")
        checks_passed += 1
    else:
        st.warning(f"⚠️ Unusual vessel dimensions - may cause physics issues")
    
    # Check 6: Physics Parameters
    st.markdown("### ✅ Check 6: Layer Physics Parameters")
    physics_ok = True
    for layer in manager.layer_stack:
        angle = layer.winding_angle_deg
        if angle < 5 or angle > 89:
            st.error(f"❌ Layer {layer.layer_set_id}: Extreme winding angle {angle}°")
            physics_ok = False
    
    if physics_ok:
        st.success("✅ All layer winding angles reasonable")
        checks_passed += 1
    
    # Final Score
    st.markdown("---")
    st.markdown("### 🎯 Final Score")
    
    score_pct = (checks_passed / total_checks) * 100
    
    if score_pct == 100:
        st.success(f"🎉 **PERFECT SCORE**: {checks_passed}/{total_checks} checks passed!")
        st.info("Your trajectory generation should be working. If you're still seeing issues, it's likely in the visualization/conversion step.")
    elif score_pct >= 80:
        st.warning(f"⚠️ **MOSTLY GOOD**: {checks_passed}/{total_checks} checks passed")
        st.info("Minor issues found - check the failed items above")
    else:
        st.error(f"❌ **CRITICAL ISSUES**: Only {checks_passed}/{total_checks} checks passed")
        st.info("Major problems found - fix the failed checks above")
    
    return score_pct >= 80

# Add this button to your trajectory planning page
if st.button("🚀 5-Minute Trajectory Check"):
    five_minute_trajectory_check()

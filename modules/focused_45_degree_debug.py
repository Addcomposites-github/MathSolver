"""
Focused 45° Geodesic Debugging
Specific tests for your exact case: 45° geodesic with full coverage
"""

import streamlit as st
import numpy as np
import math

def debug_45_degree_geodesic_case():
    """
    Focused debugging for exactly what you want:
    45° geodesic winding with full coverage on your specific vessel
    """
    st.markdown("# 🎯 45° Geodesic Case - Focused Debug")
    st.markdown("**Testing exactly what you want: 45° geodesic winding with full coverage**")
    
    if not hasattr(st.session_state, 'vessel_geometry'):
        st.error("❌ No vessel geometry - can't debug")
        return
    
    vessel = st.session_state.vessel_geometry
    
    # Step 1: Validate your vessel geometry
    st.markdown("## Step 1: 🏗️ Your Vessel Geometry")
    
    profile = vessel.get_profile_points()
    vessel_diameter_mm = vessel.inner_diameter
    vessel_length_mm = max(profile['z_mm']) - min(profile['z_mm'])
    vessel_radius_m = vessel_diameter_mm / 2000  # Convert to meters
    
    st.write(f"**Your vessel specs:**")
    st.write(f"  Diameter: {vessel_diameter_mm}mm ({vessel_radius_m:.3f}m radius)")
    st.write(f"  Length: {vessel_length_mm}mm ({vessel_length_mm/1000:.3f}m)")
    st.write(f"  Dome type: {vessel.dome_type}")
    
    # Step 2: Calculate what 45° geodesic SHOULD look like
    st.markdown("## Step 2: 🧮 What 45° Geodesic SHOULD Be")
    
    # Clairaut constant for 45°
    clairaut_45 = vessel_radius_m * math.sin(math.radians(45.0))
    
    st.write(f"**Expected 45° geodesic properties:**")
    st.write(f"  Clairaut constant C = R·sin(45°) = {clairaut_45:.6f}m")
    st.write(f"  Should wrap around vessel {vessel_length_mm/1000:.3f}m length")
    st.write(f"  Should have radius varying from ~{clairaut_45:.3f}m to {vessel_radius_m:.3f}m")
    
    # Calculate expected pattern
    circumference = 2 * math.pi * vessel_radius_m
    roving_width = 0.003  # 3mm
    expected_circuits = circumference / roving_width
    
    st.write(f"  Expected full coverage: ~{expected_circuits:.0f} circuits")
    st.write(f"  Each circuit should advance ~{360/expected_circuits:.1f}° in φ")
    
    # Step 3: Test current trajectory generation
    st.markdown("## Step 3: 🧪 Test Current Generation")
    
    if st.button("🚀 Generate Test 45° Trajectory"):
        try:
            from modules.unified_trajectory_planner import UnifiedTrajectoryPlanner
            
            # Create planner with your exact vessel
            planner = UnifiedTrajectoryPlanner(
                vessel_geometry=vessel,
                roving_width_m=0.003,  # 3mm
                payout_length_m=0.5,   # 500mm
                default_friction_coeff=0.1
            )
            
            st.write("✅ Planner created successfully")
            
            # Generate EXACTLY what you want
            result = planner.generate_trajectory(
                pattern_type='geodesic',
                coverage_mode='full_coverage',  # This should give you multiple circuits
                physics_model='clairaut',
                continuity_level=1,
                num_layers_desired=1,
                target_params={'winding_angle_deg': 45.0},
                options={'num_points': 200}
            )
            
            if result and result.points:
                points = result.points
                st.success(f"✅ Generated {len(points)} points")
                
                # Analyze what we actually got
                analyze_generated_trajectory(points, vessel_radius_m, clairaut_45, expected_circuits)
                
            else:
                st.error("❌ NO POINTS GENERATED - This is your problem!")
                st.error("🚨 The trajectory generation is completely failing")
                
                # Try to diagnose why
                st.markdown("### 🔍 Diagnosis: Why No Points?")
                
                # Check planner status
                status = planner.get_system_status()
                st.write("**Planner status:**")
                for component, state in status.items():
                    st.write(f"  {component}: {state}")
                
                # Try minimal generation
                st.markdown("**Trying minimal generation:**")
                minimal_result = planner.generate_trajectory(
                    pattern_type='geodesic',
                    coverage_mode='single_pass',
                    physics_model='clairaut',
                    continuity_level=0,
                    num_layers_desired=1,
                    target_params={'winding_angle_deg': 45.0},
                    options={'num_points': 10}
                )
                
                if minimal_result and minimal_result.points:
                    st.warning("⚠️ Minimal generation works - issue is with full coverage")
                else:
                    st.error("❌ Even minimal generation fails - core physics broken")
                
        except ImportError as e:
            st.error(f"❌ Cannot import UnifiedTrajectoryPlanner: {e}")
            st.info("💡 This might be your issue - module not available")
        except Exception as e:
            st.error(f"❌ Trajectory generation failed: {e}")
            st.info("💡 This error is preventing trajectory generation")
    
    # Step 4: Check your current trajectories
    st.markdown("## Step 4: 🔍 Analyze Your Current Trajectories")
    
    if hasattr(st.session_state, 'all_layer_trajectories') and st.session_state.all_layer_trajectories:
        trajectories = st.session_state.all_layer_trajectories
        st.write(f"**Found {len(trajectories)} existing trajectories**")
        
        for i, traj in enumerate(trajectories):
            with st.expander(f"Trajectory {i+1} Analysis"):
                analyze_existing_trajectory(traj, vessel_radius_m, clairaut_45, expected_circuits)
    else:
        st.warning("⚠️ No existing trajectories to analyze")
    
    # Step 5: Direct physics test
    st.markdown("## Step 5: 🔬 Direct Physics Test")
    
    if st.button("🧮 Test Physics Directly"):
        test_direct_physics(vessel_radius_m, clairaut_45)

def analyze_generated_trajectory(points, vessel_radius_m, expected_clairaut, expected_circuits):
    """Analyze trajectory that was just generated"""
    st.markdown("### 📊 Generated Trajectory Analysis")
    
    # Extract coordinates - handle different point formats
    rho_coords = []
    z_coords = []
    phi_coords = []
    alpha_coords = []
    
    for p in points:
        if hasattr(p, 'rho'):
            rho_coords.append(p.rho)
        if hasattr(p, 'z'):
            z_coords.append(p.z)
        if hasattr(p, 'phi'):
            phi_coords.append(p.phi)
        if hasattr(p, 'alpha_deg'):
            alpha_coords.append(p.alpha_deg)
    
    if not rho_coords or not z_coords:
        st.error("❌ Points missing coordinate data")
        return
    
    # Basic range analysis
    rho_range = max(rho_coords) - min(rho_coords)
    z_range = max(z_coords) - min(z_coords)
    phi_range = max(phi_coords) - min(phi_coords) if phi_coords else 0
    
    st.write(f"**Coordinate Analysis:**")
    st.write(f"  Radius range: {min(rho_coords):.6f} to {max(rho_coords):.6f}m (span: {rho_range:.6f}m)")
    st.write(f"  Z range: {min(z_coords):.6f} to {max(z_coords):.6f}m (span: {z_range:.6f}m)")
    st.write(f"  φ range: {math.degrees(min(phi_coords)):.1f}° to {math.degrees(max(phi_coords)):.1f}° (span: {math.degrees(phi_range):.1f}°)")
    
    # Check if ranges make sense
    issues = []
    
    if rho_range < 1e-6:
        issues.append("❌ NO RADIAL VARIATION - trajectory collapsed!")
    
    if z_range < 1e-6:
        issues.append("❌ NO AXIAL VARIATION - trajectory collapsed!")
    
    if max(rho_coords) < vessel_radius_m * 0.1:
        issues.append(f"❌ TRAJECTORY TOO SMALL - max radius {max(rho_coords):.6f}m vs vessel {vessel_radius_m:.3f}m")
    
    if min(rho_coords) > vessel_radius_m * 2:
        issues.append(f"❌ TRAJECTORY TOO LARGE - min radius {min(rho_coords):.6f}m vs vessel {vessel_radius_m:.3f}m")
    
    # Check Clairaut theorem
    if alpha_coords:
        clairaut_values = [rho * math.sin(math.radians(alpha)) for rho, alpha in zip(rho_coords, alpha_coords)]
        clairaut_mean = np.mean(clairaut_values)
        clairaut_std = np.std(clairaut_values)
        
        st.write(f"**Clairaut Theorem Check:**")
        st.write(f"  Expected C: {expected_clairaut:.6f}m")
        st.write(f"  Actual C: {clairaut_mean:.6f}m ± {clairaut_std:.6f}m")
        
        if abs(clairaut_mean - expected_clairaut) > 0.001:
            issues.append(f"❌ CLAIRAUT VIOLATION - wrong physics! Expected {expected_clairaut:.6f}, got {clairaut_mean:.6f}")
    
    # Check coverage
    if phi_coords:
        full_rotations = phi_range / (2 * math.pi)
        st.write(f"**Coverage Analysis:**")
        st.write(f"  Full rotations: {full_rotations:.2f}")
        st.write(f"  Expected circuits: ~{expected_circuits:.0f}")
        
        if full_rotations < 1:
            issues.append(f"❌ INSUFFICIENT COVERAGE - only {full_rotations:.2f} rotations")
    
    # Report issues
    if issues:
        st.error("🚨 **TRAJECTORY IS BS!** Issues found:")
        for issue in issues:
            st.write(f"  {issue}")
    else:
        st.success("✅ **TRAJECTORY LOOKS GOOD!** Passes all checks")

def analyze_existing_trajectory(traj, vessel_radius_m, expected_clairaut, expected_circuits):
    """Analyze existing trajectory from your session state"""
    
    traj_data = traj.get('trajectory_data', {})
    st.write(f"**Trajectory data keys:** {list(traj_data.keys())}")
    
    if 'points' not in traj_data or not traj_data['points']:
        st.error("❌ No points in this trajectory")
        return
    
    points = traj_data['points']
    st.write(f"**Point count:** {len(points)}")
    
    if len(points) == 0:
        st.error("❌ Empty trajectory")
        return
    
    # Check point format
    first_point = points[0]
    st.write(f"**Point type:** {type(first_point)}")
    
    if hasattr(first_point, '__dict__'):
        attrs = list(first_point.__dict__.keys())
        st.write(f"**Point attributes:** {attrs}")
        
        if hasattr(first_point, 'rho') and hasattr(first_point, 'z'):
            # Good - has proper coordinates
            analyze_generated_trajectory(points, vessel_radius_m, expected_clairaut, expected_circuits)
        else:
            st.error("❌ Points missing rho/z coordinates - wrong format")
    else:
        st.error("❌ Points are not objects - wrong format")

def test_direct_physics(vessel_radius_m, expected_clairaut):
    """Test physics calculations directly"""
    st.markdown("### 🧮 Direct Physics Calculations")
    
    st.write("**Testing basic physics formulas:**")
    
    # Test 1: Clairaut constant
    test_angle = 45.0
    calc_clairaut = vessel_radius_m * math.sin(math.radians(test_angle))
    
    st.write(f"  Clairaut constant for 45°: {calc_clairaut:.6f}m")
    
    if abs(calc_clairaut - expected_clairaut) < 1e-10:
        st.success("✅ Clairaut calculation correct")
    else:
        st.error("❌ Clairaut calculation wrong!")
    
    # Test 2: Simple geodesic point
    st.write("**Testing simple geodesic point calculation:**")
    
    # For a point on the equator at 45°
    equator_rho = vessel_radius_m
    equator_alpha = 45.0
    
    # This should give us the Clairaut constant
    test_clairaut = equator_rho * math.sin(math.radians(equator_alpha))
    
    st.write(f"  Equatorial point: rho={equator_rho:.3f}m, α=45°")
    st.write(f"  Gives C = {test_clairaut:.6f}m")
    
    # Test 3: Geodesic at different radius
    test_rho = vessel_radius_m * 0.8  # 80% of max radius
    required_alpha = math.degrees(math.asin(expected_clairaut / test_rho))
    
    st.write(f"  At radius {test_rho:.3f}m, geodesic needs α = {required_alpha:.1f}°")
    
    if 10 <= required_alpha <= 80:
        st.success("✅ Physics calculations reasonable")
    else:
        st.error("❌ Physics calculations produce unreasonable angles!")

def test_45_degree_case_now():
    """Quick test of your exact case right now"""
    st.markdown("## ⚡ Quick 45° Test")
    
    if not hasattr(st.session_state, 'vessel_geometry'):
        st.error("❌ Need vessel geometry first")
        return
    
    try:
        from modules.unified_trajectory_planner import UnifiedTrajectoryPlanner
        
        vessel = st.session_state.vessel_geometry
        
        planner = UnifiedTrajectoryPlanner(
            vessel_geometry=vessel,
            roving_width_m=0.003,
            payout_length_m=0.5,
            default_friction_coeff=0.1
        )
        
        # Minimal test
        result = planner.generate_trajectory(
            pattern_type='geodesic',
            coverage_mode='single_pass',
            physics_model='clairaut',
            continuity_level=0,
            num_layers_desired=1,
            target_params={'winding_angle_deg': 45.0},
            options={'num_points': 20}
        )
        
        if result and result.points:
            st.success(f"✅ Basic 45° generation works: {len(result.points)} points")
            
            # Quick analysis
            point = result.points[0]
            if hasattr(point, 'rho') and hasattr(point, 'z'):
                st.write(f"Sample point: rho={point.rho:.6f}m, z={point.z:.6f}m")
            else:
                st.error("❌ Points missing coordinates")
        else:
            st.error("❌ Basic 45° generation FAILED")
            
    except Exception as e:
        st.error(f"❌ Quick test failed: {e}")
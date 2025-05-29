"""
Debug utilities for trajectory generation issues
"""

def debug_trajectory_generation_failure(unified_planner, **trajectory_params):
    """
    Comprehensive debugging for trajectory generation failures
    """
    print("=== TRAJECTORY GENERATION DEBUG ===")
    
    # 1. Check vessel geometry
    print("1. Vessel Geometry Check:")
    vessel = unified_planner.vessel_geometry
    if vessel is None:
        print("   ❌ Vessel geometry is None!")
        return False
    
    print(f"   ✅ Vessel type: {type(vessel).__name__}")
    
    # Check vessel attributes
    attrs_to_check = ['inner_diameter', 'cylindrical_length', 'dome_type']
    for attr in attrs_to_check:
        if hasattr(vessel, attr):
            value = getattr(vessel, attr)
            print(f"   ✅ {attr}: {value}")
        else:
            print(f"   ⚠️ Missing {attr}")
    
    # Check profile points
    if hasattr(vessel, 'get_profile_points'):
        try:
            profile = vessel.get_profile_points()
            print(f"   ✅ Profile points available: {list(profile.keys())}")
            if 'r_inner_mm' in profile:
                radius_range = f"{np.min(profile['r_inner_mm']):.1f} - {np.max(profile['r_inner_mm']):.1f}mm"
                print(f"   ✅ Radius range: {radius_range}")
        except Exception as e:
            print(f"   ❌ Profile points error: {e}")
    
    # 2. Check planner parameters
    print("\n2. Planner Parameters:")
    print(f"   Roving width: {unified_planner.roving_width_m}m")
    print(f"   Payout length: {unified_planner.payout_length_m}m")
    print(f"   Default friction: {unified_planner.default_friction_coeff}")
    
    # 3. Check trajectory parameters
    print("\n3. Trajectory Parameters:")
    for key, value in trajectory_params.items():
        print(f"   {key}: {value}")
    
    # 4. Test pattern calculation directly
    print("\n4. Pattern Calculation Test:")
    try:
        target_params = trajectory_params.get('target_params', {})
        angle_deg = target_params.get('winding_angle_deg', 45.0)
        
        pattern_metrics = unified_planner.pattern_calc.calculate_pattern_metrics(
            vessel_geometry=unified_planner.vessel_geometry,
            roving_width_m=unified_planner.roving_width_m,
            winding_angle_deg=angle_deg,
            num_layers=1
        )
        
        print(f"   Pattern calculation success: {pattern_metrics.get('success', False)}")
        if not pattern_metrics.get('success'):
            print(f"   Error: {pattern_metrics.get('error', 'Unknown error')}")
            print(f"   Error type: {pattern_metrics.get('error_type', 'Unknown')}")
        else:
            solution = pattern_metrics.get('pattern_solution')
            if solution:
                print(f"   Solution found: {solution.get('nd_total_bands', 0)} bands")
            else:
                print("   No pattern solution found")
                
    except Exception as e:
        print(f"   ❌ Pattern calculation failed: {e}")
    
    # 5. Test physics engine
    print("\n5. Physics Engine Test:")
    try:
        vessel_radius = unified_planner._get_vessel_radius()
        print(f"   Vessel radius: {vessel_radius}m")
        
        if vessel_radius > 0:
            # Test simple geodesic calculation
            clairaut_const = vessel_radius * 0.707  # 45° angle
            print(f"   Test Clairaut constant: {clairaut_const}")
            
            # Try generating a few test points
            test_points = unified_planner.physics_engine.solve_geodesic(
                clairaut_constant=clairaut_const,
                initial_param_val=0.0,
                initial_phi_rad=0.0,
                param_end_val=vessel_radius * 0.1,
                num_points=10
            )
            print(f"   ✅ Physics engine generated {len(test_points)} test points")
        else:
            print("   ❌ Invalid vessel radius for physics test")
            
    except Exception as e:
        print(f"   ❌ Physics engine test failed: {e}")
    
    print("\n=== DEBUG COMPLETE ===")
    return True

# Usage in Streamlit app:
def add_debug_button_to_app():
    """
    Add this to your trajectory planning page for debugging
    """
    if st.button("🔍 Debug Trajectory Generation"):
        if 'vessel_geometry' in st.session_state and st.session_state.vessel_geometry:
            try:
                from modules.unified_trajectory_planner import UnifiedTrajectoryPlanner
                
                # Create test planner
                test_planner = UnifiedTrajectoryPlanner(
                    vessel_geometry=st.session_state.vessel_geometry,
                    roving_width_m=0.003,
                    payout_length_m=0.5,
                    default_friction_coeff=0.1
                )
                
                # Run debug
                debug_trajectory_generation_failure(
                    test_planner,
                    pattern_type='helical',
                    coverage_mode='full_coverage',
                    physics_model='clairaut',
                    continuity_level=1,
                    target_params={'winding_angle_deg': 45.0}
                )
                
            except Exception as e:
                st.error(f"Debug failed: {e}")
        else:
            st.error("No vessel geometry available for debugging")

# Quick fix function
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
    else:
        st.success("No fixes needed")
    
    return len(fixes_applied) == 0
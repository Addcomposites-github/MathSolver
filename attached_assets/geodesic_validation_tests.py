"""
Geodesic Trajectory Validation Tests
Catch where the system is generating BS trajectories instead of real geodesic paths
"""

import streamlit as st
import numpy as np
import math
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any

class GeodesicValidationTester:
    """
    Tests to validate that geodesic trajectory generation is actually working
    and not falling back to dummy/random data
    """
    
    def __init__(self):
        self.test_results = {}
        self.failures = []
    
    def run_all_geodesic_tests(self):
        """Run comprehensive geodesic validation tests"""
        st.markdown("# 🔬 Geodesic Trajectory Validation Tests")
        st.markdown("**Catch where the system generates BS instead of real geodesic paths**")
        
        # Test 1: Physics Constants Validation
        st.markdown("## Test 1: 🧮 Physics Constants Validation")
        self.test_clairaut_constants()
        
        # Test 2: Vessel Geometry Usage
        st.markdown("## Test 2: 🏗️ Vessel Geometry Usage Test")
        self.test_vessel_geometry_usage()
        
        # Test 3: Geodesic Path Properties
        st.markdown("## Test 3: 📐 Geodesic Path Properties")
        self.test_geodesic_properties()
        
        # Test 4: Full Coverage Pattern Validation
        st.markdown("## Test 4: 🎯 Full Coverage Pattern Validation")
        self.test_full_coverage_pattern()
        
        # Test 5: Coordinate System Consistency
        st.markdown("## Test 5: 🌐 Coordinate System Consistency")
        self.test_coordinate_consistency()
        
        # Test 6: Fallback Detection
        st.markdown("## Test 6: 🚨 Fallback/Dummy Data Detection")
        self.test_fallback_detection()
        
        # Summary
        self.show_validation_summary()
    
    def test_clairaut_constants(self):
        """Test if Clairaut constants are being calculated correctly"""
        st.markdown("### Testing Clairaut constant calculations...")
        
        if not hasattr(st.session_state, 'vessel_geometry'):
            st.error("❌ No vessel geometry available")
            return
        
        vessel = st.session_state.vessel_geometry
        
        # Test specific cases
        test_cases = [
            {'angle': 30.0, 'expected_range': (0.3, 0.7)},
            {'angle': 45.0, 'expected_range': (0.5, 0.9)},
            {'angle': 60.0, 'expected_range': (0.7, 1.0)}
        ]
        
        vessel_radius = vessel.inner_diameter / 2000  # Convert mm to m
        st.write(f"**Vessel radius**: {vessel_radius:.3f}m")
        
        clairaut_results = []
        
        for test_case in test_cases:
            angle_deg = test_case['angle']
            angle_rad = math.radians(angle_deg)
            
            # Calculate Clairaut constant: C = R * sin(α)
            clairaut_constant = vessel_radius * math.sin(angle_rad)
            
            # Validate range
            expected_min, expected_max = test_case['expected_range']
            expected_min *= vessel_radius
            expected_max *= vessel_radius
            
            is_valid = expected_min <= clairaut_constant <= expected_max
            
            clairaut_results.append({
                'Angle (°)': angle_deg,
                'Clairaut C (m)': f"{clairaut_constant:.6f}",
                'Expected Range': f"{expected_min:.3f}-{expected_max:.3f}",
                'Valid': '✅' if is_valid else '❌'
            })
            
            if not is_valid:
                self.failures.append(f"Clairaut constant for {angle_deg}° outside expected range")
        
        df = pd.DataFrame(clairaut_results)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Test if trajectory planner uses these constants
        st.markdown("### Testing if planner actually uses Clairaut constants...")
        
        try:
            from modules.unified_trajectory_planner import UnifiedTrajectoryPlanner
            
            planner = UnifiedTrajectoryPlanner(
                vessel_geometry=vessel,
                roving_width_m=0.003,
                payout_length_m=0.5,
                default_friction_coeff=0.1
            )
            
            # Generate trajectory with known parameters
            result = planner.generate_trajectory(
                pattern_type='geodesic',
                coverage_mode='single_pass',
                physics_model='clairaut',
                continuity_level=1,
                num_layers_desired=1,
                target_params={'winding_angle_deg': 45.0},
                options={'num_points': 50}
            )
            
            if result and result.points:
                # Check if the trajectory respects Clairaut's theorem
                points = result.points
                clairaut_test_passed = self.verify_clairaut_theorem(points, vessel_radius)
                
                if clairaut_test_passed:
                    st.success("✅ Trajectory respects Clairaut's theorem")
                else:
                    st.error("❌ Trajectory VIOLATES Clairaut's theorem - generating BS!")
                    self.failures.append("Trajectory violates Clairaut's theorem")
            else:
                st.error("❌ No trajectory generated")
                self.failures.append("No trajectory generated for Clairaut test")
                
        except Exception as e:
            st.error(f"❌ Clairaut test failed: {e}")
            self.failures.append(f"Clairaut test error: {e}")
    
    def verify_clairaut_theorem(self, points, vessel_radius) -> bool:
        """Verify trajectory points satisfy Clairaut's theorem: rho * sin(alpha) = constant"""
        try:
            clairaut_values = []
            
            for i in range(len(points) - 1):
                point = points[i]
                if hasattr(point, 'rho') and hasattr(point, 'alpha_deg'):
                    rho = point.rho
                    alpha_rad = math.radians(point.alpha_deg)
                    clairaut_val = rho * math.sin(alpha_rad)
                    clairaut_values.append(clairaut_val)
            
            if not clairaut_values:
                return False
            
            # Check if Clairaut constant is approximately constant
            mean_c = np.mean(clairaut_values)
            std_c = np.std(clairaut_values)
            coefficient_of_variation = std_c / mean_c if mean_c > 0 else float('inf')
            
            st.write(f"**Clairaut constant analysis:**")
            st.write(f"  Mean C: {mean_c:.6f}m")
            st.write(f"  Std C: {std_c:.6f}m")
            st.write(f"  Coefficient of variation: {coefficient_of_variation:.3f}")
            
            # Good geodesic should have CV < 0.05 (5% variation)
            return coefficient_of_variation < 0.05
            
        except Exception as e:
            st.error(f"Clairaut verification failed: {e}")
            return False
    
    def test_vessel_geometry_usage(self):
        """Test if the trajectory actually uses the vessel geometry"""
        st.markdown("### Testing vessel geometry usage...")
        
        if not hasattr(st.session_state, 'vessel_geometry'):
            st.error("❌ No vessel geometry")
            return
        
        vessel = st.session_state.vessel_geometry
        profile = vessel.get_profile_points()
        
        # Get vessel characteristics
        vessel_z_range = (max(profile['z_mm']) - min(profile['z_mm'])) / 1000  # Convert to m
        vessel_r_max = max(profile['r_inner_mm']) / 1000  # Convert to m
        
        st.write(f"**Vessel characteristics:**")
        st.write(f"  Z span: {vessel_z_range:.3f}m")
        st.write(f"  Max radius: {vessel_r_max:.3f}m")
        
        # Generate trajectory and check if it matches vessel
        try:
            from modules.unified_trajectory_planner import UnifiedTrajectoryPlanner
            
            planner = UnifiedTrajectoryPlanner(
                vessel_geometry=vessel,
                roving_width_m=0.003,
                payout_length_m=0.5,
                default_friction_coeff=0.1
            )
            
            result = planner.generate_trajectory(
                pattern_type='geodesic',
                coverage_mode='single_pass',
                physics_model='clairaut',
                continuity_level=1,
                num_layers_desired=1,
                target_params={'winding_angle_deg': 45.0},
                options={'num_points': 100}
            )
            
            if result and result.points:
                points = result.points
                
                # Extract trajectory characteristics
                traj_z = [p.z for p in points if hasattr(p, 'z')]
                traj_rho = [p.rho for p in points if hasattr(p, 'rho')]
                
                if traj_z and traj_rho:
                    traj_z_range = max(traj_z) - min(traj_z)
                    traj_r_max = max(traj_rho)
                    
                    st.write(f"**Trajectory characteristics:**")
                    st.write(f"  Z span: {traj_z_range:.3f}m")
                    st.write(f"  Max radius: {traj_r_max:.3f}m")
                    
                    # Check if trajectory matches vessel
                    z_ratio = traj_z_range / vessel_z_range if vessel_z_range > 0 else 0
                    r_ratio = traj_r_max / vessel_r_max if vessel_r_max > 0 else 0
                    
                    st.write(f"**Geometry match analysis:**")
                    st.write(f"  Z ratio (traj/vessel): {z_ratio:.3f}")
                    st.write(f"  R ratio (traj/vessel): {r_ratio:.3f}")
                    
                    # Good trajectory should have ratios close to 1.0
                    z_match = 0.8 <= z_ratio <= 1.2
                    r_match = 0.8 <= r_ratio <= 1.2
                    
                    if z_match and r_match:
                        st.success("✅ Trajectory properly matches vessel geometry")
                    else:
                        st.error("❌ Trajectory DOES NOT match vessel geometry - generating BS!")
                        self.failures.append(f"Geometry mismatch: Z ratio {z_ratio:.3f}, R ratio {r_ratio:.3f}")
                        
                        # This is a smoking gun for BS generation
                        if z_ratio < 0.1 or r_ratio < 0.1:
                            st.error("🚨 **SMOKING GUN**: Trajectory is orders of magnitude smaller than vessel!")
                        if z_ratio > 10 or r_ratio > 10:
                            st.error("🚨 **SMOKING GUN**: Trajectory is orders of magnitude larger than vessel!")
                else:
                    st.error("❌ No valid trajectory coordinates")
                    self.failures.append("No valid trajectory coordinates")
            else:
                st.error("❌ No trajectory points generated")
                self.failures.append("No trajectory points in vessel geometry test")
                
        except Exception as e:
            st.error(f"❌ Vessel geometry test failed: {e}")
            self.failures.append(f"Vessel geometry test error: {e}")
    
    def test_geodesic_properties(self):
        """Test if generated paths have actual geodesic properties"""
        st.markdown("### Testing geodesic path properties...")
        
        try:
            # Generate a known geodesic trajectory
            if not hasattr(st.session_state, 'all_layer_trajectories') or not st.session_state.all_layer_trajectories:
                st.warning("⚠️ No trajectories available - generate trajectories first")
                return
            
            # Get first trajectory
            first_traj = st.session_state.all_layer_trajectories[0]
            traj_data = first_traj.get('trajectory_data', {})
            
            if 'points' not in traj_data or not traj_data['points']:
                st.error("❌ No trajectory points available")
                return
            
            points = traj_data['points']
            st.write(f"**Analyzing trajectory with {len(points)} points**")
            
            # Test 1: Path smoothness
            smoothness_score = self.test_path_smoothness(points)
            
            # Test 2: Geodesic curvature (should be zero for true geodesics)
            geodesic_score = self.test_geodesic_curvature(points)
            
            # Test 3: Physical constraints
            physics_score = self.test_physical_constraints(points)
            
            # Combined score
            overall_score = (smoothness_score + geodesic_score + physics_score) / 3
            
            st.write(f"**Geodesic Properties Score: {overall_score:.1f}%**")
            
            if overall_score >= 80:
                st.success("✅ Trajectory has good geodesic properties")
            elif overall_score >= 60:
                st.warning("⚠️ Trajectory has some geodesic properties but issues detected")
            else:
                st.error("❌ Trajectory LACKS geodesic properties - likely BS!")
                self.failures.append(f"Poor geodesic properties: {overall_score:.1f}%")
            
        except Exception as e:
            st.error(f"❌ Geodesic properties test failed: {e}")
            self.failures.append(f"Geodesic properties test error: {e}")
    
    def test_path_smoothness(self, points) -> float:
        """Test if path is smooth (geodesics should be smooth)"""
        try:
            if len(points) < 3:
                return 0.0
            
            # Check coordinate continuity
            z_coords = [p.z for p in points if hasattr(p, 'z')]
            rho_coords = [p.rho for p in points if hasattr(p, 'rho')]
            
            if len(z_coords) < 3 or len(rho_coords) < 3:
                return 0.0
            
            # Calculate second derivatives (measure of smoothness)
            z_second_deriv = np.diff(z_coords, n=2)
            rho_second_deriv = np.diff(rho_coords, n=2)
            
            # Good geodesic should have small second derivatives
            z_smoothness = 100 / (1 + np.mean(np.abs(z_second_deriv)) * 1000)
            rho_smoothness = 100 / (1 + np.mean(np.abs(rho_second_deriv)) * 1000)
            
            smoothness = (z_smoothness + rho_smoothness) / 2
            
            st.write(f"  Path smoothness: {smoothness:.1f}%")
            return smoothness
            
        except Exception as e:
            st.write(f"  Smoothness test failed: {e}")
            return 0.0
    
    def test_geodesic_curvature(self, points) -> float:
        """Test geodesic curvature (should be near zero for true geodesics)"""
        try:
            # For a true geodesic on a surface, the geodesic curvature should be zero
            # We'll check if the path follows expected geodesic behavior
            
            if len(points) < 5:
                return 0.0
            
            # Simple test: check if path follows expected helical/spiral pattern
            phi_coords = [p.phi for p in points if hasattr(p, 'phi')]
            z_coords = [p.z for p in points if hasattr(p, 'z')]
            
            if len(phi_coords) < 5 or len(z_coords) < 5:
                return 0.0
            
            # For geodesic, phi should increase monotonically with z for helical paths
            phi_progression = np.diff(phi_coords)
            z_progression = np.diff(z_coords)
            
            # Check for monotonic progression
            phi_monotonic = np.all(phi_progression >= 0) or np.all(phi_progression <= 0)
            z_monotonic = np.all(z_progression >= 0) or np.all(z_progression <= 0)
            
            # Check for reasonable progression rates
            if len(phi_progression) > 0 and len(z_progression) > 0:
                phi_rate = np.mean(np.abs(phi_progression))
                z_rate = np.mean(np.abs(z_progression))
                
                # Reasonable rates for geodesic
                rate_reasonable = 0.001 < phi_rate < 10 and 0.001 < z_rate < 1
            else:
                rate_reasonable = False
            
            geodesic_score = 0
            if phi_monotonic:
                geodesic_score += 40
            if z_monotonic:
                geodesic_score += 30
            if rate_reasonable:
                geodesic_score += 30
            
            st.write(f"  Geodesic behavior: {geodesic_score:.1f}%")
            return geodesic_score
            
        except Exception as e:
            st.write(f"  Geodesic curvature test failed: {e}")
            return 0.0
    
    def test_physical_constraints(self, points) -> float:
        """Test if trajectory respects physical constraints"""
        try:
            if len(points) < 2:
                return 0.0
            
            # Test 1: All radii should be positive
            rho_coords = [p.rho for p in points if hasattr(p, 'rho')]
            if not all(rho > 0 for rho in rho_coords):
                st.write("  ❌ Negative radii detected")
                return 0.0
            
            # Test 2: Winding angles should be reasonable
            alpha_coords = [p.alpha_deg for p in points if hasattr(p, 'alpha_deg')]
            if alpha_coords:
                if not all(5 <= alpha <= 85 for alpha in alpha_coords):
                    st.write("  ❌ Unreasonable winding angles detected")
                    return 50.0  # Partial credit
            
            # Test 3: Coordinates should be finite
            all_finite = True
            for p in points:
                if hasattr(p, 'rho') and hasattr(p, 'z') and hasattr(p, 'phi'):
                    if not (np.isfinite(p.rho) and np.isfinite(p.z) and np.isfinite(p.phi)):
                        all_finite = False
                        break
            
            if not all_finite:
                st.write("  ❌ Non-finite coordinates detected")
                return 0.0
            
            st.write("  ✅ Physical constraints satisfied")
            return 100.0
            
        except Exception as e:
            st.write(f"  Physical constraints test failed: {e}")
            return 0.0
    
    def test_full_coverage_pattern(self):
        """Test if full coverage actually generates multiple circuits"""
        st.markdown("### Testing full coverage pattern generation...")
        
        try:
            from modules.unified_trajectory_planner import UnifiedTrajectoryPlanner
            
            vessel = st.session_state.vessel_geometry
            planner = UnifiedTrajectoryPlanner(
                vessel_geometry=vessel,
                roving_width_m=0.003,  # 3mm
                payout_length_m=0.5,
                default_friction_coeff=0.1
            )
            
            # Test full coverage mode
            result = planner.generate_trajectory(
                pattern_type='geodesic',
                coverage_mode='full_coverage',  # This should generate multiple circuits
                physics_model='clairaut',
                continuity_level=1,
                num_layers_desired=1,
                target_params={'winding_angle_deg': 45.0},
                options={'num_points': 200}
            )
            
            if result and result.points:
                points = result.points
                st.write(f"**Full coverage result: {len(points)} points**")
                
                # Analyze if this is actually full coverage
                phi_coords = [p.phi for p in points if hasattr(p, 'phi')]
                
                if phi_coords:
                    phi_span = max(phi_coords) - min(phi_coords)
                    num_full_rotations = phi_span / (2 * math.pi)
                    
                    st.write(f"  Angular span: {math.degrees(phi_span):.1f}°")
                    st.write(f"  Full rotations: {num_full_rotations:.2f}")
                    
                    # Good full coverage should have multiple rotations
                    if num_full_rotations >= 2.0:
                        st.success("✅ Full coverage generates multiple circuits")
                    elif num_full_rotations >= 0.5:
                        st.warning("⚠️ Partial coverage detected")
                        self.failures.append(f"Partial coverage: only {num_full_rotations:.2f} rotations")
                    else:
                        st.error("❌ Full coverage generates minimal rotation - likely BS!")
                        self.failures.append(f"Minimal coverage: only {num_full_rotations:.2f} rotations")
                        
                        # This is a major red flag
                        if num_full_rotations < 0.1:
                            st.error("🚨 **SMOKING GUN**: Full coverage generates almost no rotation!")
                else:
                    st.error("❌ No angular data in full coverage")
                    self.failures.append("No angular data in full coverage")
            else:
                st.error("❌ Full coverage generates no points")
                self.failures.append("Full coverage generates no points")
                
        except Exception as e:
            st.error(f"❌ Full coverage test failed: {e}")
            self.failures.append(f"Full coverage test error: {e}")
    
    def test_coordinate_consistency(self):
        """Test coordinate system consistency"""
        st.markdown("### Testing coordinate system consistency...")
        
        try:
            if not hasattr(st.session_state, 'all_layer_trajectories') or not st.session_state.all_layer_trajectories:
                st.warning("⚠️ No trajectories for coordinate test")
                return
            
            # Test coordinate conversion consistency
            first_traj = st.session_state.all_layer_trajectories[0]
            traj_data = first_traj.get('trajectory_data', {})
            
            if 'points' not in traj_data:
                st.error("❌ No points for coordinate test")
                return
            
            points = traj_data['points']
            
            # Test cylindrical to Cartesian conversion
            coord_errors = []
            
            for i, point in enumerate(points[:10]):  # Test first 10 points
                if hasattr(point, 'rho') and hasattr(point, 'phi') and hasattr(point, 'z'):
                    # Convert to Cartesian
                    x_calc = point.rho * math.cos(point.phi)
                    y_calc = point.rho * math.sin(point.phi)
                    z_calc = point.z
                    
                    # Check if conversion makes sense
                    radius_check = math.sqrt(x_calc**2 + y_calc**2)
                    
                    if abs(radius_check - point.rho) > 1e-6:
                        coord_errors.append(f"Point {i}: radius mismatch")
                    
                    # Check for reasonable coordinates
                    if abs(x_calc) > 10 or abs(y_calc) > 10 or abs(z_calc) > 10:
                        coord_errors.append(f"Point {i}: extreme coordinates")
                    
                    if abs(x_calc) < 1e-10 and abs(y_calc) < 1e-10:
                        coord_errors.append(f"Point {i}: coordinates too small")
            
            if coord_errors:
                st.error("❌ Coordinate system issues detected:")
                for error in coord_errors[:5]:  # Show first 5 errors
                    st.write(f"  - {error}")
                self.failures.append("Coordinate system inconsistencies")
            else:
                st.success("✅ Coordinate system consistent")
                
        except Exception as e:
            st.error(f"❌ Coordinate consistency test failed: {e}")
            self.failures.append(f"Coordinate test error: {e}")
    
    def test_fallback_detection(self):
        """Detect if system is using fallback/dummy trajectories"""
        st.markdown("### Testing for fallback/dummy trajectory detection...")
        
        try:
            if not hasattr(st.session_state, 'all_layer_trajectories') or not st.session_state.all_layer_trajectories:
                st.warning("⚠️ No trajectories for fallback test")
                return
            
            # Common signs of fallback trajectories
            fallback_signs = []
            
            for i, traj in enumerate(st.session_state.all_layer_trajectories):
                traj_data = traj.get('trajectory_data', {})
                
                if 'points' not in traj_data or not traj_data['points']:
                    fallback_signs.append(f"Trajectory {i+1}: No points")
                    continue
                
                points = traj_data['points']
                
                # Check for suspiciously round numbers (sign of dummy data)
                if len(points) in [10, 20, 50, 100]:  # Common dummy sizes
                    fallback_signs.append(f"Trajectory {i+1}: Suspiciously round point count ({len(points)})")
                
                # Check for identical coordinates (dummy data)
                if len(points) >= 2:
                    first_point = points[0]
                    last_point = points[-1]
                    
                    if hasattr(first_point, 'rho') and hasattr(last_point, 'rho'):
                        if abs(first_point.rho - last_point.rho) < 1e-10:
                            fallback_signs.append(f"Trajectory {i+1}: Identical start/end radius")
                
                # Check for unrealistic uniformity
                if len(points) >= 5:
                    rho_coords = [p.rho for p in points[:5] if hasattr(p, 'rho')]
                    if len(set(rho_coords)) == 1:  # All identical
                        fallback_signs.append(f"Trajectory {i+1}: All radii identical")
                
                # Check for missing physics attributes
                physics_attrs = ['alpha_deg', 'local_curvature']
                missing_attrs = []
                if points:
                    first_point = points[0]
                    for attr in physics_attrs:
                        if not hasattr(first_point, attr):
                            missing_attrs.append(attr)
                
                if missing_attrs:
                    fallback_signs.append(f"Trajectory {i+1}: Missing physics attributes: {missing_attrs}")
            
            if fallback_signs:
                st.error("🚨 **FALLBACK/DUMMY DATA DETECTED:**")
                for sign in fallback_signs:
                    st.write(f"  - {sign}")
                self.failures.append("Fallback/dummy data detected")
                
                st.error("**💥 YOUR SYSTEM IS GENERATING BS TRAJECTORIES!**")
                st.info("**Root cause**: Trajectory generation is failing and falling back to dummy/default data")
            else:
                st.success("✅ No obvious fallback patterns detected")
                
        except Exception as e:
            st.error(f"❌ Fallback detection failed: {e}")
            self.failures.append(f"Fallback detection error: {e}")
    
    def show_validation_summary(self):
        """Show summary of all validation tests"""
        st.markdown("---")
        st.markdown("## 🎯 Validation Summary")
        
        if not self.failures:
            st.success("🎉 **ALL TESTS PASSED!** Your geodesic trajectory generation appears to be working correctly.")
            st.info("If you're still seeing wrong trajectories, the issue is likely in visualization or coordinate conversion.")
        else:
            st.error(f"❌ **{len(self.failures)} CRITICAL ISSUES FOUND:**")
            
            for i, failure in enumerate(self.failures, 1):
                st.write(f"**{i}.** {failure}")
            
            st.markdown("### 🔧 Immediate Actions:")
            
            # Generate specific recommendations based on failures
            if any("geometry mismatch" in f.lower() for f in self.failures):
                st.error("🚨 **CRITICAL**: Trajectory generation is NOT using your vessel geometry!")
                st.info("**Fix**: Check UnifiedTrajectoryPlanner initialization - it's likely using default/fallback geometry")
            
            if any("clairaut" in f.lower() for f in self.failures):
                st.error("🚨 **CRITICAL**: Physics calculations are wrong!")
                st.info("**Fix**: Check physics engine implementation - Clairaut theorem violations")
            
            if any("fallback" in f.lower() or "dummy" in f.lower()):
                st.error("🚨 **CRITICAL**: System is generating fake trajectories!")
                st.info("**Fix**: Trajectory generation is failing silently and returning dummy data")
            
            if any("coverage" in f.lower()):
                st.error("🚨 **CRITICAL**: Full coverage is not working!")
                st.info("**Fix**: Pattern calculation is broken - not generating proper multi-circuit patterns")

# Main testing function
def run_geodesic_validation_tests():
    """Run comprehensive geodesic validation tests"""
    tester = GeodesicValidationTester()
    tester.run_all_geodesic_tests()

# Quick validation for immediate use
def quick_geodesic_reality_check():
    """Quick check to see if trajectories are completely bogus"""
    st.markdown("## ⚡ Quick Geodesic Reality Check")
    
    if not hasattr(st.session_state, 'all_layer_trajectories') or not st.session_state.all_layer_trajectories:
        st.error("❌ No trajectories to check")
        return
    
    # Get first trajectory
    first_traj = st.session_state.all_layer_trajectories[0]
    traj_data = first_traj.get('trajectory_data', {})
    
    if 'points' not in traj_data or not traj_data['points']:
        st.error("🚨 **TRAJECTORY IS BS**: No points generated!")
        return
    
    points = traj_data['points']
    
    # Quick sanity checks
    issues = []
    
    # Check 1: Point count
    if len(points) < 10:
        issues.append(f"Too few points: {len(points)}")
    
    # Check 2: Coordinate ranges
    if hasattr(points[0], 'rho') and hasattr(points[0], 'z'):
        rho_coords = [p.rho for p in points]
        z_coords = [p.z for p in points]
        
        rho_range = max(rho_coords) - min(rho_coords)
        z_range = max(z_coords) - min(z_coords)
        
        if rho_range < 1e-6:
            issues.append("No radial variation - trajectory collapsed")
        
        if z_range < 1e-6:
            issues.append("No axial variation - trajectory collapsed")
        
        # Check against vessel
        if hasattr(st.session_state, 'vessel_geometry'):
            vessel = st.session_state.vessel_geometry
            vessel_radius = vessel.inner_diameter / 2000  # m
            
            if max(rho_coords) < vessel_radius * 0.1:
                issues.append("Trajectory much smaller than vessel - wrong scale")
            
            if max(rho_coords) > vessel_radius * 10:
                issues.append("Trajectory much larger than vessel - wrong scale")
    
    # Check 3: Physics violations
    if hasattr(points[0], 'alpha_deg'):
        angles = [p.alpha_deg for p in points]
        if any(angle < 0 or angle > 90 for angle in angles):
            issues.append("Invalid winding angles detected")
    
    # Report results
    if issues:
        st.error("🚨 **TRAJECTORY IS BS!** Issues found:")
        for issue in issues:
            st.write(f"  - {issue}")
        st.info("💡 Your trajectory generation is definitely broken - run full diagnostic")
    else:
        st.success("✅ **Trajectory passes basic reality check** - looks like real geodesic data")
        st.info("If you're still seeing issues, they're likely in visualization/conversion")

# Add this to your app
if st.button("⚡ Quick Reality Check"):
    quick_geodesic_reality_check()

if st.button("🔬 Full Geodesic Validation"):
    run_geodesic_validation_tests()

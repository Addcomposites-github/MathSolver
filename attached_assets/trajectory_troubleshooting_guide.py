"""
Comprehensive Trajectory Generation Troubleshooting Guide
Systematic approach to identify and fix trajectory generation issues
"""

import streamlit as st
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

class TrajectoryTroubleshooter:
    """
    Systematic troubleshooting for trajectory generation issues.
    Helps identify where in the pipeline things are going wrong.
    """
    
    def __init__(self):
        self.debug_data = {}
        self.issues_found = []
        self.recommendations = []
    
    def run_full_diagnostic(self):
        """Run complete diagnostic of trajectory generation pipeline"""
        st.markdown("# 🔍 Trajectory Generation Troubleshooting Guide")
        st.markdown("Systematic analysis to identify where trajectory generation is failing")
        
        # Step 1: Environment Check
        st.markdown("## Step 1: Environment & Module Check")
        self.check_environment()
        
        # Step 2: Vessel Geometry Validation
        st.markdown("## Step 2: Vessel Geometry Validation")
        vessel_ok = self.check_vessel_geometry()
        
        # Step 3: Layer Stack Validation
        st.markdown("## Step 3: Layer Stack Validation")
        layer_stack_ok = self.check_layer_stack() if vessel_ok else False
        
        # Step 4: Trajectory Planner Validation
        st.markdown("## Step 4: Trajectory Planner Validation")
        planner_ok = self.check_trajectory_planner() if layer_stack_ok else False
        
        # Step 5: Data Flow Analysis
        st.markdown("## Step 5: Data Flow Analysis")
        data_flow_ok = self.check_data_flow() if planner_ok else False
        
        # Step 6: Physics & Mathematics Check
        st.markdown("## Step 6: Physics & Mathematics Check")
        physics_ok = self.check_physics_calculations() if data_flow_ok else False
        
        # Step 7: Output Analysis
        st.markdown("## Step 7: Output Analysis")
        if physics_ok:
            self.analyze_trajectory_output()
        
        # Summary and Recommendations
        st.markdown("## 🎯 Summary & Recommendations")
        self.show_summary()
    
    def check_environment(self):
        """Check if all required modules and dependencies are available"""
        st.markdown("### 🔧 Module & Environment Check")
        
        modules_to_check = [
            ('modules.unified_trajectory_planner', 'UnifiedTrajectoryPlanner'),
            ('modules.trajectory_data_converter', 'TrajectoryDataConverter'),
            ('modules.layer_manager', 'LayerStackManager'),
            ('modules.multi_layer_trajectory_orchestrator', 'MultiLayerTrajectoryOrchestrator'),
            ('modules.geometry', 'VesselGeometry'),
            ('modules.fixed_advanced_3d_visualizer', 'FixedAdvanced3DVisualizer')
        ]
        
        module_status = []
        for module_name, class_name in modules_to_check:
            try:
                module = __import__(module_name, fromlist=[class_name])
                getattr(module, class_name)
                module_status.append({'Module': module_name, 'Status': '✅ Available', 'Class': class_name})
            except ImportError as e:
                module_status.append({'Module': module_name, 'Status': f'❌ Missing: {e}', 'Class': class_name})
                self.issues_found.append(f"Missing module: {module_name}")
            except AttributeError as e:
                module_status.append({'Module': module_name, 'Status': f'⚠️ Class missing: {e}', 'Class': class_name})
                self.issues_found.append(f"Missing class {class_name} in {module_name}")
        
        df = pd.DataFrame(module_status)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Check session state
        st.markdown("### 📊 Session State Check")
        session_keys = [
            'vessel_geometry',
            'layer_stack_manager', 
            'all_layer_trajectories',
            'trajectory_data'
        ]
        
        session_status = []
        for key in session_keys:
            if hasattr(st.session_state, key) and getattr(st.session_state, key) is not None:
                value = getattr(st.session_state, key)
                if isinstance(value, list):
                    status = f"✅ Available ({len(value)} items)"
                elif hasattr(value, '__len__'):
                    status = f"✅ Available ({len(value)} elements)"
                else:
                    status = "✅ Available"
            else:
                status = "❌ Missing"
                self.issues_found.append(f"Missing session state: {key}")
            
            session_status.append({'Session Key': key, 'Status': status})
        
        df_session = pd.DataFrame(session_status)
        st.dataframe(df_session, use_container_width=True, hide_index=True)
    
    def check_vessel_geometry(self) -> bool:
        """Validate vessel geometry is properly defined"""
        st.markdown("### 🏗️ Vessel Geometry Analysis")
        
        if not hasattr(st.session_state, 'vessel_geometry') or st.session_state.vessel_geometry is None:
            st.error("❌ No vessel geometry found in session state")
            self.issues_found.append("Missing vessel geometry")
            return False
        
        vessel = st.session_state.vessel_geometry
        
        try:
            # Check basic properties
            basic_props = {
                'Inner Diameter': getattr(vessel, 'inner_diameter', None),
                'Cylindrical Length': getattr(vessel, 'cylindrical_length', None),
                'Wall Thickness': getattr(vessel, 'wall_thickness', None),
                'Dome Type': getattr(vessel, 'dome_type', None)
            }
            
            st.write("**Basic Properties:**")
            for prop, value in basic_props.items():
                if value is not None:
                    st.write(f"  ✅ {prop}: {value}")
                else:
                    st.write(f"  ❌ {prop}: Missing")
                    self.issues_found.append(f"Missing vessel property: {prop}")
            
            # Check profile points
            try:
                profile = vessel.get_profile_points()
                if profile and 'z_mm' in profile and 'r_inner_mm' in profile:
                    z_points = len(profile['z_mm'])
                    r_points = len(profile['r_inner_mm'])
                    z_range = (max(profile['z_mm']) - min(profile['z_mm'])) / 1000
                    r_range = (max(profile['r_inner_mm']) - min(profile['r_inner_mm'])) / 1000
                    
                    st.success(f"✅ Profile available: {z_points} points")
                    st.write(f"  Z range: {z_range:.3f}m, R variation: {r_range:.3f}m")
                    
                    if z_points < 10:
                        st.warning("⚠️ Very few profile points - may cause trajectory issues")
                        self.issues_found.append("Insufficient profile resolution")
                    
                    if z_range < 0.1:
                        st.warning("⚠️ Very short vessel - may cause trajectory scaling issues")
                    
                    return True
                else:
                    st.error("❌ Invalid profile data")
                    self.issues_found.append("Invalid vessel profile")
                    return False
                    
            except Exception as e:
                st.error(f"❌ Profile generation failed: {e}")
                self.issues_found.append(f"Profile generation error: {e}")
                return False
                
        except Exception as e:
            st.error(f"❌ Vessel geometry validation failed: {e}")
            self.issues_found.append(f"Vessel validation error: {e}")
            return False
    
    def check_layer_stack(self) -> bool:
        """Validate layer stack configuration"""
        st.markdown("### 📚 Layer Stack Analysis")
        
        if not hasattr(st.session_state, 'layer_stack_manager') or st.session_state.layer_stack_manager is None:
            st.error("❌ No layer stack manager found")
            self.issues_found.append("Missing layer stack manager")
            return False
        
        manager = st.session_state.layer_stack_manager
        
        try:
            summary = manager.get_layer_stack_summary()
            
            st.write("**Layer Stack Summary:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Layers", summary['total_layers'])
            with col2:
                st.metric("Applied Layers", summary['layers_applied_to_mandrel'])
            with col3:
                st.metric("Total Thickness", f"{summary['total_thickness_mm']:.2f}mm")
            
            if summary['total_layers'] == 0:
                st.error("❌ No layers defined")
                self.issues_found.append("No layers in stack")
                return False
            
            if summary['layers_applied_to_mandrel'] == 0:
                st.warning("⚠️ No layers applied to mandrel - trajectory planning may fail")
                self.issues_found.append("No layers applied to mandrel")
            
            # Analyze individual layers
            st.write("**Individual Layer Analysis:**")
            layer_issues = []
            
            for i, layer in enumerate(manager.layer_stack):
                layer_info = f"Layer {layer.layer_set_id}: {layer.layer_type} at {layer.winding_angle_deg}°"
                
                # Check for problematic configurations
                if layer.winding_angle_deg < 5 or layer.winding_angle_deg > 89:
                    layer_issues.append(f"{layer_info} - Extreme winding angle")
                
                if layer.num_plies <= 0:
                    layer_issues.append(f"{layer_info} - Invalid ply count")
                
                if layer.single_ply_thickness_mm <= 0:
                    layer_issues.append(f"{layer_info} - Invalid ply thickness")
                
                st.write(f"  ✅ {layer_info}")
            
            if layer_issues:
                st.warning("⚠️ **Layer Issues Found:**")
                for issue in layer_issues:
                    st.write(f"  - {issue}")
                    self.issues_found.append(issue)
            
            return True
            
        except Exception as e:
            st.error(f"❌ Layer stack analysis failed: {e}")
            self.issues_found.append(f"Layer stack error: {e}")
            return False
    
    def check_trajectory_planner(self) -> bool:
        """Test trajectory planner initialization and basic functionality"""
        st.markdown("### 🎯 Trajectory Planner Validation")
        
        try:
            from modules.unified_trajectory_planner import UnifiedTrajectoryPlanner
            
            # Test planner initialization
            vessel = st.session_state.vessel_geometry
            
            test_planner = UnifiedTrajectoryPlanner(
                vessel_geometry=vessel,
                roving_width_m=0.003,  # 3mm
                payout_length_m=0.5,   # 500mm
                default_friction_coeff=0.1
            )
            
            st.success("✅ UnifiedTrajectoryPlanner initialized successfully")
            
            # Test system status
            status = test_planner.get_system_status()
            st.write("**Planner System Status:**")
            for component, state in status.items():
                st.write(f"  ✅ {component}: {state}")
            
            # Test with simple parameters
            st.write("**Testing Simple Trajectory Generation:**")
            
            try:
                result = test_planner.generate_trajectory(
                    pattern_type='helical',
                    coverage_mode='single_pass',
                    physics_model='clairaut',
                    continuity_level=1,
                    num_layers_desired=1,
                    target_params={'winding_angle_deg': 45.0},
                    options={'num_points': 20}  # Small test
                )
                
                if result and result.points:
                    st.success(f"✅ Test trajectory generated: {len(result.points)} points")
                    
                    # Analyze test trajectory
                    sample_point = result.points[0]
                    if hasattr(sample_point, 'rho') and hasattr(sample_point, 'z'):
                        st.write(f"  Sample point: rho={sample_point.rho:.3f}, z={sample_point.z:.3f}")
                        
                        # Check for reasonable coordinate ranges
                        all_rho = [p.rho for p in result.points]
                        all_z = [p.z for p in result.points]
                        
                        rho_range = max(all_rho) - min(all_rho)
                        z_range = max(all_z) - min(all_z)
                        
                        st.write(f"  Coordinate ranges: Rho={rho_range:.3f}m, Z={z_range:.3f}m")
                        
                        if rho_range < 0.001 or z_range < 0.001:
                            st.warning("⚠️ Very small coordinate ranges - possible generation issue")
                            self.issues_found.append("Trajectory generation produces tiny coordinates")
                    
                    return True
                else:
                    st.error("❌ Test trajectory generation returned empty result")
                    self.issues_found.append("Trajectory generation returns empty result")
                    return False
                    
            except Exception as e:
                st.error(f"❌ Test trajectory generation failed: {e}")
                self.issues_found.append(f"Trajectory generation error: {e}")
                return False
                
        except ImportError as e:
            st.error(f"❌ Cannot import UnifiedTrajectoryPlanner: {e}")
            self.issues_found.append(f"Import error: {e}")
            return False
        except Exception as e:
            st.error(f"❌ Trajectory planner validation failed: {e}")
            self.issues_found.append(f"Planner validation error: {e}")
            return False
    
    def check_data_flow(self) -> bool:
        """Check data flow from layer stack to trajectory generation"""
        st.markdown("### 🔄 Data Flow Analysis")
        
        try:
            from modules.multi_layer_trajectory_orchestrator import MultiLayerTrajectoryOrchestrator
            
            if not hasattr(st.session_state, 'layer_stack_manager'):
                st.error("❌ No layer stack manager for data flow test")
                return False
            
            manager = st.session_state.layer_stack_manager
            orchestrator = MultiLayerTrajectoryOrchestrator(manager)
            
            st.write("**Testing Orchestrator Initialization:**")
            st.success("✅ MultiLayerTrajectoryOrchestrator created")
            
            # Test single layer trajectory generation
            if manager.layer_stack:
                st.write("**Testing Single Layer Generation:**")
                test_layer = manager.layer_stack[0]
                
                try:
                    # Test the internal trajectory generation
                    test_result = orchestrator._generate_single_layer_trajectory(
                        0, test_layer, 3.0, 0.125  # 3mm roving, 0.125mm thickness
                    )
                    
                    if test_result:
                        st.success("✅ Single layer trajectory generation works")
                        
                        # Analyze the result format
                        st.write(f"**Result Analysis:**")
                        st.write(f"  Result type: {type(test_result)}")
                        st.write(f"  Result keys: {list(test_result.keys()) if isinstance(test_result, dict) else 'Not a dict'}")
                        
                        # Check for expected data
                        expected_keys = ['path_points', 'x_points_m', 'y_points_m', 'z_points_m']
                        missing_keys = [key for key in expected_keys if key not in test_result]
                        
                        if missing_keys:
                            st.warning(f"⚠️ Missing expected keys: {missing_keys}")
                            self.issues_found.append(f"Missing trajectory data keys: {missing_keys}")
                        else:
                            st.success("✅ All expected data keys present")
                        
                        return True
                    else:
                        st.error("❌ Single layer trajectory generation returned None")
                        self.issues_found.append("Single layer generation returns None")
                        return False
                        
                except Exception as e:
                    st.error(f"❌ Single layer generation failed: {e}")
                    self.issues_found.append(f"Single layer generation error: {e}")
                    return False
            else:
                st.error("❌ No layers available for data flow test")
                return False
                
        except ImportError as e:
            st.error(f"❌ Cannot import orchestrator: {e}")
            self.issues_found.append(f"Orchestrator import error: {e}")
            return False
        except Exception as e:
            st.error(f"❌ Data flow check failed: {e}")
            self.issues_found.append(f"Data flow error: {e}")
            return False
    
    def check_physics_calculations(self) -> bool:
        """Validate physics calculations and mathematical models"""
        st.markdown("### 🔬 Physics & Mathematics Validation")
        
        try:
            vessel = st.session_state.vessel_geometry
            
            # Test basic physics calculations
            st.write("**Testing Physics Calculations:**")
            
            # Test 1: Clairaut constant calculation
            vessel_radius = vessel.inner_diameter / 2000  # Convert to meters
            test_angle = 45.0
            clairaut_constant = vessel_radius * np.sin(np.radians(test_angle))
            
            st.write(f"  Vessel radius: {vessel_radius:.3f}m")
            st.write(f"  Test winding angle: {test_angle}°")
            st.write(f"  Clairaut constant: {clairaut_constant:.6f}m")
            
            if clairaut_constant > 0 and clairaut_constant < vessel_radius:
                st.success("✅ Clairaut constant calculation valid")
            else:
                st.error("❌ Invalid Clairaut constant")
                self.issues_found.append("Invalid Clairaut constant calculation")
                return False
            
            # Test 2: Coordinate system consistency
            profile = vessel.get_profile_points()
            z_mm = np.array(profile['z_mm'])
            r_mm = np.array(profile['r_inner_mm'])
            
            # Convert to meters for physics
            z_m = z_mm / 1000.0
            r_m = r_mm / 1000.0
            
            st.write(f"**Coordinate System Check:**")
            st.write(f"  Z range (physics units): {min(z_m):.3f} to {max(z_m):.3f}m")
            st.write(f"  R range (physics units): {min(r_m):.3f} to {max(r_m):.3f}m")
            
            # Check for reasonable physics scales
            z_span = max(z_m) - min(z_m)
            r_span = max(r_m) - min(r_m)
            
            if z_span < 0.01 or r_span < 0.001:
                st.warning("⚠️ Very small physics scales - may cause numerical issues")
                self.issues_found.append("Physics scales too small")
            elif z_span > 10 or r_span > 5:
                st.warning("⚠️ Very large physics scales - may cause numerical issues")
                self.issues_found.append("Physics scales too large")
            else:
                st.success("✅ Physics scales reasonable")
            
            # Test 3: Numerical stability
            test_points = np.linspace(min(z_m), max(z_m), 10)
            test_radii = np.interp(test_points, z_m, r_m)
            
            # Check for NaN or infinite values
            if np.any(np.isnan(test_radii)) or np.any(np.isinf(test_radii)):
                st.error("❌ NaN or infinite values in profile interpolation")
                self.issues_found.append("Numerical instability in profile")
                return False
            else:
                st.success("✅ Numerical stability check passed")
            
            return True
            
        except Exception as e:
            st.error(f"❌ Physics validation failed: {e}")
            self.issues_found.append(f"Physics validation error: {e}")
            return False
    
    def analyze_trajectory_output(self):
        """Analyze the actual trajectory output if available"""
        st.markdown("### 📊 Trajectory Output Analysis")
        
        if not hasattr(st.session_state, 'all_layer_trajectories') or not st.session_state.all_layer_trajectories:
            st.warning("⚠️ No trajectory output available for analysis")
            return
        
        trajectories = st.session_state.all_layer_trajectories
        st.write(f"**Found {len(trajectories)} trajectory records**")
        
        for i, traj in enumerate(trajectories):
            with st.expander(f"Trajectory {i+1} Analysis - Layer {traj.get('layer_id', 'Unknown')}"):
                
                # Basic structure check
                st.write("**Structure Analysis:**")
                st.write(f"  Top-level keys: {list(traj.keys())}")
                
                if 'trajectory_data' in traj:
                    traj_data = traj['trajectory_data']
                    st.write(f"  Trajectory data keys: {list(traj_data.keys())}")
                    
                    # Analyze points data
                    if 'points' in traj_data:
                        points = traj_data['points']
                        st.write(f"  Points count: {len(points)}")
                        
                        if points:
                            # Analyze first point
                            point = points[0]
                            st.write(f"  Point type: {type(point)}")
                            
                            if hasattr(point, '__dict__'):
                                attrs = list(point.__dict__.keys())
                                st.write(f"  Point attributes: {attrs}")
                                
                                # Check coordinate values
                                if hasattr(point, 'rho') and hasattr(point, 'z'):
                                    all_rho = [p.rho for p in points if hasattr(p, 'rho')]
                                    all_z = [p.z for p in points if hasattr(p, 'z')]
                                    
                                    if all_rho and all_z:
                                        st.write(f"  Rho range: {min(all_rho):.6f} to {max(all_rho):.6f}")
                                        st.write(f"  Z range: {min(all_z):.6f} to {max(all_z):.6f}")
                                        
                                        # Flag potential issues
                                        if max(all_rho) - min(all_rho) < 1e-6:
                                            st.error("❌ No radial variation - trajectory collapsed")
                                            self.issues_found.append(f"Layer {i+1}: No radial variation")
                                        
                                        if max(all_z) - min(all_z) < 1e-6:
                                            st.error("❌ No axial variation - trajectory collapsed")
                                            self.issues_found.append(f"Layer {i+1}: No axial variation")
                                        
                                        if min(all_rho) < 0:
                                            st.error("❌ Negative radius values")
                                            self.issues_found.append(f"Layer {i+1}: Negative radius")
                                    else:
                                        st.error("❌ No valid coordinate data")
                                        self.issues_found.append(f"Layer {i+1}: No coordinate data")
                    
                    # Check for coordinate arrays
                    coord_arrays = ['x_points_m', 'y_points_m', 'z_points_m']
                    available_arrays = [key for key in coord_arrays if key in traj_data]
                    
                    if available_arrays:
                        st.write(f"  Available coordinate arrays: {available_arrays}")
                        
                        for array_key in available_arrays:
                            arr = traj_data[array_key]
                            if arr:
                                st.write(f"    {array_key}: {len(arr)} values, range {min(arr):.6f} to {max(arr):.6f}")
                    else:
                        st.warning("⚠️ No coordinate arrays found")
                        self.issues_found.append(f"Layer {i+1}: No coordinate arrays")
    
    def show_summary(self):
        """Show summary of all issues found and recommendations"""
        st.markdown("### 📋 Issues Summary")
        
        if not self.issues_found:
            st.success("🎉 **No major issues found!** Your trajectory generation pipeline appears to be working correctly.")
            st.info("If you're still seeing problems, they may be in the visualization or coordinate conversion steps.")
        else:
            st.error(f"❌ **Found {len(self.issues_found)} issues:**")
            
            # Categorize issues
            critical_issues = []
            warning_issues = []
            
            for issue in self.issues_found:
                if any(keyword in issue.lower() for keyword in ['missing', 'error', 'failed', 'none', 'empty']):
                    critical_issues.append(issue)
                else:
                    warning_issues.append(issue)
            
            if critical_issues:
                st.markdown("**🔴 Critical Issues (Must Fix):**")
                for issue in critical_issues:
                    st.write(f"  • {issue}")
            
            if warning_issues:
                st.markdown("**🟡 Warnings (Should Review):**")
                for issue in warning_issues:
                    st.write(f"  • {issue}")
        
        # Recommendations
        st.markdown("### 💡 Troubleshooting Recommendations")
        
        recommendations = self._generate_recommendations()
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                st.write(f"**{i}.** {rec}")
        else:
            st.info("No specific recommendations - system appears healthy!")
    
    def _generate_recommendations(self) -> List[str]:
        """Generate specific recommendations based on issues found"""
        recommendations = []
        
        issue_text = " ".join(self.issues_found).lower()
        
        if "missing module" in issue_text:
            recommendations.append("Install missing modules or check import paths")
        
        if "vessel geometry" in issue_text:
            recommendations.append("Regenerate vessel geometry - go to Vessel Geometry page and reconfigure")
        
        if "layer stack" in issue_text:
            recommendations.append("Check layer stack configuration - ensure layers are properly defined and applied")
        
        if "trajectory generation" in issue_text:
            recommendations.append("Check unified trajectory planner parameters and physics model selection")
        
        if "coordinate" in issue_text or "range" in issue_text:
            recommendations.append("Verify coordinate system consistency between vessel geometry and trajectory generation")
        
        if "numerical" in issue_text or "physics" in issue_text:
            recommendations.append("Check vessel dimensions for reasonable physics scales (avoid very small or very large values)")
        
        if not recommendations:
            recommendations = [
                "Try regenerating trajectories with simpler parameters first",
                "Check that vessel geometry has reasonable dimensions",
                "Verify that layer winding angles are within practical ranges (15-75°)",
                "Test with a single simple layer before complex multi-layer stacks"
            ]
        
        return recommendations

# Main troubleshooting function to add to your app
def run_trajectory_troubleshooting():
    """Main function to run trajectory troubleshooting"""
    troubleshooter = TrajectoryTroubleshooter()
    troubleshooter.run_full_diagnostic()

# Quick diagnostic function for immediate issues
def quick_trajectory_diagnostic():
    """Quick diagnostic for immediate trajectory issues"""
    st.markdown("## 🚀 Quick Trajectory Diagnostic")
    
    issues = []
    
    # Quick checks
    if not hasattr(st.session_state, 'vessel_geometry') or st.session_state.vessel_geometry is None:
        issues.append("❌ No vessel geometry")
    else:
        st.success("✅ Vessel geometry available")
    
    if not hasattr(st.session_state, 'layer_stack_manager') or st.session_state.layer_stack_manager is None:
        issues.append("❌ No layer stack")
    else:
        manager = st.session_state.layer_stack_manager
        summary = manager.get_layer_stack_summary()
        if summary['total_layers'] == 0:
            issues.append("❌ No layers defined")
        else:
            st.success(f"✅ {summary['total_layers']} layers defined")
    
    if not hasattr(st.session_state, 'all_layer_trajectories') or not st.session_state.all_layer_trajectories:
        issues.append("❌ No trajectories generated")
    else:
        st.success(f"✅ {len(st.session_state.all_layer_trajectories)} trajectories available")
    
    if issues:
        st.error("**Quick Issues Found:**")
        for issue in issues:
            st.write(f"  {issue}")
        st.info("💡 Run the full diagnostic for detailed analysis")
    else:
        st.success("🎉 Quick check passed! If you're still having issues, run the full diagnostic.")

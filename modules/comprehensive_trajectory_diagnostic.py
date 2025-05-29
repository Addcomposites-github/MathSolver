"""
Comprehensive Trajectory Generation Troubleshooting Guide
Deep analysis to identify trajectory generation and visualization issues
"""

import streamlit as st
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import traceback

class ComprehensiveTrajectoryDiagnostic:
    """
    Comprehensive diagnostic for trajectory generation and visualization pipeline
    """
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.recommendations = []
        
    def run_full_diagnostic(self):
        """Execute complete diagnostic suite"""
        st.markdown("# 🔍 Comprehensive Trajectory Diagnostic")
        st.markdown("Deep analysis of the entire trajectory generation and visualization pipeline")
        
        # Phase 1: Core System Analysis
        st.markdown("## Phase 1: Core System Analysis")
        phase1_score = self._analyze_core_systems()
        
        # Phase 2: Data Flow Analysis
        st.markdown("## Phase 2: Data Flow Analysis")
        phase2_score = self._analyze_data_flow()
        
        # Phase 3: Trajectory Generation Deep Dive
        st.markdown("## Phase 3: Trajectory Generation Analysis")
        phase3_score = self._analyze_trajectory_generation()
        
        # Phase 4: Coordinate System Analysis
        st.markdown("## Phase 4: Coordinate System Analysis")
        phase4_score = self._analyze_coordinate_systems()
        
        # Phase 5: Visualization Pipeline Analysis
        st.markdown("## Phase 5: Visualization Pipeline Analysis")
        phase5_score = self._analyze_visualization_pipeline()
        
        # Final Report
        self._generate_final_report([phase1_score, phase2_score, phase3_score, phase4_score, phase5_score])
    
    def _analyze_core_systems(self) -> float:
        """Analyze core system components"""
        st.markdown("### 🔧 Core System Components")
        
        score = 0
        total_checks = 4
        
        # Check session state completeness
        required_components = {
            'vessel_geometry': 'Vessel geometry object',
            'layer_stack_manager': 'Layer stack manager',
            'all_layer_trajectories': 'Generated trajectories'
        }
        
        for component, description in required_components.items():
            if hasattr(st.session_state, component) and getattr(st.session_state, component) is not None:
                st.success(f"✅ {description}")
                score += 1
            else:
                st.error(f"❌ Missing {description}")
                self.issues.append(f"Missing {component}")
        
        # Check module imports
        modules_to_test = [
            ('modules.unified_trajectory_planner', 'UnifiedTrajectoryPlanner'),
            ('modules.trajectory_data_converter', 'TrajectoryDataConverter'),
            ('modules.fixed_advanced_3d_visualizer', 'FixedAdvanced3DVisualizer')
        ]
        
        st.markdown("#### Module Availability")
        module_issues = []
        for module_path, class_name in modules_to_test:
            try:
                module = __import__(module_path, fromlist=[class_name])
                cls = getattr(module, class_name)
                st.success(f"✅ {class_name}")
            except ImportError as e:
                st.error(f"❌ {class_name}: Import error - {e}")
                module_issues.append(f"{class_name} import failed")
            except AttributeError as e:
                st.error(f"❌ {class_name}: Class not found - {e}")
                module_issues.append(f"{class_name} class missing")
        
        if not module_issues:
            score += 1
        
        return score / total_checks
    
    def _analyze_data_flow(self) -> float:
        """Analyze data flow through the system"""
        st.markdown("### 📊 Data Flow Analysis")
        
        score = 0
        total_checks = 3
        
        # Check vessel geometry data quality
        if hasattr(st.session_state, 'vessel_geometry') and st.session_state.vessel_geometry:
            vessel = st.session_state.vessel_geometry
            try:
                profile = vessel.get_profile_points()
                if profile and len(profile.get('z_mm', [])) > 10:
                    st.success(f"✅ Vessel profile: {len(profile['z_mm'])} points")
                    
                    # Analyze profile quality
                    z_range = max(profile['z_mm']) - min(profile['z_mm'])
                    r_range = max(profile['r_inner_mm']) - min(profile['r_inner_mm'])
                    
                    st.write(f"  Profile ranges: Z={z_range:.1f}mm, R={r_range:.1f}mm")
                    
                    if z_range > 50 and r_range >= 0:  # Reasonable ranges
                        score += 1
                    else:
                        self.warnings.append(f"Unusual profile ranges: Z={z_range:.1f}mm, R={r_range:.1f}mm")
                else:
                    st.error("❌ Invalid vessel profile")
                    self.issues.append("Invalid vessel profile")
            except Exception as e:
                st.error(f"❌ Vessel profile error: {e}")
                self.issues.append(f"Vessel profile error: {e}")
        
        # Check layer stack data quality
        if hasattr(st.session_state, 'layer_stack_manager') and st.session_state.layer_stack_manager:
            manager = st.session_state.layer_stack_manager
            summary = manager.get_layer_stack_summary()
            
            st.write(f"Layer stack: {summary['total_layers']} layers, {summary['layers_applied_to_mandrel']} applied")
            
            if summary['total_layers'] > 0 and summary['layers_applied_to_mandrel'] > 0:
                st.success("✅ Layer stack properly configured")
                score += 1
            else:
                st.error("❌ Layer stack configuration issues")
                self.issues.append("Layer stack not properly configured")
        
        # Check trajectory data quality
        if hasattr(st.session_state, 'all_layer_trajectories') and st.session_state.all_layer_trajectories:
            trajectories = st.session_state.all_layer_trajectories
            
            valid_count = 0
            total_points = 0
            
            for i, traj in enumerate(trajectories):
                traj_data = traj.get('trajectory_data', {})
                
                # Check for coordinate data
                coord_keys = ['x_points_m', 'y_points_m', 'z_points_m', 'path_points']
                found_coords = [key for key in coord_keys if key in traj_data and len(traj_data[key]) > 0]
                
                if found_coords:
                    valid_count += 1
                    if 'x_points_m' in traj_data:
                        total_points += len(traj_data['x_points_m'])
                    elif 'path_points' in traj_data:
                        total_points += len(traj_data['path_points'])
                    
                    st.write(f"  Trajectory {i+1}: {found_coords}")
                else:
                    st.error(f"  ❌ Trajectory {i+1}: No coordinate data")
            
            if valid_count > 0:
                st.success(f"✅ {valid_count} valid trajectories, {total_points} total points")
                score += 1
            else:
                st.error("❌ No valid trajectory data")
                self.issues.append("No valid trajectory coordinate data")
        
        return score / total_checks
    
    def _analyze_trajectory_generation(self) -> float:
        """Deep dive into trajectory generation process"""
        st.markdown("### 🎯 Trajectory Generation Deep Analysis")
        
        score = 0
        total_checks = 3
        
        # Test unified planner directly
        if hasattr(st.session_state, 'vessel_geometry') and st.session_state.vessel_geometry:
            try:
                from modules.unified_trajectory_planner import UnifiedTrajectoryPlanner
                
                planner = UnifiedTrajectoryPlanner(
                    vessel_geometry=st.session_state.vessel_geometry,
                    roving_width_m=0.003,
                    payout_length_m=0.5,
                    default_friction_coeff=0.1
                )
                
                # Test with conservative parameters
                test_result = planner.generate_trajectory(
                    pattern_type='helical',
                    coverage_mode='single_pass',
                    physics_model='clairaut',
                    continuity_level=1,
                    num_layers_desired=1,
                    target_params={'winding_angle_deg': 45.0},
                    options={'num_points': 50}
                )
                
                if test_result and hasattr(test_result, 'points') and test_result.points:
                    st.success(f"✅ Direct planner test: {len(test_result.points)} points")
                    score += 1
                    
                    # Analyze point quality
                    sample_point = test_result.points[0]
                    point_attrs = dir(sample_point)
                    coord_attrs = [attr for attr in point_attrs if attr in ['x', 'y', 'z', 'rho', 'phi']]
                    
                    st.write(f"  Point attributes: {coord_attrs}")
                    
                    if 'rho' in coord_attrs and 'z' in coord_attrs:
                        rho_val = getattr(sample_point, 'rho')
                        z_val = getattr(sample_point, 'z')
                        st.write(f"  Sample coordinates: rho={rho_val:.6f}, z={z_val:.6f}")
                        
                        if abs(rho_val) > 1e-6 and abs(z_val) > 1e-6:
                            score += 1
                        else:
                            self.issues.append("Generated coordinates are near zero")
                    else:
                        self.issues.append("Points missing coordinate attributes")
                else:
                    st.error("❌ Direct planner test failed")
                    self.issues.append("Direct trajectory generation failed")
                    
            except Exception as e:
                st.error(f"❌ Planner test error: {e}")
                self.issues.append(f"Planner error: {e}")
        
        # Test orchestrator if available
        try:
            from modules.multi_layer_trajectory_orchestrator import MultiLayerTrajectoryOrchestrator
            
            if (hasattr(st.session_state, 'vessel_geometry') and 
                hasattr(st.session_state, 'layer_stack_manager')):
                
                orchestrator = MultiLayerTrajectoryOrchestrator(
                    vessel_geometry=st.session_state.vessel_geometry,
                    layer_manager=st.session_state.layer_stack_manager
                )
                
                st.success("✅ Orchestrator initialized successfully")
                score += 1
            else:
                st.warning("⚠️ Cannot test orchestrator - missing dependencies")
                
        except Exception as e:
            st.error(f"❌ Orchestrator test error: {e}")
            self.issues.append(f"Orchestrator error: {e}")
        
        return score / total_checks
    
    def _analyze_coordinate_systems(self) -> float:
        """Analyze coordinate system compatibility"""
        st.markdown("### 🧭 Coordinate System Analysis")
        
        score = 0
        total_checks = 2
        
        if (hasattr(st.session_state, 'vessel_geometry') and 
            hasattr(st.session_state, 'all_layer_trajectories') and
            st.session_state.all_layer_trajectories):
            
            vessel = st.session_state.vessel_geometry
            trajectories = st.session_state.all_layer_trajectories
            
            try:
                # Get vessel coordinate system
                profile = vessel.get_profile_points()
                vessel_z_mm = np.array(profile['z_mm'])
                vessel_z_m = vessel_z_mm / 1000.0
                
                vessel_z_min = np.min(vessel_z_m)
                vessel_z_max = np.max(vessel_z_m)
                vessel_z_center = (vessel_z_min + vessel_z_max) / 2
                
                st.write(f"**Vessel coordinate system:**")
                st.write(f"  Z range: {vessel_z_min:.3f}m to {vessel_z_max:.3f}m")
                st.write(f"  Z center: {vessel_z_center:.3f}m")
                
                score += 1
                
                # Analyze trajectory coordinates
                for i, traj in enumerate(trajectories):
                    traj_data = traj.get('trajectory_data', {})
                    
                    if 'z_points_m' in traj_data:
                        traj_z = np.array(traj_data['z_points_m'])
                        traj_z_min = np.min(traj_z)
                        traj_z_max = np.max(traj_z)
                        traj_z_center = (traj_z_min + traj_z_max) / 2
                        
                        st.write(f"**Trajectory {i+1} coordinates:**")
                        st.write(f"  Z range: {traj_z_min:.3f}m to {traj_z_max:.3f}m")
                        st.write(f"  Z center: {traj_z_center:.3f}m")
                        
                        # Check alignment
                        z_offset = vessel_z_center - traj_z_center
                        st.write(f"  Z offset: {z_offset:.3f}m")
                        
                        if abs(z_offset) < 0.01:
                            st.success("✅ Coordinate systems aligned")
                            score += 1
                        else:
                            st.warning(f"⚠️ Coordinate system mismatch: {z_offset:.3f}m offset")
                            self.warnings.append(f"Trajectory {i+1} needs {z_offset:.3f}m Z-offset")
                        
                        break  # Only check first trajectory
                    elif 'x_points_m' in traj_data:
                        st.write(f"**Trajectory {i+1}:** Found x,y,z coordinates")
                        # Could add more analysis here
                        score += 0.5
                        break
                
            except Exception as e:
                st.error(f"❌ Coordinate analysis error: {e}")
                self.issues.append(f"Coordinate analysis error: {e}")
        
        return score / total_checks
    
    def _analyze_visualization_pipeline(self) -> float:
        """Analyze visualization component functionality"""
        st.markdown("### 🎨 Visualization Pipeline Analysis")
        
        score = 0
        total_checks = 3
        
        # Test trajectory data converter
        try:
            from modules.trajectory_data_converter import TrajectoryDataConverter
            
            converter = TrajectoryDataConverter()
            st.success("✅ Trajectory data converter available")
            score += 1
            
            # Test conversion if trajectory data exists
            if (hasattr(st.session_state, 'all_layer_trajectories') and 
                st.session_state.all_layer_trajectories):
                
                first_traj = st.session_state.all_layer_trajectories[0]
                raw_data = first_traj.get('trajectory_data', {})
                
                if raw_data:
                    try:
                        converted = converter.convert_unified_trajectory_to_visualization_format(raw_data)
                        
                        if converted and converted.get('success'):
                            st.success(f"✅ Data conversion successful: {converted['total_points']} points")
                            score += 1
                        else:
                            st.error("❌ Data conversion failed")
                            self.issues.append("Trajectory data conversion failed")
                    except Exception as e:
                        st.error(f"❌ Conversion test error: {e}")
                        self.issues.append(f"Conversion error: {e}")
                        
        except ImportError as e:
            st.error(f"❌ Converter import error: {e}")
            self.issues.append("Trajectory data converter not available")
        
        # Test 3D visualizer
        try:
            from modules.fixed_advanced_3d_visualizer import FixedAdvanced3DVisualizer
            
            visualizer = FixedAdvanced3DVisualizer()
            st.success("✅ 3D visualizer available")
            score += 1
            
        except ImportError as e:
            st.error(f"❌ 3D visualizer import error: {e}")
            self.issues.append("3D visualizer not available")
        
        return score / total_checks
    
    def _generate_final_report(self, phase_scores: List[float]):
        """Generate comprehensive final report"""
        st.markdown("## 📋 Final Diagnostic Report")
        
        overall_score = sum(phase_scores) / len(phase_scores) * 100
        
        # Score breakdown
        phase_names = ["Core Systems", "Data Flow", "Trajectory Generation", "Coordinate Systems", "Visualization"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Phase Scores")
            for name, score in zip(phase_names, phase_scores):
                score_pct = score * 100
                if score_pct >= 80:
                    st.success(f"{name}: {score_pct:.0f}%")
                elif score_pct >= 50:
                    st.warning(f"{name}: {score_pct:.0f}%")
                else:
                    st.error(f"{name}: {score_pct:.0f}%")
        
        with col2:
            st.markdown("### 🎯 Overall Score")
            if overall_score >= 80:
                st.success(f"**{overall_score:.0f}%** - System mostly functional")
            elif overall_score >= 50:
                st.warning(f"**{overall_score:.0f}%** - Significant issues found")
            else:
                st.error(f"**{overall_score:.0f}%** - Critical problems detected")
        
        # Issues and recommendations
        if self.issues:
            st.markdown("### ❌ Critical Issues Found")
            for issue in self.issues:
                st.error(f"• {issue}")
        
        if self.warnings:
            st.markdown("### ⚠️ Warnings")
            for warning in self.warnings:
                st.warning(f"• {warning}")
        
        # Specific recommendations based on findings
        st.markdown("### 💡 Recommendations")
        
        if overall_score >= 80:
            st.info("System appears to be working well. If you're still experiencing issues, they may be related to specific parameter combinations or edge cases.")
        elif "Layer stack not properly configured" in self.issues:
            st.error("**Priority 1**: Go to Layer Stack Definition and apply layers to mandrel")
        elif "Direct trajectory generation failed" in self.issues:
            st.error("**Priority 1**: Check unified planner configuration and vessel geometry")
        elif any("Coordinate" in issue for issue in self.issues + self.warnings):
            st.warning("**Priority 2**: Coordinate system alignment needed for visualization")
        else:
            st.info("Review the specific issues listed above and address them in order of priority")

def run_comprehensive_diagnostic():
    """Main function to run comprehensive diagnostic"""
    diagnostic = ComprehensiveTrajectoryDiagnostic()
    diagnostic.run_full_diagnostic()
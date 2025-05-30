"""
Unified Trajectory Planner - Main Integration Class
Step 6: Complete system that orchestrates all trajectory generation components
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from .unified_trajectory_core import TrajectoryPoint, TrajectoryResult
from .unified_physics_engine import PhysicsEngine
from .robust_pattern_calculator import RobustPatternCalculator
from .unified_continuity_manager import ContinuityManager, PathQualityReport
from .unified_turnaround_planner import TurnaroundPlanner, MachineCoords

class UnifiedTrajectoryPlanner:
    """
    Primary interface for generating COPV filament winding trajectories.
    Integrates physics, pattern calculation, continuity, and turnaround planning.
    """

    def __init__(self,
                 vessel_geometry,
                 roving_width_m: float,
                 payout_length_m: float,
                 default_friction_coeff: float = 0.1,
                 material_properties: Optional[Dict[str, Any]] = None):
        """
        Initializes the UnifiedTrajectoryPlanner.

        Args:
            vessel_geometry: Vessel geometry object with meridian profile
            roving_width_m: The as-laid width of the fiber roving/band in meters
            payout_length_m: The free fiber length from feed-eye to mandrel (meters)
            default_friction_coeff: Default coefficient of friction for non-geodesic paths
            material_properties: Dictionary of material properties for future use
        """
        self.vessel_geometry = vessel_geometry
        self.roving_width_m = roving_width_m
        self.payout_length_m = payout_length_m
        self.default_friction_coeff = default_friction_coeff
        self.material_properties = material_properties if material_properties else {}

        # Extract meridian points from vessel geometry
        print(f"[DEBUG] Vessel geometry type: {type(vessel_geometry)}")
        print(f"[DEBUG] Vessel geometry attributes: {dir(vessel_geometry)}")
        
        if hasattr(vessel_geometry, 'generate_profile'):
            vessel_geometry.generate_profile()
            profile_points = vessel_geometry.get_profile_points()
            print(f"[DEBUG] Profile points keys: {profile_points.keys()}")
            
            # Handle different attribute naming conventions
            if 'rho_points' in profile_points:
                rho_data = profile_points['rho_points']
                z_data = profile_points['z_points']
            elif 'r_inner_mm' in profile_points:
                # Convert from mm to m and use inner radius as rho
                rho_data = profile_points['r_inner_mm'] / 1000.0
                z_data = profile_points['z_mm'] / 1000.0
            else:
                raise ValueError("Unrecognized vessel geometry format")
            
            # Convert to [rho, z] format for physics engine and ensure proper ordering
            # Sort by z-coordinate to ensure strictly increasing sequence
            sort_indices = np.argsort(z_data)
            meridian_points = np.column_stack([rho_data[sort_indices], z_data[sort_indices]])
            
            print(f"[DEBUG] Vessel geometry input Z range: {z_data.min():.3f} to {z_data.max():.3f}m")
            print(f"[DEBUG] Meridian points Z range: {meridian_points[:, 1].min():.3f} to {meridian_points[:, 1].max():.3f}m")
            print(f"[DEBUG] Meridian points shape: {meridian_points.shape}")
            print(f"[DEBUG] First 3 meridian points: {meridian_points[:3]}")
            print(f"[DEBUG] Last 3 meridian points: {meridian_points[-3:]}")
        else:
            # Fallback: create simple cylinder profile
            print("[DEBUG] Using fallback cylinder profile - vessel geometry missing generate_profile method")
            radius = getattr(vessel_geometry, 'inner_diameter', 200) / 2000  # Convert mm to m
            length = getattr(vessel_geometry, 'cylindrical_length', 500) / 1000  # Convert mm to m
            print(f"[DEBUG] Fallback cylinder: radius={radius:.3f}m, length={length:.3f}m")
            z_points = np.linspace(-length/2, length/2, 50)
            rho_points = np.full_like(z_points, radius)
            meridian_points = np.column_stack([rho_points, z_points])
            print(f"[DEBUG] Fallback Z range: {z_points.min():.3f} to {z_points.max():.3f}m")

        # Initialize sub-components
        self.physics_engine = PhysicsEngine(vessel_meridian_points=meridian_points)
        self.pattern_calc = RobustPatternCalculator()
        self.continuity_mgr = ContinuityManager()
        self.turnaround_planner = TurnaroundPlanner(payout_length_m=self.payout_length_m)

        self.trajectory_log = []

    def _log_message(self, message: str):
        """Internal logging for debugging and analysis"""
        print(f"[UnifiedPlanner] {message}")
        self.trajectory_log.append(message)

    def _get_vessel_radius(self) -> float:
        """Get characteristic vessel radius"""
        if hasattr(self.vessel_geometry, 'inner_diameter'):
            return self.vessel_geometry.inner_diameter / 2000  # Convert mm to m
        return 0.1  # Default 100mm radius

    def _determine_clairaut_constant(self, 
                                   target_winding_angle_deg: float,
                                   vessel_radius_m: float) -> float:
        """Calculate Clairaut constant for geodesic paths"""
        alpha_rad = np.radians(target_winding_angle_deg)
        return vessel_radius_m * np.sin(alpha_rad)

    def _estimate_angular_propagation(self, 
                                    pattern_type: str,
                                    physics_model: str,
                                    **params) -> float:
        """Estimate angular propagation for one circuit"""
        # Simplified estimation based on pattern type
        if pattern_type == 'helical':
            # For helical, advancement depends on winding angle and geometry
            winding_angle = params.get('winding_angle_deg')
            if winding_angle is None:
                raise ValueError("winding_angle_deg is required for helical patterns")
            # Simple estimate: steeper angles = more advancement
            return np.radians(10 + winding_angle * 0.5)
        elif pattern_type == 'geodesic':
            # Geodesic advancement depends on Clairaut constant and geometry
            return np.radians(20)  # Typical value
        else:
            return np.radians(15)  # Default estimate

    def generate_trajectory(self,
                          pattern_type: str,      # 'geodesic', 'non_geodesic', 'helical', 'hoop'
                          coverage_mode: str,     # 'single_pass', 'full_coverage', 'custom'
                          physics_model: str,     # 'clairaut', 'friction' (constant_angle disabled)
                          continuity_level: int,  # 0, 1, 2
                          num_layers_desired: int = 1,
                          initial_conditions: Optional[Dict[str, float]] = None,
                          target_params: Optional[Dict[str, Any]] = None,
                          **options) -> TrajectoryResult:
        """
        Generates a filament winding trajectory based on the specified parameters.

        Args:
            pattern_type: Type of path ('geodesic', 'non_geodesic', 'helical', 'hoop')
            coverage_mode: How the mandrel should be covered
            physics_model: Underlying physics model
            continuity_level: Desired continuity (0, 1, or 2)
            num_layers_desired: Number of complete fiber layers
            initial_conditions: Starting state for the trajectory
            target_params: Target parameters for the winding
            options: Additional keyword arguments

        Returns:
            TrajectoryResult: The generated trajectory and metadata
        """
        self._log_message(f"Generating trajectory: {pattern_type}, {coverage_mode}, {physics_model}, C{continuity_level}")
        
        all_points: List[TrajectoryPoint] = []
        metadata_log = {
            'input_pattern_type': pattern_type,
            'input_coverage_mode': coverage_mode,
            'input_physics_model': physics_model,
            'input_continuity_level': continuity_level,
            'input_num_layers_desired': num_layers_desired,
            'initial_conditions': initial_conditions or {},
            'target_params': target_params or {},
            'vessel_geometry_type': type(self.vessel_geometry).__name__,
            'roving_width_m': self.roving_width_m
        }

        try:
            # Get basic parameters
            vessel_radius_m = self._get_vessel_radius()
            target_angle_deg = target_params.get('winding_angle_deg') if target_params else None
            if target_angle_deg is None:
                raise ValueError("winding_angle_deg is required in target_params")
            
            # Initialize starting conditions
            start_phi_rad = initial_conditions.get('start_phi_rad', 0.0) if initial_conditions else 0.0
            start_z = initial_conditions.get('start_z', 0.0) if initial_conditions else 0.0
            
            circuits_to_generate = 1
            pattern_advancement_rad = 0.0
            pattern_info = None

            # Pattern calculation for full coverage
            if coverage_mode == 'full_coverage':
                self._log_message("Calculating pattern parameters for full coverage...")
                
                # Estimate angular propagation
                estimated_propagation_rad = self._estimate_angular_propagation(
                    pattern_type, physics_model, winding_angle_deg=target_angle_deg
                )
                metadata_log['estimated_angular_propagation_rad'] = estimated_propagation_rad
                
                # Calculate pattern parameters
                try:
                    pattern_metrics = self.pattern_calc.calculate_pattern_metrics(
                        vessel_geometry=self.vessel_geometry,
                        roving_width_m=self.roving_width_m,
                        winding_angle_deg=target_angle_deg,
                        num_layers=num_layers_desired
                    )
                    
                    if pattern_metrics['success'] and pattern_metrics['pattern_solution']:
                        pattern_info = pattern_metrics['pattern_solution']
                        circuits_to_generate = max(1, int(pattern_info.get('n_actual_bands_per_layer', 1)))
                        pattern_advancement_rad = pattern_info.get('actual_angular_propagation_rad', estimated_propagation_rad)
                        metadata_log['pattern_calculation'] = pattern_metrics
                        self._log_message(f"Pattern solution: {circuits_to_generate} circuits, {np.degrees(pattern_advancement_rad):.2f}° advancement")
                    else:
                        self._log_message("Pattern calculation failed, using single pass")
                        circuits_to_generate = 1
                except Exception as e:
                    self._log_message(f"Pattern calculation error: {e}, using single pass")
                    circuits_to_generate = 1

            # Generate trajectory circuits
            current_phi_rad = start_phi_rad
            
            for circuit_num in range(circuits_to_generate):
                self._log_message(f"Generating circuit {circuit_num + 1}/{circuits_to_generate}")
                
                circuit_points = self._generate_single_circuit(
                    pattern_type=pattern_type,
                    physics_model=physics_model,
                    vessel_radius_m=vessel_radius_m,
                    target_angle_deg=target_angle_deg,
                    start_phi_rad=current_phi_rad,
                    start_z=start_z,
                    options=options
                )
                
                # Debug circuit point generation
                print(f"[DEBUG] Circuit {circuit_num + 1} generated {len(circuit_points) if circuit_points else 0} points")
                
                # Apply continuity management if needed
                if circuit_points and all_points and continuity_level > 0:
                    circuit_points = self._ensure_continuity(
                        all_points[-2:], circuit_points[:2], 
                        continuity_level, circuit_points
                    )
                
                all_points.extend(circuit_points)
                print(f"[DEBUG] Total points after circuit {circuit_num + 1}: {len(all_points)}")
                
                # Update starting position for next circuit
                if coverage_mode == 'full_coverage' and pattern_advancement_rad > 0:
                    current_phi_rad += pattern_advancement_rad
                
                if coverage_mode == 'single_pass':
                    break

            # Final quality validation
            self._log_message("Performing final quality validation...")
            quality_report = self.continuity_mgr.validate_path_smoothness(all_points)
            
            metadata_log['final_trajectory_points_count'] = len(all_points)
            metadata_log['trajectory_log'] = self.trajectory_log.copy()
            
            # Add coverage metrics if pattern info available
            if pattern_info and hasattr(self.pattern_calc, 'optimize_coverage_efficiency'):
                try:
                    coverage_metrics = self.pattern_calc.optimize_coverage_efficiency(
                        n_actual_bands_per_layer=pattern_info.get('n_actual_bands_per_layer', 1),
                        angular_band_width_rad=2*np.pi/circuits_to_generate,  # Simplified
                        vessel_radius_m=vessel_radius_m
                    )
                    metadata_log['coverage_metrics'] = coverage_metrics
                except Exception as e:
                    self._log_message(f"Coverage metrics calculation failed: {e}")

            self._log_message(f"Trajectory generation complete. Total points: {len(all_points)}")
            
            return TrajectoryResult(
                points=all_points, 
                metadata=metadata_log, 
                quality_metrics=self._convert_quality_report_to_dict(quality_report)
            )

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self._log_message(f"Trajectory generation failed: {e}")
            print(f"[ERROR] Full trajectory generation exception:")
            print(error_details)
            print(f"[ERROR] Points generated before failure: {len(all_points)}")
            return TrajectoryResult(
                points=[], 
                metadata=metadata_log, 
                quality_metrics={'error': str(e), 'success': False, 'error_details': error_details}
            )

    def _generate_single_circuit(self,
                               pattern_type: str,
                               physics_model: str,
                               vessel_radius_m: float,
                               target_angle_deg: float,
                               start_phi_rad: float,
                               start_z: float,
                               options: Dict) -> List[TrajectoryPoint]:
        """Generate a single circuit trajectory"""
        circuit_points = []
        
        try:
            # Map physics model to appropriate solver based on the physics, not pattern type
            if physics_model == 'clairaut':
                # Use geodesic solver for Clairaut physics (works for helical, geodesic patterns)
                clairaut_constant = self._determine_clairaut_constant(target_angle_deg, vessel_radius_m)
                
                # Calculate actual vessel Z bounds from geometry
                vessel_z_min = self.physics_engine.z_min if hasattr(self.physics_engine, 'z_min') else -0.5
                vessel_z_max = self.physics_engine.z_max if hasattr(self.physics_engine, 'z_max') else 0.5
                
                print(f"[DEBUG] Physics engine Z bounds: {vessel_z_min:.3f}m to {vessel_z_max:.3f}m")
                print(f"[DEBUG] Physics engine z_coords range: {self.physics_engine.z_coords.min():.3f} to {self.physics_engine.z_coords.max():.3f}")
                print(f"[DEBUG] Physics engine rho_coords range: {self.physics_engine.rho_coords.min():.3f} to {self.physics_engine.rho_coords.max():.3f}")
                
                circuit_points = self.physics_engine.solve_geodesic(
                    clairaut_constant=clairaut_constant,
                    initial_param_val=vessel_z_min,
                    initial_phi_rad=start_phi_rad,
                    param_end_val=vessel_z_max,  # Cover actual vessel height
                    num_points=options.get('num_points', 100)
                )
                
            elif physics_model == 'constant_angle':
                # DISABLED: constant_angle creates unrealistic forced helical paths
                # Use geodesic Clairaut solver instead for proper fiber behavior
                print("[WARNING] constant_angle physics model is disabled - using clairaut geodesic solver instead")
                clairaut_constant = self._determine_clairaut_constant(target_angle_deg, vessel_radius_m)
                
                # Calculate full vessel height for proper geodesic coverage
                vessel_z_min = self.physics_engine.z_min if hasattr(self.physics_engine, 'z_min') else -0.5
                vessel_z_max = self.physics_engine.z_max if hasattr(self.physics_engine, 'z_max') else 0.5
                
                circuit_points = self.physics_engine.solve_geodesic(
                    clairaut_constant=clairaut_constant,
                    initial_param_val=vessel_z_min,
                    initial_phi_rad=start_phi_rad,
                    param_end_val=vessel_z_max,  # Cover full vessel height
                    num_points=options.get('num_points', 100)
                )
                
            elif physics_model == 'friction':
                # Use non-geodesic solver for friction physics
                clairaut_constant = self._determine_clairaut_constant(target_angle_deg, vessel_radius_m)
                friction_coeff = options.get('friction_coefficient', self.default_friction_coeff)
                
                # Calculate full vessel height for proper non-geodesic coverage
                vessel_z_min = self.physics_engine.z_min if hasattr(self.physics_engine, 'z_min') else -0.5
                vessel_z_max = self.physics_engine.z_max if hasattr(self.physics_engine, 'z_max') else 0.5
                
                circuit_points = self.physics_engine.solve_non_geodesic(
                    clairaut_constant=clairaut_constant,
                    friction_coefficient=friction_coeff,
                    initial_param_val=vessel_z_min,
                    initial_phi_rad=start_phi_rad,
                    param_end_val=vessel_z_max,  # Cover full vessel height
                    num_points=options.get('num_points', 100)
                )
                
            elif pattern_type == 'hoop':
                # Special case for hoop patterns
                circuit_points = self.physics_engine.solve_hoop(
                    param_val=start_z,
                    num_points=options.get('num_points', 100)
                )
                
            else:
                self._log_message(f"Unknown physics model: {physics_model}, attempting fallback")
                # Fallback to simple trajectory generation
                circuit_points = self._generate_fallback_circuit(
                    target_angle_deg, start_phi_rad, start_z, options.get('num_points', 100)
                )
                
        except Exception as e:
            self._log_message(f"Circuit generation failed: {e}")
            # Generate fallback circuit on failure
            circuit_points = self._generate_fallback_circuit(
                target_angle_deg, start_phi_rad, start_z, options.get('num_points', 100)
            )
            
        return circuit_points

    def _generate_fallback_circuit(self, 
                                 target_angle_deg: float,
                                 start_phi_rad: float, 
                                 start_z: float,
                                 num_points: int) -> List[TrajectoryPoint]:
        """Generate a simple fallback circuit when physics solvers fail"""
        circuit_points = []
        
        try:
            # Simple helical path as fallback
            vessel_radius_m = self._get_vessel_radius()
            
            # Get full vessel profile for proper Z-coordinate range
            profile = self.vessel_geometry.get_profile_points()
            if profile and 'z_mm' in profile:
                z_min_m = min(profile['z_mm']) / 1000  # Convert mm to m
                z_max_m = max(profile['z_mm']) / 1000
            else:
                # Fallback to centered vessel calculation
                cylinder_length = getattr(self.vessel_geometry, 'cylindrical_length', 500) / 1000
                z_min_m = -cylinder_length / 2
                z_max_m = cylinder_length / 2
            
            z_values = np.linspace(z_min_m, z_max_m, num_points)
            
            for i, z in enumerate(z_values):
                # Simple helical progression - use z_min_m as reference instead of start_z
                phi = start_phi_rad + (z - z_min_m) * np.tan(np.radians(target_angle_deg)) / vessel_radius_m
                
                # Convert cylindrical to Cartesian coordinates for correct TrajectoryPoint format
                x = vessel_radius_m * np.cos(phi)
                y = vessel_radius_m * np.sin(phi)
                
                point = TrajectoryPoint(
                    position=np.array([x, y, z]),
                    winding_angle_deg=target_angle_deg,
                    surface_coords={
                        'rho': vessel_radius_m,
                        'z_cyl': z,
                        'phi_cyl': phi,
                        's_meridian': i * ((z_max_m - z_min_m) / num_points)
                    }
                )
                circuit_points.append(point)
                
        except Exception as e:
            self._log_message(f"Fallback circuit generation failed: {e}")
            
        return circuit_points

    def _ensure_continuity(self, 
                         prev_points: List[TrajectoryPoint],
                         next_points: List[TrajectoryPoint],
                         continuity_level: int,
                         full_circuit: List[TrajectoryPoint]) -> List[TrajectoryPoint]:
        """Ensure continuity between trajectory segments"""
        if not prev_points or not next_points:
            return full_circuit
            
        try:
            # Analyze continuity
            continuity_report = self.continuity_mgr.analyze_continuity(prev_points, next_points)
            
            # Check if transition is needed
            needs_transition = (
                continuity_report.c0_gap_m > self.continuity_mgr.c0_threshold_m or
                (continuity_level >= 1 and continuity_report.c1_velocity_diff_mps and 
                 continuity_report.c1_velocity_diff_mps > self.continuity_mgr.c1_threshold_mps)
            )
            
            if needs_transition:
                self._log_message(f"Generating continuity transition (C{continuity_level})")
                transition_points = self.continuity_mgr.generate_smooth_transition(
                    prev_points[-1], next_points[0], continuity_level
                )
                return transition_points + full_circuit
                
        except Exception as e:
            self._log_message(f"Continuity check failed: {e}")
            
        return full_circuit

    def _convert_quality_report_to_dict(self, quality_report: PathQualityReport) -> Dict[str, Any]:
        """Convert PathQualityReport to dictionary for TrajectoryResult"""
        return {
            'max_c0_gap_mm': quality_report.max_c0_gap_mm,
            'avg_c0_gap_mm': quality_report.avg_c0_gap_mm,
            'max_c1_velocity_jump_mps': quality_report.max_c1_velocity_jump_mps,
            'avg_c1_velocity_jump_mps': quality_report.avg_c1_velocity_jump_mps,
            'max_c2_acceleration_mps2': quality_report.max_c2_acceleration_mps2,
            'is_smooth_c0': quality_report.is_smooth_c0,
            'is_smooth_c1': quality_report.is_smooth_c1,
            'is_smooth_c2': quality_report.is_smooth_c2,
            'total_length_m': quality_report.total_length_m,
            'notes': quality_report.notes
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all subsystems"""
        return {
            'physics_engine': 'Ready',
            'pattern_calculator': 'Ready', 
            'continuity_manager': 'Ready',
            'turnaround_planner': 'Ready',
            'vessel_geometry': type(self.vessel_geometry).__name__,
            'roving_width_m': self.roving_width_m,
            'payout_length_m': self.payout_length_m,
            'default_friction_coeff': self.default_friction_coeff
        }
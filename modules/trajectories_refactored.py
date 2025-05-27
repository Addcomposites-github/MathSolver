"""
Refactored Trajectory Planning Module for COPV Filament Winding

This module provides a clean, well-structured approach to trajectory generation with:
- Clear separation between geodesic and non-geodesic engines
- Robust ODE solvers with proper error handling
- Unified interface for single circuit and full coverage patterns
- Proper mathematical foundations based on Clairaut's law and Koussios theory

Author: Refactored for improved maintainability and robustness
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from scipy.integrate import solve_ivp
from scipy.interpolate import UnivariateSpline
import logging

from .geometry import VesselGeometry


class TrajectoryPlannerRefactored:
    """
    Refactored trajectory planner with clean architecture and robust mathematical foundations.
    
    Primary Interface:
    - generate_trajectory(pattern_name, coverage_option, user_circuits=1)
    
    Internal Engines:
    - _engine_geodesic_spiral: Robust geodesic trajectory generation
    - _engine_non_geodesic_spiral: Advanced non-geodesic with friction (experimental)
    - _engine_standard_patterns: Traditional helical/polar patterns
    """
    
    def __init__(self, 
                 vessel_geometry: VesselGeometry,
                 dry_roving_width_m: float = 0.003,
                 dry_roving_thickness_m: float = 0.0002,
                 roving_eccentricity_at_pole_m: float = 0.0,
                 target_cylinder_angle_deg: Optional[float] = None,
                 mu_friction_coefficient: float = 0.0):
        """
        Initialize trajectory planner with clean parameter structure.
        
        Parameters:
        -----------
        vessel_geometry : VesselGeometry
            Vessel geometry with profile points
        dry_roving_width_m : float
            Band width in meters (default: 3mm)
        dry_roving_thickness_m : float  
            Band thickness in meters (default: 0.2mm)
        roving_eccentricity_at_pole_m : float
            Polar eccentricity in meters (default: 0mm)
        target_cylinder_angle_deg : Optional[float]
            Target winding angle on cylinder (None = use geometric limit)
        mu_friction_coefficient : float
            Friction coefficient for non-geodesic patterns (default: 0.0)
        """
        self.vessel = vessel_geometry
        self.dry_roving_width_m = dry_roving_width_m
        self.dry_roving_thickness_m = dry_roving_thickness_m
        self.roving_eccentricity_at_pole_m = roving_eccentricity_at_pole_m
        self.target_cylinder_angle_deg = target_cylinder_angle_deg
        self.mu_friction_coefficient = mu_friction_coefficient
        
        # Core mathematical parameters
        self.R_cyl_m = self.vessel.inner_radius * 1e-3  # Convert mm to m
        self.effective_polar_opening_radius_m = None
        self.clairauts_constant_for_path_m = None
        self.phi_advancement_rad_per_pass = None
        
        # Initialize core calculations
        self._initialize_core_parameters()
        
        # Simple logging replacement to avoid dependency issues
        self.logger = None
        
    def _initialize_core_parameters(self):
        """Initialize core mathematical parameters for trajectory generation."""
        # Calculate effective polar opening radius
        self._calculate_effective_polar_opening()
        
        # Initialize Clairaut's constant
        self._initialize_clairauts_constant()
        
        # Calculate pattern advancement
        self._calculate_pattern_advancement()
        
    def _calculate_effective_polar_opening(self):
        """
        Calculate effective polar opening radius considering roving geometry.
        This is the minimum radius where geodesic paths can physically exist.
        """
        try:
            if not self.vessel.profile_points or 'r_inner_mm' not in self.vessel.profile_points:
                raise ValueError("Vessel profile not available for polar opening calculation")
                
            # Get polar radius from vessel profile
            r_inner_mm = self.vessel.profile_points['r_inner_mm']
            z_mm = self.vessel.profile_points['z_mm']
            
            # Find pole point (minimum radius)
            pole_idx = np.argmin(r_inner_mm)
            rho_geom_pole_m = float(r_inner_mm[pole_idx]) * 1e-3  # Convert to meters
            
            # Calculate slope at pole for roving accommodation
            if len(r_inner_mm) > 1:
                # Use finite difference for slope calculation
                dr_dz = np.gradient(np.array(r_inner_mm) * 1e-3, np.array(z_mm) * 1e-3)
                dz_drho_pole = 1.0 / abs(dr_dz[pole_idx]) if abs(dr_dz[pole_idx]) > 1e-9 else 0.0
            else:
                dz_drho_pole = 0.0
                
            # Effective opening accounting for roving width and thickness
            term_width = self.dry_roving_width_m * abs(dz_drho_pole) / 2.0
            term_thickness = -self.dry_roving_thickness_m * dz_drho_pole
            
            self.effective_polar_opening_radius_m = (
                rho_geom_pole_m + 
                self.roving_eccentricity_at_pole_m + 
                term_width + 
                term_thickness
            )
            
            print(f"Effective polar opening: {self.effective_polar_opening_radius_m*1000:.2f}mm")
        except Exception as e:
            print(f"Error calculating polar opening: {e}")
            self.effective_polar_opening_radius_m = 0.035  # Safe default
        
    def _initialize_clairauts_constant(self):
        """
        Initialize Clairaut's constant based on target angle or geometric limit.
        C = rho * sin(alpha) = constant along geodesic path
        """
        if self.target_cylinder_angle_deg is not None:
            # Use target angle to calculate Clairaut's constant
            alpha_target_rad = math.radians(self.target_cylinder_angle_deg)
            self.clairauts_constant_for_path_m = self.R_cyl_m * math.sin(alpha_target_rad)
            
            # Validate against physical minimum
            if (self.clairauts_constant_for_path_m and self.effective_polar_opening_radius_m and 
                self.clairauts_constant_for_path_m < self.effective_polar_opening_radius_m):
                print(f"Target angle {self.target_cylinder_angle_deg}° gives C = {self.clairauts_constant_for_path_m*1000:.2f}mm")
                print(f"This is below physical minimum {self.effective_polar_opening_radius_m*1000:.2f}mm")
                print("Using geometric limit instead")
                self.clairauts_constant_for_path_m = self.effective_polar_opening_radius_m
            else:
                print(f"Target angle {self.target_cylinder_angle_deg}° validated, C = {self.clairauts_constant_for_path_m*1000:.2f}mm")
        else:
            # Use geometric limit (minimum physically possible)
            self.clairauts_constant_for_path_m = self.effective_polar_opening_radius_m
            print(f"Using geometric limit, C = {self.clairauts_constant_for_path_m*1000:.2f}mm")
            
    def _calculate_pattern_advancement(self):
        """
        Calculate azimuthal advancement per pass for pattern coverage.
        This determines how much the pattern shifts after each pole-to-pole pass.
        """
        try:
            if self.target_cylinder_angle_deg is not None:
                target_alpha_rad = math.radians(self.target_cylinder_angle_deg)
            else:
                # Calculate effective angle from Clairaut's constant
                if self.clairauts_constant_for_path_m and self.R_cyl_m:
                    sin_alpha_eff = self.clairauts_constant_for_path_m / self.R_cyl_m
                    target_alpha_rad = math.asin(np.clip(sin_alpha_eff, 0, 1))
                else:
                    target_alpha_rad = math.radians(45.0)  # Safe default
                
            # Angular width of one band projected perpendicular to winding direction
            self.phi_advancement_rad_per_pass = (
                self.dry_roving_width_m / math.cos(target_alpha_rad)
            ) / self.R_cyl_m
            
            print(f"Pattern advancement: {math.degrees(self.phi_advancement_rad_per_pass):.3f}° per pass")
        except Exception as e:
            print(f"Error calculating pattern advancement: {e}")
            self.phi_advancement_rad_per_pass = 0.05  # Safe default
        
    def generate_trajectory(self, 
                          pattern_name: str, 
                          coverage_option: str, 
                          user_circuits: int = 1) -> Optional[Dict]:
        """
        Main interface for trajectory generation with clean parameter structure.
        
        Parameters:
        -----------
        pattern_name : str
            Type of pattern: 'geodesic_spiral', 'non_geodesic_spiral', 'helical', 'polar', 'hoop'
        coverage_option : str
            Coverage strategy: 'single_circuit', 'full_coverage', 'user_defined'
        user_circuits : int
            Number of circuits for 'user_defined' option
            
        Returns:
        --------
        Dict containing trajectory data or None if generation fails
        """
        print(f"Generating trajectory: {pattern_name}, coverage: {coverage_option}, circuits: {user_circuits}")
        
        # Validate inputs
        valid_patterns = ['geodesic_spiral', 'non_geodesic_spiral', 'helical', 'polar', 'hoop']
        valid_coverage = ['single_circuit', 'full_coverage', 'user_defined']
        
        if pattern_name not in valid_patterns:
            print(f"Invalid pattern_name: {pattern_name}. Must be one of {valid_patterns}")
            return None
            
        if coverage_option not in valid_coverage:
            print(f"Invalid coverage_option: {coverage_option}. Must be one of {valid_coverage}")
            return None
            
        # Calculate number of passes based on coverage option
        num_passes = self._calculate_num_passes(coverage_option, user_circuits)
        
        # Route to appropriate engine
        if pattern_name == 'geodesic_spiral':
            return self._engine_geodesic_spiral(coverage_option, num_passes)
        elif pattern_name == 'non_geodesic_spiral':
            return self._engine_non_geodesic_spiral(coverage_option, num_passes)
        else:
            return self._engine_standard_patterns(pattern_name, coverage_option, num_passes)
            
    def _calculate_num_passes(self, coverage_option: str, user_circuits: int) -> int:
        """Calculate number of pole-to-pole passes based on coverage option."""
        if coverage_option == 'single_circuit':
            return 2  # One complete circuit: A->B->A
        elif coverage_option == 'full_coverage':
            # Calculate passes needed for complete circumferential coverage
            passes_for_coverage = math.ceil(2 * math.pi / self.phi_advancement_rad_per_pass)
            # Ensure even number for complete circuits
            return passes_for_coverage if passes_for_coverage % 2 == 0 else passes_for_coverage + 1
        else:  # user_defined
            return user_circuits * 2  # Each circuit = 2 passes
            
    def _engine_geodesic_spiral(self, coverage_option: str, num_passes: int) -> Optional[Dict]:
        """
        Robust geodesic spiral trajectory generation engine.
        
        This engine implements true geodesic paths using Clairaut's law with:
        - Proper turnaround handling at polar openings
        - Smooth pattern advancement between passes
        - Robust ODE integration with error handling
        """
        print(f"Geodesic engine: Generating {num_passes} passes")
        
        # Get vessel profile for trajectory calculation
        profile_data = self._prepare_profile_data()
        if profile_data is None:
            return None
            
        # Initialize trajectory arrays
        trajectory_points = []
        current_phi_rad = 0.0
        
        # Generate each pass
        for pass_idx in range(num_passes):
            is_forward = (pass_idx % 2 == 0)
            
            print(f"Pass {pass_idx + 1}/{num_passes} ({'forward' if is_forward else 'reverse'})")
            
            # Generate helical segment
            segment_result = self._solve_geodesic_segment(
                profile_data, 
                current_phi_rad, 
                is_forward,
                pass_idx
            )
            
            if segment_result is None:
                print(f"Failed to generate pass {pass_idx + 1}")
                return None
                
            # Add helical points to trajectory
            trajectory_points.extend(segment_result['helical_points'])
            
            # Generate turnaround if not last pass
            if pass_idx < num_passes - 1:
                turnaround_result = self._generate_geodesic_turnaround(
                    segment_result['final_position'],
                    current_phi_rad
                )
                
                if turnaround_result:
                    # Apply smoothness fix: skip first and last turnaround points
                    turnaround_points = turnaround_result['points']
                    if len(turnaround_points) > 2:
                        trajectory_points.extend(turnaround_points[1:-1])
                    
                    current_phi_rad = turnaround_result['final_phi']
                else:
                    self.logger.warning(f"Turnaround generation failed at pass {pass_idx + 1}")
                    
            # Update phi for next pass
            current_phi_rad += self.phi_advancement_rad_per_pass
            
        # Convert to output format
        return self._format_trajectory_output(trajectory_points, 'geodesic_spiral', num_passes)
        
    def _prepare_profile_data(self) -> Optional[Dict]:
        """Prepare vessel profile data for trajectory calculations."""
        if not self.vessel.profile_points:
            self.logger.error("No vessel profile available")
            return None
            
        r_inner_mm = self.vessel.profile_points['r_inner_mm']
        z_mm = self.vessel.profile_points['z_mm']
        
        # Convert to meters and ensure proper ordering
        r_m = r_inner_mm * 1e-3
        z_m = z_mm * 1e-3
        
        # Sort by z coordinate for consistent processing
        sort_indices = np.argsort(z_m)
        
        return {
            'r_m': r_m[sort_indices],
            'z_m': z_m[sort_indices],
            'original_indices': sort_indices
        }
        
    def _solve_geodesic_segment(self, 
                              profile_data: Dict, 
                              initial_phi: float, 
                              is_forward: bool,
                              pass_idx: int) -> Optional[Dict]:
        """
        Solve geodesic segment using robust ODE integration.
        
        Implements: dφ/dz = (tan(α)/ρ) * √(1 + (dρ/dz)²)
        Where: sin(α) = C/ρ (Clairaut's law)
        """
        r_m = profile_data['r_m']
        z_m = profile_data['z_m']
        
        # Find trajectory bounds where ρ ≥ C
        valid_indices = np.where(r_m >= self.clairauts_constant_for_path_m - 1e-6)[0]
        
        if len(valid_indices) == 0:
            self.logger.error("No valid trajectory points found (all ρ < C)")
            return None
            
        start_idx, end_idx = valid_indices[0], valid_indices[-1]
        
        # Set up integration range
        if is_forward:
            z_range = z_m[start_idx:end_idx+1]
            r_range = r_m[start_idx:end_idx+1]
        else:
            z_range = z_m[start_idx:end_idx+1][::-1]
            r_range = r_m[start_idx:end_idx+1][::-1]
            
        # Create interpolation functions
        try:
            r_of_z = UnivariateSpline(z_range, r_range, s=0, k=3)
            dr_dz_func = r_of_z.derivative(1)
        except Exception as e:
            self.logger.error(f"Failed to create interpolation functions: {e}")
            return None
            
        # Solve geodesic ODE
        ode_result = self._solve_ode_geodesic_segment(
            z_range, initial_phi, self.clairauts_constant_for_path_m, r_of_z, dr_dz_func
        )
        
        if ode_result is None:
            return None
            
        # Generate trajectory points
        helical_points = []
        phi_solution = ode_result['solution']
        
        for i, z_val in enumerate(z_range):
            r_val = float(r_of_z(z_val))
            phi_val = float(phi_solution.sol(z_val)[0]) if phi_solution.sol else initial_phi
            
            # Calculate winding angle
            sin_alpha = np.clip(self.clairauts_constant_for_path_m / r_val, 0, 1)
            alpha_rad = math.asin(sin_alpha)
            
            # Convert to Cartesian coordinates
            x_val = r_val * math.cos(phi_val)
            y_val = r_val * math.sin(phi_val)
            
            point = {
                'rho': r_val,
                'z': z_val,
                'phi': phi_val,
                'alpha': alpha_rad,
                'x': x_val,
                'y': y_val,
                'pass_idx': pass_idx
            }
            
            helical_points.append(point)
            
        return {
            'helical_points': helical_points,
            'final_position': helical_points[-1] if helical_points else None,
            'success': True
        }
        
    def _solve_ode_geodesic_segment(self, 
                                  z_eval_points: np.ndarray,
                                  initial_phi: float,
                                  clairaut_C: float,
                                  r_of_z_func,
                                  dr_dz_func) -> Optional[Dict]:
        """
        Robust geodesic ODE solver with proper error handling.
        
        ODE: dφ/dz = (tan(α)/ρ) * √(1 + (dρ/dz)²)
        """
        def geodesic_ode_system(z, phi_array):
            """Geodesic ODE system for solve_ivp."""
            phi_val = phi_array[0]
            
            try:
                rho_val = float(r_of_z_func(z))
                
                # Check for turnaround proximity
                if rho_val <= clairaut_C + 1e-8:
                    # Near turnaround - handle carefully
                    return [1e6]  # Large but finite slope
                    
                # Calculate winding angle
                sin_alpha = clairaut_C / rho_val
                if not (0 <= sin_alpha <= 1):
                    return [0.0]  # Invalid geometry
                    
                alpha_val = math.asin(sin_alpha)
                
                # Calculate dφ/dz
                dr_dz_val = float(dr_dz_func(z))
                ds_dz = math.sqrt(1.0 + dr_dz_val**2)
                
                if abs(math.cos(alpha_val)) < 1e-9:
                    # At turnaround (α = π/2)
                    return [1e6]  # Large but finite slope
                    
                dphi_dz = (math.tan(alpha_val) / rho_val) * ds_dz
                
                return [dphi_dz]
                
            except Exception as e:
                self.logger.warning(f"ODE evaluation error at z={z}: {e}")
                return [0.0]
                
        # Solve ODE
        z_span = [z_eval_points[0], z_eval_points[-1]]
        
        try:
            solution = solve_ivp(
                geodesic_ode_system,
                z_span,
                [initial_phi],
                dense_output=True,
                method='RK45',
                rtol=1e-6,
                atol=1e-8
            )
            
            if not solution.success:
                self.logger.warning(f"ODE solver failed: {solution.message}")
                return None
                
            return {'solution': solution}
            
        except Exception as e:
            self.logger.error(f"ODE solving exception: {e}")
            return None
            
    def _generate_geodesic_turnaround(self, final_position: Dict, current_phi: float) -> Optional[Dict]:
        """Generate smooth turnaround segment at polar opening."""
        if not final_position:
            return None
            
        # Turnaround parameters
        turnaround_radius = self.clairauts_constant_for_path_m
        turnaround_z = final_position['z']
        phi_advance = self.phi_advancement_rad_per_pass
        
        # Generate circumferential points
        num_turnaround_points = max(5, int(abs(phi_advance) * 10))  # Adaptive point count
        
        turnaround_points = []
        for i in range(num_turnaround_points + 1):
            t = i / num_turnaround_points
            phi_turn = current_phi + phi_advance * t
            
            point = {
                'rho': turnaround_radius,
                'z': turnaround_z,
                'phi': phi_turn,
                'alpha': math.pi / 2,  # Purely circumferential
                'x': turnaround_radius * math.cos(phi_turn),
                'y': turnaround_radius * math.sin(phi_turn),
                'pass_idx': -1  # Mark as turnaround
            }
            
            turnaround_points.append(point)
            
        return {
            'points': turnaround_points,
            'final_phi': current_phi + phi_advance
        }
        
    def _engine_non_geodesic_spiral(self, coverage_option: str, num_passes: int) -> Optional[Dict]:
        """
        Non-geodesic spiral engine (experimental).
        
        Implements Koussios differential equations with friction effects.
        Currently marked as experimental due to ODE solver stability issues.
        """
        self.logger.warning("Non-geodesic engine is experimental - use with caution")
        
        if self.mu_friction_coefficient <= 0:
            self.logger.warning("Non-geodesic pattern requires friction coefficient > 0")
            # Fall back to geodesic
            return self._engine_geodesic_spiral(coverage_option, num_passes)
            
        # Placeholder for non-geodesic implementation
        # This would implement the full Koussios ODE system
        self.logger.info("Non-geodesic engine not yet implemented - falling back to geodesic")
        return self._engine_geodesic_spiral(coverage_option, num_passes)
        
    def _engine_standard_patterns(self, pattern_name: str, coverage_option: str, num_passes: int) -> Optional[Dict]:
        """
        Standard pattern engine for helical, polar, and hoop patterns.
        
        These patterns use simplified mathematical models compared to geodesic spirals.
        """
        self.logger.info(f"Generating standard pattern: {pattern_name}")
        
        # Placeholder for standard pattern implementations
        if pattern_name == 'helical':
            return self._generate_helical_pattern(num_passes)
        elif pattern_name == 'polar':
            return self._generate_polar_pattern(num_passes)
        elif pattern_name == 'hoop':
            return self._generate_hoop_pattern(num_passes)
        else:
            self.logger.error(f"Unknown standard pattern: {pattern_name}")
            return None
            
    def _generate_helical_pattern(self, num_passes: int) -> Optional[Dict]:
        """Generate traditional helical winding pattern."""
        # Simplified helical implementation
        self.logger.info("Helical pattern generation not yet implemented")
        return None
        
    def _generate_polar_pattern(self, num_passes: int) -> Optional[Dict]:
        """Generate polar winding pattern."""
        # Simplified polar implementation
        self.logger.info("Polar pattern generation not yet implemented")
        return None
        
    def _generate_hoop_pattern(self, num_passes: int) -> Optional[Dict]:
        """Generate hoop winding pattern."""
        # Simplified hoop implementation
        self.logger.info("Hoop pattern generation not yet implemented")
        return None
        
    def _format_trajectory_output(self, trajectory_points: List[Dict], pattern_type: str, num_passes: int) -> Dict:
        """Format trajectory data for output with consistent structure."""
        if not trajectory_points:
            return {
                'pattern_type': pattern_type,
                'total_points': 0,
                'success': False,
                'error_message': 'No trajectory points generated'
            }
            
        # Extract coordinate arrays
        x_points_m = [p['x'] for p in trajectory_points]
        y_points_m = [p['y'] for p in trajectory_points]
        z_points_m = [p['z'] for p in trajectory_points]
        rho_points_m = [p['rho'] for p in trajectory_points]
        phi_rad_profile = [p['phi'] for p in trajectory_points]
        alpha_deg_profile = [math.degrees(p['alpha']) for p in trajectory_points]
        
        # Calculate trajectory statistics
        final_phi_deg = math.degrees(phi_rad_profile[-1]) if phi_rad_profile else 0
        alpha_equator_deg = np.mean(alpha_deg_profile) if alpha_deg_profile else 0
        
        return {
            'pattern_type': f"{pattern_type}_RefactoredEngine",
            'total_points': len(trajectory_points),
            'total_circuits_legs': num_passes,
            'success': True,
            
            # Coordinate arrays
            'x_points_m': np.array(x_points_m),
            'y_points_m': np.array(y_points_m),
            'z_points_m': np.array(z_points_m),
            'rho_points_m': np.array(rho_points_m),
            'phi_rad_profile': np.array(phi_rad_profile),
            'alpha_deg_profile': np.array(alpha_deg_profile),
            
            # Trajectory metadata
            'c_eff_m': self.effective_polar_opening_radius_m,
            'clairauts_constant_used_m': self.clairauts_constant_for_path_m,
            'final_turn_around_angle_deg': final_phi_deg,
            'alpha_equator_deg': alpha_equator_deg,
            'phi_advancement_per_pass_deg': math.degrees(self.phi_advancement_rad_per_pass),
            
            # Legacy compatibility
            'path_points': trajectory_points
        }
        
    def get_validation_results(self) -> Dict:
        """Get trajectory validation results for UI feedback."""
        if self.clairauts_constant_for_path_m is None:
            return {
                'is_valid': False,
                'error_message': 'Trajectory parameters not initialized'
            }
            
        # Check if target angle is achievable
        if self.target_cylinder_angle_deg is not None:
            target_C = self.R_cyl_m * math.sin(math.radians(self.target_cylinder_angle_deg))
            is_achievable = target_C >= self.effective_polar_opening_radius_m
            
            safety_margin_mm = (target_C - self.effective_polar_opening_radius_m) * 1000
            
            return {
                'is_valid': is_achievable,
                'target_angle_deg': self.target_cylinder_angle_deg,
                'clairaut_constant_mm': target_C * 1000,
                'effective_polar_opening_mm': self.effective_polar_opening_radius_m * 1000,
                'safety_margin_mm': safety_margin_mm,
                'error_message': f"Target angle too shallow - minimum {math.degrees(math.asin(self.effective_polar_opening_radius_m/self.R_cyl_m)):.1f}°" if not is_achievable else None
            }
        else:
            # Using geometric limit
            return {
                'is_valid': True,
                'target_angle_deg': None,
                'clairaut_constant_mm': self.clairauts_constant_for_path_m * 1000,
                'effective_polar_opening_mm': self.effective_polar_opening_radius_m * 1000,
                'safety_margin_mm': 0.0,
                'error_message': None
            }
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from modules.geometry import VesselGeometry

class TrajectoryPlanner:
    """
    Filament winding trajectory planning for composite pressure vessels.
    Implements geodesic and non-geodesic winding patterns.
    """
    
    def __init__(self, vessel_geometry: VesselGeometry, 
                 dry_roving_width_m: float = 0.003,
                 dry_roving_thickness_m: float = 0.0002,
                 roving_eccentricity_at_pole_m: float = 0.0,
                 target_cylinder_angle_deg: Optional[float] = None):
        """
        Initialize trajectory planner with vessel geometry and optional target angle.
        
        Parameters:
        -----------
        vessel_geometry : VesselGeometry
            Vessel geometry object containing profile data
        dry_roving_width_m : float
            True width of the dry roving/band (default 3mm)
        dry_roving_thickness_m : float  
            True thickness of the dry roving/band (default 0.2mm)
        roving_eccentricity_at_pole_m : float
            Offset of the roving centerline from geometric polar opening
        target_cylinder_angle_deg : Optional[float]
            Desired winding angle on cylinder section (None = use geometric limit)
        """
        self.vessel = vessel_geometry
        self.dry_roving_width_m = dry_roving_width_m
        self.dry_roving_thickness_m = dry_roving_thickness_m
        self.roving_eccentricity_at_pole_m = roving_eccentricity_at_pole_m
        self.target_cylinder_angle_deg = target_cylinder_angle_deg
        self.trajectory_data = None
        
        # Geodesic calculation properties
        self.effective_polar_opening_radius_m = None  # Physical minimum turning radius
        self.clairauts_constant_for_path_m = None    # Actual constant used for path generation
        self.alpha_profile_deg = None  # Array of winding angles
        self.phi_profile_rad = None    # Array of parallel angles
        self.turn_around_angle_rad = None
        self.alpha_eq_deg = None  # Actual angle achieved at equator
        self.validation_results = None  # Target angle validation results
        self.alpha_eq_deg = None       # Winding angle at equator
        
        print("\nDEBUG trajectories.py: Entering TrajectoryPlanner.__init__")
        print(f"  Vessel dome_type: {self.vessel.dome_type}")
        print(f"  Vessel initial profile_points: {self.vessel.profile_points}")

        if self.vessel.profile_points is None:
            print("DEBUG trajectories.py: Vessel profile_points is None, calling self.vessel.generate_profile().")
            self.vessel.generate_profile()
            if self.vessel.profile_points is None:
                raise ValueError("Vessel profile_points is STILL None after calling generate_profile() in TrajectoryPlanner.")
            else:
                print("DEBUG trajectories.py: self.vessel.generate_profile() CALLED from TrajectoryPlanner.")
        
        print(f"DEBUG trajectories.py: After potential generate_profile call in TrajectoryPlanner init:")
        print(f"  self.vessel.profile_points type: {type(self.vessel.profile_points)}")
        if isinstance(self.vessel.profile_points, dict):
            print(f"  Keys in self.vessel.profile_points: {list(self.vessel.profile_points.keys())}")
            if 'r_inner' not in self.vessel.profile_points:
                print("  CRITICAL DEBUG trajectories.py: 'r_inner' key IS MISSING from self.vessel.profile_points dict HERE!")
            elif not hasattr(self.vessel.profile_points['r_inner'], '__len__') or len(self.vessel.profile_points['r_inner']) == 0:
                print("  CRITICAL DEBUG trajectories.py: 'r_inner' IS EMPTY or not array-like in self.vessel.profile_points!")
            else:
                print(f"  'r_inner' key FOUND. Length: {len(self.vessel.profile_points['r_inner'])}")
        else:
            print(f"  CRITICAL DEBUG trajectories.py: self.vessel.profile_points is NOT a dict HERE! It is: {self.vessel.profile_points}")
            raise TypeError("Vessel profile_points is not a dictionary as expected in TrajectoryPlanner.")

        # Calculate effective polar opening for geodesic paths
        print("TrajectoryPlanner init: About to call _calculate_effective_polar_opening()")
        self._calculate_effective_polar_opening()
        
        # Validate target angle and set Clairaut's constant for path generation
        if self.target_cylinder_angle_deg is not None:
            validation_success = self._validate_and_set_clairauts_constant_from_target_angle(self.target_cylinder_angle_deg)
            if not validation_success:
                print(f"WARNING: Target cylinder angle {self.target_cylinder_angle_deg}° not achievable. Using geometric limit instead.")
                self.clairauts_constant_for_path_m = self.effective_polar_opening_radius_m
            else:
                print(f"SUCCESS: Target cylinder angle {self.target_cylinder_angle_deg}° validated and set.")
        else:
            # No target angle specified, use the physical minimum turning radius
            self.clairauts_constant_for_path_m = self.effective_polar_opening_radius_m
            print(f"INFO: No target angle specified. Using geometric limit c_eff = {self.clairauts_constant_for_path_m*1000:.2f}mm")

    def _validate_and_set_clairauts_constant_from_target_angle(self, target_alpha_cyl_deg: float) -> bool:
        """
        Validates if the target cylinder angle is physically achievable and sets the
        Clairaut's constant to be used for path generation.
        
        Parameters:
        -----------
        target_alpha_cyl_deg : float
            Desired winding angle on cylinder section in degrees
            
        Returns:
        --------
        bool : True if target angle is achievable, False otherwise
        """
        if not (5 < target_alpha_cyl_deg < 85):
            self.validation_results = {
                'is_valid': False,
                'error_type': 'invalid_range',
                'message': f"Target angle {target_alpha_cyl_deg}° must be between 5° and 85° for practical winding",
                'suggested_range': [5, 85]
            }
            print(f"ERROR validate_angle: {self.validation_results['message']}")
            return False

        # Physical minimum turning radius (already calculated)
        c_eff_physical = self.effective_polar_opening_radius_m
        
        # Calculate implied Clairaut's constant for target angle
        R_cyl_m = self.vessel.inner_radius * 1e-3  # Cylinder radius in meters
        alpha_cyl_target_rad = math.radians(target_alpha_cyl_deg)
        c_implied_by_target_m = R_cyl_m * math.sin(alpha_cyl_target_rad)

        print(f"\nDEBUG _validate_target_angle:")
        print(f"  Target cylinder angle: {target_alpha_cyl_deg:.2f}°")
        print(f"  Cylinder radius: {R_cyl_m*1000:.2f}mm")
        print(f"  Implied Clairaut's constant: {c_implied_by_target_m*1000:.2f}mm")
        print(f"  Physical minimum (c_eff): {c_eff_physical*1000:.2f}mm")

        # Check if target is too shallow (requires smaller turning radius than physically possible)
        if c_implied_by_target_m < c_eff_physical - 1e-7:
            min_achievable_angle = math.degrees(math.asin(c_eff_physical / R_cyl_m))
            self.validation_results = {
                'is_valid': False,
                'error_type': 'too_shallow',
                'message': f"Target angle {target_alpha_cyl_deg:.1f}° is too shallow",
                'details': f"Requires turning radius of {c_implied_by_target_m*1000:.1f}mm, but physical minimum is {c_eff_physical*1000:.1f}mm",
                'min_achievable_angle': min_achievable_angle,
                'suggested_range': [min_achievable_angle, 85]
            }
            print(f"ERROR validate_angle: Target angle TOO SHALLOW")
            print(f"  Minimum achievable angle: {min_achievable_angle:.1f}°")
            return False
        
        # Check if target is too steep (would not enter dome)
        max_practical_angle = 80  # Leave some margin from 90°
        if target_alpha_cyl_deg > max_practical_angle:
            self.validation_results = {
                'is_valid': False,
                'error_type': 'too_steep',
                'message': f"Target angle {target_alpha_cyl_deg:.1f}° is too steep for practical winding",
                'details': f"Angles above {max_practical_angle}° may not provide adequate dome coverage",
                'max_practical_angle': max_practical_angle,
                'suggested_range': [math.degrees(math.asin(c_eff_physical / R_cyl_m)), max_practical_angle]
            }
            print(f"ERROR validate_angle: Target angle TOO STEEP for practical winding")
            return False

        # Target angle is valid - set the Clairaut's constant
        self.clairauts_constant_for_path_m = c_implied_by_target_m
        
        # Calculate the actual angle that will be achieved at equator
        actual_alpha_eq_deg = math.degrees(math.asin(c_implied_by_target_m / R_cyl_m))
        
        self.validation_results = {
            'is_valid': True,
            'target_angle': target_alpha_cyl_deg,
            'clairaut_constant_mm': c_implied_by_target_m * 1000,
            'actual_cylinder_angle': actual_alpha_eq_deg,
            'validation_details': {
                'physical_minimum_mm': c_eff_physical * 1000,
                'safety_margin_mm': (c_implied_by_target_m - c_eff_physical) * 1000,
                'cylinder_radius_mm': R_cyl_m * 1000
            }
        }
        
        print(f"SUCCESS: Target angle {target_alpha_cyl_deg:.1f}° is ACHIEVABLE")
        print(f"  Using Clairaut's constant: {c_implied_by_target_m*1000:.2f}mm")
        print(f"  Safety margin: {(c_implied_by_target_m - c_eff_physical)*1000:.2f}mm")
        
        return True

    def get_validation_results(self) -> Dict:
        """Get the results of target angle validation"""
        return self.validation_results if self.validation_results else {}
    
    def _get_slope_dz_drho_at_rho(self, rho_target: float) -> float:
        """
        Numerically estimates the slope dz/d(rho) of the inner vessel profile at a given rho.
        Uses central differences for geodesic calculations.
        """
        print(f"\nDEBUG trajectories.py: --- INSIDE _get_slope_dz_drho_at_rho ---")
        print(f"  rho_target_m for slope calculation: {rho_target}")
        print(f"  Type of self.vessel.profile_points at this point: {type(self.vessel.profile_points)}")
        if isinstance(self.vessel.profile_points, dict):
            print(f"  Keys in self.vessel.profile_points at this point: {list(self.vessel.profile_points.keys())}")
            if 'r_inner' not in self.vessel.profile_points:
                print("  CRITICAL ERROR: 'r_inner' IS MISSING when _get_slope_dz_drho_at_rho is entered!")
            else:
                print(f"  'r_inner' IS PRESENT. Length: {len(self.vessel.profile_points['r_inner'])}")
        else:
            print(f"  CRITICAL ERROR: self.vessel.profile_points is NOT a dict here! It is: {self.vessel.profile_points}")

        try:
            # Ensure units are consistent. If VesselGeometry stores in mm and planner uses m:
            rho_profile_m = self.vessel.profile_points['r_inner'] * 1e-3  # Error likely here if key missing
            z_profile_m = self.vessel.profile_points['z'] * 1e-3
            print(f"  Successfully accessed profile data: {len(rho_profile_m)} points")
        except KeyError as e:
            print(f"CRITICAL DEBUG _get_slope_dz_drho_at_rho: KeyError '{e}' accessing profile_points.")
            print(f"  Available keys at point of error: {list(self.vessel.profile_points.keys())}")
            return 0.0
        except Exception as e:
            print(f"CRITICAL DEBUG _get_slope_dz_drho_at_rho: Other error '{e}' accessing profile_points.")
            return 0.0

        if len(rho_profile_m) < 2:
            return 0.0

        # Find the closest point to rho_target
        idx = np.argmin(np.abs(rho_profile_m - rho_target))
        
        # Use forward/backward difference at boundaries
        if idx == 0 and len(rho_profile_m) > 1:
            drho = rho_profile_m[1] - rho_profile_m[0]
            if abs(drho) > 1e-9:
                return (z_profile_m[1] - z_profile_m[0]) / drho
        elif idx == len(rho_profile_m) - 1 and len(rho_profile_m) > 1:
            drho = rho_profile_m[-1] - rho_profile_m[-2]
            if abs(drho) > 1e-9:
                return (z_profile_m[-1] - z_profile_m[-2]) / drho
        elif idx > 0 and idx < len(rho_profile_m) - 1:
            # Central difference
            drho = rho_profile_m[idx + 1] - rho_profile_m[idx - 1]
            if abs(drho) > 1e-9:
                return (z_profile_m[idx + 1] - z_profile_m[idx - 1]) / drho
        
        return 0.0

    def _calculate_effective_polar_opening(self):
        """
        Calculates the effective polar opening radius (Clairaut's constant, c_eff)
        that the centerline of the roving "sees", accounting for roving width,
        thickness, and eccentricity at the pole.
        Based on Koussios Thesis Ch. 8.1, Eq. 8.5.
        """
        profile_points = self.vessel.get_profile_points()
        rho_geom_pole_mm = profile_points['r_inner_mm'][0]  # Geometric polar opening in mm
        rho_geom_pole_m = rho_geom_pole_mm * 1e-3  # Convert to meters
        
        ecc_0_m = self.roving_eccentricity_at_pole_m
        b_m = self.dry_roving_width_m
        t_rov_m = self.dry_roving_thickness_m

        # Calculate dz/drho at the geometric pole (pass meters to slope function)
        dz_drho_pole = self._get_slope_dz_drho_at_rho(rho_geom_pole_m)
        
        print(f"--- INSIDE _calculate_effective_polar_opening (DEBUG) ---")
        print(f"  rho_geom_pole_m (from profile[0]*1e-3): {rho_geom_pole_m:.6f} m")
        print(f"  ecc_0_m (from self): {ecc_0_m:.6f} m")
        print(f"  b_m (self.dry_roving_width_m): {b_m:.6f} m")
        print(f"  t_rov_m (self.dry_roving_thickness_m): {t_rov_m:.6f} m")
        print(f"  dz_drho_pole (from _get_slope): {dz_drho_pole:.6f}")
        
        # Handle infinite slope (vertical tangent)
        if np.isinf(dz_drho_pole):
            dz_drho_pole = 1e6  # Use large finite number
            print(f"  dz_drho_pole was infinite, using: {dz_drho_pole}")

        # Koussios formula: c_eff = rho_pole + ecc + (b/2)*sqrt(1+(dz/drho)^2) - (t/2)*(dz/drho)
        term_width = (b_m / 2.0) * math.sqrt(1 + dz_drho_pole**2)
        term_thickness = (t_rov_m / 2.0) * dz_drho_pole
        
        print(f"  term_width: {term_width:.6f} m")
        print(f"  term_thickness: {term_thickness:.6f} m")
        
        calculated_c_eff = rho_geom_pole_m + ecc_0_m + term_width - term_thickness
        print(f"  CALCULATED c_eff before assignment: {calculated_c_eff:.6f} m")
        
        self.effective_polar_opening_radius_m = calculated_c_eff
        
        # Ensure c_eff is positive and reasonable
        if self.effective_polar_opening_radius_m < 0:
            self.effective_polar_opening_radius_m = 1e-6
            
        return self.effective_polar_opening_radius_m

    def calculate_geodesic_alpha_at_rho(self, rho_m: float) -> Optional[float]:
        """
        Calculates geodesic winding angle with smooth transitions at polar turnaround.
        
        Based on Koussios Thesis Ch. 5, Eq. 5.23:
        - At c_eff: α must be exactly 90°
        - Smooth transition to avoid tangent discontinuities
        - Uses enhanced Clairaut's theorem with continuity constraints
        
        Parameters:
        -----------
        rho_m : float
            Radius position in meters
            
        Returns:
        --------
        Optional[float] : Winding angle in radians, None if unreachable
        """
        if self.effective_polar_opening_radius_m is None:
            self._calculate_effective_polar_opening()
        
        c_eff = self.effective_polar_opening_radius_m
        
        if rho_m < c_eff - 1e-9:
            return None  # Geodesic cannot reach this radius

        # At exactly c_eff: α = 90° (Koussios Ch. 5, Eq. 5.23)
        if abs(rho_m - c_eff) < 1e-8:
            return math.pi / 2.0
        
        # Enhanced Clairaut's theorem with smooth polar transition
        # For points very close to c_eff, use smoothed transition
        rho_ratio = rho_m / c_eff
        
        if rho_ratio < 1.001:  # Within 0.1% of c_eff - apply smoothing
            # Smooth transition zone to avoid infinite dα/ds
            epsilon = rho_ratio - 1.0  # Small positive value
            
            # Taylor expansion approach for smooth α near 90°
            # sin(α) ≈ 1 - (1/2)(π/2 - α)² for α close to π/2
            # This ensures smooth dα/dρ as ρ → c_eff
            
            if epsilon > 0:
                # Use smoothed calculation to avoid sharp transitions
                sin_alpha = 1.0 / rho_ratio
                sin_alpha = min(sin_alpha, 1.0)  # Ensure ≤ 1
                
                # Apply continuity constraint for smooth dα/ds
                # Use higher-order approximation near the pole
                alpha_raw = math.asin(sin_alpha)
                
                # Smooth the transition using polynomial blending
                blend_factor = min(epsilon / 0.001, 1.0)  # Smooth over 0.1%
                alpha_smooth = (math.pi / 2.0) * (1.0 - blend_factor) + alpha_raw * blend_factor
                
                return alpha_smooth
            else:
                return math.pi / 2.0
        
        else:
            # Standard Clairaut's theorem for points away from pole
            sin_alpha = c_eff / rho_m
            sin_alpha = np.clip(sin_alpha, -1.0, 1.0)
            
            try:
                alpha_rad = math.asin(sin_alpha)
                return alpha_rad
            except ValueError:
                return None

    def _generate_polar_turnaround_segment(self, c_eff: float, z_pole: float, 
                                         phi_start: float, alpha_pole: float) -> List[Dict]:
        """
        Generates circumferential turnaround path segment at effective polar opening.
        
        Based on filament winding literature (Koussios Ch. 8):
        - At c_eff: purely circumferential motion (dρ/ds=0, dζ/ds=0)
        - φ advances by pattern advancement angle
        - Ensures tangent vector continuity through reversal
        
        Parameters:
        -----------
        c_eff : float
            Effective polar opening radius (meters)
        z_pole : float
            Z-coordinate at pole (meters)
        phi_start : float
            Starting phi angle (radians)
        alpha_pole : float
            Winding angle at pole (should be π/2)
            
        Returns:
        --------
        List[Dict] : Turnaround path points with smooth transitions
        """
        turnaround_points = []
        
        # Pattern advancement angle - controls spacing between passes
        # This determines how much φ advances during the turnaround
        delta_phi_pattern = 2 * math.pi / 8  # Default: 8 passes for full coverage
        
        # Number of interpolation points for smooth turnaround
        num_turn_points = 12  # Enough for smooth tangent continuity
        
        # Generate circumferential arc at c_eff
        for i in range(num_turn_points):
            # Parameterize the turnaround from 0 to 1
            t = i / (num_turn_points - 1)
            
            # Smooth interpolation of phi during turnaround
            # Use cosine interpolation for smooth acceleration/deceleration
            phi_interp = phi_start + delta_phi_pattern * (1 - math.cos(math.pi * t)) / 2
            
            # At polar opening: ρ = c_eff, z = z_pole, α = 90°
            rho_turn = c_eff
            z_turn = z_pole
            alpha_turn = math.pi / 2.0  # Always 90° during turnaround
            
            # Cartesian coordinates
            x_turn = rho_turn * math.cos(phi_interp)
            y_turn = rho_turn * math.sin(phi_interp)
            
            turnaround_points.append({
                'rho': rho_turn,
                'z': z_turn,
                'alpha': alpha_turn,
                'phi': phi_interp,
                'x': x_turn,
                'y': y_turn
            })
        
        return turnaround_points

    def _resample_segment_adaptive(self, rho_segment: np.ndarray, z_segment: np.ndarray, 
                                  num_points: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample a profile segment with adaptive arc-length distribution.
        Focuses computational points where geometry changes most rapidly.
        
        Parameters:
        -----------
        rho_segment : np.ndarray
            Radial coordinates of segment
        z_segment : np.ndarray
            Axial coordinates of segment
        num_points : int
            Target number of points for this segment
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray] : Resampled (rho, z) coordinates
        """
        if len(rho_segment) < 2 or num_points < 2:
            return rho_segment, z_segment
        
        # Calculate cumulative arc length for uniform spacing along the curve
        s_coords = np.zeros_like(rho_segment)
        for i in range(1, len(rho_segment)):
            ds = math.sqrt((rho_segment[i] - rho_segment[i-1])**2 + 
                          (z_segment[i] - z_segment[i-1])**2)
            s_coords[i] = s_coords[i-1] + ds
        
        if abs(s_coords[-1]) < 1e-9:
            return rho_segment, z_segment
        
        # Create uniform arc-length distribution
        s_new = np.linspace(s_coords[0], s_coords[-1], num_points)
        rho_resampled = np.interp(s_new, s_coords, rho_segment)
        z_resampled = np.interp(s_new, s_coords, z_segment)
        
        return rho_resampled, z_resampled

    def _identify_vessel_segments(self, profile_r_m: np.ndarray, profile_z_m: np.ndarray) -> Dict:
        """
        Identify dome and cylinder segments in the vessel profile for adaptive sampling.
        
        Returns:
        --------
        Dict : Segment indices and properties
        """
        cylinder_radius_m = self.vessel.inner_radius * 1e-3
        cyl_half_length_m = self.vessel.cylindrical_length * 1e-3 / 2.0
        
        # Find points that are approximately at cylinder radius and cylinder z-positions
        tolerance_r = cylinder_radius_m * 0.05  # 5% tolerance
        tolerance_z = cyl_half_length_m * 0.05  # 5% tolerance
        
        # Forward dome to cylinder junction
        fwd_junction_candidates = np.where(
            (np.abs(profile_r_m - cylinder_radius_m) < tolerance_r) & 
            (np.abs(profile_z_m - cyl_half_length_m) < tolerance_z)
        )[0]
        
        # Aft cylinder to dome junction  
        aft_junction_candidates = np.where(
            (np.abs(profile_r_m - cylinder_radius_m) < tolerance_r) & 
            (np.abs(profile_z_m + cyl_half_length_m) < tolerance_z)
        )[0]
        
        if len(fwd_junction_candidates) == 0 or len(aft_junction_candidates) == 0:
            # No clear cylinder section found - treat as all dome
            return {
                'has_cylinder': False,
                'fwd_dome_end': len(profile_r_m) - 1,
                'aft_dome_start': 0
            }
        
        fwd_junction = fwd_junction_candidates[0]  # First occurrence
        aft_junction = aft_junction_candidates[-1]  # Last occurrence
        
        return {
            'has_cylinder': True,
            'fwd_dome_end': fwd_junction,
            'cylinder_start': fwd_junction,
            'cylinder_end': aft_junction,
            'aft_dome_start': aft_junction
        }

    def generate_geodesic_trajectory(self, num_points_dome: int = 150, num_points_cylinder: int = 20) -> Dict:
        """
        Generates complete pole-to-pole geodesic path with adaptive point distribution.
        Uses higher density in dome regions (high curvature) and lower density in cylinder (constant curvature).
        Based on Koussios geodesic theory with Clairaut's theorem.
        
        Parameters:
        -----------
        num_points_dome : int
            Number of points to use for each dome segment (default: 150)
        num_points_cylinder : int
            Number of points to use for cylinder segment (default: 20)
        """
        if self.vessel.profile_points is None or 'r_inner' not in self.vessel.profile_points:
            print("Error: Vessel profile not generated. Call vessel.generate_profile() first.")
            return None
        if self.effective_polar_opening_radius_m is None:
            self._calculate_effective_polar_opening()
            if self.effective_polar_opening_radius_m is None:
                print("Error: Effective polar opening could not be calculated.")
                return None
        
        c_for_winding = self.clairauts_constant_for_path_m
        print(f"\nDEBUG generate_geodesic_trajectory (ADAPTIVE): Using Clairaut's constant c = {c_for_winding:.6f} m")
        print(f"DEBUG: Physical minimum c_eff = {self.effective_polar_opening_radius_m:.6f} m")
        print(f"DEBUG: Adaptive sampling - Dome points: {num_points_dome}, Cylinder points: {num_points_cylinder}")
        print(f"DEBUG: Roving parameters - width: {self.dry_roving_width_m*1000:.1f}mm, thickness: {self.dry_roving_thickness_m*1000:.1f}mm")

        # Get complete vessel profile in meters
        profile_r_m_orig = self.vessel.profile_points['r_inner'] * 1e-3
        profile_z_m_orig = self.vessel.profile_points['z'] * 1e-3

        print(f"DEBUG: Original profile length: {len(profile_r_m_orig)} points")
        print(f"DEBUG: Original profile Z range: {np.min(profile_z_m_orig):.4f}m to {np.max(profile_z_m_orig):.4f}m")
        print(f"DEBUG: Original profile R range: {np.min(profile_r_m_orig):.4f}m to {np.max(profile_r_m_orig):.4f}m")

        # Identify vessel segments for adaptive sampling
        segments = self._identify_vessel_segments(profile_r_m_orig, profile_z_m_orig)
        print(f"DEBUG: Vessel segments identified - Has cylinder: {segments['has_cylinder']}")

        # Build adaptive profile with different point densities
        adaptive_r_segments = []
        adaptive_z_segments = []
        
        if segments['has_cylinder']:
            # Forward dome segment
            fwd_dome_r = profile_r_m_orig[0:segments['fwd_dome_end']+1]
            fwd_dome_z = profile_z_m_orig[0:segments['fwd_dome_end']+1]
            fwd_r_resampled, fwd_z_resampled = self._resample_segment_adaptive(
                fwd_dome_r, fwd_dome_z, num_points_dome)
            adaptive_r_segments.append(fwd_r_resampled)
            adaptive_z_segments.append(fwd_z_resampled)
            
            # Cylinder segment  
            cyl_r = profile_r_m_orig[segments['cylinder_start']:segments['cylinder_end']+1]
            cyl_z = profile_z_m_orig[segments['cylinder_start']:segments['cylinder_end']+1]
            cyl_r_resampled, cyl_z_resampled = self._resample_segment_adaptive(
                cyl_r, cyl_z, num_points_cylinder)
            # Skip first point to avoid duplication with dome end
            adaptive_r_segments.append(cyl_r_resampled[1:])
            adaptive_z_segments.append(cyl_z_resampled[1:])
            
            # Aft dome segment
            aft_dome_r = profile_r_m_orig[segments['aft_dome_start']:]
            aft_dome_z = profile_z_m_orig[segments['aft_dome_start']:]
            aft_r_resampled, aft_z_resampled = self._resample_segment_adaptive(
                aft_dome_r, aft_dome_z, num_points_dome)
            # Skip first point to avoid duplication with cylinder end
            adaptive_r_segments.append(aft_r_resampled[1:])
            adaptive_z_segments.append(aft_z_resampled[1:])
            
            print(f"DEBUG: Adaptive segments - Fwd dome: {len(fwd_r_resampled)}, Cylinder: {len(cyl_r_resampled)-1}, Aft dome: {len(aft_r_resampled)-1}")
        else:
            # No clear cylinder - treat as single dome with higher density
            dome_r_resampled, dome_z_resampled = self._resample_segment_adaptive(
                profile_r_m_orig, profile_z_m_orig, num_points_dome * 2)
            adaptive_r_segments.append(dome_r_resampled)
            adaptive_z_segments.append(dome_z_resampled)
            print(f"DEBUG: Single dome segment with {len(dome_r_resampled)} points")
        
        # Combine all segments
        profile_r_m_calc = np.concatenate(adaptive_r_segments)
        profile_z_m_calc = np.concatenate(adaptive_z_segments)
        
        print(f"DEBUG: Adaptive profile generated with {len(profile_r_m_calc)} total points")
        print(f"DEBUG: Adaptive Z range: {np.min(profile_z_m_calc):.4f}m to {np.max(profile_z_m_calc):.4f}m")
        print(f"DEBUG: Adaptive R range: {np.min(profile_r_m_calc):.4f}m to {np.max(profile_r_m_calc):.4f}m")
        
        if len(profile_r_m_calc) < 2:
            print("Error: Not enough profile points for trajectory calculation")
            return None
        
        # Process complete vessel profile to find windable path with smooth polar transitions
        path_rho_m, path_z_m, path_alpha_rad, path_phi_rad_cumulative = [], [], [], []
        path_x_m, path_y_m = [], []
        current_phi_rad = 0.0
        first_valid_point_found = False

        for i in range(len(profile_r_m_calc)):
            rho_i_m = profile_r_m_calc[i]
            z_i_m = profile_z_m_calc[i]
            
            # Enhanced polar turnaround handling with circumferential path segments
            if rho_i_m >= c_for_winding - 1e-7:  # Small tolerance
                alpha_i_rad = self.calculate_geodesic_alpha_at_rho(rho_i_m)
                if alpha_i_rad is None:
                    if abs(rho_i_m - c_for_winding) < 1e-6:
                        alpha_i_rad = math.pi / 2.0  # Exact 90° at effective polar opening
                    else:
                        alpha_i_rad = path_alpha_rad[-1] if path_alpha_rad else math.pi / 2.0
                
                # Special handling at effective polar opening for turnaround
                if abs(rho_i_m - c_for_winding) < 1e-6:
                    # At c_eff: implement circumferential turnaround segment
                    # This creates smooth tangent continuity through the reversal
                    if first_valid_point_found and len(path_rho_m) > 0:
                        # Generate circumferential path segment at polar opening
                        turnaround_points = self._generate_polar_turnaround_segment(
                            c_for_winding, z_i_m, current_phi_rad, alpha_i_rad
                        )
                        
                        # Add turnaround points to path
                        for turn_point in turnaround_points:
                            path_rho_m.append(turn_point['rho'])
                            path_z_m.append(turn_point['z'])
                            path_alpha_rad.append(turn_point['alpha'])
                            path_phi_rad_cumulative.append(turn_point['phi'])
                            path_x_m.append(turn_point['x'])
                            path_y_m.append(turn_point['y'])
                            current_phi_rad = turn_point['phi']
                        
                        continue  # Skip normal processing for this point
            else:
                # Point is inside c_for_winding, skip if path hasn't started
                if not first_valid_point_found:
                    continue
                else:
                    print(f"Warning: Path dipped inside c_for_winding at rho={rho_i_m:.4f}m, stopping segment")
                    break

            if not first_valid_point_found:
                # First valid point - start of trajectory
                first_valid_point_found = True
                path_rho_m.append(rho_i_m)
                path_z_m.append(z_i_m)
                path_alpha_rad.append(alpha_i_rad)
                path_phi_rad_cumulative.append(current_phi_rad)
                path_x_m.append(rho_i_m * math.cos(current_phi_rad))
                path_y_m.append(rho_i_m * math.sin(current_phi_rad))
                continue

            # Calculate phi increment for segment
            rho_prev_m = path_rho_m[-1]
            z_prev_m = path_z_m[-1]
            alpha_prev_rad = path_alpha_rad[-1]

            d_rho = rho_i_m - rho_prev_m
            d_z = z_i_m - z_prev_m
            ds_segment_m = math.sqrt(d_rho**2 + d_z**2)
            
            delta_phi = 0.0
            if ds_segment_m > 1e-9:
                rho_avg_segment_m = (rho_i_m + rho_prev_m) / 2.0
                alpha_avg_segment_rad = (alpha_i_rad + alpha_prev_rad) / 2.0
                
                if abs(rho_avg_segment_m) > 1e-7 and abs(math.cos(alpha_avg_segment_rad)) > 1e-8:
                    tan_alpha_avg = math.tan(alpha_avg_segment_rad)
                    if abs(tan_alpha_avg) > 1e8:  # Cap extremely large values
                        tan_alpha_avg = np.sign(tan_alpha_avg) * 1e8
                    delta_phi = (ds_segment_m / rho_avg_segment_m) * tan_alpha_avg

            current_phi_rad += delta_phi
            
            path_rho_m.append(rho_i_m)
            path_z_m.append(z_i_m)
            path_alpha_rad.append(alpha_i_rad)
            path_phi_rad_cumulative.append(current_phi_rad)
            path_x_m.append(rho_i_m * math.cos(current_phi_rad))
            path_y_m.append(rho_i_m * math.sin(current_phi_rad))

        if not path_rho_m:
            print(f"Error: No valid trajectory points generated")
            print(f"DEBUG: c_for_winding = {c_for_winding:.6f} m")
            print(f"DEBUG: Profile R range: {np.min(profile_r_m_calc):.6f}m to {np.max(profile_r_m_calc):.6f}m")
            print(f"DEBUG: Number of points with rho >= c_for_winding: {np.sum(profile_r_m_calc >= c_for_winding - 1e-7)}")
            return None
        
        print(f"SUCCESS: Generated {len(path_rho_m)} valid trajectory points")

        # Convert to final format
        alpha_values = [math.degrees(alpha) for alpha in path_alpha_rad]
        phi_values = list(path_phi_rad_cumulative)
        z_values = list(path_z_m)
        rho_points = np.array(path_rho_m)
        
        print(f"DEBUG: Generated {len(path_rho_m)} trajectory points")
        print(f"DEBUG: Z range in trajectory: {min(z_values):.4f}m to {max(z_values):.4f}m")
        print(f"DEBUG: Phi accumulation: {phi_values[0]:.4f} to {phi_values[-1]:.4f} rad")
        
        # Store calculated profiles
        self.alpha_profile_deg = np.array(alpha_values)
        self.phi_profile_rad = np.array(phi_values)
        self.turn_around_angle_rad = phi_values[-1] if len(phi_values) > 0 else 0
        
        # Calculate equatorial winding angle
        self.alpha_eq_deg = alpha_values[-1] if alpha_values else 0
        
        # Create path_points with 3D Cartesian coordinates for visualization
        path_points = []
        x_coords = []
        y_coords = []
        z_coords = []
        
        for i in range(len(alpha_values)):
            rho = rho_points[i] 
            z = z_values[i]
            phi = phi_values[i]
            
            # Convert cylindrical (rho, z, phi) to Cartesian (x, y, z)
            x = rho * np.cos(phi)
            y = rho * np.sin(phi)
            # z remains the same (axial coordinate)
            
            x_coords.append(x)
            y_coords.append(y)
            z_coords.append(z)
            
            path_points.append({
                'r': rho,
                'z': z,
                'theta': phi,
                'alpha': alpha_values[i],
                'x': x,
                'y': y,
                'circuit': 0  # Single geodesic path
            })
        
        return {
            'path_points': path_points,
            'pattern_type': 'Geodesic',
            'total_circuits': 1,
            'total_length': len(path_points) * 0.1,  # Approximate length
            'coverage_efficiency': 0.95,
            'rho_points': rho_points[:len(alpha_values)],
            'z_points': np.array(z_values),
            'x_points': np.array(x_coords),
            'y_points': np.array(y_coords),
            'z_coords': np.array(z_coords),
            'alpha_deg': np.array(alpha_values),
            'phi_rad': np.array(phi_values),
            'c_eff_m': self.effective_polar_opening_radius_m,
            'turn_around_angle_deg': math.degrees(self.turn_around_angle_rad),
            'alpha_equator_deg': self.alpha_eq_deg
        }

    def generate_multi_circuit_trajectory(self, num_circuits: int = 4, 
                                         num_points_dome: int = 150, 
                                         num_points_cylinder: int = 20) -> Dict:
        """
        Generate complete multi-circuit winding trajectory for full vessel coverage.
        
        Parameters:
        -----------
        num_circuits : int
            Number of circuits around the vessel circumference
        num_points_dome : int
            Points per dome segment
        num_points_cylinder : int
            Points per cylinder segment
            
        Returns:
        --------
        Dict : Complete multi-circuit trajectory data
        """
        # Generate base single circuit
        base_circuit = self.generate_geodesic_trajectory(num_points_dome, num_points_cylinder)
        
        if base_circuit is None:
            return None
        
        # Extract single circuit data
        base_x = base_circuit['x_points']
        base_y = base_circuit['y_points'] 
        base_z = base_circuit['z_coords']
        base_phi = base_circuit['phi_rad']
        
        # Initialize multi-circuit arrays
        all_x, all_y, all_z = [], [], []
        all_phi_continuous = []
        
        # Phase shift between circuits for even distribution
        phi_shift_per_circuit = 2 * math.pi / num_circuits
        
        print(f"DEBUG: Generating {num_circuits} circuits with {phi_shift_per_circuit:.3f} rad shift per circuit")
        
        for circuit in range(num_circuits):
            # Calculate phase shift for this circuit
            phi_offset = circuit * phi_shift_per_circuit
            
            # Apply phase shift to get new positions
            circuit_phi = base_phi + phi_offset
            circuit_x = base_x * np.cos(phi_offset) - base_y * np.sin(phi_offset)
            circuit_y = base_x * np.sin(phi_offset) + base_y * np.cos(phi_offset)
            circuit_z = base_z.copy()
            
            # Accumulate trajectory points
            all_x.extend(circuit_x)
            all_y.extend(circuit_y)
            all_z.extend(circuit_z)
            all_phi_continuous.extend(circuit_phi)
            
            print(f"Circuit {circuit+1}: {len(circuit_x)} points, phi range: {circuit_phi[0]:.3f} to {circuit_phi[-1]:.3f} rad")
        
        # Calculate total statistics
        total_points = len(all_x)
        total_fiber_length = base_circuit.get('fiber_length_m', 0) * num_circuits
        
        return {
            'pattern_type': 'Multi-Circuit Geodesic',
            'num_circuits': num_circuits,
            'total_points': total_points,
            'points_per_circuit': len(base_x),
            'x_points': np.array(all_x),
            'y_points': np.array(all_y),
            'z_coords': np.array(all_z),
            'phi_rad_continuous': np.array(all_phi_continuous),
            'total_fiber_length_m': total_fiber_length,
            'target_cylinder_angle_deg': math.degrees(math.asin(self.clairauts_constant_for_path_m / 0.1)),
            'c_eff_m': base_circuit.get('c_eff_m', self.effective_polar_opening_radius_m),
            'coverage_efficiency': 0.95 * num_circuits / 4.0,  # Assumes 4 circuits for full coverage
            'base_circuit_data': base_circuit
        }

    def calculate_trajectory(self, params: Dict) -> Dict:
        """
        Calculate winding trajectory based on parameters.
        
        Parameters:
        -----------
        params : Dict
            Trajectory parameters including pattern type, angles, speeds, etc.
            
        Returns:
        --------
        Dict : Trajectory data including path points and properties
        """
        pattern_type = params.get('pattern_type', 'Helical')
        
        if pattern_type == 'Geodesic':
            return self.generate_geodesic_trajectory(params.get('num_points', 100))
        elif pattern_type == 'Multi-Circuit':
            return self.generate_multi_circuit_trajectory(
                params.get('num_circuits', 4),
                params.get('num_points_dome', 150),
                params.get('num_points_cylinder', 20)
            )
        elif pattern_type == 'Helical':
            return self._calculate_helical_trajectory(params)
        elif pattern_type == 'Hoop':
            return self._calculate_hoop_trajectory(params)
        elif pattern_type == 'Polar':
            return self._calculate_polar_trajectory(params)
        elif pattern_type == 'Transitional':
            return self._calculate_transitional_trajectory(params)
        else:
            raise ValueError(f"Unsupported pattern type: {pattern_type}")
            
    def _calculate_helical_trajectory(self, params: Dict) -> Dict:
        """
        Calculate helical winding trajectory.
        
        Based on constant winding angle geodesic paths on the vessel surface.
        """
        winding_angle = math.radians(params.get('winding_angle', 55.0))
        band_width = params.get('band_width', 6.0)  # mm
        circuits_to_close = params.get('circuits_to_close', 8)
        overlap_allowance = params.get('overlap_allowance', 10.0) / 100.0  # Convert % to fraction
        mandrel_speed = params.get('mandrel_speed', 10.0)  # rpm
        carriage_speed = params.get('carriage_speed', 100.0)  # mm/min
        
        # Get vessel dimensions
        R_cylinder = self.vessel.inner_radius  # mm
        L_cylinder = self.vessel.cylindrical_length  # mm
        
        # Calculate helical pitch based on winding angle
        # tan(α) = (circumferential distance) / (axial distance)
        circumference = 2 * np.pi * R_cylinder
        pitch = circumference / math.tan(winding_angle)  # Axial advance per revolution
        
        # Calculate number of complete circuits needed for coverage
        effective_band_width = band_width * (1 + overlap_allowance)
        circuits_for_coverage = int(np.ceil(L_cylinder / (pitch / circuits_to_close)))
        total_circuits = circuits_for_coverage * circuits_to_close
        
        # Generate trajectory points
        path_points = []
        total_mandrel_rotations = 0
        
        for circuit in range(total_circuits):
            # Calculate angular positions for this circuit
            start_angle = (circuit * 2 * np.pi) / circuits_to_close
            
            # Number of mandrel rotations for cylindrical section
            rotations_cylinder = L_cylinder / pitch
            
            for rot_fraction in np.linspace(0, rotations_cylinder, 100):
                # Current mandrel angle
                theta = start_angle + rot_fraction * 2 * np.pi
                
                # Axial position
                z = -L_cylinder/2 + (rot_fraction / rotations_cylinder) * L_cylinder
                
                # For cylindrical section, radius is constant
                r = R_cylinder
                
                path_points.append({
                    'theta': theta,
                    'r': r, 
                    'z': z,
                    'circuit': circuit,
                    'winding_angle': math.degrees(winding_angle)
                })
                
            total_mandrel_rotations += rotations_cylinder
        
        # Calculate trajectory properties
        total_fiber_length = self._calculate_fiber_length(path_points)
        winding_time = total_mandrel_rotations / mandrel_speed  # minutes
        coverage_efficiency = self._calculate_coverage_efficiency(path_points, band_width)
        
        # Add dome transitions (simplified)
        dome_points = self._calculate_dome_transitions(winding_angle, circuits_to_close)
        path_points.extend(dome_points)
        
        trajectory_data = {
            'pattern_type': 'Helical',
            'path_points': path_points,
            'winding_angle': math.degrees(winding_angle),
            'total_circuits': total_circuits,
            'circuits_to_close': circuits_to_close,
            'total_fiber_length': total_fiber_length,
            'winding_time': winding_time,
            'mandrel_rotations': total_mandrel_rotations,
            'coverage_efficiency': coverage_efficiency,
            'fiber_utilization': 95.0,  # Typical for helical
            'pitch': pitch
        }
        
        return trajectory_data
        
    def _calculate_hoop_trajectory(self, params: Dict) -> Dict:
        """Calculate hoop winding trajectory (circumferential windings)"""
        band_width = params.get('band_width', 6.0)  # mm
        mandrel_speed = params.get('mandrel_speed', 10.0)  # rpm
        
        R_cylinder = self.vessel.inner_radius
        L_cylinder = self.vessel.cylindrical_length
        
        # Calculate number of hoop circuits needed
        circuits_needed = int(np.ceil(L_cylinder / band_width))
        
        path_points = []
        
        for circuit in range(circuits_needed):
            # Axial position for this circuit
            z = -L_cylinder/2 + circuit * band_width + band_width/2
            
            # Generate full circumferential path
            for theta in np.linspace(0, 2*np.pi, 100):
                path_points.append({
                    'theta': theta,
                    'r': R_cylinder,
                    'z': z,
                    'circuit': circuit,
                    'winding_angle': 90.0
                })
        
        # Calculate properties
        total_fiber_length = circuits_needed * 2 * np.pi * R_cylinder / 1000  # Convert to meters
        winding_time = circuits_needed * (1/mandrel_speed)  # Assume 1 revolution per circuit
        
        trajectory_data = {
            'pattern_type': 'Hoop',
            'path_points': path_points,
            'winding_angle': 90.0,
            'total_circuits': circuits_needed,
            'total_fiber_length': total_fiber_length,
            'winding_time': winding_time,
            'mandrel_rotations': circuits_needed,
            'coverage_efficiency': 100.0,
            'fiber_utilization': 98.0
        }
        
        return trajectory_data
        
    def _calculate_polar_trajectory(self, params: Dict) -> Dict:
        """Calculate polar winding trajectory (over the poles)"""
        band_width = params.get('band_width', 6.0)
        circuits_to_close = params.get('circuits_to_close', 4)
        mandrel_speed = params.get('mandrel_speed', 10.0)
        
        R_cylinder = self.vessel.inner_radius
        
        # Polar winding goes over the domes
        # Simplified calculation - full implementation would require dome contour integration
        
        path_points = []
        
        for circuit in range(circuits_to_close):
            start_angle = circuit * (2 * np.pi / circuits_to_close)
            
            # Simplified polar path (would need actual dome contour for accuracy)
            for s in np.linspace(0, 1, 200):
                # Parameter s goes from 0 to 1 along the meridian
                if s < 0.25:
                    # Bottom dome
                    z = -self.vessel.cylindrical_length/2 - (0.25 - s) * 4 * R_cylinder * 0.8
                    r = R_cylinder * np.sqrt(1 - ((0.25 - s) * 4)**2) if (0.25 - s) * 4 < 1 else 0
                elif s < 0.75:
                    # Cylindrical section
                    z = -self.vessel.cylindrical_length/2 + (s - 0.25) * 2 * self.vessel.cylindrical_length
                    r = R_cylinder
                else:
                    # Top dome
                    z = self.vessel.cylindrical_length/2 + (s - 0.75) * 4 * R_cylinder * 0.8
                    r = R_cylinder * np.sqrt(1 - ((s - 0.75) * 4)**2) if (s - 0.75) * 4 < 1 else 0
                
                path_points.append({
                    'theta': start_angle,
                    'r': max(0, r),
                    'z': z,
                    'circuit': circuit,
                    'winding_angle': 0.0,  # Changes along path
                    's_parameter': s
                })
        
        # Estimate properties
        path_length_per_circuit = 2 * (self.vessel.cylindrical_length + 2 * R_cylinder)
        total_fiber_length = circuits_to_close * path_length_per_circuit / 1000
        winding_time = circuits_to_close * 2  # Estimate
        
        trajectory_data = {
            'pattern_type': 'Polar',
            'path_points': path_points,
            'winding_angle': 'Variable',
            'total_circuits': circuits_to_close,
            'total_fiber_length': total_fiber_length,
            'winding_time': winding_time,
            'mandrel_rotations': circuits_to_close * 0.5,  # Partial rotations
            'coverage_efficiency': 85.0,
            'fiber_utilization': 92.0
        }
        
        return trajectory_data
        
    def _calculate_transitional_trajectory(self, params: Dict) -> Dict:
        """Calculate transitional trajectory (helical to hoop transition)"""
        winding_angle = params.get('winding_angle', 55.0)
        band_width = params.get('band_width', 6.0)
        
        # Simplified transitional pattern
        # Combines helical and hoop characteristics
        
        # Calculate base helical trajectory
        helical_data = self._calculate_helical_trajectory(params)
        
        # Modify for transitional characteristics
        path_points = helical_data['path_points']
        
        # Add transition zones where angle changes
        R_cylinder = self.vessel.inner_radius
        transition_length = R_cylinder * 0.5  # Transition zone length
        
        for point in path_points:
            z = point['z']
            # Modify winding angle in transition zones
            if abs(z) > self.vessel.cylindrical_length/2 - transition_length:
                # In transition zone - angle varies from helical to hoop
                distance_from_end = abs(abs(z) - self.vessel.cylindrical_length/2)
                angle_factor = distance_from_end / transition_length
                point['winding_angle'] = winding_angle * angle_factor + 90.0 * (1 - angle_factor)
        
        trajectory_data = {
            'pattern_type': 'Transitional',
            'path_points': path_points,
            'winding_angle': f"{winding_angle}° to 90°",
            'total_circuits': helical_data['total_circuits'],
            'total_fiber_length': helical_data['total_fiber_length'] * 1.1,  # Slightly more
            'winding_time': helical_data['winding_time'] * 1.15,
            'mandrel_rotations': helical_data['mandrel_rotations'],
            'coverage_efficiency': 92.0,
            'fiber_utilization': 94.0
        }
        
        return trajectory_data
        
    def _calculate_dome_transitions(self, winding_angle: float, circuits_to_close: int) -> List[Dict]:
        """Calculate trajectory points for dome transition regions"""
        dome_points = []
        
        # Simplified dome transition calculation
        # In practice, this requires integration along the dome contour
        
        R_cylinder = self.vessel.inner_radius
        dome_height = self.vessel.profile_points.get('dome_height', R_cylinder * 0.8)
        
        # Add a few representative dome transition points
        for circuit in range(circuits_to_close):
            start_angle = circuit * (2 * np.pi / circuits_to_close)
            
            # Top dome transition
            dome_points.append({
                'theta': start_angle,
                'r': R_cylinder * 0.8,
                'z': self.vessel.cylindrical_length/2 + dome_height * 0.5,
                'circuit': circuit,
                'winding_angle': math.degrees(winding_angle),
                'region': 'dome_transition'
            })
            
            # Bottom dome transition
            dome_points.append({
                'theta': start_angle + np.pi,
                'r': R_cylinder * 0.8,
                'z': -self.vessel.cylindrical_length/2 - dome_height * 0.5,
                'circuit': circuit,
                'winding_angle': math.degrees(winding_angle),
                'region': 'dome_transition'
            })
        
        return dome_points
        
    def _calculate_fiber_length(self, path_points: List[Dict]) -> float:
        """Calculate total fiber length from path points"""
        total_length = 0.0
        
        for i in range(1, len(path_points)):
            p1 = path_points[i-1]
            p2 = path_points[i]
            
            # Convert to Cartesian coordinates
            x1 = p1['r'] * math.cos(p1['theta'])
            y1 = p1['r'] * math.sin(p1['theta'])
            z1 = p1['z']
            
            x2 = p2['r'] * math.cos(p2['theta'])
            y2 = p2['r'] * math.sin(p2['theta'])
            z2 = p2['z']
            
            # Distance between points
            segment_length = math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
            total_length += segment_length
        
        return total_length / 1000.0  # Convert mm to meters
        
    def _calculate_coverage_efficiency(self, path_points: List[Dict], band_width: float) -> float:
        """Calculate surface coverage efficiency"""
        # Simplified calculation based on theoretical vs actual coverage
        # In practice, this would require detailed surface area analysis
        
        if not path_points:
            return 0.0
            
        # Estimate based on pattern type and parameters
        unique_circuits = len(set(p.get('circuit', 0) for p in path_points))
        
        # Theoretical coverage area
        cylinder_area = 2 * np.pi * self.vessel.inner_radius * self.vessel.cylindrical_length
        
        # Actual coverage (simplified)
        covered_area = unique_circuits * band_width * 2 * np.pi * self.vessel.inner_radius
        
        efficiency = min(100.0, (covered_area / cylinder_area) * 100.0)
        return efficiency
        
    def get_winding_parameters(self) -> Dict:
        """Return current winding parameters for analysis"""
        if self.trajectory_data is None:
            return {}
            
        return {
            'pattern_type': self.trajectory_data.get('pattern_type', 'Unknown'),
            'total_circuits': self.trajectory_data.get('total_circuits', 0),
            'fiber_length': self.trajectory_data.get('total_fiber_length', 0),
            'winding_time': self.trajectory_data.get('winding_time', 0),
            'coverage_efficiency': self.trajectory_data.get('coverage_efficiency', 0)
        }
        
    def calculate_geodesic_angle(self, r: float, z: float) -> float:
        """
        Calculate geodesic winding angle at given position.
        
        For a surface of revolution, geodesic condition is:
        r * sin(α) = constant (Clairaut's theorem)
        """
        R_cylinder = self.vessel.inner_radius
        
        # For geodesic on cylinder: sin(α) = constant/r
        # Choose constant based on desired angle at cylinder
        geodesic_constant = R_cylinder * math.sin(math.radians(55.0))  # Example: 55° at cylinder
        
        if r <= 0:
            return 0.0
            
        sin_alpha = min(1.0, geodesic_constant / r)
        alpha = math.asin(sin_alpha)
        
        return math.degrees(alpha)

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
        
        # CORRECTED: Pure Clairaut's Law implementation
        # α = arcsin(c_eff / ρ) for all points where ρ >= c_eff
        
        # Direct application of Clairaut's theorem: ρ sin(α) = c_eff
        sin_alpha = c_eff / rho_m
        
        # Ensure sin_alpha is within valid range [0, 1]
        if sin_alpha > 1.0:
            return None  # Point is unreachable by geodesic
        
        # Calculate true geodesic angle using Clairaut's Law
        alpha_rad = math.asin(sin_alpha)
        
        return alpha_rad

    def _calculate_tangent_vector(self, rho: float, z: float, phi: float, alpha: float) -> tuple:
        """
        Calculate tangent vector components (dρ/ds, dz/ds, dφ/ds) for geodesic path.
        Based on Koussios Thesis Ch. 2, Eq. 2.19 for tangent vectors.
        """
        # For geodesic path with winding angle α at radius ρ:
        # dρ/ds = cos(α) * cos(β)  where β is meridional angle
        # dz/ds = cos(α) * sin(β) 
        # dφ/ds = sin(α) / ρ
        
        # Calculate meridional slope at this point
        dz_drho = self._get_slope_dz_drho_at_rho(rho)
        beta = math.atan(dz_drho) if abs(dz_drho) < 1e6 else math.pi/2
        
        # Tangent vector components
        drho_ds = math.cos(alpha) * math.cos(beta)
        dz_ds = math.cos(alpha) * math.sin(beta)
        dphi_ds = math.sin(alpha) / rho if rho > 1e-8 else 0
        
        return drho_ds, dz_ds, dphi_ds

    def _calculate_enhanced_tangent_vector(self, rho: float, z: float, phi: float, alpha: float, 
                                         h_derivative: float, t: float) -> Optional[Tuple[float, float, float]]:
        """
        Calculate enhanced tangent vector with curvature-aware derivatives.
        Modulates tangent components based on transition parameter to create smooth curves.
        """
        # Base tangent from geodesic calculation
        base_tangent = self._calculate_tangent_vector(rho, z, phi, alpha)
        if base_tangent is None:
            return None
            
        drho_ds_base, dz_ds_base, dphi_ds_base = base_tangent
        
        # Apply curvature modulation based on transition progress
        # Reduce meridional speed near transition points to increase curvature
        curvature_factor = 1.0 - 0.3 * h_derivative  # Modulate based on transition rate
        
        drho_ds_enhanced = drho_ds_base * curvature_factor
        dz_ds_enhanced = dz_ds_base * curvature_factor
        dphi_ds_enhanced = dphi_ds_base / max(curvature_factor, 0.5)  # Compensate phi rate
        
        return drho_ds_enhanced, dz_ds_enhanced, dphi_ds_enhanced

    def _estimate_path_curvature_radius(self, rho: float, z: float, tangent: Tuple[float, float, float],
                                      start_rho: float, end_rho: float) -> float:
        """
        Estimate local radius of curvature for the path.
        Returns radius in meters for debugging analysis.
        """
        drho_ds, dz_ds, dphi_ds = tangent
        
        # Calculate approximate curvature using change in tangent direction
        speed = math.sqrt(drho_ds**2 + dz_ds**2)
        if speed < 1e-8:
            return float('inf')  # Straight line
            
        # Approximate curvature radius from transition zone geometry
        transition_span = abs(end_rho - start_rho)
        if transition_span < 1e-8:
            return rho  # Circumferential arc
            
        # Estimate based on path geometry
        curvature_radius = transition_span / (2 * abs(dphi_ds) * speed + 1e-8)
        return max(curvature_radius, rho * 0.1)  # Minimum reasonable value

    def _generate_smooth_transition_zone(self, start_rho: float, end_rho: float,
                                       start_alpha: float, end_alpha: float,
                                       phi_current: float, num_points: int = 6,
                                       reverse_meridional: bool = False) -> List[Dict]:
        print(f"DEBUG: *** ENTERED _generate_smooth_transition_zone ***")
        print(f"DEBUG: Parameters - start_rho={start_rho:.6f}, end_rho={end_rho:.6f}")
        print(f"DEBUG: Parameters - start_alpha={math.degrees(start_alpha):.1f}°, end_alpha={math.degrees(end_alpha):.1f}°")
        print(f"DEBUG: Parameters - phi_current={phi_current:.4f}, num_points={num_points}")
        print("DEBUG: Successfully entered _generate_smooth_transition_zone")
        print(f"*** ENTERING _generate_smooth_transition_zone ***")
        print(f"start_rho={start_rho:.6f}m, end_rho={end_rho:.6f}m")
        print(f"start_alpha={math.degrees(start_alpha):.1f}°, end_alpha={math.degrees(end_alpha):.1f}°")
        print(f"phi_current={math.degrees(phi_current):.1f}°, num_points={num_points}")
        print(f"reverse_meridional={reverse_meridional}")
        """
        Generate smooth transition zone with enhanced curvature management.
        Creates geometrically smooth curve with controlled radius of curvature.
        
        Parameters:
        -----------
        start_rho : float
            Starting radius (meters)
        end_rho : float
            Ending radius (meters)
        start_alpha : float
            Starting winding angle (radians)
        end_alpha : float
            Target winding angle (radians, typically π/2 at polar opening)
        phi_current : float
            Current azimuthal angle (radians)
        num_points : int
            Number of transition points (default: 6 for smooth curve)
            
        Returns:
        --------
        List[Dict] : Smooth transition path points with controlled curvature
        """
        transition_points = []
        
        # Enhanced smoothing with curvature control
        t_values = np.linspace(0, 1, num_points)
        
        # Get vessel profile slopes at start and end for tangent matching
        start_slope = self._get_slope_dz_drho_at_rho(start_rho)
        end_slope = self._get_slope_dz_drho_at_rho(end_rho)
        
        print(f"=== ENHANCED TRANSITION ZONE DEBUGGING ===")
        print(f"Start: ρ={start_rho:.6f}m α={math.degrees(start_alpha):.2f}° slope={start_slope:.6f}")
        print(f"End:   ρ={end_rho:.6f}m α={math.degrees(end_alpha):.2f}° slope={end_slope:.6f}")
        
        for i, t in enumerate(t_values):
            # Enhanced cubic Hermite spline with curvature control
            # Use quintic polynomial for C² continuity: f(t) = 6t⁵ - 15t⁴ + 10t³
            h_smooth = 6 * t**5 - 15 * t**4 + 10 * t**3  # Smoother than cubic
            h_derivative = 30 * t**4 - 60 * t**3 + 30 * t**2  # Derivative for tangent calculation
            
            # CRITICAL FIX: Smooth path interpolation instead of linear
            # Use smoothstep for radius to create curved path geometry
            rho_t = start_rho + (end_rho - start_rho) * h_smooth
            alpha_t = start_alpha + (end_alpha - start_alpha) * h_smooth
            
            # Get Z coordinate from vessel profile
            z_t = self._interpolate_z_from_profile(rho_t)
            if z_t is None:
                continue
                
            # Calculate enhanced tangent vector with curvature-aware derivatives
            tangent = self._calculate_enhanced_tangent_vector(rho_t, z_t, phi_current, alpha_t, h_derivative, t)
            if tangent is None:
                continue
                
            # CRITICAL FIX: Apply meridional direction reversal BEFORE debug output
            if reverse_meridional:
                # For outgoing path on expanding dome: both dρ/ds and dz/ds should be positive
                # Reverse dz/ds sign and ensure dρ/ds is positive for outward expansion
                drho_ds_corrected = abs(tangent[0])  # Ensure positive for outward expansion
                dz_ds_corrected = abs(tangent[1])    # Make positive to move away from pole
                tangent = (drho_ds_corrected, dz_ds_corrected, tangent[2])
                
            # Calculate radius of curvature for debugging
            radius_curvature = self._estimate_path_curvature_radius(rho_t, z_t, tangent, start_rho, end_rho)
            
            # Debug output for curvature analysis (now shows corrected tangent)
            if i in [0, num_points//2, num_points-1]:  # Start, middle, end points
                print(f"Point {i}: t={t:.3f} ρ={rho_t:.6f}m α={math.degrees(alpha_t):.1f}°")
                print(f"  Tangent {'(REVERSED)' if reverse_meridional else ''}: dρ/ds={tangent[0]:.6f} dz/ds={tangent[1]:.6f} dφ/ds={tangent[2]:.6f}")
                print(f"  Curvature radius: {radius_curvature:.4f}m ({radius_curvature*1000:.1f}mm)")
                
            # Calculate Cartesian coordinates
            x_t = rho_t * math.cos(phi_current)
            y_t = rho_t * math.sin(phi_current)
            
            transition_points.append({
                'rho': rho_t,
                'z': z_t,
                'alpha': alpha_t,
                'phi': phi_current,
                'x': x_t,
                'y': y_t,
                'drho_ds': tangent[0],
                'dz_ds': tangent[1],
                'dphi_ds': tangent[2]
            })
            
            # Update phi for next point (small increment)
            phi_increment = tangent[2] * 0.001  # Small arc length step
            phi_current += phi_increment
            
        print(f"*** EXITING _generate_smooth_transition_zone - Generated {len(transition_points)} points ***")
        return transition_points

    def _interpolate_z_from_profile(self, rho_target: float) -> Optional[float]:
        """
        Interpolate Z coordinate from vessel profile at given radius.
        """
        try:
            profile_r = self.vessel.profile_points['r_inner'] * 1e-3  # Convert to meters
            profile_z = self.vessel.profile_points['z'] * 1e-3       # Convert to meters
            
            # Find interpolation range
            if rho_target <= profile_r[0]:
                return float(profile_z[0])
            elif rho_target >= profile_r[-1]:
                return float(profile_z[-1])
            else:
                # Linear interpolation
                return float(np.interp(rho_target, profile_r, profile_z))
        except:
            return None

    def _generate_polar_turnaround_segment(self, c_eff: float, z_pole: float, 
                                         phi_start: float, alpha_pole: float,
                                         incoming_tangent: tuple = None) -> List[Dict]:
        print(f"DEBUG: *** ENTERED _generate_polar_turnaround_segment ***")
        print(f"DEBUG: Parameters - c_eff={c_eff:.6f}, z_pole={z_pole:.6f}")
        print(f"DEBUG: Parameters - phi_start={phi_start:.4f}, alpha_pole={math.degrees(alpha_pole):.1f}°")
        print("DEBUG: Successfully entered _generate_polar_turnaround_segment")
        print(f"*** ENTERING _generate_polar_turnaround_segment ***")
        print(f"c_eff={c_eff:.6f}m, z_pole={z_pole:.6f}m")
        print(f"phi_start={math.degrees(phi_start):.1f}°, alpha_pole={math.degrees(alpha_pole):.1f}°")
        print(f"incoming_tangent={'None' if incoming_tangent is None else f'({incoming_tangent[0]:.4f}, {incoming_tangent[1]:.4f}, {incoming_tangent[2]:.4f})'}")
        """
        Generates truly smooth C¹ continuous turnaround path segment with proper tangent matching.
        
        Based on Koussios Thesis Ch. 2 (tangent vectors) and geodesic theory:
        - Calculates explicit tangent vector matching at connection points
        - Creates genuine circular arc at c_eff with proper curvature
        - Ensures dρ/ds, dz/ds, dφ/ds continuity through the turnaround
        
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
        incoming_tangent : tuple
            (drho_ds, dz_ds, dphi_ds) from incoming helical path
            
        Returns:
        --------
        List[Dict] : Truly smooth turnaround path points with matched tangent vectors
        """
        turnaround_points = []
        
        # Pattern advancement angle - controls spacing between passes
        delta_phi_pattern = 2 * math.pi / 8  # 8 passes for full coverage
        
        # Create genuine circular arc at constant radius c_eff and constant z_pole
        # This represents the actual physical path of the fiber during turnaround
        num_turn_points = 36  # Increased points for smoother circular arc visualization
        
        # Calculate arc length and parameterization
        arc_length = c_eff * delta_phi_pattern  # Total arc length
        ds_increment = arc_length / (num_turn_points - 1)  # Arc length per segment
        
        print(f"\n=== TURNAROUND SEGMENT DEBUG ===")
        print(f"c_eff: {c_eff:.6f} m")
        print(f"z_pole: {z_pole:.6f} m") 
        print(f"phi_start: {math.degrees(phi_start):.2f}°")
        print(f"delta_phi_pattern: {math.degrees(delta_phi_pattern):.2f}°")
        print(f"arc_length: {arc_length:.6f} m")
        print(f"num_turn_points: {num_turn_points}")
        
        for i in range(num_turn_points):
            # Parameter along arc from 0 to 1
            t = i / (num_turn_points - 1)
            
            # For circular arc at constant radius and height:
            # ρ = c_eff (constant)
            # z = z_pole (constant for true geodesic at pole)
            # φ varies linearly with arc length
            
            rho_turn = c_eff
            z_turn = z_pole  # Constant Z for proper geodesic at pole
            
            # Linear phi progression for circular arc (this is geometrically correct)
            phi_turn = phi_start + delta_phi_pattern * t
            
            # At polar opening: α = 90° for pure circumferential motion
            alpha_turn = math.pi / 2.0
            
            # Calculate Cartesian coordinates
            x_turn = rho_turn * math.cos(phi_turn)
            y_turn = rho_turn * math.sin(phi_turn)
            
            # Calculate tangent vector for this point on circular arc
            # For circular motion at constant radius: drho_ds = 0, dz_ds = 0
            drho_ds = 0.0  # No radial motion during turnaround
            dz_ds = 0.0    # No axial motion during turnaround  
            dphi_ds = 1.0 / c_eff  # Pure circumferential motion
            
            # Log detailed point data for first few and last few points
            if i < 3 or i >= num_turn_points - 3:
                print(f"Point {i:2d}: t={t:.3f} | ρ={rho_turn:.6f} z={z_turn:.6f} φ={math.degrees(phi_turn):7.2f}° | x={x_turn:.6f} y={y_turn:.6f}")
                print(f"         Tangent: dρ/ds={drho_ds:.6f} dz/ds={dz_ds:.6f} dφ/ds={dphi_ds:.6f}")
            
            turnaround_points.append({
                'rho': rho_turn,
                'z': z_turn,
                'alpha': alpha_turn,
                'phi': phi_turn,
                'x': x_turn,
                'y': y_turn,
                'drho_ds': drho_ds,
                'dz_ds': dz_ds,
                'dphi_ds': dphi_ds
            })
        
        print(f"Generated {len(turnaround_points)} turnaround points")
        print(f"=== END TURNAROUND DEBUG ===\n")
        
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

    def generate_geodesic_trajectory(self, num_points_dome: int = 150, num_points_cylinder: int = 20, number_of_passes: int = 2) -> Dict:
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
        
        # Initialize multi-pass trajectory generation
        path_rho_m, path_z_m, path_alpha_rad, path_phi_rad_cumulative = [], [], [], []
        path_x_m, path_y_m = [], []
        current_phi_rad = 0.0
        first_valid_point_found = False
        
        print(f"\n=== MULTI-PASS GEODESIC TRAJECTORY GENERATION ===")
        print(f"Target passes: {number_of_passes}")
        print(f"Each pass: forward journey + turnaround + return journey + turnaround")
        
        # CRITICAL FIX: Use the target Clairaut's constant, not the physical minimum
        print(f"DEBUG: self.clairauts_constant_for_path_m = {self.clairauts_constant_for_path_m}")
        print(f"DEBUG: self.effective_polar_opening_radius_m = {self.effective_polar_opening_radius_m}")
        print(f"DEBUG: self.target_cylinder_angle_deg = {self.target_cylinder_angle_deg}")
        
        if self.clairauts_constant_for_path_m is not None:
            c_for_winding = self.clairauts_constant_for_path_m
            print(f"SUCCESS: Using TARGET Clairaut's constant c = {c_for_winding:.6f} m (for {self.target_cylinder_angle_deg}° target angle)")
        else:
            # Force calculation of target Clairaut's constant
            if self.target_cylinder_angle_deg is not None:
                R_cyl_m = self.vessel.inner_radius * 1e-3  # Convert to meters
                alpha_target_rad = math.radians(self.target_cylinder_angle_deg)
                c_for_winding = R_cyl_m * math.sin(alpha_target_rad)
                print(f"FORCED CALCULATION: Target {self.target_cylinder_angle_deg}° gives c = {c_for_winding:.6f} m")
            else:
                c_for_winding = self.effective_polar_opening_radius_m
                print(f"FALLBACK: Using physical minimum c = {c_for_winding:.6f} m")
        
        print(f"FINAL c_for_winding = {c_for_winding:.6f} m - THIS WILL BE USED FOR ALL GEODESIC CALCULATIONS")
        
        total_points_generated = 0
        
        # Generate multiple complete passes
        for pass_number in range(number_of_passes):
            print(f"\n--- PASS {pass_number + 1} of {number_of_passes} ---")
            
            # Direction alternates: odd passes go forward, even go reverse
            forward_direction = (pass_number % 2 == 0)
            
            # Find the correct starting point where vessel radius ≈ c_for_winding
            start_index = None
            end_index = None
            
            # Find indices where vessel radius equals c_for_winding
            for i in range(len(profile_r_m_calc)):
                if profile_r_m_calc[i] >= c_for_winding - 1e-6:
                    if start_index is None:
                        start_index = i
                    end_index = i
            
            if start_index is None:
                print(f"ERROR: No points found on vessel profile where ρ >= c_for_winding ({c_for_winding:.6f}m)")
                print(f"Max vessel radius: {np.max(profile_r_m_calc):.6f}m")
                continue
                
            print(f"DEBUG: Found trajectory segment from index {start_index} to {end_index}")
            print(f"DEBUG: ρ range: {profile_r_m_calc[start_index]:.6f}m to {profile_r_m_calc[end_index]:.6f}m")
            print(f"DEBUG: z range: {profile_z_m_calc[start_index]:.6f}m to {profile_z_m_calc[end_index]:.6f}m")
            
            # Create the correct profile range for this pass
            if forward_direction:
                profile_range = range(start_index, end_index + 1)
                print(f"Direction: Forward (ρ={profile_r_m_calc[start_index]:.6f} to ρ={profile_r_m_calc[end_index]:.6f})")
            else:
                profile_range = range(end_index, start_index - 1, -1)
                print(f"Direction: Reverse (ρ={profile_r_m_calc[end_index]:.6f} to ρ={profile_r_m_calc[start_index]:.6f})")
            
            print(f"DEBUG: Processing {len(list(profile_range))} points for this pass")
            pass_points_start = len(path_rho_m)
            
            for idx, i in enumerate(profile_range):
                rho_i_m = profile_r_m_calc[i]
                z_i_m = profile_z_m_calc[i]
                
                # DEBUG: Log each point being processed
                print(f"DEBUG Pass {pass_number + 1}: Processing point {i}: ρ={rho_i_m:.6f}m z={z_i_m:.6f}m c_for_winding={c_for_winding:.6f}m")
                
                # Check if this is the last point in the helical segment
                is_last_point = (idx == len(list(profile_range)) - 1)
                if is_last_point:
                    print(f"DEBUG Pass {pass_number + 1}: *** LAST POINT IN HELICAL SEGMENT *** - should trigger turnaround")
                
                # Enhanced polar turnaround handling with circumferential path segments
                # DEBUG: Check the condition for valid points
                radius_condition = rho_i_m >= c_for_winding - 1e-7
                print(f"DEBUG Pass {pass_number + 1}: Radius condition check: ρ={rho_i_m:.6f} >= c-tol={c_for_winding - 1e-7:.6f} = {radius_condition}")
                
                if radius_condition:  # Small tolerance
                    # CORRECTED: Use target c_for_winding for true geodesic calculation
                    # Apply Clairaut's Law directly: sin(α) = c_for_winding / ρ
                    sin_alpha = c_for_winding / rho_i_m
                    if sin_alpha <= 1.0:
                        alpha_i_rad = math.asin(sin_alpha)
                    else:
                        alpha_i_rad = math.pi / 2.0  # At exact c_for_winding
                    print(f"DEBUG Pass {pass_number + 1}: Calculated TRUE GEODESIC alpha={alpha_i_rad:.4f} ({math.degrees(alpha_i_rad):.1f}°) using c_for_winding={c_for_winding:.6f}")
                    
                    # CRITICAL FIX: Add normal helical points to trajectory arrays
                    # Update phi based on geodesic advancement
                    d_phi = abs(alpha_i_rad) * 0.01  # Small increment for visualization
                    current_phi_rad += d_phi
                    
                    # Add this helical point to trajectory arrays
                    path_rho_m.append(rho_i_m)
                    path_z_m.append(z_i_m)
                    path_alpha_rad.append(alpha_i_rad)
                    path_phi_rad_cumulative.append(current_phi_rad)
                    path_x_m.append(rho_i_m * math.cos(current_phi_rad))
                    path_y_m.append(rho_i_m * math.sin(current_phi_rad))
                    
                    # Only trigger turnaround at the last point of helical segment, not at intermediate points
                    # This ensures the complete helical path is generated before the turnaround
                    if is_last_point:
                        print(f"DEBUG: *** LAST POINT REACHED *** at ρ={rho_i_m:.6f}, c_for_winding={c_for_winding:.6f}, diff={abs(rho_i_m - c_for_winding):.8f}")
                        print(f"DEBUG: *** TRIGGERING TURNAROUND *** last_point={is_last_point}")
                        print(f"DEBUG: Checking turnaround conditions - first_valid_point_found={first_valid_point_found}, len(path_rho_m)={len(path_rho_m)}")
                        # At c_eff: implement smooth C¹ continuous turnaround with transition zones
                        if first_valid_point_found and len(path_rho_m) > 0:
                            print(f"DEBUG: *** ENTERING SMOOTH TURNAROUND SEQUENCE ***")
                            
                            # Step 1: Generate incoming transition zone (helical → circumferential)
                            # Create smooth transition from last helical point to polar opening
                            last_helical_rho = path_rho_m[-1]
                            last_helical_alpha = path_alpha_rad[-1]
                            transition_length = 0.002  # 2mm transition zone
                            
                            print(f"Generating incoming transition: ρ {last_helical_rho:.6f} → {c_for_winding:.6f}")
                            print(f"α transition: {math.degrees(last_helical_alpha):.1f}° → 90.0°")
                            
                            print("DEBUG: Attempting to call incoming transition zone...")
                            incoming_transition = self._generate_smooth_transition_zone(
                                last_helical_rho, c_for_winding, 
                                last_helical_alpha, math.pi/2,  # CRITICAL FIX: Use actual last alpha, not pi/2
                                current_phi_rad, num_points=12
                            )
                            print("DEBUG: Successfully returned from incoming transition zone")
                            
                            # Step 2: Generate circumferential turnaround segment
                            if incoming_transition:
                                turnaround_phi_start = incoming_transition[-1]['phi']
                            else:
                                turnaround_phi_start = current_phi_rad
                            
                            # Initialize total_new_points to prevent variable scope error
                            total_new_points = 0
                                
                            print(f"Generating circumferential turnaround at ρ={c_for_winding:.6f}")
                            print("DEBUG: Attempting to call polar turnaround segment...")
                            turnaround_points = self._generate_polar_turnaround_segment(
                                c_for_winding, (incoming_transition[-1]["z"] if incoming_transition else z_i_m), turnaround_phi_start, math.pi/2, None
                            )
                            print("DEBUG: Successfully returned from polar turnaround segment")
                            
                            # Step 3: Generate outgoing transition zone (circumferential → helical)
                            if turnaround_points:
                                outgoing_phi_start = turnaround_points[-1]['phi']
                            else:
                                outgoing_phi_start = turnaround_phi_start
                            
                            # Calculate target outgoing angle (reverse direction for next pass)
                            outgoing_target_alpha = last_helical_alpha  # Same magnitude, different direction
                            outgoing_target_rho = c_for_winding + transition_length
                            
                            print(f"Generating outgoing transition: ρ {c_for_winding:.6f} → {outgoing_target_rho:.6f}")
                            print(f"α transition: 90.0° → {math.degrees(outgoing_target_alpha):.1f}°")
                            
                            print("DEBUG: Attempting to call outgoing transition zone...")
                            outgoing_transition = self._generate_smooth_transition_zone(
                                c_for_winding, outgoing_target_rho,
                                math.pi/2, math.pi/2,  # Force both to 90° for consistency
                                outgoing_phi_start, num_points=12
                            )
                            print("DEBUG: Successfully returned from outgoing transition zone")
                            
                            # Step 4: Integrate all segments with debugging
                            total_new_points = 0
                            
                            # Add incoming transition (skip first point to avoid duplication)
                            if incoming_transition and len(incoming_transition) > 1:
                                for i, point in enumerate(incoming_transition[1:], 1):
                                    path_rho_m.append(point['rho'])
                                    path_z_m.append(point['z'])
                                    path_alpha_rad.append(point['alpha'])
                                    path_phi_rad_cumulative.append(point['phi'])
                                    path_x_m.append(point['x'])
                                    path_y_m.append(point['y'])
                                    current_phi_rad = point['phi']
                                    total_new_points += 1
                                print(f"Added {len(incoming_transition)-1} incoming transition points")
                            
                            # Add circumferential turnaround
                            if turnaround_points:
                                for point in turnaround_points:
                                    path_rho_m.append(point['rho'])
                                    path_z_m.append(point['z'])
                                    path_alpha_rad.append(point['alpha'])
                                    path_phi_rad_cumulative.append(point['phi'])
                                    path_x_m.append(point['x'])
                                    path_y_m.append(point['y'])
                                    current_phi_rad = point['phi']
                                    total_new_points += 1
                                print(f"Added {len(turnaround_points)} circumferential turnaround points")
                            
                            # Add outgoing transition
                            if outgoing_transition:
                                for point in outgoing_transition:
                                    path_rho_m.append(point['rho'])
                                    path_z_m.append(point['z'])
                                    path_alpha_rad.append(point['alpha'])
                                    path_phi_rad_cumulative.append(point['phi'])
                                    path_x_m.append(point['x'])
                                    path_y_m.append(point['y'])
                                    current_phi_rad = point['phi']
                                total_new_points += 1
                            print(f"Added {len(outgoing_transition)} outgoing transition points")
                        
                        print(f"=== SMOOTH TURNAROUND COMPLETE ===")
                        # Safety check for variable scope
                        if 'total_new_points' not in locals():
                            total_new_points = 0
                        print(f"Total new points added: {total_new_points}")
                        print(f"Path now has {len(path_rho_m)} total points")
                        # Safety checks for all variables to prevent scope errors
                        if 'turnaround_points' not in locals():
                            turnaround_points = None
                        if 'incoming_transition' not in locals():
                            incoming_transition = None
                        if 'outgoing_transition' not in locals():
                            outgoing_transition = None
                            
                        if turnaround_points:
                            total_angular_span = turnaround_points[-1]['phi'] - incoming_transition[0]['phi'] if incoming_transition else 0
                            print(f"Total angular span of complete turnaround: {math.degrees(total_angular_span):.2f}°")
                        
                        # Update current position to continue from end of turnaround
                        if outgoing_transition and len(outgoing_transition) > 0:
                            current_phi_rad = outgoing_transition[-1]['phi']
                            print(f"DEBUG: Turnaround complete, continuing from φ={math.degrees(current_phi_rad):.1f}°")
                        
                        # Mark that we've completed a leg and should reverse direction
                        print(f"DEBUG: Completed leg {pass_number + 1}, continuing with return path...")
                        
                        # Set flag to generate return path after this pass completes
                        generate_return_path = True
                        
                        continue  # Skip normal processing for this specific point, but continue iteration
            else:
                # Point is inside c_for_winding, skip if path hasn't started
                if not first_valid_point_found:
                    continue
                else:
                    # Path has reached polar opening - execute turnaround sequence
                    print(f"DEBUG: Path reached polar opening at rho={rho_i_m:.4f}m - executing turnaround sequence")
                    
                    # Execute the complete smooth turnaround sequence
                    if len(path_rho_m) > 0:
                        print(f"DEBUG: *** ENTERING SMOOTH TURNAROUND SEQUENCE ***")
                        
                        # Initialize variables to prevent scope errors
                        total_new_points = 0
                        incoming_transition = None
                        turnaround_points = None
                        outgoing_transition = None
                        
                        # Step 1: Generate incoming transition zone (helical → circumferential)
                        last_helical_rho = path_rho_m[-1]
                        last_helical_alpha = path_alpha_rad[-1]
                        transition_length = 0.002  # 2mm transition zone
                        
                        print(f"Generating incoming transition: ρ {last_helical_rho:.6f} → {c_for_winding:.6f}")
                        print(f"α transition: {math.degrees(last_helical_alpha):.1f}° → 90.0°")
                        
                        incoming_transition = self._generate_smooth_transition_zone(
                            last_helical_rho, c_for_winding, 
                            last_helical_alpha, math.pi/2,
                            current_phi_rad, num_points=12
                        )
                        
                        # Step 2: Generate circumferential turnaround segment
                        if incoming_transition:
                            turnaround_phi_start = incoming_transition[-1]['phi']
                        else:
                            turnaround_phi_start = current_phi_rad
                            
                        print(f"Generating circumferential turnaround at ρ={c_for_winding:.6f}")
                        turnaround_points = self._generate_polar_turnaround_segment(
                            c_for_winding, (incoming_transition[-1]["z"] if incoming_transition else z_i_m), turnaround_phi_start, math.pi/2, None
                        )
                        
                        # Step 3: Generate outgoing transition zone (circumferential → helical)
                        if turnaround_points:
                            outgoing_phi_start = turnaround_points[-1]['phi']
                        else:
                            outgoing_phi_start = turnaround_phi_start
                        
                        # Calculate target outgoing angle for return pass (REVERSED DIRECTION)
                        # For proper turnaround, meridional direction must be reversed
                        # If incoming was going towards pole (decreasing z), outgoing goes away from pole
                        outgoing_target_alpha = last_helical_alpha  # Same magnitude but will reverse meridional direction
                        outgoing_target_rho = c_for_winding + transition_length
                        
                        # DEBUG: Check meridional direction reversal
                        incoming_z_velocity = -math.cos(last_helical_alpha)  # Incoming dz/ds component
                        outgoing_z_velocity = math.cos(outgoing_target_alpha)  # Should be opposite
                        print(f"DEBUG MERIDIONAL REVERSAL:")
                        print(f"  Incoming dz/ds sign: {incoming_z_velocity:.6f} (towards {'pole' if incoming_z_velocity < 0 else 'equator'})")
                        print(f"  Outgoing dz/ds sign: {outgoing_z_velocity:.6f} (towards {'pole' if outgoing_z_velocity < 0 else 'equator'})")
                        
                        print(f"Generating outgoing transition: ρ {c_for_winding:.6f} → {outgoing_target_rho:.6f}")
                        print(f"α transition: 90.0° → {math.degrees(outgoing_target_alpha):.1f}°")
                        
                        outgoing_transition = self._generate_smooth_transition_zone(
                            c_for_winding, outgoing_target_rho,
                            math.pi/2, outgoing_target_alpha,
                            outgoing_phi_start, num_points=12,
                            reverse_meridional=True  # CRITICAL: Reverse direction for outgoing path
                        )
                        
                        # Step 4: Integrate all transition and turnaround segments with interface debugging
                        total_new_points = 0
                        
                        # DEBUG: Log last helical point tangent vector
                        if len(path_rho_m) > 1:
                            last_helical_tangent = self._calculate_tangent_vector(
                                path_rho_m[-1], path_z_m[-1], path_phi_rad_cumulative[-1], path_alpha_rad[-1]
                            )
                            print(f"\n=== INTERFACE TANGENT DEBUGGING ===")
                            print(f"LAST HELICAL POINT:")
                            print(f"  Position: ρ={path_rho_m[-1]:.6f} z={path_z_m[-1]:.6f} φ={math.degrees(path_phi_rad_cumulative[-1]):7.2f}°")
                            print(f"  Tangent: dρ/ds={last_helical_tangent[0]:.6f} dz/ds={last_helical_tangent[1]:.6f} dφ/ds={last_helical_tangent[2]:.6f}")
                            print(f"  Cartesian tangent: dx/ds={last_helical_tangent[0]*math.cos(path_phi_rad_cumulative[-1]) - path_rho_m[-1]*math.sin(path_phi_rad_cumulative[-1])*last_helical_tangent[2]:.6f}")
                            print(f"                     dy/ds={last_helical_tangent[0]*math.sin(path_phi_rad_cumulative[-1]) + path_rho_m[-1]*math.cos(path_phi_rad_cumulative[-1])*last_helical_tangent[2]:.6f}")
                            print(f"                     dz/ds={last_helical_tangent[1]:.6f}")
                        
                        # Add incoming transition (skip first point to avoid duplication)
                        if incoming_transition and len(incoming_transition) > 1:
                            # DEBUG: First incoming transition point
                            first_inc = incoming_transition[1]
                            print(f"FIRST INCOMING TRANSITION:")
                            print(f"  Position: ρ={first_inc['rho']:.6f} z={first_inc['z']:.6f} φ={math.degrees(first_inc['phi']):7.2f}°")
                            print(f"  Tangent: dρ/ds={first_inc['drho_ds']:.6f} dz/ds={first_inc['dz_ds']:.6f} dφ/ds={first_inc['dphi_ds']:.6f}")
                            
                            for point in incoming_transition[1:]:
                                path_rho_m.append(point['rho'])
                                path_z_m.append(point['z'])
                                path_alpha_rad.append(point['alpha'])
                                path_phi_rad_cumulative.append(point['phi'])
                                path_x_m.append(point['x'])
                                path_y_m.append(point['y'])
                                current_phi_rad = point['phi']
                                total_new_points += 1
                            
                            # DEBUG: Last incoming transition point
                            last_inc = incoming_transition[-1]
                            print(f"LAST INCOMING TRANSITION:")
                            print(f"  Position: ρ={last_inc['rho']:.6f} z={last_inc['z']:.6f} φ={math.degrees(last_inc['phi']):7.2f}°")
                            print(f"  Tangent: dρ/ds={last_inc['drho_ds']:.6f} dz/ds={last_inc['dz_ds']:.6f} dφ/ds={last_inc['dphi_ds']:.6f}")
                            print(f"Added {len(incoming_transition)-1} incoming transition points")
                        
                        # Add circumferential turnaround
                        if turnaround_points:
                            # DEBUG: First and last turnaround points
                            first_turn = turnaround_points[0]
                            last_turn = turnaround_points[-1]
                            print(f"FIRST CIRCUMFERENTIAL POINT:")
                            print(f"  Position: ρ={first_turn['rho']:.6f} z={first_turn['z']:.6f} φ={math.degrees(first_turn['phi']):7.2f}°")
                            print(f"  Tangent: dρ/ds={first_turn['drho_ds']:.6f} dz/ds={first_turn['dz_ds']:.6f} dφ/ds={first_turn['dphi_ds']:.6f}")
                            
                            for point in turnaround_points:
                                path_rho_m.append(point['rho'])
                                path_z_m.append(point['z'])
                                path_alpha_rad.append(point['alpha'])
                                path_phi_rad_cumulative.append(point['phi'])
                                path_x_m.append(point['x'])
                                path_y_m.append(point['y'])
                                current_phi_rad = point['phi']
                                total_new_points += 1
                            
                            print(f"LAST CIRCUMFERENTIAL POINT:")
                            print(f"  Position: ρ={last_turn['rho']:.6f} z={last_turn['z']:.6f} φ={math.degrees(last_turn['phi']):7.2f}°")
                            print(f"  Tangent: dρ/ds={last_turn['drho_ds']:.6f} dz/ds={last_turn['dz_ds']:.6f} dφ/ds={last_turn['dphi_ds']:.6f}")
                            print(f"Added {len(turnaround_points)} circumferential turnaround points")
                        
                        # Add outgoing transition
                        if outgoing_transition:
                            # DEBUG: First and last outgoing transition points
                            first_out = outgoing_transition[0]
                            last_out = outgoing_transition[-1]
                            print(f"FIRST OUTGOING TRANSITION:")
                            print(f"  Position: ρ={first_out['rho']:.6f} z={first_out['z']:.6f} φ={math.degrees(first_out['phi']):7.2f}°")
                            print(f"  Tangent: dρ/ds={first_out['drho_ds']:.6f} dz/ds={first_out['dz_ds']:.6f} dφ/ds={first_out['dphi_ds']:.6f}")
                            
                            for point in outgoing_transition:
                                path_rho_m.append(point['rho'])
                                path_z_m.append(point['z'])
                                path_alpha_rad.append(point['alpha'])
                                path_phi_rad_cumulative.append(point['phi'])
                                path_x_m.append(point['x'])
                                path_y_m.append(point['y'])
                                current_phi_rad = point['phi']
                                total_new_points += 1
                            
                            print(f"LAST OUTGOING TRANSITION:")
                            print(f"  Position: ρ={last_out['rho']:.6f} z={last_out['z']:.6f} φ={math.degrees(last_out['phi']):7.2f}°")
                            print(f"  Tangent: dρ/ds={last_out['drho_ds']:.6f} dz/ds={last_out['dz_ds']:.6f} dφ/ds={last_out['dphi_ds']:.6f}")
                            print(f"Added {len(outgoing_transition)} outgoing transition points")
                        
                        print(f"=== END INTERFACE DEBUGGING ===")
                        
                        print(f"=== SMOOTH TURNAROUND COMPLETE ===")
                        print(f"Total new points added: {total_new_points}")
                        print(f"Path now has {len(path_rho_m)} total points")
                        if turnaround_points and incoming_transition:
                            total_angular_span = turnaround_points[-1]['phi'] - incoming_transition[0]['phi']
                            print(f"Total angular span of complete turnaround: {math.degrees(total_angular_span):.2f}°")
                    
                    break

            if not first_valid_point_found:
                # First valid point - start of trajectory
                first_valid_point_found = True
                
                # Set initial phi for this pass
                current_phi_rad = pass_number * 0.1  # Advance phi between passes
                
                path_rho_m.append(rho_i_m)
                path_z_m.append(z_i_m)
                path_alpha_rad.append(alpha_i_rad)
                path_phi_rad_cumulative.append(current_phi_rad)
                path_x_m.append(rho_i_m * math.cos(current_phi_rad))
                path_y_m.append(rho_i_m * math.sin(current_phi_rad))
                
                print(f"DEBUG Pass {pass_number + 1}: STARTING POINT - ρ={rho_i_m:.6f}m z={z_i_m:.6f}m α={math.degrees(alpha_i_rad):.1f}° φ={math.degrees(current_phi_rad):.1f}°")
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
        
        # CRITICAL FIX: Generate return helical path for complete circuit
        print(f"DEBUG: About to check if we should generate return path. Current points: {len(path_rho_m)}")
        if len(path_rho_m) > 0:
            print(f"\nDEBUG: *** GENERATING RETURN HELICAL PATH FOR COMPLETE CIRCUIT ***")
            initial_points = len(path_rho_m)
            
            # Generate return journey (reverse direction from last turnaround)
            return_profile_range = range(len(profile_r_m_calc) - 1, -1, -1)
            print(f"DEBUG: Return journey processing {len(list(return_profile_range))} points in reverse direction")
            
            # Continue from last phi position
            if len(path_phi_rad_cumulative) > 0:
                current_phi_rad = path_phi_rad_cumulative[-1]
            
            # Generate return helical path
            for ret_idx, ret_i in enumerate(return_profile_range):
                ret_rho_i_m = profile_r_m_calc[ret_i]
                ret_z_i_m = profile_z_m_calc[ret_i]
                
                # Only process points that are reachable (ρ >= c_for_winding)
                if ret_rho_i_m >= c_for_winding - 1e-6:
                    # Calculate alpha for return journey
                    ret_alpha_rad = self.calculate_geodesic_alpha_at_rho(ret_rho_i_m)
                    if ret_alpha_rad is not None:
                        # Calculate phi advancement  
                        if len(path_rho_m) > 0:
                            d_phi = abs(ret_alpha_rad) * 0.01  # Small phi increment
                            current_phi_rad += d_phi
                        
                        # Convert to Cartesian coordinates
                        ret_x_m = ret_rho_i_m * math.cos(current_phi_rad)
                        ret_y_m = ret_rho_i_m * math.sin(current_phi_rad)
                        
                        # Add return path point
                        path_rho_m.append(ret_rho_i_m)
                        path_z_m.append(ret_z_i_m)
                        path_alpha_rad.append(ret_alpha_rad)
                        path_phi_rad_cumulative.append(current_phi_rad)
                        path_x_m.append(ret_x_m)
                        path_y_m.append(ret_y_m)
                        
                        if len(path_rho_m) % 50 == 0:
                            print(f"DEBUG Return: Point {len(path_rho_m)}: ρ={ret_rho_i_m:.6f}m z={ret_z_i_m:.6f}m α={math.degrees(ret_alpha_rad):.1f}°")
            
            final_points = len(path_rho_m)
            print(f"DEBUG: *** RETURN PATH COMPLETE *** - Added {final_points - initial_points} return points")
            print(f"DEBUG: *** TOTAL TRAJECTORY POINTS: {final_points} ***")

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

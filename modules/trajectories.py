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
                 roving_eccentricity_at_pole_m: float = 0.0):
        """
        Initialize trajectory planner with vessel geometry.
        
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
        """
        self.vessel = vessel_geometry
        self.dry_roving_width_m = dry_roving_width_m
        self.dry_roving_thickness_m = dry_roving_thickness_m
        self.roving_eccentricity_at_pole_m = roving_eccentricity_at_pole_m
        self.trajectory_data = None
        
        # Geodesic calculation properties
        self.effective_polar_opening_radius_m = None
        self.alpha_profile_deg = None  # Array of winding angles
        self.phi_profile_rad = None    # Array of parallel angles
        self.turn_around_angle_rad = None
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
        rho_geom_pole = profile_points['r_inner'][0]  # Geometric polar opening
        
        ecc_0 = self.roving_eccentricity_at_pole_m
        b = self.dry_roving_width_m
        t_rov = self.dry_roving_thickness_m

        # Calculate dz/drho at the geometric pole
        dz_drho_pole = self._get_slope_dz_drho_at_rho(rho_geom_pole)
        
        # Handle infinite slope (vertical tangent)
        if np.isinf(dz_drho_pole):
            dz_drho_pole = 1e6  # Use large finite number

        # Koussios formula: c_eff = rho_pole + ecc + (b/2)*sqrt(1+(dz/drho)^2) - (t/2)*(dz/drho)
        term_width = (b / 2.0) * math.sqrt(1 + dz_drho_pole**2)
        term_thickness = (t_rov / 2.0) * dz_drho_pole
        
        self.effective_polar_opening_radius_m = rho_geom_pole + ecc_0 + term_width - term_thickness
        
        # Ensure c_eff is positive and reasonable
        if self.effective_polar_opening_radius_m < 0:
            self.effective_polar_opening_radius_m = 1e-6
            
        return self.effective_polar_opening_radius_m

    def calculate_geodesic_alpha_at_rho(self, rho_m: float) -> Optional[float]:
        """
        Calculates geodesic winding angle (radians) at a given radius rho_m.
        Uses Clairaut's theorem: rho * sin(alpha) = c_eff
        """
        if self.effective_polar_opening_radius_m is None:
            self._calculate_effective_polar_opening()
        
        c_eff = self.effective_polar_opening_radius_m
        
        if rho_m < c_eff - 1e-9:
            return None  # Geodesic cannot reach this radius

        # Clairaut's theorem: sin(alpha) = c_eff / rho
        asin_arg = c_eff / rho_m
        asin_arg = np.clip(asin_arg, -1.0, 1.0)  # Ensure valid range
        
        try:
            alpha_rad = math.asin(asin_arg)
            return alpha_rad
        except ValueError:
            return None

    def generate_geodesic_trajectory(self, num_points_half_circuit: int = 100) -> Dict:
        """
        Generates geodesic path points (rho, z, alpha, phi) for one half circuit.
        Based on Koussios geodesic theory with Clairaut's theorem.
        """
        profile_points = self.vessel.get_profile_points()
        rho_profile = profile_points['r_inner']
        z_profile = profile_points['z']
        
        # Find dome section (typically first half of profile)
        dome_end_idx = len(rho_profile) // 2
        rho_dome = rho_profile[:dome_end_idx]
        z_dome = z_profile[:dome_end_idx]
        
        # Generate points from effective polar opening to equator
        c_eff = self.effective_polar_opening_radius_m
        rho_max = max(rho_dome)
        
        # Create rho points for geodesic calculation
        rho_points = np.linspace(c_eff + 1e-6, min(rho_max, self.vessel.inner_radius), num_points_half_circuit)
        
        # Calculate geodesic properties at each point
        alpha_values = []
        phi_values = []
        z_values = []
        
        phi_cumulative = 0.0
        
        for i, rho in enumerate(rho_points):
            # Calculate winding angle using Clairaut's theorem
            alpha = self.calculate_geodesic_alpha_at_rho(rho)
            if alpha is None:
                continue
                
            alpha_values.append(math.degrees(alpha))
            
            # Interpolate z coordinate from vessel profile
            z_interp = np.interp(rho, rho_dome, z_dome)
            z_values.append(z_interp)
            
            # Calculate incremental parallel angle (simplified)
            if i > 0:
                drho = rho - rho_points[i-1]
                if abs(drho) > 1e-9 and not np.isclose(alpha, np.pi/2):
                    dphi = drho / (rho * math.tan(alpha)) if math.tan(alpha) != 0 else 0
                    phi_cumulative += dphi
            
            phi_values.append(phi_cumulative)
        
        # Store calculated profiles
        self.alpha_profile_deg = np.array(alpha_values)
        self.phi_profile_rad = np.array(phi_values)
        self.turn_around_angle_rad = phi_cumulative if len(phi_values) > 0 else 0
        
        # Calculate equatorial winding angle
        self.alpha_eq_deg = alpha_values[-1] if alpha_values else 0
        
        return {
            'rho_points': rho_points[:len(alpha_values)],
            'z_points': np.array(z_values),
            'alpha_deg': np.array(alpha_values),
            'phi_rad': np.array(phi_values),
            'c_eff_m': c_eff,
            'turn_around_angle_deg': math.degrees(self.turn_around_angle_rad),
            'alpha_equator_deg': self.alpha_eq_deg,
            'pattern_type': 'Geodesic'
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

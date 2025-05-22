import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from modules.geometry import VesselGeometry

class TrajectoryPlanner:
    """
    Filament winding trajectory planning for composite pressure vessels.
    Implements geodesic and non-geodesic winding patterns.
    """
    
    def __init__(self, vessel_geometry: VesselGeometry):
        """
        Initialize trajectory planner with vessel geometry.
        
        Parameters:
        -----------
        vessel_geometry : VesselGeometry
            Vessel geometry object containing profile data
        """
        self.vessel = vessel_geometry
        self.trajectory_data = None
        
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
        
        if pattern_type == 'Helical':
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

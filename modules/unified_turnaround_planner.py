"""
Unified Trajectory Planner - Turnaround Planner
Step 5: Polar region transitions and feed-eye kinematics for filament winding machines
"""

import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from .unified_trajectory_core import TrajectoryPoint

@dataclass
class MachineCoords:
    """
    Represents the state of the filament winding machine axes.
    Units: radians for rotation, meters for linear.
    """
    mandrel_rotation_rad: float = 0.0      # machine_X_mandrel_rotation_rad
    carriage_axial_m: float = 0.0          # machine_Y_carriage_axial_m
    feed_eye_radial_m: float = 0.0         # machine_Z_feed_eye_radial_m
    feed_eye_yaw_rad: float = 0.0          # machine_A_feed_eye_yaw_rad
    timestamp_s: Optional[float] = None    # For motion optimization

class TurnaroundPlanner:
    """
    Handles the generation of fiber paths and machine motions for
    turnarounds in the polar regions of a COPV.
    """

    def __init__(self,
                 payout_length_m: float,
                 default_turnaround_points: int = 20):
        """
        Initializes the TurnaroundPlanner.

        Args:
            payout_length_m: The free fiber length from feed-eye to mandrel contact point (meters).
            default_turnaround_points: Default number of points for a generated turnaround path.
        """
        self.payout_length_m = payout_length_m
        self.default_turnaround_points = default_turnaround_points

    def generate_polar_turnaround_on_mandrel(self,
                                             entry_point: TrajectoryPoint,
                                             polar_opening_radius_m: float,
                                             pattern_advancement_angle_rad: float,
                                             num_points: Optional[int] = None
                                             ) -> List[TrajectoryPoint]:
        """
        Generates the fiber path on the mandrel surface during a polar turnaround.
        This path achieves the required pattern advancement.

        Args:
            entry_point: The TrajectoryPoint where the fiber enters the turnaround region.
            polar_opening_radius_m: The radius of the polar opening around which the turnaround occurs.
            pattern_advancement_angle_rad: The azimuthal shift (in phi) required for the pattern.
            num_points: Number of points to generate for the turnaround path.

        Returns:
            List[TrajectoryPoint]: A list of TrajectoryPoints on the mandrel surface.
        """
        if num_points is None:
            num_points = self.default_turnaround_points

        turnaround_path_on_mandrel: List[TrajectoryPoint] = []

        # Extract starting conditions from entry point
        start_phi = entry_point.surface_coords.get('phi_rad', 0.0)
        start_rho = entry_point.surface_coords.get('rho', polar_opening_radius_m)
        start_z = entry_point.surface_coords.get('z_cyl', entry_point.position[2])

        # Ensure entry rho is close to polar opening radius for a standard turnaround
        if not math.isclose(start_rho, polar_opening_radius_m, rel_tol=0.1):
            print(f"Warning: Turnaround entry rho ({start_rho:.4f}m) is not close to polar opening radius ({polar_opening_radius_m:.4f}m).")
            start_rho = polar_opening_radius_m

        # Generate turnaround path as circular arc at polar opening
        for i in range(num_points + 1):
            u = i / num_points  # Parameter from 0 to 1
            current_phi = start_phi + u * pattern_advancement_angle_rad
            
            # Position on the mandrel (turnaround at polar opening radius)
            current_rho = polar_opening_radius_m
            current_z = start_z  # Simplified: assuming constant z during arc
            
            pos_3d = np.array([current_rho * np.cos(current_phi),
                               current_rho * np.sin(current_phi),
                               current_z])
            
            # Tangent vector on the mandrel surface for this circular path
            tangent_on_mandrel = np.array([-current_rho * np.sin(current_phi),
                                           current_rho * np.cos(current_phi),
                                           0.0])
            if np.linalg.norm(tangent_on_mandrel) > 1e-9:
                tangent_on_mandrel = tangent_on_mandrel / np.linalg.norm(tangent_on_mandrel)
            
            # Surface normal (simplified axial direction)
            normal_vec = np.array([0.0, 0.0, 1.0 if current_z >= 0 else -1.0])

            tp = TrajectoryPoint(
                position=pos_3d,
                winding_angle_deg=90.0,  # Circumferential at polar opening
                surface_coords={'rho': current_rho, 'z_cyl': current_z, 'phi_rad': current_phi},
                tangent_vector=tangent_on_mandrel,
                normal_vector=normal_vec,
                arc_length_from_start=entry_point.arc_length_from_start + u * pattern_advancement_angle_rad * current_rho
            )
            turnaround_path_on_mandrel.append(tp)
            
        return turnaround_path_on_mandrel

    def calculate_feed_eye_motion(self,
                                  mandrel_path_segment: List[TrajectoryPoint],
                                  vessel_geometry=None
                                 ) -> Tuple[List[MachineCoords], List[bool]]:
        """
        Calculates the required feed-eye machine coordinates for a given fiber path
        segment on the mandrel surface.

        Args:
            mandrel_path_segment: List of TrajectoryPoints on the mandrel surface.
            vessel_geometry: Vessel geometry object (optional for collision checking).

        Returns:
            Tuple[List[MachineCoords], List[bool]]:
                - List of MachineCoords for each point in the mandrel_path_segment.
                - List of booleans indicating collision status for each feed-eye point.
        """
        machine_coords_path: List[MachineCoords] = []
        collision_flags: List[bool] = []

        for point_on_mandrel in mandrel_path_segment:
            if point_on_mandrel.tangent_vector is None:
                raise ValueError("TrajectoryPoint in mandrel_path_segment must have a valid tangent_vector.")

            # Feed-eye position: P_eye = P_mandrel - payout_length * Tangent_fiber_in_space
            feed_eye_position = point_on_mandrel.position - self.payout_length_m * point_on_mandrel.tangent_vector

            # Basic Machine Coordinates mapping
            # Mandrel rotation from phi coordinate
            phi_contact = point_on_mandrel.surface_coords.get(
                'phi_rad', 
                np.arctan2(point_on_mandrel.position[1], point_on_mandrel.position[0])
            )
            
            # Carriage axial position from feed-eye z position
            y_carriage_m = feed_eye_position[2]
            
            # Feed-eye radial position (distance from z-axis)
            z_feed_eye_radial_m = np.sqrt(feed_eye_position[0]**2 + feed_eye_position[1]**2)
            
            # Feed-eye yaw calculation using Andrianov's formula
            # A = π/2 - β_srf, where β_srf = π/2 - α_srf
            # Therefore A = α_srf (winding angle with meridian)
            alpha_srf_rad = np.radians(point_on_mandrel.winding_angle_deg)
            a_feed_eye_yaw_rad = alpha_srf_rad

            machine_coord = MachineCoords(
                mandrel_rotation_rad=phi_contact,
                carriage_axial_m=y_carriage_m,
                feed_eye_radial_m=z_feed_eye_radial_m,
                feed_eye_yaw_rad=a_feed_eye_yaw_rad
            )
            machine_coords_path.append(machine_coord)

            # Basic collision checking (simplified)
            is_colliding = False
            if vessel_geometry is not None:
                # Check if feed-eye position interferes with vessel
                feed_eye_radius_from_axis = np.sqrt(feed_eye_position[0]**2 + feed_eye_position[1]**2)
                vessel_radius_at_z = getattr(vessel_geometry, 'inner_diameter', 200) / 2000  # Convert mm to m
                
                # Simple collision check: feed-eye too close to vessel surface
                if feed_eye_radius_from_axis < vessel_radius_at_z + 0.01:  # 1cm clearance
                    is_colliding = True
                    
            collision_flags.append(is_colliding)
            if is_colliding:
                print(f"Warning: Feed-eye collision detected at mandrel point {point_on_mandrel.position}")

        return machine_coords_path, collision_flags

    def optimize_turnaround_time(self,
                                 machine_coords_path: List[MachineCoords],
                                 motion_constraints: Dict[str, Dict[str, float]]) -> List[MachineCoords]:
        """
        Optimizes the timing of machine movements during a turnaround to minimize time
        while respecting motion constraints.

        Args:
            machine_coords_path: The sequence of MachineCoords defining the turnaround.
            motion_constraints: Dict specifying {'axis_name': {'max_vel': float, 'max_accel': float}}

        Returns:
            List[MachineCoords]: The machine_coords_path with optimized timestamps.
        """
        if not machine_coords_path:
            return []

        # Simplified time optimization approach
        # Calculate required times for each axis based on motion constraints
        
        num_points = len(machine_coords_path)
        axis_times = []
        
        # Analyze each axis motion
        axes_data = {
            'mandrel_rotation': [mc.mandrel_rotation_rad for mc in machine_coords_path],
            'carriage_axial': [mc.carriage_axial_m for mc in machine_coords_path],
            'feed_eye_radial': [mc.feed_eye_radial_m for mc in machine_coords_path],
            'feed_eye_yaw': [mc.feed_eye_yaw_rad for mc in machine_coords_path]
        }
        
        for axis_name, positions in axes_data.items():
            if len(positions) < 2:
                continue
                
            max_vel = motion_constraints.get(axis_name, {}).get('max_vel', 1.0)
            max_accel = motion_constraints.get(axis_name, {}).get('max_accel', 2.0)
            
            # Calculate total distance
            total_distance = sum(abs(positions[i+1] - positions[i]) for i in range(len(positions)-1))
            
            # Simple time estimation: trapezoidal velocity profile
            if total_distance > 0:
                # Time to accelerate to max velocity
                t_accel = max_vel / max_accel
                # Distance during acceleration/deceleration
                accel_distance = 0.5 * max_accel * t_accel**2
                
                if total_distance <= 2 * accel_distance:
                    # Triangular profile (never reach max velocity)
                    t_total = 2 * np.sqrt(total_distance / max_accel)
                else:
                    # Trapezoidal profile
                    constant_vel_distance = total_distance - 2 * accel_distance
                    t_constant = constant_vel_distance / max_vel
                    t_total = 2 * t_accel + t_constant
                
                axis_times.append(t_total)
        
        # Use the slowest axis to determine overall time
        estimated_total_time = max(axis_times) if axis_times else 1.0
        
        print(f"Estimated turnaround time: {estimated_total_time:.2f}s for {num_points} points")
        
        # Assign timestamps based on uniform distribution
        if num_points > 1:
            dt = estimated_total_time / (num_points - 1)
            for i, mc_point in enumerate(machine_coords_path):
                mc_point.timestamp_s = i * dt
        else:
            machine_coords_path[0].timestamp_s = 0.0
                
        return machine_coords_path

    def validate_turnaround_feasibility(self,
                                      turnaround_path: List[TrajectoryPoint],
                                      machine_coords: List[MachineCoords],
                                      motion_constraints: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Validates whether a turnaround path is feasible given machine constraints.

        Args:
            turnaround_path: The generated turnaround path on mandrel
            machine_coords: Corresponding machine coordinates
            motion_constraints: Machine motion limits

        Returns:
            Dict with feasibility analysis results
        """
        validation_results = {
            'is_feasible': True,
            'issues': [],
            'warnings': [],
            'metrics': {}
        }

        if len(turnaround_path) != len(machine_coords):
            validation_results['is_feasible'] = False
            validation_results['issues'].append("Mismatch between path and machine coordinate lengths")
            return validation_results

        # Check for collision issues
        collision_count = sum(1 for tp in turnaround_path 
                            if tp.position[0]**2 + tp.position[1]**2 < (0.01)**2)  # Very basic check
        
        if collision_count > 0:
            validation_results['warnings'].append(f"Potential collision points detected: {collision_count}")

        # Check motion constraint violations
        if len(machine_coords) > 1:
            for i in range(len(machine_coords) - 1):
                mc1, mc2 = machine_coords[i], machine_coords[i+1]
                dt = (mc2.timestamp_s or 0) - (mc1.timestamp_s or 0)
                
                if dt > 0:
                    # Check velocities
                    vel_mandrel = abs(mc2.mandrel_rotation_rad - mc1.mandrel_rotation_rad) / dt
                    vel_carriage = abs(mc2.carriage_axial_m - mc1.carriage_axial_m) / dt
                    vel_radial = abs(mc2.feed_eye_radial_m - mc1.feed_eye_radial_m) / dt
                    vel_yaw = abs(mc2.feed_eye_yaw_rad - mc1.feed_eye_yaw_rad) / dt
                    
                    # Check against constraints
                    constraints_check = {
                        'mandrel_rotation': vel_mandrel,
                        'carriage_axial': vel_carriage,
                        'feed_eye_radial': vel_radial,
                        'feed_eye_yaw': vel_yaw
                    }
                    
                    for axis, velocity in constraints_check.items():
                        max_vel = motion_constraints.get(axis, {}).get('max_vel', float('inf'))
                        if velocity > max_vel:
                            validation_results['is_feasible'] = False
                            validation_results['issues'].append(
                                f"{axis} velocity {velocity:.3f} exceeds limit {max_vel:.3f}"
                            )

        # Calculate useful metrics
        if turnaround_path:
            path_length = turnaround_path[-1].arc_length_from_start - turnaround_path[0].arc_length_from_start
            validation_results['metrics']['path_length_m'] = path_length
            validation_results['metrics']['num_points'] = len(turnaround_path)
            
        if machine_coords and machine_coords[-1].timestamp_s is not None:
            validation_results['metrics']['total_time_s'] = machine_coords[-1].timestamp_s

        return validation_results

    def generate_complete_turnaround(self,
                                   entry_point: TrajectoryPoint,
                                   polar_opening_radius_m: float,
                                   pattern_advancement_angle_rad: float,
                                   motion_constraints: Dict[str, Dict[str, float]],
                                   vessel_geometry=None) -> Dict[str, Any]:
        """
        Complete turnaround generation workflow combining all methods.

        Args:
            entry_point: Starting point for turnaround
            polar_opening_radius_m: Polar opening radius
            pattern_advancement_angle_rad: Required pattern advancement
            motion_constraints: Machine motion limits
            vessel_geometry: Optional vessel geometry for collision checking

        Returns:
            Dict containing complete turnaround analysis
        """
        try:
            # Generate mandrel path
            mandrel_path = self.generate_polar_turnaround_on_mandrel(
                entry_point, polar_opening_radius_m, pattern_advancement_angle_rad
            )
            
            # Calculate machine coordinates
            machine_coords, collision_flags = self.calculate_feed_eye_motion(
                mandrel_path, vessel_geometry
            )
            
            # Optimize timing
            optimized_coords = self.optimize_turnaround_time(machine_coords, motion_constraints)
            
            # Validate feasibility
            validation = self.validate_turnaround_feasibility(
                mandrel_path, optimized_coords, motion_constraints
            )
            
            return {
                'mandrel_path': mandrel_path,
                'machine_coordinates': optimized_coords,
                'collision_flags': collision_flags,
                'validation': validation,
                'success': True,
                'payout_length_m': self.payout_length_m,
                'polar_opening_radius_m': polar_opening_radius_m,
                'pattern_advancement_deg': np.degrees(pattern_advancement_angle_rad)
            }
            
        except Exception as e:
            return {
                'mandrel_path': [],
                'machine_coordinates': [],
                'collision_flags': [],
                'validation': {'is_feasible': False, 'issues': [str(e)]},
                'success': False,
                'error': str(e)
            }
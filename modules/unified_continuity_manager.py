"""
Unified Trajectory Planner - Continuity Manager
Step 4: Mathematical smoothness analysis and transition generation for trajectory segments
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from scipy.interpolate import CubicSpline
from .unified_trajectory_core import TrajectoryPoint

@dataclass
class ContinuityReport:
    """
    Reports the analysis of continuity between the end of one segment
    and the start of another.
    """
    c0_gap_m: float = field(metadata={"unit": "meters", "description": "Positional gap (Euclidean distance)."})
    c0_gap_vector_m: Optional[np.ndarray] = field(default=None, metadata={"unit": "meters", "description": "Positional gap vector."})
    
    c1_velocity_diff_mps: Optional[float] = field(default=None, metadata={"unit": "m/s", "description": "Magnitude of velocity vector difference."})
    c1_velocity_diff_vector_mps: Optional[np.ndarray] = field(default=None, metadata={"unit": "m/s", "description": "Velocity difference vector."})
    c1_speed_diff_mps: Optional[float] = field(default=None, metadata={"unit": "m/s", "description": "Difference in speed magnitudes."})
    c1_direction_angle_diff_rad: Optional[float] = field(default=None, metadata={"unit": "radians", "description": "Angle between velocity vectors."})

    c2_acceleration_diff_mps2: Optional[float] = field(default=None, metadata={"unit": "m/s^2", "description": "Magnitude of acceleration vector difference."})
    c2_acceleration_diff_vector_mps2: Optional[np.ndarray] = field(default=None, metadata={"unit": "m/s^2", "description": "Acceleration difference vector."})

    notes: List[str] = field(default_factory=list)

@dataclass
class PathQualityReport:
    """
    Reports on the overall smoothness and quality of a trajectory.
    """
    max_c0_gap_mm: float = field(default=0.0, metadata={"unit": "mm", "description": "Maximum positional gap between consecutive points."})
    avg_c0_gap_mm: float = field(default=0.0, metadata={"unit": "mm", "description": "Average positional gap between consecutive points."})
    
    max_c1_velocity_jump_mps: float = field(default=0.0, metadata={"unit": "m/s", "description": "Maximum velocity jump magnitude."})
    avg_c1_velocity_jump_mps: float = field(default=0.0, metadata={"unit": "m/s", "description": "Average velocity jump magnitude."})
    
    max_c2_acceleration_mps2: float = field(default=0.0, metadata={"unit": "m/s^2", "description": "Maximum acceleration magnitude encountered."})
    max_c2_acceleration_jump_mps2: float = field(default=0.0, metadata={"unit": "m/s^2", "description": "Maximum acceleration jump magnitude."})
    
    points_exceeding_c0_threshold: int = 0
    points_exceeding_c1_threshold: int = 0
    points_exceeding_c2_accel_limit: int = 0
    
    is_smooth_c0: bool = True
    is_smooth_c1: bool = True
    is_smooth_c2: bool = True
    
    total_length_m: Optional[float] = None
    notes: List[str] = field(default_factory=list)

class ContinuityManager:
    """
    Manages and ensures mathematical smoothness (C0, C1, C2 continuity)
    between trajectory segments.
    """

    def __init__(self,
                 c0_pos_gap_threshold_m: float = 0.0001,  # 0.1mm
                 c1_vel_jump_threshold_mps: float = 0.05,
                 machine_max_accel_mps2: float = 5.0):
        """
        Initializes the ContinuityManager with smoothness thresholds.

        Args:
            c0_pos_gap_threshold_m: Maximum allowable position gap for C0 continuity.
            c1_vel_jump_threshold_mps: Maximum allowable velocity jump for C1 continuity.
            machine_max_accel_mps2: Machine's maximum allowable acceleration.
        """
        self.c0_threshold_m = c0_pos_gap_threshold_m
        self.c1_threshold_mps = c1_vel_jump_threshold_mps
        self.machine_max_accel_mps2 = machine_max_accel_mps2

    def _estimate_velocity(self, points: List[TrajectoryPoint], index: int, dt: float = 0.01) -> Optional[np.ndarray]:
        """
        Estimates velocity at a point using finite differences if not available in TrajectoryPoint.
        """
        if index < 0 or index >= len(points):
            return None
            
        # Check if velocity is already available
        if hasattr(points[index], 'velocity_vector_mps') and points[index].velocity_vector_mps is not None:
            return points[index].velocity_vector_mps
            
        # Finite difference estimation
        if 0 < index < len(points) - 1:
            return (points[index+1].position - points[index-1].position) / (2 * dt)
        elif index == 0 and len(points) > 1:
            return (points[index+1].position - points[index].position) / dt
        elif index == len(points) - 1 and len(points) > 1:
            return (points[index].position - points[index-1].position) / dt
        
        return None

    def _estimate_acceleration(self, points: List[TrajectoryPoint], index: int, dt: float = 0.01) -> Optional[np.ndarray]:
        """
        Estimates acceleration at a point using finite differences.
        """
        if index < 0 or index >= len(points):
            return None
            
        # Check if acceleration is already available
        if hasattr(points[index], 'acceleration_vector_mps2') and points[index].acceleration_vector_mps2 is not None:
            return points[index].acceleration_vector_mps2
            
        # Estimate from position using second derivative
        if 0 < index < len(points) - 1:
            return (points[index+1].position - 2 * points[index].position + points[index-1].position) / (dt**2)
        
        return None

    def analyze_continuity(self,
                          segment1_end_points: List[TrajectoryPoint],
                          segment2_start_points: List[TrajectoryPoint],
                          dt: float = 0.01) -> ContinuityReport:
        """
        Analyzes the continuity between the end of segment1 and the start of segment2.
        """
        if not segment1_end_points or not segment2_start_points:
            raise ValueError("Segment end/start points lists cannot be empty.")

        pt1_end = segment1_end_points[-1]
        pt2_start = segment2_start_points[0]

        # C0 Continuity (Position) - calculate gap first
        c0_gap_vector_m = pt2_start.position - pt1_end.position
        c0_gap_m = float(np.linalg.norm(c0_gap_vector_m))

        # Create report with required parameter
        report = ContinuityReport(c0_gap_m=c0_gap_m)
        report.c0_gap_vector_m = c0_gap_vector_m
        
        if report.c0_gap_m > self.c0_threshold_m:
            report.notes.append(f"C0 Fail: Position gap {report.c0_gap_m*1000:.2f}mm > threshold {self.c0_threshold_m*1000:.2f}mm.")
        else:
            report.notes.append(f"C0 Pass: Position gap {report.c0_gap_m*1000:.2f}mm.")

        # C1 Continuity (Velocity)
        vel1_end = self._estimate_velocity(segment1_end_points, len(segment1_end_points)-1, dt)
        vel2_start = self._estimate_velocity(segment2_start_points, 0, dt)
            
        if vel1_end is not None and vel2_start is not None:
            report.c1_velocity_diff_vector_mps = vel2_start - vel1_end
            report.c1_velocity_diff_mps = float(np.linalg.norm(report.c1_velocity_diff_vector_mps))
            report.c1_speed_diff_mps = float(np.linalg.norm(vel2_start) - np.linalg.norm(vel1_end))
            
            # Angle between velocity vectors
            norm_v1 = np.linalg.norm(vel1_end)
            norm_v2 = np.linalg.norm(vel2_start)
            if norm_v1 > 1e-9 and norm_v2 > 1e-9:
                cos_angle = np.dot(vel1_end, vel2_start) / (norm_v1 * norm_v2)
                report.c1_direction_angle_diff_rad = float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
            
            if report.c1_velocity_diff_mps > self.c1_threshold_mps:
                report.notes.append(f"C1 Fail: Velocity jump {report.c1_velocity_diff_mps:.3f}m/s > threshold {self.c1_threshold_mps:.3f}m/s.")
            else:
                report.notes.append(f"C1 Pass: Velocity jump {report.c1_velocity_diff_mps:.3f}m/s.")
        else:
            report.notes.append("C1 Info: Could not calculate velocity jump (insufficient data).")

        # C2 Continuity (Acceleration)
        accel1_end = self._estimate_acceleration(segment1_end_points, len(segment1_end_points)-1, dt)
        accel2_start = self._estimate_acceleration(segment2_start_points, 0, dt)

        if accel1_end is not None and accel2_start is not None:
            report.c2_acceleration_diff_vector_mps2 = accel2_start - accel1_end
            report.c2_acceleration_diff_mps2 = float(np.linalg.norm(report.c2_acceleration_diff_vector_mps2))
            report.notes.append(f"C2 Info: Acceleration diff {report.c2_acceleration_diff_mps2:.2f}m/s^2.")
        else:
            report.notes.append("C2 Info: Could not calculate acceleration jump.")
            
        return report

    def generate_smooth_transition(self,
                                  point1: TrajectoryPoint,
                                  point2: TrajectoryPoint,
                                  continuity_level: int,
                                  num_transition_points: int = 10) -> List[TrajectoryPoint]:
        """
        Generates a smooth transition segment between point1 and point2.
        Uses cubic spline interpolation for position and tangent matching.

        Args:
            point1: End TrajectoryPoint of the first segment
            point2: Start TrajectoryPoint of the second segment
            continuity_level: 0 for C0 (position), 1 for C1 (position and velocity)
            num_transition_points: Number of points to generate for the transition

        Returns:
            List[TrajectoryPoint]: The generated transition points
        """
        transition_points: List[TrajectoryPoint] = []
        
        p0 = point1.position
        p1 = point2.position

        if continuity_level == 0:  # C0 - Linear interpolation
            for i in range(1, num_transition_points + 1):
                u = i / (num_transition_points + 1)
                pos = p0 * (1 - u) + p1 * u
                
                # Interpolate winding angle
                angle_interp = point1.winding_angle_deg * (1 - u) + point2.winding_angle_deg * u
                
                tp = TrajectoryPoint(
                    position=pos,
                    winding_angle_deg=angle_interp,
                    surface_coords={},
                    arc_length_from_start=point1.arc_length_from_start + u * np.linalg.norm(p1 - p0)
                )
                transition_points.append(tp)
            return transition_points

        elif continuity_level >= 1:  # C1 - Cubic Hermite Spline
            # Get velocity vectors
            v0 = self._estimate_velocity([point1], 0) if not hasattr(point1, 'velocity_vector_mps') or point1.velocity_vector_mps is None else point1.velocity_vector_mps
            v1 = self._estimate_velocity([point2], 0) if not hasattr(point2, 'velocity_vector_mps') or point2.velocity_vector_mps is None else point2.velocity_vector_mps
            
            if v0 is None or v1 is None:
                print("Warning: Velocity info missing for C1 transition, falling back to C0 linear interpolation.")
                return self.generate_smooth_transition(point1, point2, 0, num_transition_points)

            # Cubic Hermite spline interpolation
            for i in range(1, num_transition_points + 1):
                u = i / (num_transition_points + 1)
                
                # Hermite basis functions
                h00 = 2 * u**3 - 3 * u**2 + 1
                h10 = -2 * u**3 + 3 * u**2
                h01 = u**3 - 2 * u**2 + u
                h11 = u**3 - u**2
                
                # Position interpolation
                pos = h00 * p0 + h10 * p1 + h01 * v0 + h11 * v1
                
                # Interpolate other properties
                angle_interp = point1.winding_angle_deg * (1 - u) + point2.winding_angle_deg * u
                
                # Calculate tangent vector for this point
                d_h00_du = 6 * u**2 - 6 * u
                d_h10_du = -6 * u**2 + 6 * u
                d_h01_du = 3 * u**2 - 4 * u + 1
                d_h11_du = 3 * u**2 - 2 * u
                
                tangent = d_h00_du * p0 + d_h10_du * p1 + d_h01_du * v0 + d_h11_du * v1
                tangent_normalized = tangent / np.linalg.norm(tangent) if np.linalg.norm(tangent) > 1e-9 else np.array([1.0, 0.0, 0.0])
                
                tp = TrajectoryPoint(
                    position=pos,
                    winding_angle_deg=angle_interp,
                    surface_coords={},
                    tangent_vector=tangent_normalized,
                    arc_length_from_start=point1.arc_length_from_start + u * np.linalg.norm(p1 - p0)
                )
                transition_points.append(tp)

            if continuity_level == 2:
                print("Warning: C2 continuity requested. This cubic spline ensures C1. "
                      "True C2 typically requires higher-order splines.")
            
            return transition_points
        else:
            raise ValueError(f"Unsupported continuity level: {continuity_level}")

    def validate_path_smoothness(self, trajectory: List[TrajectoryPoint], time_step_s: float = 0.01) -> PathQualityReport:
        """
        Validates the overall smoothness of a given trajectory.
        """
        report = PathQualityReport()
        if not trajectory or len(trajectory) < 2:
            report.notes.append("Trajectory too short to validate smoothness.")
            return report

        report.total_length_m = trajectory[-1].arc_length_from_start if trajectory[-1].arc_length_from_start > 0 else None

        positions = np.array([p.position for p in trajectory])
        velocities = []
        accelerations = []

        # Calculate velocities and accelerations
        for i, point in enumerate(trajectory):
            vel = self._estimate_velocity(trajectory, i, time_step_s)
            accel = self._estimate_acceleration(trajectory, i, time_step_s)
            velocities.append(vel)
            accelerations.append(accel)

        # Analyze continuity metrics
        max_c0_gap = 0.0
        c0_gaps = []
        max_c1_jump = 0.0
        c1_jumps = []
        max_c2_accel = 0.0
        max_c2_jump = 0.0
        c2_jumps = []

        for i in range(len(trajectory) - 1):
            # C0 Analysis
            gap = np.linalg.norm(positions[i+1] - positions[i])
            c0_gaps.append(gap)
            if gap > max_c0_gap:
                max_c0_gap = gap
            if gap > self.c0_threshold_m * 10:  # Allow reasonable step size
                report.points_exceeding_c0_threshold += 1
                report.is_smooth_c0 = False
            
            # C1 Analysis
            if velocities[i] is not None and velocities[i+1] is not None:
                vel_jump = np.linalg.norm(velocities[i+1] - velocities[i])
                c1_jumps.append(vel_jump)
                if vel_jump > max_c1_jump:
                    max_c1_jump = vel_jump
                if vel_jump > self.c1_threshold_mps:
                    report.points_exceeding_c1_threshold += 1
                    report.is_smooth_c1 = False
            
            # C2 Analysis
            if accelerations[i] is not None:
                accel_mag = np.linalg.norm(accelerations[i])
                if accel_mag > max_c2_accel:
                    max_c2_accel = accel_mag
                if accel_mag > self.machine_max_accel_mps2:
                    report.points_exceeding_c2_accel_limit += 1
                    report.is_smooth_c2 = False
            
            if accelerations[i] is not None and accelerations[i+1] is not None:
                accel_jump = np.linalg.norm(accelerations[i+1] - accelerations[i])
                c2_jumps.append(accel_jump)
                if accel_jump > max_c2_jump:
                    max_c2_jump = accel_jump

        # Populate report metrics
        report.max_c0_gap_mm = max_c0_gap * 1000
        report.avg_c0_gap_mm = np.mean(c0_gaps) * 1000 if c0_gaps else 0.0
        report.max_c1_velocity_jump_mps = max_c1_jump
        report.avg_c1_velocity_jump_mps = np.mean(c1_jumps) if c1_jumps else 0.0
        report.max_c2_acceleration_mps2 = max_c2_accel
        report.max_c2_acceleration_jump_mps2 = max_c2_jump

        # Summary assessment
        if report.is_smooth_c0 and report.is_smooth_c1 and report.is_smooth_c2:
            report.notes.append("Path meets specified C0, C1, and C2 criteria.")
        else:
            issues = []
            if not report.is_smooth_c0:
                issues.append("C0 (position)")
            if not report.is_smooth_c1:
                issues.append("C1 (velocity)")
            if not report.is_smooth_c2:
                issues.append("C2 (acceleration)")
            report.notes.append(f"Path has smoothness issues: {', '.join(issues)}")
            
        return report

    def repair_trajectory_continuity(self, 
                                   trajectory: List[TrajectoryPoint],
                                   continuity_level: int = 1) -> List[TrajectoryPoint]:
        """
        Repairs continuity issues in a trajectory by inserting smooth transitions
        where discontinuities are detected.

        Args:
            trajectory: Original trajectory with potential discontinuities
            continuity_level: Desired continuity level (0, 1, or 2)

        Returns:
            List[TrajectoryPoint]: Repaired trajectory with smooth transitions
        """
        if len(trajectory) < 2:
            return trajectory

        repaired_trajectory = [trajectory[0]]  # Start with first point
        
        for i in range(1, len(trajectory)):
            # Analyze continuity between current and previous point
            continuity_report = self.analyze_continuity(
                [repaired_trajectory[-1]], [trajectory[i]]
            )
            
            # Check if repair is needed
            needs_repair = False
            if continuity_report.c0_gap_m > self.c0_threshold_m:
                needs_repair = True
            if (continuity_level >= 1 and continuity_report.c1_velocity_diff_mps is not None and 
                continuity_report.c1_velocity_diff_mps > self.c1_threshold_mps):
                needs_repair = True
            
            if needs_repair:
                # Generate smooth transition
                transition_points = self.generate_smooth_transition(
                    repaired_trajectory[-1], trajectory[i], 
                    continuity_level, num_transition_points=5
                )
                repaired_trajectory.extend(transition_points)
            
            repaired_trajectory.append(trajectory[i])
        
        return repaired_trajectory
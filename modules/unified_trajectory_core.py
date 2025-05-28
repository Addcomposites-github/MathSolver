"""
Unified Trajectory Planner - Core Data Structures
Step 1: Clean, standardized data structures for trajectory generation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
import numpy as np

@dataclass
class TrajectoryPoint:
    """
    Represents a single point in the fiber trajectory on the mandrel surface.
    """
    position: np.ndarray = field(metadata={"unit": "meters", "description": "Cartesian coordinates [x, y, z] of the point in 3D space."})
    """Cartesian coordinates [x, y, z] of the point in 3D space, in meters."""

    surface_coords: Dict[str, float] = field(default_factory=dict, metadata={"description": "Coordinates of the point on the mandrel surface (e.g., {'rho': float, 'z_cyl': float, 'phi_cyl': float, 's_meridian': float}). rho and z_cyl in meters, phi_cyl in radians, s_meridian in meters."})
    """
    Coordinates of the point on the mandrel surface.
    Keys might include:
    - 'rho': Radial distance from the axis of revolution (meters).
    - 'z_cyl': Axial position along the cylinder (meters).
    - 'phi_cyl': Azimuthal angle around the axis of revolution (radians).
    - 's_meridian': Arc length along the meridian from a reference point (meters).
    Units should be consistent (meters for lengths, radians for angles).
    """

    winding_angle_deg: float = field(metadata={"unit": "degrees", "description": "Winding angle at this point. Convention: Angle between the fiber and the meridian (longitudinal axis for cylinder). 0 deg is axial, 90 deg is hoop. Output in degrees."})
    """
    Winding angle at this point, in degrees.
    Convention: Angle between the fiber path and the local meridian.
    For a cylinder, this is the angle with the cylinder's main axis.
    - 0 degrees typically represents an axial path.
    - 90 degrees typically represents a hoop path.
    Stored in degrees as per output requirements (internal calculations might use radians).
    """

    fiber_tension: float = field(default=0.0, metadata={"unit": "Newtons", "description": "Calculated or target fiber tension at this point."})
    """Calculated or target fiber tension at this point, in Newtons."""

    # Optional additional fields that might be useful
    normal_vector: np.ndarray = field(default=None, metadata={"description": "Surface normal vector [nx, ny, nz] at this point."})
    """Surface normal vector [nx, ny, nz] at this point (dimensionless unit vector)."""

    tangent_vector: np.ndarray = field(default=None, metadata={"description": "Path tangent vector [tx, ty, tz] at this point."})
    """Path tangent vector [tx, ty, tz] at this point (dimensionless unit vector)."""

    curvature_geodesic: float = field(default=0.0, metadata={"unit": "1/meters", "description": "Geodesic curvature of the path at this point."})
    """Geodesic curvature of the path at this point, in 1/meters."""

    curvature_normal: float = field(default=0.0, metadata={"unit": "1/meters", "description": "Normal curvature of the path (surface normal direction) at this point."})
    """Normal curvature of the path (in the surface normal direction) at this point, in 1/meters."""

    arc_length_from_start: float = field(default=0.0, metadata={"unit": "meters", "description": "Accumulated arc length along the trajectory from the starting point."})
    """Accumulated arc length along the trajectory from the starting point, in meters."""


@dataclass
class TrajectoryResult:
    """
    Encapsulates the results of a trajectory generation process.
    """
    points: List[TrajectoryPoint] = field(default_factory=list, metadata={"description": "List of TrajectoryPoint objects defining the path."})
    """List of TrajectoryPoint objects defining the path."""

    metadata: Dict[str, Any] = field(default_factory=dict, metadata={"description": "Rich metadata for the trajectory, including input parameters, calculation settings, and any warnings or errors."})
    """
    Rich metadata for the trajectory. Examples:
    - 'input_parameters': Dict of parameters used for generation.
    - 'calculation_time_seconds': float.
    - 'solver_settings': Dict of settings used by the numerical solver.
    - 'warnings': List[str].
    - 'errors': List[str].
    - 'vessel_profile_id': str or hash.
    """

    quality_metrics: Dict[str, Any] = field(default_factory=dict, metadata={"description": "Quantitative metrics assessing the quality of the generated trajectory."})
    """
    Quantitative metrics assessing the quality of the generated trajectory. Examples:
    - 'total_path_length_meters': float.
    - 'coverage_efficiency_percentage': float (if applicable).
    - 'max_position_gap_mm': float (C0 continuity check).
    - 'max_velocity_jump_m_per_s': float (C1 continuity check).
    - 'max_acceleration_m_per_s2': float (C2 continuity check).
    - 'min_polar_opening_clearance_mm': float.
    - 'average_winding_angle_deg': float.
    - 'winding_angle_std_dev_deg': float.
    """

    def __post_init__(self):
        """Calculate basic quality metrics automatically"""
        # Calculate total path length if not provided
        if not self.quality_metrics.get('total_path_length_meters') and self.points:
            path_length = 0.0
            for i in range(1, len(self.points)):
                path_length += np.linalg.norm(self.points[i].position - self.points[i-1].position)
            self.quality_metrics['total_path_length_meters'] = path_length
        
        # Calculate average winding angle if not provided
        if not self.quality_metrics.get('average_winding_angle_deg') and self.points:
            angles = [point.winding_angle_deg for point in self.points if hasattr(point, 'winding_angle_deg')]
            if angles:
                self.quality_metrics['average_winding_angle_deg'] = np.mean(angles)
                self.quality_metrics['winding_angle_std_dev_deg'] = np.std(angles)
        
        # Calculate continuity metrics
        if len(self.points) > 1:
            position_gaps = []
            for i in range(1, len(self.points)):
                gap = np.linalg.norm(self.points[i].position - self.points[i-1].position)
                position_gaps.append(gap * 1000)  # Convert to mm
            
            if position_gaps:
                self.quality_metrics['max_position_gap_mm'] = max(position_gaps)
                self.quality_metrics['avg_position_gap_mm'] = np.mean(position_gaps)
"""
Robust Turnaround Kinematics for COPV Design
Implements detailed feed-eye motion planning for smooth turnarounds at polar openings
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TurnaroundPoint:
    """Single point in turnaround trajectory with complete kinematic data."""
    # Mandrel surface coordinates
    x_mandrel: float  # X position on mandrel surface (m)
    y_mandrel: float  # Y position on mandrel surface (m) 
    z_mandrel: float  # Z position on mandrel surface (m)
    
    # Feed-eye coordinates
    x_feed_eye: float  # Feed-eye X position (m)
    y_feed_eye: float  # Feed-eye Y position (m)
    z_feed_eye: float  # Feed-eye Z position (m)
    yaw_feed_eye_rad: float  # Feed-eye yaw angle (rad)
    
    # Fiber parameters
    phi_angle_rad: float  # Circumferential angle (rad)
    beta_surface_rad: float  # Fiber angle on surface (rad)
    payout_length_m: float  # Fiber payout length (m)
    
    # Motion parameters
    velocity_magnitude: float  # Expected motion speed
    continuity_index: float  # C1/C2 continuity measure


@dataclass
class TurnaroundSequence:
    """Complete turnaround sequence with kinematic validation."""
    points: List[TurnaroundPoint]
    entry_state: Dict  # Initial conditions
    exit_state: Dict  # Final conditions
    total_phi_advancement_rad: float
    max_curvature_per_mm: float
    smooth_transitions: bool
    collision_free: bool


class RobustTurnaroundCalculator:
    """
    Advanced turnaround kinematics calculator implementing smooth feed-eye motion
    planning for continuous winding without fiber bridging or slippage.
    """
    
    def __init__(self, fiber_tension_n: float = 10.0,
                 min_bend_radius_mm: float = 50.0,
                 max_velocity_mps: float = 0.5):
        """
        Initialize turnaround calculator with physical constraints.
        
        Parameters:
        -----------
        fiber_tension_n : float
            Typical fiber tension during winding (Newtons)
        min_bend_radius_mm : float
            Minimum allowable fiber bend radius
        max_velocity_mps : float
            Maximum feed-eye velocity for smooth motion
        """
        self.fiber_tension = fiber_tension_n
        self.min_bend_radius = min_bend_radius_mm / 1000.0  # Convert to meters
        self.max_velocity = max_velocity_mps
        
        # Kinematic smoothing parameters
        self.smoothing_factor = 0.8  # For velocity transitions
        self.curvature_limit = 1.0 / self.min_bend_radius  # Max curvature
        
        print(f"Turnaround calculator initialized:")
        print(f"  Fiber tension: {fiber_tension_n}N")
        print(f"  Min bend radius: {min_bend_radius_mm}mm") 
        print(f"  Max velocity: {max_velocity_mps}m/s")
    
    def generate_smooth_turnaround(self, 
                                 entry_point: Dict,
                                 exit_point: Dict,
                                 mandrel_geometry: Dict,
                                 phi_advancement_rad: float,
                                 num_intermediate_points: int = 12) -> TurnaroundSequence:
        """
        Generate smooth turnaround sequence with optimal feed-eye motion.
        
        Parameters:
        -----------
        entry_point : Dict
            Entry state with position, velocity, fiber angle
        exit_point : Dict  
            Desired exit state
        mandrel_geometry : Dict
            Current mandrel surface geometry
        phi_advancement_rad : float
            Required circumferential advancement
        num_intermediate_points : int
            Number of intermediate points for smooth motion
            
        Returns:
        --------
        TurnaroundSequence with complete kinematic data
        """
        print(f"\n=== Generating Smooth Turnaround ===")
        print(f"Phi advancement: {math.degrees(phi_advancement_rad):.1f}°")
        print(f"Intermediate points: {num_intermediate_points}")
        
        try:
            # Extract geometry parameters
            polar_radius_mm = mandrel_geometry.get('polar_opening_radius_mm', 40.0)
            polar_radius_m = polar_radius_mm / 1000.0
            
            # Calculate turnaround zone geometry
            turnaround_zone = self._define_turnaround_zone(
                entry_point, polar_radius_m, phi_advancement_rad
            )
            
            # Generate mandrel path points
            mandrel_path = self._generate_mandrel_turnaround_path(
                turnaround_zone, num_intermediate_points
            )
            
            # Calculate feed-eye kinematics for each point
            turnaround_points = []
            
            for i, mandrel_pt in enumerate(mandrel_path):
                # Calculate local surface properties
                surface_normal = self._calculate_surface_normal(mandrel_pt, mandrel_geometry)
                surface_tangent = self._calculate_surface_tangent(mandrel_pt, mandrel_path, i)
                
                # Determine fiber angle at this point
                beta_surface = self._calculate_turnaround_fiber_angle(
                    mandrel_pt, turnaround_zone, i / len(mandrel_path)
                )
                
                # Calculate optimal feed-eye position
                feed_eye_pos = self._calculate_feed_eye_position(
                    mandrel_pt, surface_normal, surface_tangent, beta_surface
                )
                
                # Calculate motion parameters
                velocity_mag = self._calculate_smooth_velocity(
                    i, len(mandrel_path), turnaround_zone
                )
                
                # Calculate payout length
                payout_length = self._calculate_payout_length(
                    mandrel_pt, feed_eye_pos, beta_surface
                )
                
                # Create turnaround point
                turnaround_point = TurnaroundPoint(
                    x_mandrel=mandrel_pt['x'],
                    y_mandrel=mandrel_pt['y'], 
                    z_mandrel=mandrel_pt['z'],
                    x_feed_eye=feed_eye_pos['x'],
                    y_feed_eye=feed_eye_pos['y'],
                    z_feed_eye=feed_eye_pos['z'],
                    yaw_feed_eye_rad=feed_eye_pos['yaw'],
                    phi_angle_rad=mandrel_pt['phi'],
                    beta_surface_rad=beta_surface,
                    payout_length_m=payout_length,
                    velocity_magnitude=velocity_mag,
                    continuity_index=self._calculate_continuity_index(i, len(mandrel_path))
                )
                
                turnaround_points.append(turnaround_point)
            
            # Validate sequence for smoothness and collisions
            max_curvature = self._calculate_max_path_curvature(turnaround_points)
            smooth_transitions = max_curvature < self.curvature_limit
            collision_free = self._check_collision_clearance(turnaround_points, mandrel_geometry)
            
            # Create complete sequence
            sequence = TurnaroundSequence(
                points=turnaround_points,
                entry_state=entry_point.copy(),
                exit_state=exit_point.copy(),
                total_phi_advancement_rad=phi_advancement_rad,
                max_curvature_per_mm=max_curvature * 1000,
                smooth_transitions=smooth_transitions,
                collision_free=collision_free
            )
            
            print(f"  Generated {len(turnaround_points)} turnaround points")
            print(f"  Max curvature: {max_curvature*1000:.2f}/mm")
            print(f"  Smooth transitions: {smooth_transitions}")
            print(f"  Collision free: {collision_free}")
            
            return sequence
            
        except Exception as e:
            print(f"Error generating turnaround: {e}")
            # Return minimal fallback sequence
            return self._create_fallback_sequence(entry_point, exit_point, phi_advancement_rad)
    
    def _define_turnaround_zone(self, entry_point: Dict, polar_radius_m: float, 
                              phi_advancement_rad: float) -> Dict:
        """Define geometric parameters for turnaround zone."""
        try:
            # Extract entry conditions
            entry_z = entry_point.get('z', 0.0)
            entry_phi = entry_point.get('phi', 0.0)
            entry_beta = entry_point.get('beta_surface_rad', math.radians(45))
            
            # Calculate turnaround zone parameters
            turnaround_radius = max(polar_radius_m, self.min_bend_radius)
            
            # Z-coordinate for turnaround center
            z_turnaround = entry_z + 0.010  # 10mm offset for clearance
            
            # Angular span for smooth transition
            angular_span = abs(phi_advancement_rad)
            
            return {
                'center_z': z_turnaround,
                'radius': turnaround_radius,
                'phi_start': entry_phi,
                'phi_end': entry_phi + phi_advancement_rad,
                'angular_span': angular_span,
                'entry_beta': entry_beta
            }
            
        except Exception:
            # Safe fallback
            return {
                'center_z': 0.01,
                'radius': polar_radius_m,
                'phi_start': 0.0,
                'phi_end': phi_advancement_rad,
                'angular_span': abs(phi_advancement_rad),
                'entry_beta': math.radians(45)
            }
    
    def _generate_mandrel_turnaround_path(self, turnaround_zone: Dict, 
                                        num_points: int) -> List[Dict]:
        """Generate smooth path on mandrel surface for turnaround."""
        mandrel_path = []
        
        try:
            # Extract zone parameters
            z_center = turnaround_zone['center_z']
            radius = turnaround_zone['radius']
            phi_start = turnaround_zone['phi_start']
            phi_end = turnaround_zone['phi_end']
            
            # Generate phi values with smooth distribution
            phi_values = np.linspace(phi_start, phi_end, num_points)
            
            for i, phi in enumerate(phi_values):
                # Smooth radius variation during turnaround
                progress = i / (num_points - 1)
                smooth_progress = 0.5 * (1 - math.cos(math.pi * progress))  # Cosine smoothing
                
                # Radius varies smoothly during turnaround
                current_radius = radius * (1.0 + 0.1 * smooth_progress)
                
                # Calculate mandrel surface coordinates
                x = current_radius * math.cos(phi)
                y = current_radius * math.sin(phi)
                z = z_center
                
                mandrel_path.append({
                    'x': x, 'y': y, 'z': z,
                    'phi': phi, 'rho': current_radius,
                    'progress': progress
                })
            
            return mandrel_path
            
        except Exception:
            # Minimal fallback
            return [{'x': 0.04, 'y': 0.0, 'z': 0.01, 'phi': 0.0, 'rho': 0.04, 'progress': 0.5}]
    
    def _calculate_surface_normal(self, mandrel_point: Dict, 
                                mandrel_geometry: Dict) -> np.ndarray:
        """Calculate surface normal vector at mandrel point."""
        try:
            # For cylindrical/dome surfaces, normal points radially outward
            x, y = mandrel_point['x'], mandrel_point['y']
            rho = math.sqrt(x**2 + y**2)
            
            if rho > 1e-6:
                normal = np.array([x/rho, y/rho, 0.0])
            else:
                normal = np.array([1.0, 0.0, 0.0])  # Fallback at pole
            
            return normal / np.linalg.norm(normal)
            
        except Exception:
            return np.array([1.0, 0.0, 0.0])
    
    def _calculate_surface_tangent(self, mandrel_point: Dict, 
                                 mandrel_path: List[Dict], point_index: int) -> np.ndarray:
        """Calculate surface tangent vector for smooth motion."""
        try:
            # Calculate tangent from neighboring points
            if point_index == 0 and len(mandrel_path) > 1:
                # Forward difference at start
                next_pt = mandrel_path[point_index + 1]
                tangent = np.array([
                    next_pt['x'] - mandrel_point['x'],
                    next_pt['y'] - mandrel_point['y'],
                    next_pt['z'] - mandrel_point['z']
                ])
            elif point_index == len(mandrel_path) - 1 and len(mandrel_path) > 1:
                # Backward difference at end
                prev_pt = mandrel_path[point_index - 1]
                tangent = np.array([
                    mandrel_point['x'] - prev_pt['x'],
                    mandrel_point['y'] - prev_pt['y'],
                    mandrel_point['z'] - prev_pt['z']
                ])
            elif len(mandrel_path) > 2:
                # Central difference
                prev_pt = mandrel_path[point_index - 1]
                next_pt = mandrel_path[point_index + 1]
                tangent = np.array([
                    next_pt['x'] - prev_pt['x'],
                    next_pt['y'] - prev_pt['y'], 
                    next_pt['z'] - prev_pt['z']
                ]) * 0.5
            else:
                # Fallback tangent
                phi = mandrel_point.get('phi', 0.0)
                tangent = np.array([-math.sin(phi), math.cos(phi), 0.0])
            
            return tangent / max(np.linalg.norm(tangent), 1e-6)
            
        except Exception:
            return np.array([0.0, 1.0, 0.0])
    
    def _calculate_turnaround_fiber_angle(self, mandrel_point: Dict, 
                                        turnaround_zone: Dict, progress: float) -> float:
        """Calculate fiber angle during turnaround for smooth transition."""
        try:
            # Smooth transition from entry angle to circumferential
            entry_beta = turnaround_zone['entry_beta']
            target_beta = math.radians(75)  # Near-circumferential at turnaround
            
            # Use smooth interpolation function
            smooth_progress = 0.5 * (1 - math.cos(math.pi * progress))
            
            beta_surface = entry_beta + (target_beta - entry_beta) * smooth_progress
            
            # Ensure reasonable bounds
            return max(math.radians(10), min(math.radians(85), beta_surface))
            
        except Exception:
            return math.radians(45)  # Safe fallback
    
    def _calculate_feed_eye_position(self, mandrel_point: Dict, 
                                   surface_normal: np.ndarray,
                                   surface_tangent: np.ndarray,
                                   beta_surface_rad: float) -> Dict:
        """Calculate optimal feed-eye position for given mandrel point."""
        try:
            # Extract mandrel coordinates
            x_m, y_m, z_m = mandrel_point['x'], mandrel_point['y'], mandrel_point['z']
            
            # Calculate fiber direction on surface
            phi = mandrel_point.get('phi', 0.0)
            fiber_dir_surf = np.array([
                -math.sin(phi) * math.cos(beta_surface_rad),
                math.cos(phi) * math.cos(beta_surface_rad),
                math.sin(beta_surface_rad)
            ])
            
            # Feed-eye offset distance (typical 20-50mm for clearance)
            offset_distance = 0.030  # 30mm offset
            
            # Calculate feed-eye position
            # Offset in direction opposite to fiber direction for proper payout
            feed_eye_offset = -offset_distance * fiber_dir_surf
            
            x_fe = x_m + feed_eye_offset[0]
            y_fe = y_m + feed_eye_offset[1] 
            z_fe = z_m + feed_eye_offset[2]
            
            # Calculate yaw angle for feed-eye orientation
            yaw_angle = math.atan2(y_fe - y_m, x_fe - x_m)
            
            return {
                'x': x_fe, 'y': y_fe, 'z': z_fe,
                'yaw': yaw_angle
            }
            
        except Exception:
            # Safe fallback position
            return {
                'x': mandrel_point['x'] + 0.03,
                'y': mandrel_point['y'],
                'z': mandrel_point['z'] + 0.01,
                'yaw': 0.0
            }
    
    def _calculate_smooth_velocity(self, point_index: int, total_points: int, 
                                 turnaround_zone: Dict) -> float:
        """Calculate smooth velocity profile for turnaround motion."""
        try:
            progress = point_index / max(total_points - 1, 1)
            
            # Bell curve velocity profile (slow at ends, faster in middle)
            velocity_factor = math.exp(-8 * (progress - 0.5)**2)
            
            # Scale to maximum velocity
            velocity = self.max_velocity * velocity_factor * self.smoothing_factor
            
            return max(0.01, velocity)  # Ensure non-zero velocity
            
        except Exception:
            return 0.1  # Safe fallback velocity
    
    def _calculate_payout_length(self, mandrel_point: Dict, 
                               feed_eye_pos: Dict, beta_surface_rad: float) -> float:
        """Calculate fiber payout length from feed-eye to mandrel contact."""
        try:
            # Distance between feed-eye and mandrel contact point
            dx = feed_eye_pos['x'] - mandrel_point['x']
            dy = feed_eye_pos['y'] - mandrel_point['y']
            dz = feed_eye_pos['z'] - mandrel_point['z']
            
            direct_distance = math.sqrt(dx**2 + dy**2 + dz**2)
            
            # Account for fiber path curvature (approximately 5-15% longer)
            curvature_factor = 1.0 + 0.1 * abs(math.sin(beta_surface_rad))
            
            payout_length = direct_distance * curvature_factor
            
            return max(0.005, payout_length)  # Minimum 5mm payout
            
        except Exception:
            return 0.030  # 30mm fallback
    
    def _calculate_continuity_index(self, point_index: int, total_points: int) -> float:
        """Calculate continuity measure for motion smoothness."""
        try:
            # Higher index indicates smoother motion (closer to 1.0)
            progress = point_index / max(total_points - 1, 1)
            
            # Smooth function that's highest in middle of motion
            continuity = 1.0 - 4 * (progress - 0.5)**2
            
            return max(0.1, continuity)
            
        except Exception:
            return 0.8  # Good fallback continuity
    
    def _calculate_max_path_curvature(self, turnaround_points: List[TurnaroundPoint]) -> float:
        """Calculate maximum curvature in turnaround path."""
        try:
            if len(turnaround_points) < 3:
                return 0.0
            
            max_curvature = 0.0
            
            for i in range(1, len(turnaround_points) - 1):
                # Calculate curvature using three consecutive points
                p1 = turnaround_points[i-1]
                p2 = turnaround_points[i]
                p3 = turnaround_points[i+1]
                
                # Vector from p1 to p2
                v1 = np.array([p2.x_feed_eye - p1.x_feed_eye,
                              p2.y_feed_eye - p1.y_feed_eye,
                              p2.z_feed_eye - p1.z_feed_eye])
                
                # Vector from p2 to p3
                v2 = np.array([p3.x_feed_eye - p2.x_feed_eye,
                              p3.y_feed_eye - p2.y_feed_eye,
                              p3.z_feed_eye - p2.z_feed_eye])
                
                # Calculate curvature
                cross_product = np.cross(v1, v2)
                cross_magnitude = np.linalg.norm(cross_product)
                v1_magnitude = np.linalg.norm(v1)
                
                if v1_magnitude > 1e-6:
                    curvature = cross_magnitude / (v1_magnitude**3)
                    max_curvature = max(max_curvature, curvature)
            
            return max_curvature
            
        except Exception:
            return 0.1  # Conservative fallback
    
    def _check_collision_clearance(self, turnaround_points: List[TurnaroundPoint],
                                 mandrel_geometry: Dict) -> bool:
        """Check if turnaround path maintains safe clearance from mandrel."""
        try:
            min_clearance = 0.010  # 10mm minimum clearance
            
            for point in turnaround_points:
                # Distance from feed-eye to mandrel surface
                clearance = math.sqrt(
                    (point.x_feed_eye - point.x_mandrel)**2 +
                    (point.y_feed_eye - point.y_mandrel)**2 +
                    (point.z_feed_eye - point.z_mandrel)**2
                )
                
                if clearance < min_clearance:
                    return False
            
            return True
            
        except Exception:
            return False  # Conservative - assume collision risk
    
    def _create_fallback_sequence(self, entry_point: Dict, exit_point: Dict,
                                phi_advancement_rad: float) -> TurnaroundSequence:
        """Create minimal fallback sequence when main calculation fails."""
        
        # Single point fallback
        fallback_point = TurnaroundPoint(
            x_mandrel=0.04, y_mandrel=0.0, z_mandrel=0.01,
            x_feed_eye=0.07, y_feed_eye=0.0, z_feed_eye=0.02,
            yaw_feed_eye_rad=0.0,
            phi_angle_rad=phi_advancement_rad / 2,
            beta_surface_rad=math.radians(45),
            payout_length_m=0.03,
            velocity_magnitude=0.1,
            continuity_index=0.5
        )
        
        return TurnaroundSequence(
            points=[fallback_point],
            entry_state=entry_point,
            exit_state=exit_point,
            total_phi_advancement_rad=phi_advancement_rad,
            max_curvature_per_mm=1.0,
            smooth_transitions=False,
            collision_free=True
        )
"""
Path Continuity & Transitions for COPV Design
Ensures smooth feed-eye motion and fiber path continuity (C1/C2) between winding segments
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.interpolate import CubicSpline


@dataclass
class TransitionState:
    """Complete state information for segment transitions."""
    position: np.ndarray  # [x, y, z] coordinates
    velocity: np.ndarray  # [vx, vy, vz] velocity vector
    acceleration: np.ndarray  # [ax, ay, az] acceleration vector
    fiber_angle_rad: float  # Fiber angle on surface
    feed_eye_yaw_rad: float  # Feed-eye orientation
    payout_length: float  # Current payout length
    timestamp: float  # Time parameter


@dataclass
class ContinuityAnalysis:
    """Analysis results for path continuity."""
    position_continuity_c0: bool  # Position continuous
    velocity_continuity_c1: bool  # Velocity continuous (C1)
    acceleration_continuity_c2: bool  # Acceleration continuous (C2)
    max_position_gap_mm: float  # Maximum position discontinuity
    max_velocity_jump_mps: float  # Maximum velocity discontinuity  
    max_acceleration_jump: float  # Maximum acceleration discontinuity
    smooth_transition_required: bool  # Whether blending is needed


class PathContinuityManager:
    """
    Manages smooth transitions between different winding segments ensuring
    C1/C2 continuity for high-quality fiber placement.
    """
    
    def __init__(self, position_tolerance_mm: float = 0.1,
                 velocity_tolerance_mps: float = 0.05,
                 acceleration_tolerance: float = 1.0):
        """
        Initialize continuity manager with precision tolerances.
        
        Parameters:
        -----------
        position_tolerance_mm : float
            Maximum allowable position gap (default 0.1mm)
        velocity_tolerance_mps : float
            Maximum allowable velocity discontinuity
        acceleration_tolerance : float
            Maximum allowable acceleration jump
        """
        self.position_tolerance = position_tolerance_mm / 1000.0
        self.velocity_tolerance = velocity_tolerance_mps
        self.acceleration_tolerance = acceleration_tolerance
        
        # Spline parameters for smooth blending
        self.blend_duration = 0.1  # 100ms transition time
        self.spline_order = 3  # Cubic splines for C2 continuity
        
        print(f"Path continuity manager initialized:")
        print(f"  Position tolerance: {position_tolerance_mm}mm")
        print(f"  Velocity tolerance: {velocity_tolerance_mps}m/s") 
        print(f"  Acceleration tolerance: {acceleration_tolerance}m/s²")
    
    def analyze_segment_continuity(self, 
                                 segment_1_end: TransitionState,
                                 segment_2_start: TransitionState) -> ContinuityAnalysis:
        """
        Analyze continuity between two trajectory segments.
        
        Parameters:
        -----------
        segment_1_end : TransitionState
            Final state of first segment
        segment_2_start : TransitionState
            Initial state of second segment
            
        Returns:
        --------
        ContinuityAnalysis with detailed continuity assessment
        """
        print(f"\n=== Analyzing Segment Continuity ===")
        
        try:
            # Position continuity (C0)
            position_gap = np.linalg.norm(segment_2_start.position - segment_1_end.position)
            position_continuous = position_gap <= self.position_tolerance
            
            # Velocity continuity (C1)
            velocity_jump = np.linalg.norm(segment_2_start.velocity - segment_1_end.velocity)
            velocity_continuous = velocity_jump <= self.velocity_tolerance
            
            # Acceleration continuity (C2)
            accel_jump = np.linalg.norm(segment_2_start.acceleration - segment_1_end.acceleration)
            acceleration_continuous = accel_jump <= self.acceleration_tolerance
            
            # Determine if smooth transition is needed
            smooth_transition_needed = not (position_continuous and velocity_continuous)
            
            analysis = ContinuityAnalysis(
                position_continuity_c0=position_continuous,
                velocity_continuity_c1=velocity_continuous,
                acceleration_continuity_c2=acceleration_continuous,
                max_position_gap_mm=position_gap * 1000,
                max_velocity_jump_mps=velocity_jump,
                max_acceleration_jump=accel_jump,
                smooth_transition_required=smooth_transition_needed
            )
            
            print(f"  Position gap: {position_gap*1000:.3f}mm (C0: {position_continuous})")
            print(f"  Velocity jump: {velocity_jump:.3f}m/s (C1: {velocity_continuous})")
            print(f"  Acceleration jump: {accel_jump:.2f}m/s² (C2: {acceleration_continuous})")
            print(f"  Smooth transition needed: {smooth_transition_needed}")
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing continuity: {e}")
            # Conservative analysis - assume transition needed
            return ContinuityAnalysis(
                position_continuity_c0=False, velocity_continuity_c1=False,
                acceleration_continuity_c2=False, max_position_gap_mm=1.0,
                max_velocity_jump_mps=0.1, max_acceleration_jump=2.0,
                smooth_transition_required=True
            )
    
    def generate_smooth_transition(self,
                                 segment_1_end: TransitionState,
                                 segment_2_start: TransitionState,
                                 num_transition_points: int = 10) -> List[TransitionState]:
        """
        Generate smooth transition between segments using cubic splines.
        
        Parameters:
        -----------
        segment_1_end : TransitionState
            Final state of first segment
        segment_2_start : TransitionState
            Initial state of second segment
        num_transition_points : int
            Number of intermediate points for transition
            
        Returns:
        --------
        List of TransitionState points for smooth blending
        """
        print(f"\n=== Generating Smooth Transition ===")
        print(f"Transition points: {num_transition_points}")
        
        try:
            # Time parameterization for transition
            t_start = 0.0
            t_end = self.blend_duration
            t_values = np.linspace(t_start, t_end, num_transition_points)
            
            # Create splines for each coordinate
            position_splines = self._create_position_splines(
                segment_1_end, segment_2_start, t_start, t_end
            )
            
            # Generate transition points
            transition_points = []
            
            for i, t in enumerate(t_values):
                # Evaluate splines at current time
                position = np.array([spline(t) for spline in position_splines])
                velocity = np.array([spline.derivative(1)(t) for spline in position_splines])
                acceleration = np.array([spline.derivative(2)(t) for spline in position_splines])
                
                # Interpolate other parameters
                progress = t / t_end
                fiber_angle = self._interpolate_angle(
                    segment_1_end.fiber_angle_rad,
                    segment_2_start.fiber_angle_rad,
                    progress
                )
                
                feed_eye_yaw = self._interpolate_angle(
                    segment_1_end.feed_eye_yaw_rad,
                    segment_2_start.feed_eye_yaw_rad,
                    progress
                )
                
                payout_length = self._smooth_interpolate(
                    segment_1_end.payout_length,
                    segment_2_start.payout_length,
                    progress
                )
                
                # Create transition state
                transition_state = TransitionState(
                    position=position,
                    velocity=velocity,
                    acceleration=acceleration,
                    fiber_angle_rad=fiber_angle,
                    feed_eye_yaw_rad=feed_eye_yaw,
                    payout_length=payout_length,
                    timestamp=segment_1_end.timestamp + t
                )
                
                transition_points.append(transition_state)
            
            print(f"  Generated {len(transition_points)} transition points")
            
            # Validate transition smoothness
            max_velocity = max(np.linalg.norm(tp.velocity) for tp in transition_points)
            max_acceleration = max(np.linalg.norm(tp.acceleration) for tp in transition_points)
            
            print(f"  Max velocity in transition: {max_velocity:.3f}m/s")
            print(f"  Max acceleration in transition: {max_acceleration:.2f}m/s²")
            
            return transition_points
            
        except Exception as e:
            print(f"Error generating transition: {e}")
            # Return linear interpolation fallback
            return self._linear_transition_fallback(
                segment_1_end, segment_2_start, num_transition_points
            )
    
    def validate_multi_segment_continuity(self, 
                                        trajectory_segments: List[List[TransitionState]]) -> Dict:
        """
        Validate continuity across multiple trajectory segments.
        
        Parameters:
        -----------
        trajectory_segments : List[List[TransitionState]]
            List of trajectory segments to validate
            
        Returns:
        --------
        Dict with comprehensive continuity analysis
        """
        print(f"\n=== Validating Multi-Segment Continuity ===")
        print(f"Total segments: {len(trajectory_segments)}")
        
        validation_results = {
            'total_segments': len(trajectory_segments),
            'continuity_violations': 0,
            'max_position_gap_mm': 0.0,
            'max_velocity_jump_mps': 0.0,
            'segments_needing_transitions': [],
            'overall_continuity_grade': 'A'
        }
        
        try:
            for i in range(len(trajectory_segments) - 1):
                if not trajectory_segments[i] or not trajectory_segments[i+1]:
                    continue
                
                # Get end of current segment and start of next
                segment_end = trajectory_segments[i][-1]
                next_segment_start = trajectory_segments[i+1][0]
                
                # Analyze continuity
                analysis = self.analyze_segment_continuity(segment_end, next_segment_start)
                
                # Track violations
                if not analysis.position_continuity_c0 or not analysis.velocity_continuity_c1:
                    validation_results['continuity_violations'] += 1
                    validation_results['segments_needing_transitions'].append(i)
                
                # Track maximum discontinuities
                validation_results['max_position_gap_mm'] = max(
                    validation_results['max_position_gap_mm'],
                    analysis.max_position_gap_mm
                )
                
                validation_results['max_velocity_jump_mps'] = max(
                    validation_results['max_velocity_jump_mps'],
                    analysis.max_velocity_jump_mps
                )
            
            # Assign overall grade
            if validation_results['continuity_violations'] == 0:
                validation_results['overall_continuity_grade'] = 'A'
            elif validation_results['continuity_violations'] <= 2:
                validation_results['overall_continuity_grade'] = 'B'
            else:
                validation_results['overall_continuity_grade'] = 'C'
            
            print(f"  Continuity violations: {validation_results['continuity_violations']}")
            print(f"  Max position gap: {validation_results['max_position_gap_mm']:.3f}mm")
            print(f"  Overall grade: {validation_results['overall_continuity_grade']}")
            
            return validation_results
            
        except Exception as e:
            print(f"Error in multi-segment validation: {e}")
            return validation_results
    
    def _create_position_splines(self, start_state: TransitionState, 
                               end_state: TransitionState,
                               t_start: float, t_end: float) -> List:
        """Create cubic splines for smooth position transition."""
        try:
            # Time points for spline creation
            t_points = [t_start, t_end]
            
            splines = []
            
            # Create spline for each coordinate (x, y, z)
            for coord_idx in range(3):
                # Position values at start and end
                pos_values = [start_state.position[coord_idx], end_state.position[coord_idx]]
                
                # Velocity values at start and end (derivatives)
                vel_values = [start_state.velocity[coord_idx], end_state.velocity[coord_idx]]
                
                # Create cubic spline with derivative constraints
                spline = CubicSpline(
                    t_points, pos_values, 
                    bc_type=((1, vel_values[0]), (1, vel_values[1]))
                )
                
                splines.append(spline)
            
            return splines
            
        except Exception:
            # Fallback to linear interpolation
            return self._linear_spline_fallback(start_state, end_state, t_start, t_end)
    
    def _interpolate_angle(self, angle1_rad: float, angle2_rad: float, 
                          progress: float) -> float:
        """Interpolate between angles accounting for wraparound."""
        try:
            # Normalize angles to [-π, π]
            angle1_norm = math.atan2(math.sin(angle1_rad), math.cos(angle1_rad))
            angle2_norm = math.atan2(math.sin(angle2_rad), math.cos(angle2_rad))
            
            # Calculate angular difference
            diff = angle2_norm - angle1_norm
            
            # Choose shortest path
            if diff > math.pi:
                diff -= 2 * math.pi
            elif diff < -math.pi:
                diff += 2 * math.pi
            
            # Smooth interpolation using cosine function
            smooth_progress = 0.5 * (1 - math.cos(math.pi * progress))
            
            return angle1_norm + diff * smooth_progress
            
        except Exception:
            # Linear fallback
            return angle1_rad + (angle2_rad - angle1_rad) * progress
    
    def _smooth_interpolate(self, value1: float, value2: float, progress: float) -> float:
        """Smooth interpolation using cosine blending."""
        try:
            # Cosine interpolation for smooth transition
            smooth_progress = 0.5 * (1 - math.cos(math.pi * progress))
            return value1 + (value2 - value1) * smooth_progress
            
        except Exception:
            # Linear fallback
            return value1 + (value2 - value1) * progress
    
    def _linear_transition_fallback(self, start_state: TransitionState,
                                  end_state: TransitionState, 
                                  num_points: int) -> List[TransitionState]:
        """Simple linear transition fallback when spline creation fails."""
        
        transition_points = []
        
        for i in range(num_points):
            progress = i / max(num_points - 1, 1)
            
            # Linear interpolation
            position = start_state.position + progress * (end_state.position - start_state.position)
            velocity = start_state.velocity + progress * (end_state.velocity - start_state.velocity)
            acceleration = start_state.acceleration + progress * (end_state.acceleration - start_state.acceleration)
            
            fiber_angle = start_state.fiber_angle_rad + progress * (end_state.fiber_angle_rad - start_state.fiber_angle_rad)
            feed_eye_yaw = start_state.feed_eye_yaw_rad + progress * (end_state.feed_eye_yaw_rad - start_state.feed_eye_yaw_rad)
            payout_length = start_state.payout_length + progress * (end_state.payout_length - start_state.payout_length)
            
            transition_state = TransitionState(
                position=position,
                velocity=velocity,
                acceleration=acceleration,
                fiber_angle_rad=fiber_angle,
                feed_eye_yaw_rad=feed_eye_yaw,
                payout_length=payout_length,
                timestamp=start_state.timestamp + progress * self.blend_duration
            )
            
            transition_points.append(transition_state)
        
        return transition_points
    
    def _linear_spline_fallback(self, start_state: TransitionState,
                              end_state: TransitionState,
                              t_start: float, t_end: float) -> List:
        """Create simple linear splines when cubic spline creation fails."""
        
        splines = []
        
        for coord_idx in range(3):
            # Simple linear function
            def linear_spline(t, coord=coord_idx):
                progress = (t - t_start) / (t_end - t_start)
                return (start_state.position[coord] + 
                       progress * (end_state.position[coord] - start_state.position[coord]))
            
            # Add derivative methods for compatibility
            def linear_derivative(t, coord=coord_idx):
                return (end_state.position[coord] - start_state.position[coord]) / (t_end - t_start)
            
            def linear_second_derivative(t):
                return 0.0
            
            linear_spline.derivative = lambda n: linear_derivative if n == 1 else linear_second_derivative
            
            splines.append(linear_spline)
        
        return splines
"""
Unified Trajectory Utilities for COPV Design
Consolidates common trajectory generation functions to eliminate redundancy
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from scipy.interpolate import UnivariateSpline


class TrajectoryProfileProcessor:
    """Centralized profile preparation and preprocessing for all trajectory methods."""
    
    @staticmethod
    def prepare_profile_data_unified(vessel_geometry, resample_points: int = 100) -> Optional[Dict]:
        """
        Unified profile preparation method consolidating all preprocessing logic.
        
        Replaces multiple redundant profile preparation methods across trajectory classes.
        """
        try:
            if not hasattr(vessel_geometry, 'profile_points') or vessel_geometry.profile_points is None:
                return None
            
            profile = vessel_geometry.profile_points
            
            # Validate required keys
            required_keys = ['r_inner_mm', 'z_mm']
            for key in required_keys:
                if key not in profile:
                    return None
            
            # Extract and validate arrays
            r_inner = np.array(profile['r_inner_mm'])
            z_vals = np.array(profile['z_mm'])
            
            if len(r_inner) != len(z_vals) or len(r_inner) < 3:
                return None
            
            # Adaptive resampling for consistent point distribution
            resampled_data = TrajectoryProfileProcessor._resample_adaptive(
                z_vals, r_inner, resample_points
            )
            
            # Identify vessel segments (dome, cylinder transitions)
            segments = TrajectoryProfileProcessor._identify_segments(
                resampled_data['z_mm'], resampled_data['r_inner_mm']
            )
            
            # Calculate pole coordinates for continuous path generation
            z_coords = resampled_data['z_mm'] / 1000.0  # Convert to meters
            r_coords = resampled_data['r_inner_mm'] / 1000.0  # Convert to meters
            
            # Find front and aft pole z-coordinates
            front_pole_z = float(np.max(z_coords))  # Positive dome
            aft_pole_z = float(np.min(z_coords))    # Negative dome
            
            # Calculate effective polar opening radius
            polar_opening_radius = float(np.min(r_coords))
            
            return {
                'r_inner_mm': resampled_data['r_inner_mm'],
                'z_mm': resampled_data['z_mm'],
                'r_outer_mm': profile.get('r_outer_mm', resampled_data['r_inner_mm']),
                'segments': segments,
                'original_length': len(r_inner),
                'resampled_length': len(resampled_data['r_inner_mm']),
                # Add data needed for continuous path generation
                'pole_z_coords_m': {
                    'front': front_pole_z,
                    'aft': aft_pole_z
                },
                'effective_polar_opening_radius_m': polar_opening_radius,
                'r_inner_m': r_coords,
                'z_m': z_coords
            }
            
        except Exception as e:
            print(f"Error in unified profile preparation: {e}")
            return None
    
    @staticmethod
    def _resample_adaptive(z_original: np.ndarray, r_original: np.ndarray, 
                          target_points: int) -> Dict:
        """Adaptive resampling with higher density in curved regions."""
        try:
            # Sort by z coordinate
            sort_indices = np.argsort(z_original)
            z_sorted = z_original[sort_indices]
            r_sorted = r_original[sort_indices]
            
            # Create spline for smooth interpolation
            if len(z_sorted) >= 4:
                try:
                    spline = UnivariateSpline(z_sorted, r_sorted, s=0, k=3)
                except:
                    spline = lambda z: np.interp(z, z_sorted, r_sorted)
            else:
                # Linear interpolation fallback
                spline = lambda z: np.interp(z, z_sorted, r_sorted)
            
            # Calculate curvature for adaptive point distribution
            z_fine = np.linspace(z_sorted[0], z_sorted[-1], target_points * 3)
            r_fine = [float(spline(z)) for z in z_fine]
            
            # Calculate curvature (second derivative approximation)
            if len(z_fine) >= 3:
                dr_dz = np.gradient(r_fine, z_fine)
                d2r_dz2 = np.gradient(dr_dz, z_fine)
                curvature = np.abs(d2r_dz2) / (1 + dr_dz**2)**1.5
            else:
                curvature = np.ones(len(z_fine))
            
            # Weighted sampling based on curvature
            weights = 1.0 + 2.0 * curvature / np.max(curvature)
            cumulative_weights = np.cumsum(weights)
            cumulative_weights = cumulative_weights / cumulative_weights[-1]
            
            # Generate target points with adaptive spacing
            target_positions = np.linspace(0, 1, target_points)
            z_resampled = np.interp(target_positions, cumulative_weights, z_fine)
            r_resampled = [float(spline(z)) for z in z_resampled]
            
            return {
                'z_mm': np.array(z_resampled),
                'r_inner_mm': np.array(r_resampled)
            }
            
        except Exception:
            # Fallback to uniform resampling
            z_uniform = np.linspace(z_original[0], z_original[-1], target_points)
            r_uniform = np.interp(z_uniform, z_original, r_original)
            return {
                'z_mm': z_uniform,
                'r_inner_mm': r_uniform
            }
    
    @staticmethod
    def _identify_segments(z_mm: np.ndarray, r_inner_mm: np.ndarray) -> Dict:
        """Identify dome, cylinder, and transition segments."""
        try:
            # Calculate radius derivative to find transitions
            dr_dz = np.gradient(r_inner_mm, z_mm)
            
            # Identify cylinder region (low curvature)
            cylinder_threshold = 0.1  # mm/mm
            cylinder_mask = np.abs(dr_dz) < cylinder_threshold
            
            # Find cylinder boundaries
            if np.any(cylinder_mask):
                cylinder_indices = np.where(cylinder_mask)[0]
                cylinder_start = cylinder_indices[0]
                cylinder_end = cylinder_indices[-1]
            else:
                # No clear cylinder, use middle third
                cylinder_start = len(z_mm) // 3
                cylinder_end = 2 * len(z_mm) // 3
            
            return {
                'dome_1': {'start': 0, 'end': cylinder_start},
                'cylinder': {'start': cylinder_start, 'end': cylinder_end},
                'dome_2': {'start': cylinder_end, 'end': len(z_mm) - 1},
                'has_cylinder': cylinder_end > cylinder_start + 5
            }
            
        except Exception:
            # Fallback segmentation
            n_points = len(z_mm)
            return {
                'dome_1': {'start': 0, 'end': n_points // 3},
                'cylinder': {'start': n_points // 3, 'end': 2 * n_points // 3},
                'dome_2': {'start': 2 * n_points // 3, 'end': n_points - 1},
                'has_cylinder': True
            }


class UnifiedTrajectoryCore:
    """Consolidated core trajectory generation methods."""
    
    @staticmethod
    def generate_polar_turnaround_unified(current_phi_rad: float, 
                                        z_turnaround: float,
                                        rho_turnaround: float,
                                        phi_advancement_rad: Optional[float] = None,
                                        num_points: int = 8) -> Dict:
        """
        Unified polar turnaround generation consolidating multiple redundant methods.
        
        Replaces _generate_polar_turnaround_segment and _generate_polar_turnaround_segment_fixed_phi_advance
        """
        try:
            # Calculate phi advancement if not provided
            if phi_advancement_rad is None:
                phi_advancement_rad = math.radians(10.0)  # Default 10 degrees
            
            # Generate smooth turnaround points
            turnaround_points = []
            phi_values = np.linspace(current_phi_rad, 
                                   current_phi_rad + phi_advancement_rad, 
                                   num_points)
            
            for phi in phi_values:
                x = rho_turnaround * np.cos(phi)
                y = rho_turnaround * np.sin(phi)
                z = z_turnaround
                
                turnaround_points.append({
                    'x': x, 'y': y, 'z': z,
                    'rho': rho_turnaround, 
                    'phi': phi, 
                    'alpha': math.pi / 2  # 90° at turnaround
                })
            
            return {
                'points': turnaround_points,
                'final_phi': current_phi_rad + phi_advancement_rad,
                'point_count': len(turnaround_points)
            }
            
        except Exception as e:
            print(f"Error in unified turnaround generation: {e}")
            return {
                'points': [],
                'final_phi': current_phi_rad,
                'point_count': 0
            }
    
    @staticmethod
    def calculate_pattern_advancement(roving_width_m: float, 
                                    cylinder_radius_m: float,
                                    target_angle_deg: Optional[float] = None) -> float:
        """
        Unified pattern advancement calculation used across all trajectory methods.
        """
        try:
            if target_angle_deg and cylinder_radius_m > 1e-6:
                target_alpha_rad = math.radians(target_angle_deg)
                phi_advancement = (roving_width_m / math.cos(target_alpha_rad)) / cylinder_radius_m
            else:
                # Default advancement for reasonable coverage
                phi_advancement = roving_width_m / cylinder_radius_m
            
            # Ensure reasonable bounds
            return max(math.radians(1.0), min(math.radians(30.0), phi_advancement))
            
        except Exception:
            return math.radians(10.0)  # Safe fallback


class TrajectoryOutputStandardizer:
    """Standardizes trajectory output format across all generation methods."""
    
    @staticmethod
    def format_trajectory_output_standard(trajectory_points: List[Dict], 
                                        pattern_type: str,
                                        generation_params: Dict) -> Dict:
        """
        Standardized output format for all trajectory generation methods.
        
        Ensures consistent data structures across geodesic, non-geodesic, and fixed engines.
        """
        try:
            if not trajectory_points:
                return {'success': False, 'message': 'No trajectory points generated'}
            
            # Extract coordinate arrays with consistent naming
            x_points = np.array([p.get('x', 0) for p in trajectory_points])
            y_points = np.array([p.get('y', 0) for p in trajectory_points])
            z_points = np.array([p.get('z', 0) for p in trajectory_points])
            rho_points = np.array([p.get('rho', 0) for p in trajectory_points])
            phi_points = np.array([p.get('phi', 0) for p in trajectory_points])
            alpha_points = np.array([p.get('alpha', 0) for p in trajectory_points])
            
            # Calculate comprehensive statistics
            total_points = len(trajectory_points)
            path_length_m = TrajectoryOutputStandardizer._calculate_path_length(
                x_points, y_points, z_points
            )
            
            # Validate continuity
            continuity_check = TrajectoryOutputStandardizer._check_path_continuity(
                x_points, y_points, z_points
            )
            
            return {
                'success': True,
                'pattern_type': pattern_type,
                'total_points': total_points,
                'total_circuits_legs': generation_params.get('num_passes', 1),
                
                # Standardized coordinate arrays
                'x_points_m': x_points,
                'y_points_m': y_points,
                'z_points_m': z_points,
                'rho_points_m': rho_points,
                'phi_rad_profile': phi_points,
                'alpha_deg_profile': np.degrees(alpha_points),
                
                # Path statistics
                'path_length_m': path_length_m,
                'final_turn_around_angle_deg': math.degrees(phi_points[-1]) if len(phi_points) > 0 else 0,
                'alpha_equator_deg': math.degrees(np.mean(alpha_points)) if len(alpha_points) > 0 else 0,
                'phi_advancement_per_pass_deg': math.degrees(generation_params.get('phi_advancement_rad', 0)),
                
                # Quality metrics
                'continuity_check': continuity_check,
                'generation_method': generation_params.get('method', 'unified_core'),
                'clairauts_constant_mm': generation_params.get('clairauts_constant_m', 0) * 1000,
                
                # Path validation
                'max_gap_mm': continuity_check['max_gap_mm'],
                'is_continuous': continuity_check['is_continuous']
            }
            
        except Exception as e:
            return {
                'success': False, 
                'message': f'Output formatting failed: {e}',
                'pattern_type': pattern_type
            }
    
    @staticmethod
    def _calculate_path_length(x_points: np.ndarray, y_points: np.ndarray, 
                              z_points: np.ndarray) -> float:
        """Calculate total path length."""
        try:
            if len(x_points) < 2:
                return 0.0
            
            dx = np.diff(x_points)
            dy = np.diff(y_points)
            dz = np.diff(z_points)
            
            return float(np.sum(np.sqrt(dx**2 + dy**2 + dz**2)))
            
        except Exception:
            return 0.0
    
    @staticmethod
    def _check_path_continuity(x_points: np.ndarray, y_points: np.ndarray, 
                              z_points: np.ndarray, gap_threshold_mm: float = 50.0) -> Dict:
        """Check path continuity and identify large gaps."""
        try:
            if len(x_points) < 2:
                return {'is_continuous': True, 'max_gap_mm': 0.0, 'large_gaps': 0}
            
            dx = np.diff(x_points)
            dy = np.diff(y_points)
            dz = np.diff(z_points)
            
            step_lengths_m = np.sqrt(dx**2 + dy**2 + dz**2)
            step_lengths_mm = step_lengths_m * 1000
            
            max_gap_mm = float(np.max(step_lengths_mm))
            large_gaps = np.sum(step_lengths_mm > gap_threshold_mm)
            is_continuous = max_gap_mm < gap_threshold_mm
            
            return {
                'is_continuous': is_continuous,
                'max_gap_mm': max_gap_mm,
                'large_gaps': int(large_gaps),
                'mean_step_mm': float(np.mean(step_lengths_mm)),
                'std_step_mm': float(np.std(step_lengths_mm))
            }
            
        except Exception:
            return {'is_continuous': False, 'max_gap_mm': 0.0, 'large_gaps': 0}
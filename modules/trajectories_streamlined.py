"""
Streamlined Trajectory Planner for COPV Design
Consolidates all trajectory generation methods into a unified, efficient interface
"""

import numpy as np
import math
from typing import Dict, List, Optional
from scipy.integrate import solve_ivp
from scipy.interpolate import UnivariateSpline

from .geometry import VesselGeometry
from .trajectory_utils import (
    TrajectoryProfileProcessor,
    UnifiedTrajectoryCore, 
    TrajectoryOutputStandardizer
)


class StreamlinedTrajectoryPlanner:
    """
    Unified trajectory planner eliminating code redundancy and providing
    a single, robust interface for all trajectory generation needs.
    """
    
    def __init__(self, vessel_geometry: VesselGeometry, 
                 dry_roving_width_m: float = 0.003,
                 dry_roving_thickness_m: float = 0.0002,
                 roving_eccentricity_at_pole_m: float = 0.0,
                 target_cylinder_angle_deg: Optional[float] = None,
                 mu_friction_coefficient: float = 0.0):
        """Initialize streamlined trajectory planner."""
        
        self.vessel = vessel_geometry
        self.dry_roving_width_m = dry_roving_width_m
        self.dry_roving_thickness_m = dry_roving_thickness_m
        self.roving_eccentricity_at_pole_m = roving_eccentricity_at_pole_m
        self.target_cylinder_angle_deg = target_cylinder_angle_deg
        self.mu_friction_coefficient = mu_friction_coefficient
        
        # Initialize unified parameters
        self._initialize_parameters()
        
    def _initialize_parameters(self):
        """Initialize core parameters using unified methods."""
        try:
            # Get vessel dimensions
            if hasattr(self.vessel, 'inner_radius_m'):
                self.inner_radius_m = self.vessel.inner_radius_m
            elif hasattr(self.vessel, 'profile_points'):
                r_inner_mm = self.vessel.profile_points.get('r_inner_mm', [100])
                self.inner_radius_m = np.max(r_inner_mm) / 1000.0
            else:
                self.inner_radius_m = 0.1
            
            # Calculate effective polar opening radius
            self.effective_polar_opening_radius_m = max(
                self.roving_eccentricity_at_pole_m,
                self.dry_roving_width_m / 2.0
            )
            
            # Set Clairaut's constant
            self.clairauts_constant_for_path_m = self.effective_polar_opening_radius_m
            
            # Calculate unified pattern advancement
            self.phi_advancement_rad_per_pass = UnifiedTrajectoryCore.calculate_pattern_advancement(
                self.dry_roving_width_m,
                self.inner_radius_m,
                self.target_cylinder_angle_deg
            )
            
            print(f"Streamlined planner initialized:")
            print(f"  Inner radius: {self.inner_radius_m*1000:.1f}mm")
            print(f"  Clairaut constant: {self.clairauts_constant_for_path_m*1000:.1f}mm")
            print(f"  Pattern advancement: {math.degrees(self.phi_advancement_rad_per_pass):.1f}°")
            
        except Exception as e:
            print(f"Parameter initialization error: {e}")
            self._set_safe_defaults()
    
    def _set_safe_defaults(self):
        """Set safe default parameters."""
        self.inner_radius_m = 0.1
        self.effective_polar_opening_radius_m = 0.005
        self.clairauts_constant_for_path_m = 0.005
        self.phi_advancement_rad_per_pass = math.radians(10.0)
    
    def generate_trajectory(self, pattern_name: str, coverage_option: str, 
                          user_circuits: int = 1) -> Optional[Dict]:
        """
        Single unified dispatch method for all trajectory generation.
        
        Replaces calculate_trajectory and multiple pattern-specific methods.
        """
        print(f"\n=== Streamlined Trajectory Generation ===")
        print(f"Pattern: {pattern_name}, Coverage: {coverage_option}, Circuits: {user_circuits}")
        
        try:
            # Unified profile preparation
            profile_data = TrajectoryProfileProcessor.prepare_profile_data_unified(
                self.vessel, resample_points=100
            )
            
            if profile_data is None:
                return {'success': False, 'message': 'Failed to prepare vessel profile'}
            
            # Calculate number of passes
            num_passes = self._calculate_passes_unified(coverage_option, user_circuits)
            
            # Route to appropriate core engine
            if pattern_name.lower() in ['geodesic', 'geodesic_spiral']:
                return self._generate_geodesic_unified(profile_data, num_passes)
            elif pattern_name.lower() in ['non_geodesic', 'non_geodesic_spiral']:
                return self._generate_non_geodesic_unified(profile_data, num_passes)
            else:
                return self._generate_helical_unified(profile_data, num_passes)
                
        except Exception as e:
            print(f"Trajectory generation error: {e}")
            return {'success': False, 'message': f'Generation failed: {e}'}
    
    def _calculate_passes_unified(self, coverage_option: str, user_circuits: int) -> int:
        """Unified pass calculation logic."""
        if coverage_option == 'single_circuit':
            return 2
        elif coverage_option == 'full_coverage':
            if abs(self.phi_advancement_rad_per_pass) > 1e-7:
                passes_needed = math.ceil(2 * math.pi / abs(self.phi_advancement_rad_per_pass))
                return passes_needed if passes_needed % 2 == 0 else passes_needed + 1
            return 20
        else:  # user_defined
            return user_circuits * 2
    
    def _generate_geodesic_unified(self, profile_data: Dict, num_passes: int) -> Dict:
        """
        Unified geodesic trajectory generation consolidating all geodesic methods.
        """
        print(f"Generating geodesic trajectory: {num_passes} passes")
        
        all_points = []
        current_phi_rad = 0.0
        current_sin_alpha = self.clairauts_constant_for_path_m / self.inner_radius_m
        
        for pass_idx in range(num_passes):
            is_forward = (pass_idx % 2 == 0)
            
            # Generate geodesic segment using robust ODE
            segment_result = self._solve_geodesic_segment_unified(
                profile_data, current_phi_rad, current_sin_alpha, is_forward
            )
            
            if not segment_result.get('success', False):
                print(f"Pass {pass_idx + 1} failed, stopping generation")
                break
            
            # Add points (avoid duplicates)
            segment_points = segment_result['points']
            start_idx = 1 if len(all_points) > 0 else 0
            all_points.extend(segment_points[start_idx:])
            
            # Update state for next pass
            current_phi_rad = segment_result['final_phi']
            current_sin_alpha = segment_result['final_sin_alpha']
            
            # Pattern advancement
            current_phi_rad += self.phi_advancement_rad_per_pass
        
        # Standardized output
        return TrajectoryOutputStandardizer.format_trajectory_output_standard(
            all_points, 
            "geodesic_unified",
            {
                'num_passes': num_passes,
                'phi_advancement_rad': self.phi_advancement_rad_per_pass,
                'clairauts_constant_m': self.clairauts_constant_for_path_m,
                'method': 'streamlined_geodesic'
            }
        )
    
    def _solve_geodesic_segment_unified(self, profile_data: Dict, initial_phi: float,
                                      initial_sin_alpha: float, is_forward: bool) -> Dict:
        """Unified geodesic segment solver with robust error handling."""
        try:
            # Prepare interpolation functions
            z_mm = profile_data['z_mm']
            r_mm = profile_data['r_inner_mm']
            
            z_m = z_mm / 1000.0
            r_m = r_mm / 1000.0
            
            # Sort for interpolation
            sort_indices = np.argsort(z_m)
            z_sorted = z_m[sort_indices]
            r_sorted = r_m[sort_indices]
            
            # Create interpolation functions
            if len(z_sorted) >= 4:
                r_of_z = UnivariateSpline(z_sorted, r_sorted, s=0, k=3)
                dr_dz_func = r_of_z.derivative(1)
            else:
                def r_of_z(z): return np.interp(z, z_sorted, r_sorted)
                def dr_dz_func(z): 
                    h = 1e-6
                    return (r_of_z(z + h) - r_of_z(z - h)) / (2 * h)
            
            # Integration range
            z_start = z_sorted[0] if is_forward else z_sorted[-1]
            z_end = z_sorted[-1] if is_forward else z_sorted[0]
            z_eval = np.linspace(z_start, z_end, 50)
            
            # Geodesic ODE system
            def geodesic_ode(z, y):
                sin_alpha, phi = y
                try:
                    rho = float(r_of_z(z))
                    if rho <= self.clairauts_constant_for_path_m + 1e-6:
                        return [0.0, 1e6]
                    
                    sin_alpha_clairaut = self.clairauts_constant_for_path_m / rho
                    if sin_alpha_clairaut > 1.0:
                        return [0.0, 1e6]
                    
                    alpha = np.arcsin(sin_alpha_clairaut)
                    drho_dz = float(dr_dz_func(z))
                    
                    if abs(np.cos(alpha)) < 1e-9:
                        dphi_dz = 1e6
                    else:
                        ds_dz = np.sqrt(1.0 + drho_dz**2)
                        dphi_dz = (np.tan(alpha) / rho) * ds_dz
                    
                    return [0.0, dphi_dz]
                except:
                    return [0.0, 0.0]
            
            # Solve ODE
            sol = solve_ivp(
                geodesic_ode,
                [z_start, z_end],
                [initial_sin_alpha, initial_phi],
                t_eval=z_eval,
                method='RK45',
                rtol=1e-6,
                atol=1e-8
            )
            
            if not sol.success or len(sol.t) < 2:
                return {'success': False}
            
            # Generate trajectory points
            points = []
            for z, sin_alpha, phi in zip(sol.t, sol.y[0], sol.y[1]):
                try:
                    rho = float(r_of_z(z))
                    alpha = np.arcsin(np.clip(sin_alpha, -1.0, 1.0))
                    
                    x = rho * np.cos(phi)
                    y = rho * np.sin(phi)
                    
                    points.append({
                        'x': x, 'y': y, 'z': z,
                        'rho': rho, 'phi': phi, 'alpha': alpha
                    })
                except:
                    continue
            
            return {
                'success': True,
                'points': points,
                'final_phi': sol.y[1][-1],
                'final_sin_alpha': sol.y[0][-1]
            }
            
        except Exception as e:
            print(f"Geodesic segment error: {e}")
            return {'success': False}
    
    def _generate_non_geodesic_unified(self, profile_data: Dict, num_passes: int) -> Dict:
        """Unified non-geodesic generation - enhanced with friction effects."""
        print(f"Generating non-geodesic trajectory: {num_passes} passes, μ={self.mu_friction_coefficient}")
        
        # For now, fall back to geodesic with notification
        print("Note: Advanced non-geodesic with friction is under development")
        return self._generate_geodesic_unified(profile_data, num_passes)
    
    def _generate_helical_unified(self, profile_data: Dict, num_passes: int) -> Dict:
        """Unified helical pattern generation consolidating multiple helical methods."""
        print(f"Generating helical trajectory: {num_passes} passes")
        
        try:
            all_points = []
            current_phi = 0.0
            
            # Simple helical generation for demonstration
            z_mm = profile_data['z_mm']
            r_mm = profile_data['r_inner_mm']
            
            for pass_idx in range(num_passes):
                for i, (z_val, r_val) in enumerate(zip(z_mm, r_mm)):
                    z_m = z_val / 1000.0
                    r_m = r_val / 1000.0
                    
                    phi = current_phi + i * 0.1
                    x = r_m * np.cos(phi)
                    y = r_m * np.sin(phi)
                    
                    all_points.append({
                        'x': x, 'y': y, 'z': z_m,
                        'rho': r_m, 'phi': phi, 'alpha': math.radians(45)
                    })
                
                current_phi += self.phi_advancement_rad_per_pass
            
            return TrajectoryOutputStandardizer.format_trajectory_output_standard(
                all_points,
                "helical_unified", 
                {
                    'num_passes': num_passes,
                    'phi_advancement_rad': self.phi_advancement_rad_per_pass,
                    'method': 'streamlined_helical'
                }
            )
            
        except Exception as e:
            print(f"Helical generation error: {e}")
            return {'success': False, 'message': f'Helical generation failed: {e}'}


# Backward compatibility wrapper
class TrajectoryPlanner(StreamlinedTrajectoryPlanner):
    """
    Backward compatibility wrapper maintaining existing interface
    while using streamlined internals.
    """
    
    def calculate_trajectory(self, trajectory_params: Dict) -> Dict:
        """Legacy interface compatibility."""
        pattern_type = trajectory_params.get('pattern_type', 'geodesic')
        coverage = 'single_circuit'  # Default for legacy calls
        
        return self.generate_trajectory(pattern_type, coverage, 1)
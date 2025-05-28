"""
Full Feed-Eye Kinematics for Non-Geodesic Paths
Implements machine coordinate calculations (X,Y,Z,A) for paths with continuously 
varying winding angles derived from Koussios friction models.
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.integrate import solve_ivp


@dataclass
class NonGeodesicState:
    """Complete state for non-geodesic winding point."""
    # Mandrel surface coordinates
    z_mandrel: float  # Axial position on mandrel (m)
    rho_mandrel: float  # Radial position on mandrel (m)
    phi_mandrel: float  # Circumferential angle (rad)
    
    # Fiber parameters
    beta_surface_rad: float  # Fiber angle on surface (rad)
    sin_alpha: float  # Clairaut-like parameter for non-geodesic
    
    # Machine coordinates
    x_machine: float  # Feed-eye X position (m)
    y_machine: float  # Feed-eye Y position (m) 
    z_machine: float  # Feed-eye Z position (m)
    a_machine_rad: float  # Feed-eye yaw angle (rad)
    
    # Physical parameters
    payout_length: float  # Fiber payout length (m)
    friction_force: float  # Lateral friction force (N)
    stability_margin: float  # Path stability measure


class NonGeodesicKinematicsCalculator:
    """
    Advanced calculator for non-geodesic winding kinematics implementing
    Koussios friction model (Eq. 5.62) for continuously varying winding angles.
    """
    
    def __init__(self, friction_coefficient: float = 0.3,
                 fiber_tension_n: float = 10.0,
                 integration_tolerance: float = 1e-6):
        """
        Initialize non-geodesic kinematics calculator.
        
        Parameters:
        -----------
        friction_coefficient : float
            Static friction coefficient between fiber and mandrel
        fiber_tension_n : float
            Fiber tension during winding (Newtons)
        integration_tolerance : float
            Numerical integration tolerance for ODE solving
        """
        self.mu_friction = friction_coefficient
        self.fiber_tension = fiber_tension_n
        self.integration_tol = integration_tolerance
        
        # Physical constants
        self.gravity = 9.81  # m/s²
        self.fiber_density = 1600  # kg/m³ (typical carbon fiber)
        self.fiber_diameter = 7e-6  # 7 microns typical fiber diameter
        
        print(f"Non-geodesic kinematics calculator initialized:")
        print(f"  Friction coefficient: {friction_coefficient}")
        print(f"  Fiber tension: {fiber_tension_n}N")
        print(f"  Integration tolerance: {integration_tolerance}")
    
    def calculate_cylinder_non_geodesic(self, 
                                      cylinder_radius_m: float,
                                      cylinder_length_m: float,
                                      initial_angle_rad: float,
                                      num_points: int = 50) -> List[NonGeodesicState]:
        """
        Calculate non-geodesic path on cylinder using Koussios friction model.
        
        Based on Koussios Eq. 5.62: (sin α)' = c₂f² + c₁f + c₀
        where f = sin α and coefficients depend on surface curvature and friction.
        
        Parameters:
        -----------
        cylinder_radius_m : float
            Cylinder radius in meters
        cylinder_length_m : float
            Cylinder length in meters
        initial_angle_rad : float
            Initial fiber angle at cylinder start
        num_points : int
            Number of points along cylinder
            
        Returns:
        --------
        List of NonGeodesicState points along cylinder
        """
        print(f"\n=== Calculating Cylinder Non-Geodesic Path ===")
        print(f"Radius: {cylinder_radius_m*1000:.1f}mm, Length: {cylinder_length_m*1000:.1f}mm")
        print(f"Initial angle: {math.degrees(initial_angle_rad):.1f}°")
        
        try:
            # Koussios coefficients for cylinder (Eq. 5.62 specialized)
            # For cylinder: k_m = 0 (no meridional curvature), k_p = 1/R
            k_principal = 1.0 / cylinder_radius_m
            
            # Calculate Koussios coefficients
            coeffs = self._calculate_koussios_coefficients_cylinder(
                cylinder_radius_m, k_principal
            )
            
            # Set up ODE for sin(α) evolution
            z_span = np.linspace(0, cylinder_length_m, num_points)
            initial_sin_alpha = math.sin(initial_angle_rad)
            
            # Solve Koussios ODE
            ode_result = self._solve_koussios_ode_cylinder(
                z_span, initial_sin_alpha, coeffs, cylinder_radius_m
            )
            
            if not ode_result['success']:
                print("Warning: ODE integration failed, using fallback")
                return self._cylinder_fallback_path(
                    cylinder_radius_m, cylinder_length_m, initial_angle_rad, num_points
                )
            
            # Convert ODE solution to machine coordinates
            cylinder_states = []
            phi_current = 0.0
            
            for i, (z, sin_alpha) in enumerate(zip(ode_result['z_values'], ode_result['sin_alpha_values'])):
                # Calculate current fiber angle
                sin_alpha_clamped = max(-1.0, min(1.0, sin_alpha))
                beta_surface = math.asin(sin_alpha_clamped)
                
                # Update circumferential position
                if i > 0:
                    dz = z - ode_result['z_values'][i-1]
                    dphi = self._calculate_phi_increment_cylinder(
                        dz, beta_surface, cylinder_radius_m
                    )
                    phi_current += dphi
                
                # Calculate machine coordinates
                machine_coords = self._cylinder_to_machine_coordinates(
                    z, cylinder_radius_m, phi_current, beta_surface
                )
                
                # Calculate physical parameters
                friction_force = self._calculate_friction_force(beta_surface, cylinder_radius_m)
                stability = self._calculate_path_stability(beta_surface, self.mu_friction)
                
                # Create state
                state = NonGeodesicState(
                    z_mandrel=z,
                    rho_mandrel=cylinder_radius_m,
                    phi_mandrel=phi_current,
                    beta_surface_rad=beta_surface,
                    sin_alpha=sin_alpha_clamped,
                    x_machine=machine_coords['x'],
                    y_machine=machine_coords['y'],
                    z_machine=machine_coords['z'],
                    a_machine_rad=machine_coords['a'],
                    payout_length=machine_coords['payout'],
                    friction_force=friction_force,
                    stability_margin=stability
                )
                
                cylinder_states.append(state)
            
            print(f"  Generated {len(cylinder_states)} cylinder points")
            print(f"  Final angle: {math.degrees(beta_surface):.1f}°")
            print(f"  Total phi advancement: {math.degrees(phi_current):.1f}°")
            
            return cylinder_states
            
        except Exception as e:
            print(f"Error in cylinder calculation: {e}")
            return self._cylinder_fallback_path(
                cylinder_radius_m, cylinder_length_m, initial_angle_rad, num_points
            )
    
    def calculate_dome_non_geodesic(self,
                                  dome_profile: Dict,
                                  initial_angle_rad: float,
                                  num_points: int = 50) -> List[NonGeodesicState]:
        """
        Calculate non-geodesic path on dome using advanced numerical methods.
        
        Implements Koussios Eq. 5.62 for general surfaces of revolution with
        varying curvature along the meridian.
        
        Parameters:
        -----------
        dome_profile : Dict
            Dome geometry with z_mm, r_inner_mm arrays
        initial_angle_rad : float
            Initial fiber angle at dome start
        num_points : int
            Number of points along dome
            
        Returns:
        --------
        List of NonGeodesicState points along dome
        """
        print(f"\n=== Calculating Dome Non-Geodesic Path ===")
        print(f"Profile points: {len(dome_profile.get('z_mm', []))}")
        print(f"Initial angle: {math.degrees(initial_angle_rad):.1f}°")
        
        try:
            # Extract and validate dome profile
            z_mm = np.array(dome_profile['z_mm'])
            r_mm = np.array(dome_profile['r_inner_mm'])
            
            if len(z_mm) != len(r_mm) or len(z_mm) < 3:
                raise ValueError("Invalid dome profile data")
            
            # Convert to meters and sort
            z_m = z_mm / 1000.0
            r_m = r_mm / 1000.0
            sort_indices = np.argsort(z_m)
            z_sorted = z_m[sort_indices]
            r_sorted = r_m[sort_indices]
            
            # Resample for uniform spacing
            z_uniform = np.linspace(z_sorted[0], z_sorted[-1], num_points)
            r_uniform = np.interp(z_uniform, z_sorted, r_sorted)
            
            # Calculate curvatures at each point
            curvatures = self._calculate_dome_curvatures(z_uniform, r_uniform)
            
            # Solve non-geodesic ODE for dome
            dome_states = []
            phi_current = 0.0
            sin_alpha_current = math.sin(initial_angle_rad)
            
            for i, (z, r, curv) in enumerate(zip(z_uniform, r_uniform, curvatures)):
                # Update sin(α) using Koussios model
                if i > 0:
                    dz = z - z_uniform[i-1]
                    sin_alpha_current = self._update_sin_alpha_dome(
                        sin_alpha_current, dz, r, curv
                    )
                
                # Calculate current fiber angle
                sin_alpha_clamped = max(-1.0, min(1.0, sin_alpha_current))
                beta_surface = math.asin(sin_alpha_clamped)
                
                # Update circumferential position
                if i > 0:
                    dphi = self._calculate_phi_increment_dome(
                        z - z_uniform[i-1], r, beta_surface
                    )
                    phi_current += dphi
                
                # Calculate machine coordinates
                machine_coords = self._dome_to_machine_coordinates(
                    z, r, phi_current, beta_surface, curv
                )
                
                # Calculate physical parameters
                friction_force = self._calculate_friction_force(beta_surface, r)
                stability = self._calculate_path_stability(beta_surface, self.mu_friction)
                
                # Create state
                state = NonGeodesicState(
                    z_mandrel=z,
                    rho_mandrel=r,
                    phi_mandrel=phi_current,
                    beta_surface_rad=beta_surface,
                    sin_alpha=sin_alpha_clamped,
                    x_machine=machine_coords['x'],
                    y_machine=machine_coords['y'],
                    z_machine=machine_coords['z'],
                    a_machine_rad=machine_coords['a'],
                    payout_length=machine_coords['payout'],
                    friction_force=friction_force,
                    stability_margin=stability
                )
                
                dome_states.append(state)
            
            print(f"  Generated {len(dome_states)} dome points")
            print(f"  Final angle: {math.degrees(beta_surface):.1f}°")
            print(f"  Total phi advancement: {math.degrees(phi_current):.1f}°")
            
            return dome_states
            
        except Exception as e:
            print(f"Error in dome calculation: {e}")
            return self._dome_fallback_path(dome_profile, initial_angle_rad, num_points)
    
    def _calculate_koussios_coefficients_cylinder(self, radius: float, 
                                                k_principal: float) -> Dict:
        """Calculate Koussios coefficients for cylinder surface."""
        try:
            # For cylinder: E = 1, G = ρ² (where ρ is radius)
            # Koussios Eq. 5.62 coefficients
            E = 1.0
            G = radius**2
            
            # Principal curvatures: k_m = 0, k_p = 1/R for cylinder
            k_m = 0.0  # No meridional curvature
            k_p = k_principal
            
            # Friction-related terms
            mu = self.mu_friction
            
            # Calculate coefficients (simplified for cylinder)
            c2 = mu * k_p / math.sqrt(E * G)
            c1 = 0.0  # Linear term (typically small for cylinder)
            c0 = -mu * k_m / math.sqrt(E * G)
            
            return {
                'c2': c2, 'c1': c1, 'c0': c0,
                'E': E, 'G': G, 'k_m': k_m, 'k_p': k_p
            }
            
        except Exception:
            # Safe fallback coefficients
            return {'c2': 0.01, 'c1': 0.0, 'c0': 0.0, 'E': 1.0, 'G': radius**2, 'k_m': 0.0, 'k_p': 1/radius}
    
    def _solve_koussios_ode_cylinder(self, z_span: np.ndarray, 
                                   initial_sin_alpha: float,
                                   coeffs: Dict, radius: float) -> Dict:
        """Solve Koussios ODE for cylinder non-geodesic path."""
        try:
            def koussios_ode(z, y):
                """Koussios ODE: (sin α)' = c₂f² + c₁f + c₀ where f = sin α"""
                sin_alpha = y[0]
                
                # Clamp to valid range
                f = max(-1.0, min(1.0, sin_alpha))
                
                # Koussios equation
                dsinalpha_dz = coeffs['c2'] * f**2 + coeffs['c1'] * f + coeffs['c0']
                
                return [dsinalpha_dz]
            
            # Solve ODE
            sol = solve_ivp(
                koussios_ode,
                [z_span[0], z_span[-1]],
                [initial_sin_alpha],
                t_eval=z_span,
                method='RK45',
                rtol=self.integration_tol,
                atol=self.integration_tol * 0.1
            )
            
            return {
                'success': sol.success,
                'z_values': sol.t,
                'sin_alpha_values': sol.y[0] if sol.success else np.full_like(z_span, initial_sin_alpha)
            }
            
        except Exception:
            # Fallback to constant angle
            return {
                'success': False,
                'z_values': z_span,
                'sin_alpha_values': np.full_like(z_span, initial_sin_alpha)
            }
    
    def _calculate_phi_increment_cylinder(self, dz: float, beta: float, radius: float) -> float:
        """Calculate circumferential increment for cylinder."""
        try:
            if abs(math.cos(beta)) < 1e-6:
                return 0.0  # Nearly circumferential
            
            # For cylinder: dphi/dz = tan(β)/ρ
            dphi = (math.tan(beta) / radius) * dz
            
            return dphi
            
        except Exception:
            return 0.0
    
    def _cylinder_to_machine_coordinates(self, z: float, radius: float, 
                                       phi: float, beta: float) -> Dict:
        """Convert cylinder coordinates to machine coordinates."""
        try:
            # Mandrel surface position
            x_mandrel = radius * math.cos(phi)
            y_mandrel = radius * math.sin(phi)
            z_mandrel = z
            
            # Feed-eye offset calculation
            offset_distance = 0.030  # 30mm typical offset
            
            # Calculate fiber direction on surface
            fiber_dir = np.array([
                -math.sin(phi) * math.cos(beta),
                math.cos(phi) * math.cos(beta),
                math.sin(beta)
            ])
            
            # Feed-eye position (offset opposite to fiber direction)
            x_machine = x_mandrel - offset_distance * fiber_dir[0]
            y_machine = y_mandrel - offset_distance * fiber_dir[1]
            z_machine = z_mandrel - offset_distance * fiber_dir[2]
            
            # Feed-eye yaw angle
            a_machine = math.atan2(y_machine - y_mandrel, x_machine - x_mandrel)
            
            # Payout length
            payout_length = offset_distance * (1.0 + 0.1 * abs(math.sin(beta)))
            
            return {
                'x': x_machine, 'y': y_machine, 'z': z_machine,
                'a': a_machine, 'payout': payout_length
            }
            
        except Exception:
            # Safe fallback
            return {
                'x': radius + 0.03, 'y': 0.0, 'z': z,
                'a': 0.0, 'payout': 0.03
            }
    
    def _calculate_dome_curvatures(self, z_array: np.ndarray, 
                                 r_array: np.ndarray) -> np.ndarray:
        """Calculate curvatures along dome meridian."""
        try:
            # Calculate first and second derivatives
            dr_dz = np.gradient(r_array, z_array)
            d2r_dz2 = np.gradient(dr_dz, z_array)
            
            # Principal curvatures for surface of revolution
            # k_m = d²r/ds² / (1 + (dr/dz)²)^(3/2) (meridional)
            # k_p = (1/r) * dr/ds / sqrt(1 + (dr/dz)²) (circumferential)
            
            curvatures = []
            for i, (r, dr, d2r) in enumerate(zip(r_array, dr_dz, d2r_dz2)):
                if r < 1e-6:
                    r = 1e-6  # Avoid division by zero
                
                denominator = (1 + dr**2)**(3/2)
                k_m = d2r / max(denominator, 1e-6)
                k_p = dr / (r * math.sqrt(1 + dr**2))
                
                curvatures.append({'k_m': k_m, 'k_p': k_p})
            
            return curvatures
            
        except Exception:
            # Fallback to spherical curvature
            avg_radius = np.mean(r_array)
            return [{'k_m': 1/avg_radius, 'k_p': 1/avg_radius} for _ in r_array]
    
    def _update_sin_alpha_dome(self, sin_alpha: float, dz: float, 
                             radius: float, curvature: Dict) -> float:
        """Update sin(α) for dome using Koussios model."""
        try:
            # Simplified Koussios update for dome
            k_m = curvature['k_m']
            k_p = curvature['k_p']
            
            # Friction coefficient effect
            mu = self.mu_friction
            
            # Simplified coefficient calculation
            c_eff = mu * (k_p - k_m) / radius
            
            # Update sin(α)
            delta_sin_alpha = c_eff * sin_alpha * dz
            new_sin_alpha = sin_alpha + delta_sin_alpha
            
            # Clamp to valid range
            return max(-1.0, min(1.0, new_sin_alpha))
            
        except Exception:
            return sin_alpha  # No change if calculation fails
    
    def _calculate_phi_increment_dome(self, dz: float, radius: float, beta: float) -> float:
        """Calculate circumferential increment for dome."""
        try:
            if abs(math.cos(beta)) < 1e-6 or radius < 1e-6:
                return 0.0
            
            # For dome: dphi/dz ≈ tan(β)/ρ (simplified)
            dphi = (math.tan(beta) / radius) * dz
            
            return dphi
            
        except Exception:
            return 0.0
    
    def _dome_to_machine_coordinates(self, z: float, radius: float, 
                                   phi: float, beta: float, curvature: Dict) -> Dict:
        """Convert dome coordinates to machine coordinates."""
        try:
            # Similar to cylinder but with curvature adjustments
            x_mandrel = radius * math.cos(phi)
            y_mandrel = radius * math.sin(phi)
            z_mandrel = z
            
            # Adjust offset based on curvature
            base_offset = 0.030
            curvature_factor = 1.0 + 0.2 * abs(curvature.get('k_m', 0.0))
            offset_distance = base_offset * curvature_factor
            
            # Calculate fiber direction (simplified)
            fiber_dir = np.array([
                -math.sin(phi) * math.cos(beta),
                math.cos(phi) * math.cos(beta),
                math.sin(beta)
            ])
            
            # Feed-eye position
            x_machine = x_mandrel - offset_distance * fiber_dir[0]
            y_machine = y_mandrel - offset_distance * fiber_dir[1]
            z_machine = z_mandrel - offset_distance * fiber_dir[2]
            
            # Feed-eye yaw
            a_machine = math.atan2(y_machine - y_mandrel, x_machine - x_mandrel)
            
            # Payout length with curvature adjustment
            payout_length = offset_distance * (1.0 + 0.15 * abs(math.sin(beta)))
            
            return {
                'x': x_machine, 'y': y_machine, 'z': z_machine,
                'a': a_machine, 'payout': payout_length
            }
            
        except Exception:
            return {
                'x': radius + 0.03, 'y': 0.0, 'z': z,
                'a': 0.0, 'payout': 0.03
            }
    
    def _calculate_friction_force(self, beta: float, radius: float) -> float:
        """Calculate lateral friction force on fiber."""
        try:
            # Simplified friction force calculation
            normal_force = self.fiber_tension * math.cos(beta)
            friction_force = self.mu_friction * normal_force
            
            return max(0.0, friction_force)
            
        except Exception:
            return 0.0
    
    def _calculate_path_stability(self, beta: float, mu: float) -> float:
        """Calculate path stability margin."""
        try:
            # Stability based on friction angle vs fiber angle
            friction_angle = math.atan(mu)
            stability = friction_angle - abs(beta)
            
            # Normalize to [0, 1] range
            return max(0.0, min(1.0, stability / friction_angle))
            
        except Exception:
            return 0.5  # Neutral stability
    
    def _cylinder_fallback_path(self, radius: float, length: float, 
                              initial_angle: float, num_points: int) -> List[NonGeodesicState]:
        """Create fallback cylinder path when main calculation fails."""
        fallback_states = []
        
        z_values = np.linspace(0, length, num_points)
        phi_current = 0.0
        
        for z in z_values:
            # Simple linear angle progression
            progress = z / length
            beta = initial_angle * (1.0 - 0.2 * progress)  # Slight angle reduction
            
            # Update phi
            if len(fallback_states) > 0:
                dz = z - z_values[max(0, len(fallback_states)-1)]
                dphi = (math.tan(beta) / radius) * dz
                phi_current += dphi
            
            # Simple machine coordinates
            machine_coords = self._cylinder_to_machine_coordinates(z, radius, phi_current, beta)
            
            state = NonGeodesicState(
                z_mandrel=z, rho_mandrel=radius, phi_mandrel=phi_current,
                beta_surface_rad=beta, sin_alpha=math.sin(beta),
                x_machine=machine_coords['x'], y_machine=machine_coords['y'],
                z_machine=machine_coords['z'], a_machine_rad=machine_coords['a'],
                payout_length=machine_coords['payout'], friction_force=1.0, stability_margin=0.7
            )
            
            fallback_states.append(state)
        
        return fallback_states
    
    def _dome_fallback_path(self, dome_profile: Dict, initial_angle: float, 
                          num_points: int) -> List[NonGeodesicState]:
        """Create fallback dome path when main calculation fails."""
        try:
            z_mm = np.array(dome_profile.get('z_mm', [0, 50]))
            r_mm = np.array(dome_profile.get('r_inner_mm', [50, 100]))
            
            z_m = z_mm / 1000.0
            r_m = r_mm / 1000.0
            
            z_uniform = np.linspace(z_m[0], z_m[-1], num_points)
            r_uniform = np.interp(z_uniform, z_m, r_m)
            
            fallback_states = []
            phi_current = 0.0
            
            for i, (z, r) in enumerate(zip(z_uniform, r_uniform)):
                beta = initial_angle * (1.0 - 0.1 * i / num_points)
                
                if i > 0:
                    dz = z - z_uniform[i-1]
                    dphi = (math.tan(beta) / r) * dz
                    phi_current += dphi
                
                machine_coords = self._dome_to_machine_coordinates(
                    z, r, phi_current, beta, {'k_m': 1/r, 'k_p': 1/r}
                )
                
                state = NonGeodesicState(
                    z_mandrel=z, rho_mandrel=r, phi_mandrel=phi_current,
                    beta_surface_rad=beta, sin_alpha=math.sin(beta),
                    x_machine=machine_coords['x'], y_machine=machine_coords['y'],
                    z_machine=machine_coords['z'], a_machine_rad=machine_coords['a'],
                    payout_length=machine_coords['payout'], friction_force=1.0, stability_margin=0.7
                )
                
                fallback_states.append(state)
            
            return fallback_states
            
        except Exception:
            # Minimal fallback
            return [NonGeodesicState(
                z_mandrel=0.0, rho_mandrel=0.05, phi_mandrel=0.0,
                beta_surface_rad=initial_angle, sin_alpha=math.sin(initial_angle),
                x_machine=0.08, y_machine=0.0, z_machine=0.0, a_machine_rad=0.0,
                payout_length=0.03, friction_force=1.0, stability_margin=0.5
            )]
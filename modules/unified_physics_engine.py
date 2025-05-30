"""
Unified Trajectory Planner - Physics Engine
Step 2: Core trajectory mathematics engine for geodesic, non-geodesic, and helical paths
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
from .unified_trajectory_core import TrajectoryPoint

class PhysicsEngine:
    """
    Core engine for calculating fiber trajectories on a COPV mandrel surface.
    It handles geodesic, non-geodesic, and standard helical/hoop paths.
    """

    def __init__(self, vessel_meridian_points: np.ndarray):
        """
        Initializes the PhysicsEngine with the vessel's meridian profile.

        Args:
            vessel_meridian_points: A NumPy array of shape (N, 2) where each row
                                    is [rho, z] defining the meridian of the
                                    axisymmetric vessel. Points should be ordered,
                                    e.g., from bottom pole to top pole.
                                    Units are expected in meters.
        """
        if vessel_meridian_points.shape[1] != 2:
            raise ValueError("vessel_meridian_points must be of shape (N, 2)")

        # Ensure z is monotonically increasing for spline fitting rho(z)
        sorted_indices = np.argsort(vessel_meridian_points[:, 1])
        self.z_coords = vessel_meridian_points[sorted_indices, 1]
        self.rho_coords = vessel_meridian_points[sorted_indices, 0]

        # Remove duplicate z values to ensure strictly increasing sequence
        unique_indices = []
        prev_z = None
        tolerance = 1e-9
        
        for i, z_val in enumerate(self.z_coords):
            if prev_z is None or abs(z_val - prev_z) > tolerance:
                unique_indices.append(i)
                prev_z = z_val
        
        if len(unique_indices) < len(self.z_coords):
            print(f"Removed {len(self.z_coords) - len(unique_indices)} duplicate z-coordinates for interpolation")
            self.z_coords = self.z_coords[unique_indices]
            self.rho_coords = self.rho_coords[unique_indices]

        # Check if we have enough points for interpolation
        if len(self.z_coords) < 2:
            raise ValueError("Need at least 2 unique points for spline interpolation")

        if not np.all(np.diff(self.z_coords) > 0):
            # If z is still not strictly increasing, parameterize by cumulative arc length instead
            s_coords = np.zeros_like(self.z_coords)
            s_coords[1:] = np.cumsum(np.sqrt(np.diff(self.z_coords)**2 + np.diff(self.rho_coords)**2))
            
            # Ensure s_coords is also strictly increasing
            unique_s_indices = []
            prev_s = None
            for i, s_val in enumerate(s_coords):
                if prev_s is None or abs(s_val - prev_s) > tolerance:
                    unique_s_indices.append(i)
                    prev_s = s_val
            
            if len(unique_s_indices) < len(s_coords):
                s_coords = s_coords[unique_s_indices]
                self.z_coords = self.z_coords[unique_s_indices]
                self.rho_coords = self.rho_coords[unique_s_indices]
            
            self.meridian_param = s_coords
            self._rho_of_s = CubicSpline(s_coords, self.rho_coords, extrapolate=False)
            self._z_of_s = CubicSpline(s_coords, self.z_coords, extrapolate=False)
            self.parameterization_var = 's' # 's' for arc length
            self.s_min = s_coords[0]
            self.s_max = s_coords[-1]
        else:
            # If z is monotonic, we can use z as the parameter for rho(z)
            self.meridian_param = self.z_coords
            self._rho_of_z = CubicSpline(self.z_coords, self.rho_coords, extrapolate=False)
            self.parameterization_var = 'z' # 'z' for axial coordinate
            self.z_min = self.z_coords[0]
            self.z_max = self.z_coords[-1]

    def _get_rho_and_derivatives(self, param_val: float) -> Tuple[float, float, float]:
        """
        Calculates rho, d(rho)/d(param), and d^2(rho)/d(param)^2 at a given parameter value.
        The parameter is 'z' or 's' (meridian arc length) based on initialization.
        """
        if self.parameterization_var == 'z':
            rho = self._rho_of_z(param_val)
            d_rho_d_param = self._rho_of_z(param_val, 1)
            d2_rho_d_param2 = self._rho_of_z(param_val, 2)
            return float(rho), float(d_rho_d_param), float(d2_rho_d_param2)
        elif self.parameterization_var == 's':
            rho = self._rho_of_s(param_val)
            z = self._z_of_s(param_val)
            
            d_rho_d_s = self._rho_of_s(param_val, 1)
            d_z_d_s = self._z_of_s(param_val, 1)

            # Convert to d(rho)/dz from parametric form
            if abs(d_z_d_s) < 1e-9:
                d_rho_d_z = np.inf
                d2_rho_d_z2 = np.inf
            else:
                d_rho_d_z = d_rho_d_s / d_z_d_s
                d_rho_d_s_2 = self._rho_of_s(param_val, 2)
                d_z_d_s_2 = self._z_of_s(param_val, 2)
                d_d_rho_d_z_ds = (d_rho_d_s_2 * d_z_d_s - d_rho_d_s * d_z_d_s_2) / (d_z_d_s**2)
                d2_rho_d_z2 = d_d_rho_d_z_ds / d_z_d_s

            return float(rho), float(d_rho_d_z), float(d2_rho_d_z2)
        else:
            raise ValueError("Invalid parameterization variable.")

    def _calculate_surface_properties(self, param_val: float) -> Dict[str, float]:
        """
        Calculates local surface properties at a given parameter value (z or s).

        Returns a dictionary with:
            'rho': radius at param_val
            'd_rho_d_param': first derivative
            'd2_rho_d_param2': second derivative
            'k_m': meridional curvature
            'k_p': parallel curvature
            'G_metric': G metric (ds_meridian^2 / d_param^2)
            'E_metric': E metric (rho^2)
            'E_prime_metric': dE/d_param
        """
        rho, d_rho_d_p, d2_rho_d_p2 = self._get_rho_and_derivatives(param_val)

        if np.isinf(d_rho_d_p) or np.isinf(d2_rho_d_p2):
            k_m_val = 0
            k_p_val = 1.0 / rho if rho > 1e-9 else np.inf
            G_metric_val = np.inf
        elif abs(1 + d_rho_d_p**2) < 1e-9:
            k_m_val = np.inf
            k_p_val = np.inf
            G_metric_val = 0
        else:
            # Meridional curvature k_m = -rho_zz / (1 + rho_z^2)^(3/2)
            k_m_val = -d2_rho_d_p2 / ((1 + d_rho_d_p**2)**1.5)
            # Parallel curvature
            if self.parameterization_var == 'z':
                k_p_val = 1.0 / (rho * np.sqrt(1 + d_rho_d_p**2)) if rho > 1e-9 else np.inf
            elif self.parameterization_var == 's':
                k_p_val = 1.0 / (rho * np.sqrt(1 + d_rho_d_p**2)) if rho > 1e-9 else np.inf

        # Metrics for Koussios parameterization
        G_metric_val = 1 + d_rho_d_p**2
        E_metric_val = rho**2
        E_prime_metric_val = 2 * rho * d_rho_d_p

        return {
            "rho": rho,
            "d_rho_d_param": d_rho_d_p,
            "d2_rho_d_param2": d2_rho_d_p2,
            "k_m": k_m_val,
            "k_p": k_p_val,
            "G_metric": G_metric_val,
            "E_metric": E_metric_val,
            "E_prime_metric": E_prime_metric_val
        }

    def solve_geodesic(self,
                       clairaut_constant: float,
                       initial_param_val: float,
                       initial_phi_rad: float,
                       param_end_val: float,
                       num_points: int = 100,
                       **params) -> List[TrajectoryPoint]:
        """
        Solves for a geodesic path using Clairaut's theorem and ODE integration.
        """
        trajectory_points: List[TrajectoryPoint] = []
        
        def geodesic_ode(param, y):
            phi, s_path = y
            props = self._calculate_surface_properties(param)
            rho = props["rho"]

            if rho <= clairaut_constant or rho < 1e-6:
                print(f"[DEBUG] Geodesic hit boundary: rho={rho:.6f} <= clairaut={clairaut_constant:.6f} at param={param:.6f}")
                return [np.inf, np.inf]

            sin_alpha = clairaut_constant / rho
            if abs(sin_alpha) > 1.0:
                return [np.inf, np.inf]
            
            cos_alpha = np.sqrt(1.0 - sin_alpha**2)
            
            if self.parameterization_var == 'z':
                sqrt_G_metric = np.sqrt(props["G_metric"])
                d_phi_d_param = (sin_alpha / cos_alpha) / rho * sqrt_G_metric if cos_alpha > 1e-9 else np.copysign(np.inf, sin_alpha)
                d_s_path_d_param = sqrt_G_metric / cos_alpha if cos_alpha > 1e-9 else np.inf
            elif self.parameterization_var == 's':
                d_phi_d_param = sin_alpha / rho if rho > 1e-9 else np.copysign(np.inf, sin_alpha)
                d_s_path_d_param = 1.0 / cos_alpha if cos_alpha > 1e-9 else np.inf
            
            return [d_phi_d_param, d_s_path_d_param]

        # Integration parameters
        param_span = np.sort([initial_param_val, param_end_val])
        eval_params = np.linspace(param_span[0], param_span[1], num_points)
        y0 = [initial_phi_rad, 0.0]

        try:
            print(f"[DEBUG] Starting geodesic ODE integration: param {initial_param_val:.6f} to {param_end_val:.6f}, clairaut={clairaut_constant:.6f}")
            
            # Use robust ODE solver
            sol = solve_ivp(geodesic_ode, [initial_param_val, param_end_val], y0, 
                          t_eval=eval_params, method='RK45', atol=1e-8, rtol=1e-8)
            
            print(f"[DEBUG] ODE solver completed: success={sol.success}, message='{sol.message}'")
            
            if sol.success:
                for i, p_eval in enumerate(eval_params):
                    phi_val = sol.y[0][i]
                    s_path_val = sol.y[1][i]
                    
                    props = self._calculate_surface_properties(p_eval)
                    rho_val = props["rho"]
                    z_pos = p_eval if self.parameterization_var == 'z' else self._z_of_s(p_eval)
                    
                    pos_3d = np.array([rho_val * np.cos(phi_val), rho_val * np.sin(phi_val), z_pos])
                    
                    current_winding_angle_rad = np.arcsin(np.clip(clairaut_constant / rho_val, -1.0, 1.0)) if rho_val > 1e-9 else np.pi/2
                    
                    surf_coords = {'rho': rho_val, self.parameterization_var: p_eval, 'phi_rad': phi_val}
                    
                    point = TrajectoryPoint(
                        position=pos_3d,
                        surface_coords=surf_coords,
                        winding_angle_deg=np.degrees(current_winding_angle_rad),
                        arc_length_from_start=s_path_val
                    )
                    trajectory_points.append(point)
            
        except Exception as e:
            print(f"Warning: Geodesic integration failed: {e}")
            # Fallback to simple generation
            trajectory_points = self._fallback_geodesic_generation(
                clairaut_constant, initial_param_val, initial_phi_rad, param_end_val, num_points)

        return trajectory_points

    def solve_non_geodesic(self,
                          clairaut_constant: float,
                          friction_coefficient: float,
                          initial_param_val: float,
                          initial_phi_rad: float,
                          param_end_val: float,
                          num_points: int = 100,
                          **params) -> List[TrajectoryPoint]:
        """
        Solves for a non-geodesic path considering friction effects.
        Uses modified Clairaut's equation with friction terms.
        """
        trajectory_points: List[TrajectoryPoint] = []
        
        def non_geodesic_ode(param, y):
            phi, s_path, C_current = y  # C_current is the evolving "Clairaut constant"
            props = self._calculate_surface_properties(param)
            rho = props["rho"]
            k_m = props["k_m"]
            k_p = props["k_p"]

            if rho < 1e-6:
                return [np.inf, np.inf, 0]

            sin_alpha = C_current / rho
            if abs(sin_alpha) > 1.0:
                return [np.inf, np.inf, 0]
            
            cos_alpha = np.sqrt(1.0 - sin_alpha**2)
            
            # Friction force modification to Clairaut's theorem
            # dC/d(param) = friction_effects
            friction_term = friction_coefficient * rho * k_m * cos_alpha
            dC_d_param = friction_term
            
            if self.parameterization_var == 'z':
                sqrt_G_metric = np.sqrt(props["G_metric"])
                d_phi_d_param = (sin_alpha / cos_alpha) / rho * sqrt_G_metric if cos_alpha > 1e-9 else np.copysign(np.inf, sin_alpha)
                d_s_path_d_param = sqrt_G_metric / cos_alpha if cos_alpha > 1e-9 else np.inf
            elif self.parameterization_var == 's':
                d_phi_d_param = sin_alpha / rho if rho > 1e-9 else np.copysign(np.inf, sin_alpha)
                d_s_path_d_param = 1.0 / cos_alpha if cos_alpha > 1e-9 else np.inf
            
            return [d_phi_d_param, d_s_path_d_param, dC_d_param]

        # Integration parameters
        param_span = np.sort([initial_param_val, param_end_val])
        eval_params = np.linspace(param_span[0], param_span[1], num_points)
        y0 = [initial_phi_rad, 0.0, clairaut_constant]

        try:
            sol = solve_ivp(non_geodesic_ode, [initial_param_val, param_end_val], y0, 
                          t_eval=eval_params, method='RK45', atol=1e-8, rtol=1e-8)
            
            if sol.success:
                for i, p_eval in enumerate(eval_params):
                    phi_val = sol.y[0][i]
                    s_path_val = sol.y[1][i]
                    C_val = sol.y[2][i]
                    
                    props = self._calculate_surface_properties(p_eval)
                    rho_val = props["rho"]
                    z_pos = p_eval if self.parameterization_var == 'z' else self._z_of_s(p_eval)
                    
                    pos_3d = np.array([rho_val * np.cos(phi_val), rho_val * np.sin(phi_val), z_pos])
                    
                    current_winding_angle_rad = np.arcsin(np.clip(C_val / rho_val, -1.0, 1.0)) if rho_val > 1e-9 else np.pi/2
                    
                    surf_coords = {'rho': rho_val, self.parameterization_var: p_eval, 'phi_rad': phi_val}
                    
                    point = TrajectoryPoint(
                        position=pos_3d,
                        surface_coords=surf_coords,
                        winding_angle_deg=np.degrees(current_winding_angle_rad),
                        arc_length_from_start=s_path_val
                    )
                    trajectory_points.append(point)
                    
        except Exception as e:
            print(f"Warning: Non-geodesic integration failed: {e}")
            # Fallback to geodesic
            trajectory_points = self.solve_geodesic(clairaut_constant, initial_param_val, 
                                                  initial_phi_rad, param_end_val, num_points)

        return trajectory_points

    def solve_helical(self,
                     winding_angle_deg: float,
                     initial_param_val: float,
                     initial_phi_rad: float,
                     param_end_val: float,
                     num_points: int = 100,
                     **params) -> List[TrajectoryPoint]:
        """
        Solves for a constant winding angle helical path.
        """
        trajectory_points: List[TrajectoryPoint] = []
        winding_angle_rad = np.radians(winding_angle_deg)
        
        param_span = np.sort([initial_param_val, param_end_val])
        eval_params = np.linspace(param_span[0], param_span[1], num_points)
        
        s_path_val = 0.0
        prev_param = initial_param_val
        
        for i, p_eval in enumerate(eval_params):
            props = self._calculate_surface_properties(p_eval)
            rho_val = props["rho"]
            
            # For constant winding angle helical path
            # tan(alpha) = rho * d(phi)/d(z) for cylinder sections
            if self.parameterization_var == 'z':
                delta_param = p_eval - prev_param
                d_phi = np.tan(winding_angle_rad) * delta_param / rho_val if rho_val > 1e-6 else 0
                phi_val = initial_phi_rad + np.sum([np.tan(winding_angle_rad) * (eval_params[j] - (eval_params[j-1] if j > 0 else initial_param_val)) / 
                                                  self._calculate_surface_properties(eval_params[j])["rho"] 
                                                  for j in range(i+1)])
                
                sqrt_G_metric = np.sqrt(props["G_metric"])
                ds_increment = sqrt_G_metric * delta_param / np.cos(winding_angle_rad)
                s_path_val += ds_increment
            else:
                # For s parameterization
                delta_param = p_eval - prev_param
                d_phi = np.tan(winding_angle_rad) * delta_param / rho_val if rho_val > 1e-6 else 0
                phi_val = initial_phi_rad + d_phi * i  # Simplified
                s_path_val += delta_param / np.cos(winding_angle_rad)
            
            z_pos = p_eval if self.parameterization_var == 'z' else self._z_of_s(p_eval)
            pos_3d = np.array([rho_val * np.cos(phi_val), rho_val * np.sin(phi_val), z_pos])
            
            surf_coords = {'rho': rho_val, self.parameterization_var: p_eval, 'phi_rad': phi_val}
            
            point = TrajectoryPoint(
                position=pos_3d,
                surface_coords=surf_coords,
                winding_angle_deg=winding_angle_deg,
                arc_length_from_start=s_path_val
            )
            trajectory_points.append(point)
            prev_param = p_eval
            
        return trajectory_points

    def solve_hoop(self,
                   param_val: float,
                   num_points: int = 100,
                   **params) -> List[TrajectoryPoint]:
        """
        Solves for a pure hoop path at constant parameter value (z or s).
        """
        trajectory_points: List[TrajectoryPoint] = []
        
        props = self._calculate_surface_properties(param_val)
        rho_val = props["rho"]
        z_pos = param_val if self.parameterization_var == 'z' else self._z_of_s(param_val)
        
        # Generate points around the circumference
        phi_values = np.linspace(0, 2*np.pi, num_points)
        circumference = 2 * np.pi * rho_val
        
        for i, phi_val in enumerate(phi_values):
            pos_3d = np.array([rho_val * np.cos(phi_val), rho_val * np.sin(phi_val), z_pos])
            arc_length = (i / (num_points - 1)) * circumference if num_points > 1 else 0
            
            surf_coords = {'rho': rho_val, self.parameterization_var: param_val, 'phi_rad': phi_val}
            
            point = TrajectoryPoint(
                position=pos_3d,
                surface_coords=surf_coords,
                winding_angle_deg=90.0,  # Pure hoop is 90 degrees
                arc_length_from_start=arc_length
            )
            trajectory_points.append(point)
            
        return trajectory_points

    def _fallback_geodesic_generation(self, clairaut_constant, initial_param_val, 
                                    initial_phi_rad, param_end_val, num_points):
        """Fallback method for geodesic generation when ODE solver fails"""
        trajectory_points = []
        eval_params = np.linspace(initial_param_val, param_end_val, num_points)
        
        s_path_val = 0.0
        for i, p_eval in enumerate(eval_params):
            props = self._calculate_surface_properties(p_eval)
            rho_val = props["rho"]
            
            # Simple phi progression based on Clairaut's theorem
            if rho_val > clairaut_constant:
                sin_alpha = clairaut_constant / rho_val
                phi_increment = sin_alpha * (p_eval - initial_param_val) / rho_val
                phi_val = initial_phi_rad + phi_increment
            else:
                phi_val = initial_phi_rad
            
            z_pos = p_eval if self.parameterization_var == 'z' else self._z_of_s(p_eval)
            pos_3d = np.array([rho_val * np.cos(phi_val), rho_val * np.sin(phi_val), z_pos])
            
            if i > 0:
                s_path_val += np.linalg.norm(pos_3d - trajectory_points[-1].position)
            
            winding_angle_rad = np.arcsin(np.clip(clairaut_constant / rho_val, -1.0, 1.0)) if rho_val > 1e-9 else np.pi/2
            
            surf_coords = {'rho': rho_val, self.parameterization_var: p_eval, 'phi_rad': phi_val}
            
            point = TrajectoryPoint(
                position=pos_3d,
                surface_coords=surf_coords,
                winding_angle_deg=np.degrees(winding_angle_rad),
                arc_length_from_start=s_path_val
            )
            trajectory_points.append(point)
            
        return trajectory_points
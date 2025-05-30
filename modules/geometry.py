import numpy as np
import math
from typing import Dict, List, Tuple, Optional

class VesselGeometry:
    """
    Composite pressure vessel geometry calculations based on Koussios qrs-parameterization
    and other standard dome configurations.
    """
    
    def __init__(self, inner_diameter: float, wall_thickness: float, 
                 cylindrical_length: float, dome_type: str = "Isotensoid"):
        """
        Initialize vessel geometry.
        
        Parameters:
        -----------
        inner_diameter : float
            Inner diameter in mm
        wall_thickness : float  
            Wall thickness in mm
        cylindrical_length : float
            Length of cylindrical section in mm
        dome_type : str
            Type of dome ("Isotensoid", "Geodesic", "Elliptical", "Hemispherical")
        """
        self.inner_diameter = inner_diameter
        self.wall_thickness = wall_thickness
        self.cylindrical_length = cylindrical_length
        self.dome_type = dome_type
        
        # Derived basic parameters
        self.inner_radius = inner_diameter / 2
        self.outer_radius = self.inner_radius + wall_thickness
        self.outer_diameter = 2 * self.outer_radius
        
        # Dome parameters (will be set based on dome type)
        self.q_factor = 1.0
        self.r_factor = 1.0  
        self.s_factor = 0.5
        self.elliptical_aspect_ratio = 1.0
        
        # Generated profile data
        self.profile_points = None
        self.geometric_properties = None
        
        # Profile override flag for mandrel geometry
        self._use_profile_override = False
        
    def set_qrs_parameters(self, q: float, r: float, s: float):
        """Set qrs parameters for isotensoid dome design (Koussios method)"""
        self.q_factor = q
        self.r_factor = r
        self.s_factor = s
        
    def set_elliptical_parameters(self, aspect_ratio: float):
        """Set parameters for elliptical dome"""
        self.elliptical_aspect_ratio = aspect_ratio
        
    def generate_profile(self, num_points_per_dome: int = 50):
        """Generate the complete 2D meridian profile of the vessel.
           Origin (z=0) is at the center of the cylindrical section.
           Profile points are for the inner surface.
        """
        profile_r_inner = []
        profile_z_values = []

        # Generate dome profile (local coordinates: rho from 0 at pole, z_local from height at pole to 0 at base)
        if self.dome_type == "Hemispherical":
            local_dome_pts, dome_h = self._generate_hemispherical_profile(num_points_per_dome)
        elif self.dome_type == "Elliptical":
            local_dome_pts, dome_h = self._generate_elliptical_profile(num_points_per_dome)
        elif self.dome_type == "Isotensoid":
            local_dome_pts, dome_h = self._generate_isotensoid_profile(num_points_per_dome)
        elif self.dome_type == "Geodesic":
            # Use hemispherical as placeholder for now
            local_dome_pts, dome_h = self._generate_hemispherical_profile(num_points_per_dome)
        else:
            raise ValueError(f"Unsupported dome type: {self.dome_type}")
        
        self.dome_height = dome_h

        # Forward Dome (Top: z positive)
        # local_dome_pts[:,0] is rho (pole to cyl_radius)
        # local_dome_pts[:,1] is z_local (dome_height at pole, to 0 at cyl_radius)
        # We need to plot from z = cyl_len/2 + dome_height (pole) down to z = cyl_len/2 (cyl_junction)
        # So, z_abs = self.cylindrical_length / 2.0 + local_dome_pts[:,1]
        # And rho is local_dome_pts[:,0]
        profile_r_inner.extend(local_dome_pts[:,0])
        profile_z_values.extend(self.cylindrical_length / 2.0 + local_dome_pts[:,1])

        # Cylindrical Section
        # Last point of dome should be (self.inner_radius, self.cylindrical_length / 2.0)
        if not np.isclose(profile_r_inner[-1], self.inner_radius):  # Add if not perfectly connected
            profile_r_inner.append(self.inner_radius)
            profile_z_values.append(self.cylindrical_length / 2.0)
        
        profile_r_inner.append(self.inner_radius)  # Start of cylinder body
        profile_z_values.append(self.cylindrical_length / 2.0)
        profile_r_inner.append(self.inner_radius)  # End of cylinder body
        profile_z_values.append(-self.cylindrical_length / 2.0)

        # Aft Dome (Bottom: z negative)
        # We need points from (cyl_radius, -cyl_len/2) to (pole_radius_rho, -cyl_len/2 - dome_height)
        # local_dome_pts is still ordered pole to cyl_radius for rho, and dome_height to 0 for z_local
        # So we take it in reverse for rho: local_dome_pts[::-1,0]
        # And z_abs = -self.cylindrical_length/2.0 - local_dome_pts[::-1,1] (reversed z_local)
        profile_r_inner.extend(local_dome_pts[::-1,0])
        profile_z_values.extend(-self.cylindrical_length / 2.0 - local_dome_pts[::-1,1])

        # Standardized profile_points structure (all dimensions in mm)
        self.profile_points = {
            'r_inner_mm': np.array(profile_r_inner),
            'z_mm': np.array(profile_z_values), 
            'r_outer_mm': np.array(profile_r_inner) + self.wall_thickness,
            'dome_height_mm': self.dome_height
        }
        
        print(f"DEBUG geometry.py, generate_profile(): self.profile_points JUST ASSIGNED.")
        print(f"  Type: {type(self.profile_points)}")
        if isinstance(self.profile_points, dict):
            print(f"  Keys: {list(self.profile_points.keys())}")
            for key, value in self.profile_points.items():
                print(f"    Key '{key}' type: {type(value)}, Length (if applicable): {len(value) if hasattr(value, '__len__') else 'N/A'}")
                if key == 'r_inner' and hasattr(value, '__len__') and len(value) == 0:
                    print(f"    WARNING: '{key}' is empty!")
        else:
            print(f"  CRITICAL: self.profile_points is NOT a dict here!")
        
        self._calculate_geometric_properties()
        
    def _generate_isotensoid_profile(self, num_points_dome: int = 100):
        """
        Generate isotensoid dome profile using Koussios qrs-parameterization.
        Based on Koussios Thesis Chapter 4, Equations 4.12, 4.13, 4.20.
        """
        print("\n--- Debugging _generate_isotensoid_profile ---")
        from scipy.special import ellipkinc, ellipeinc
        
        q = self.q_factor
        r = self.r_factor
        print(f"Input q: {q}, r: {r}")
        print(f"Vessel inner_radius (target equatorial for dome): {self.inner_radius} mm")

        # Calculate dimensionless Y_min and Y_eq based on q, r (Koussios Thesis, Eq. 4.13)
        den_Y_calc = 1 + q + 2 * q * r
        if abs(den_Y_calc) < 1e-9:
            print(f"ERROR: Denominator for Y_min/Y_eq calculation is near zero (den_Y_calc={den_Y_calc}). Fallback.")
            R_dome_fallback = self.inner_radius
            dome_h_fallback = R_dome_fallback * 0.1  # Make it very flat to be obvious
            t_fallback = np.linspace(0, np.pi/2, num_points_dome)
            return np.vstack((R_dome_fallback * np.sin(t_fallback), dome_h_fallback * np.cos(t_fallback))).T, dome_h_fallback

        num_Y_calc = 1 + q + 2 * q * r + q**2 * (1 + r**2)

        r_limit = -(1 + q) / (2 * q) if q != 0 else (0 if 1 + q == 0 else -np.inf)
        print(f"Calculated r_limit for q={q}: {r_limit}")
        if r < r_limit - 1e-6:
            print(f"ERROR: r_factor ({r:.4f}) is below the limit ({r_limit:.4f}) for q={q:.4f}. num_Y_calc might be negative.")
            if num_Y_calc < 0:
                print(f"ERROR: num_Y_calc is negative ({num_Y_calc:.4f}). Cannot proceed with sqrt. Fallback.")
                R_dome_fallback = self.inner_radius
                dome_h_fallback = R_dome_fallback * 0.1
                t_fallback = np.linspace(0, np.pi/2, num_points_dome)
                return np.vstack((R_dome_fallback * np.sin(t_fallback), dome_h_fallback * np.cos(t_fallback))).T, dome_h_fallback
        
        if num_Y_calc < 0:
            num_Y_calc = 0  # Avoid math domain error if r is exactly at limit

        Y_min_dimless_raw = math.sqrt(num_Y_calc / den_Y_calc)
        print(f"Raw Y_min_dimless_raw: {Y_min_dimless_raw:.4f}")

        if q <= 1e-9:
            print("ERROR: q_factor is zero or too small for Y_eq calculation in isotensoid. Fallback.")
            R_dome_fallback = self.inner_radius
            dome_h_fallback = R_dome_fallback * 0.1
            t_fallback = np.linspace(0, np.pi/2, num_points_dome)
            return np.vstack((R_dome_fallback * np.sin(t_fallback), dome_h_fallback * np.cos(t_fallback))).T, dome_h_fallback

        # CORRECTED Y_eq calculation: Y_eq = Y_min * sqrt(q)
        Y_eq_dimless_raw = Y_min_dimless_raw * math.sqrt(q)
        print(f"Raw Y_eq_dimless_raw (equator/rho_0): {Y_eq_dimless_raw:.4f}")

        if abs(Y_eq_dimless_raw) < 1e-6:
            print("ERROR: Calculated Y_eq_dimless_raw is too small in isotensoid. Fallback.")
            R_dome_fallback = self.inner_radius
            dome_h_fallback = R_dome_fallback * 0.1
            t_fallback = np.linspace(0, np.pi/2, num_points_dome)
            return np.vstack((R_dome_fallback * np.sin(t_fallback), dome_h_fallback * np.cos(t_fallback))).T, dome_h_fallback

        rho_0_adjusted = self.inner_radius / Y_eq_dimless_raw
        print(f"Adjusted rho_0_adjusted: {rho_0_adjusted:.4f} mm")
        
        actual_dome_polar_radius_calc = Y_min_dimless_raw * rho_0_adjusted
        actual_dome_equatorial_radius_calc = Y_eq_dimless_raw * rho_0_adjusted
        print(f"Calculated dome polar radius: {actual_dome_polar_radius_calc:.4f} mm")
        print(f"Calculated dome equatorial radius (should be inner_radius): {actual_dome_equatorial_radius_calc:.4f} mm")

        theta_values = np.linspace(np.pi / 2.0, 0.0, num_points_dome)  # From pole (pi/2) to equator (0)

        denom_m_elliptic = 1 + 2 * q * (1 + r)
        if abs(denom_m_elliptic) < 1e-9:
            print(f"ERROR: Denominator for m_elliptic is near zero ({denom_m_elliptic}). Fallback.")
            R_dome_fallback = self.inner_radius
            dome_h_fallback = R_dome_fallback * 0.1
            t_fallback = np.linspace(0, np.pi/2, num_points_dome)
            return np.vstack((R_dome_fallback * np.sin(t_fallback), dome_h_fallback * np.cos(t_fallback))).T, dome_h_fallback

        m_elliptic = (q - 1) / denom_m_elliptic
        print(f"Elliptic parameter m_elliptic: {m_elliptic:.4f}")

        m_elliptic_clamped = np.clip(m_elliptic, 0.0, 1.0)  # Ensure m is in [0,1] for scipy
        if not np.isclose(m_elliptic, m_elliptic_clamped):
            print(f"Warning: m_elliptic ({m_elliptic:.4f}) was clamped to {m_elliptic_clamped:.4f}.")

        try:
            ell_F_values = ellipkinc(theta_values, m_elliptic_clamped)
            ell_E_values = ellipeinc(theta_values, m_elliptic_clamped)
        except Exception as e:
            print(f"Error calculating elliptic integrals: {e}. Fallback.")
            R_dome_fallback = self.inner_radius
            dome_h_fallback = R_dome_fallback * 0.1
            t_fallback = np.linspace(0, np.pi/2, num_points_dome)
            return np.vstack((R_dome_fallback * np.sin(t_fallback), dome_h_fallback * np.cos(t_fallback))).T, dome_h_fallback

        coeff_Z_den_sq_term = 1 + 2 * q * (1 + r)
        if coeff_Z_den_sq_term < 0:
            print(f"ERROR: Term under square root for Z coefficient is negative ({coeff_Z_den_sq_term}). Fallback.")
            R_dome_fallback = self.inner_radius
            dome_h_fallback = R_dome_fallback * 0.1
            t_fallback = np.linspace(0, np.pi/2, num_points_dome)
            return np.vstack((R_dome_fallback * np.sin(t_fallback), dome_h_fallback * np.cos(t_fallback))).T, dome_h_fallback
        
        coeff_Z_factor = Y_min_dimless_raw / math.sqrt(coeff_Z_den_sq_term)
        
        term_E_component = (1 + 2 * q * (1 + r)) * ell_E_values
        term_F_component = (1 + q + q * r) * ell_F_values
        
        Z_dimless_profile = coeff_Z_factor * (term_E_component - term_F_component)
        print(f"Sample Z_dimless_profile (pole to equator): {Z_dimless_profile[:3]}...{Z_dimless_profile[-3:]}")

        dome_height_dimless = Z_dimless_profile[0]  # Value at theta = pi/2 (pole)
        if Z_dimless_profile[-1] > 1e-6:  # Value at theta = 0 (equator) should be close to 0
            print(f"Warning: Z_dimless at equator (theta=0) is {Z_dimless_profile[-1]:.4f}, expected ~0.")
        
        actual_dome_height_m = dome_height_dimless * rho_0_adjusted
        print(f"Calculated actual_dome_height_m: {actual_dome_height_m:.4f} mm")

        # FIXED: Proper dome radius calculation for isotensoid profile
        # The dome should vary from polar radius at pole to equatorial radius at base
        # Correct isotensoid radius profile calculation
        cos_theta = np.cos(theta_values)
        sin_theta = np.sin(theta_values)
        
        # Proper meridional radius calculation for isotensoid dome
        # At pole (theta=pi/2): rho should be small (polar radius)
        # At equator (theta=0): rho should be inner_radius (equatorial radius)
        
        # Use proper isotensoid radius formula
        rho_dimless_profile = np.sqrt(
            Y_eq_dimless_raw**2 * cos_theta**2 + 
            Y_min_dimless_raw**2 * sin_theta**2
        )
        
        # Additional correction to ensure proper dome curvature
        # Blend between polar and equatorial radius based on theta
        theta_normalized = (np.pi/2 - theta_values) / (np.pi/2)  # 0 at pole, 1 at equator
        
        # Define polar radius as fraction of equatorial radius
        polar_radius_ratio = Y_min_dimless_raw / Y_eq_dimless_raw if Y_eq_dimless_raw > 0 else 0.3
        polar_radius_ratio = np.clip(polar_radius_ratio, 0.1, 0.9)  # Ensure reasonable dome shape
        
        # Apply curvature correction
        curvature_factor = polar_radius_ratio + (1 - polar_radius_ratio) * theta_normalized
        rho_dimless_profile = rho_dimless_profile * curvature_factor
        
        print(f"[DomeFix] Polar radius ratio: {polar_radius_ratio:.3f}")
        print(f"[DomeFix] Theta range: {np.min(theta_values):.3f} to {np.max(theta_values):.3f}")
        print(f"[DomeFix] Curvature factor range: {np.min(curvature_factor):.3f} to {np.max(curvature_factor):.3f}")
        rho_abs_profile = rho_dimless_profile * rho_0_adjusted
        print(f"Sample rho_abs_profile (pole to equator): {rho_abs_profile[:3]}...{rho_abs_profile[-3:]}")

        z_local_dome_abs = Z_dimless_profile * rho_0_adjusted  # From dome_height at pole, to 0 at equator
        print(f"Sample z_local_dome_abs (pole to equator): {z_local_dome_abs[:3]}...{z_local_dome_abs[-3:]}")
        
        if actual_dome_height_m < 1e-3 or np.any(np.isnan(rho_abs_profile)) or np.any(np.isnan(z_local_dome_abs)):
            print("ERROR: Calculated dome height is too small or NaN in profile. Fallback.")
            R_dome_fallback = self.inner_radius
            dome_h_fallback = R_dome_fallback * 0.1
            t_fallback = np.linspace(0, np.pi/2, num_points_dome)
            return np.vstack((R_dome_fallback * np.sin(t_fallback), dome_h_fallback * np.cos(t_fallback))).T, dome_h_fallback

        print("--- End Debugging _generate_isotensoid_profile ---\n")
        return np.vstack((rho_abs_profile, z_local_dome_abs)).T, actual_dome_height_m
    
    def _calculate_isotensoid_koussios_qrs_profile(self, num_points_dome: int):
        """
        Calculates the Koussios qrs-parameterized isotensoid dome profile.
        Profile is from (polar_opening_radius, dome_height) to (cylinder_radius, 0 - local z).
        Uses dimensionless coordinates Y = rho/rho_0, Z = z/rho_0.
        Based on Koussios Thesis Ch. 4, Equations 4.12, 4.13, and 4.20.
        """
        from scipy.special import ellipkinc, ellipeinc  # F(phi, m), E(phi, m)
        
        q = self.q_factor
        r = self.r_factor

        # Calculate Y_min and Y_eq based on q, r (Koussios Thesis, Eq. 4.13)
        den_Y_calc = 1 + q + 2 * q * r
        if abs(den_Y_calc) < 1e-9:
            print("WARNING: Invalid q,r parameters for qrs calculation.")
            return self._generate_simplified_isotensoid_profile(num_points_dome)

        num_Y_calc = 1 + q + 2 * q * r + q**2 * (1 + r**2)
        
        # Check r_limit based on Koussios Eq. 4.15
        r_limit = -(1 + q) / (2 * q) if q > 1e-9 else 0
        if r < r_limit - 1e-6:
            print(f"WARNING: r_factor ({r:.3f}) below limit ({r_limit:.3f}) for q={q:.3f}.")
            return self._generate_simplified_isotensoid_profile(num_points_dome)

        if num_Y_calc < 0:
            print("WARNING: Invalid geometry parameters for isotensoid.")
            return self._generate_simplified_isotensoid_profile(num_points_dome)

        Y_min_dimless = math.sqrt(max(0, num_Y_calc) / den_Y_calc)
        if q <= 1e-9:
            print("WARNING: q_factor too small for Y_eq calculation.")
            return self._generate_simplified_isotensoid_profile(num_points_dome)
        
        Y_eq_dimless = Y_min_dimless / math.sqrt(q)

        # Adjust reference radius so that Y_eq matches the cylinder radius
        if abs(Y_eq_dimless) < 1e-6:
            print("WARNING: Calculated Y_eq_dimless too small.")
            return self._generate_simplified_isotensoid_profile(num_points_dome)
            
        rho_0_adjusted = self.inner_radius / Y_eq_dimless
        actual_dome_polar_radius = Y_min_dimless * rho_0_adjusted
        actual_dome_equatorial_radius = Y_eq_dimless * rho_0_adjusted  # Should equal self.inner_radius

        # Elliptical coordinate theta (0 at equator, pi/2 at pole)
        theta_values = np.linspace(np.pi / 2.0, 0.0, num_points_dome)  # From pole to equator

        # Parameter m for elliptic integrals (Koussios Thesis, Eq. 4.20)
        denom_m_elliptic = 1 + 2 * q * (1 + r)
        if abs(denom_m_elliptic) < 1e-9:
            print("WARNING: Invalid parameters for elliptic integral calculation.")
            return self._generate_simplified_isotensoid_profile(num_points_dome)
            
        m_elliptic = (q - 1) / denom_m_elliptic
        m_elliptic_clamped = np.clip(m_elliptic, 0, 1)  # Ensure valid range for elliptic integrals

        try:
            # Calculate elliptic integrals using SciPy
            ell_F_values = ellipkinc(theta_values, m_elliptic_clamped)  # F(theta | m)
            ell_E_values = ellipeinc(theta_values, m_elliptic_clamped)  # E(theta | m)
        except Exception as e:
            print(f"WARNING: Error calculating elliptic integrals: {e}")
            return self._generate_simplified_isotensoid_profile(num_points_dome)

        # Calculate Z profile (Koussios Thesis, Eq. 4.20)
        coeff_Z_den_sq_term = 1 + 2 * q * (1 + r)
        if coeff_Z_den_sq_term < 0:
            print("WARNING: Negative coefficient in Z calculation.")
            return self._generate_simplified_isotensoid_profile(num_points_dome)

        coeff_Z_factor = Y_min_dimless / math.sqrt(coeff_Z_den_sq_term)
        
        term_E_component = (1 + 2 * q * (1 + r)) * ell_E_values
        term_F_component = (1 + q + q * r) * ell_F_values
        
        Z_dimless_profile = coeff_Z_factor * (term_E_component - term_F_component)
        
        # Scale to physical coordinates
        dome_height_dimless = Z_dimless_profile[0]  # Value at theta = pi/2 (pole)
        actual_dome_height_m = dome_height_dimless * rho_0_adjusted
        z_local_dome_abs = Z_dimless_profile * rho_0_adjusted

        # Calculate rho values using Koussios Thesis Eq. 4.12
        rho_dimless_profile = np.sqrt(
            Y_eq_dimless**2 * np.cos(theta_values)**2 + 
            Y_min_dimless**2 * np.sin(theta_values)**2
        )
        rho_abs_profile = rho_dimless_profile * rho_0_adjusted

        return np.vstack((rho_abs_profile, z_local_dome_abs)).T, actual_dome_height_m
    
    def _incomplete_elliptic_integral_first_kind(self, theta, m):
        """Approximation for incomplete elliptic integral of the first kind F(theta, m)."""
        # Simple approximation using series expansion for small m
        if isinstance(theta, np.ndarray):
            result = np.zeros_like(theta)
            for i, t in enumerate(theta):
                if abs(m) < 0.5:
                    # Series approximation
                    result[i] = t + m/4 * (t - np.sin(t)*np.cos(t)) + \
                               m**2/64 * (3*t - 3*np.sin(t)*np.cos(t) - np.sin(t)**3*np.cos(t))
                else:
                    # Crude approximation for larger m
                    result[i] = t * (1 + m/4)
            return result
        else:
            if abs(m) < 0.5:
                return theta + m/4 * (theta - np.sin(theta)*np.cos(theta))
            else:
                return theta * (1 + m/4)
    
    def _incomplete_elliptic_integral_second_kind(self, theta, m):
        """Approximation for incomplete elliptic integral of the second kind E(theta, m)."""
        # Simple approximation using series expansion
        if isinstance(theta, np.ndarray):
            result = np.zeros_like(theta)
            for i, t in enumerate(theta):
                if abs(m) < 0.5:
                    # Series approximation
                    result[i] = t - m/4 * (t - np.sin(t)*np.cos(t)) - \
                               m**2/64 * (t - np.sin(t)*np.cos(t) + np.sin(t)**3*np.cos(t)/3)
                else:
                    # Crude approximation for larger m
                    result[i] = t * (1 - m/4)
            return result
        else:
            if abs(m) < 0.5:
                return theta - m/4 * (theta - np.sin(theta)*np.cos(theta))
            else:
                return theta * (1 - m/4)
    
    def _generate_simplified_isotensoid_profile(self, num_points_dome: int = 100):
        """Simplified isotensoid profile as fallback."""
        R_dome = self.inner_radius
        actual_dome_height = R_dome * 0.6 * self.q_factor / (self.q_factor * 0.5 + 1)
        
        phi_angles = np.linspace(0, np.pi / 2, num_points_dome)
        dome_rho = R_dome * np.sin(phi_angles)
        dome_z_local = actual_dome_height * np.cos(phi_angles)
        return np.vstack((dome_rho, dome_z_local)).T, actual_dome_height
        
    def _calculate_isotensoid_z(self, Y: float) -> float:
        """
        Calculate Z coordinate for isotensoid profile at given Y.
        Simplified implementation of Koussios isotensoid equations.
        """
        # This is a simplified approximation
        # Full implementation would require numerical integration of differential equations
        
        q, r, s = self.q_factor, self.r_factor, self.s_factor
        
        # Approximate isotensoid shape using polynomial fit influenced by qrs parameters
        if Y <= 0.1:
            return 0.0
        
        # Empirical approximation for isotensoid profile
        # Adjusted by qrs parameters
        Y_norm = (Y - 0.1) / 0.9  # Normalize to [0,1]
        
        # Base profile modified by qrs parameters
        Z = (q * Y_norm**2 + r * Y_norm**3 + s * Y_norm) * (1 - Y_norm)**0.5
        
        return Z * 0.5  # Scale factor for realistic proportions
        
    def _generate_hemispherical_profile(self, num_points_dome: int = 50):
        """
        Generates points for one hemispherical dome.
        Profile is ordered from the polar opening (smallest rho) to the cylinder junction (largest rho).
        z-coordinates are local to the dome, starting from 0 at the cylinder tangent plane
        and going up to dome_height at the polar opening plane.

        Returns:
            Tuple[np.ndarray, float]: Array of (rho, z_local_dome) points, and dome_height
        """
        R_dome = self.inner_radius  # Hemisphere radius is the cylinder's inner radius

        # phi is the angle from the Z-axis (pole). phi=0 at pole, phi=pi/2 at cylinder junction.
        phi_angles = np.linspace(0, np.pi / 2, num_points_dome)
        dome_rho = R_dome * np.sin(phi_angles)  # rho from 0 to R_dome
        # z_local from R_dome (at pole) down to 0 (at cylinder junction plane)
        dome_z_local = R_dome * np.cos(phi_angles)
        actual_dome_height = R_dome  # For a full hemisphere from pole
        # Order from pole (rho=0) to cylinder junction (rho=R_dome)
        # z_local will go from R_dome down to 0.
        return np.vstack((dome_rho, dome_z_local)).T, actual_dome_height

    def _generate_elliptical_profile(self, num_points_dome: int = 50):
        """
        Generates points for one elliptical dome.
        Profile is ordered from the polar opening (rho=0 at pole) to the cylinder junction.
        z-coordinates are local to the dome, from 0 at the cylinder tangent plane
        up to dome_height at the pole.
        """
        a_ellipse = self.inner_radius  # Semi-major axis (along rho)
        b_ellipse = a_ellipse * self.elliptical_aspect_ratio  # Semi-minor axis (along z, this is the dome height)
        actual_dome_height = b_ellipse

        # Parametric equation for ellipse: rho = a*sin(t), z_local_from_apex = b*(1-cos(t))
        # t from 0 (pole, rho=0, z_local_from_apex=0) to pi/2 (cyl junction, rho=a, z_local_from_apex=b)
        t_angles = np.linspace(0, np.pi / 2, num_points_dome)
        dome_rho = a_ellipse * np.sin(t_angles)
        # z_local goes from actual_dome_height (at pole) down to 0 (at cylinder junction)
        dome_z_local = actual_dome_height * np.cos(t_angles)  # z measured from cylinder plane towards pole

        return np.vstack((dome_rho, dome_z_local)).T, actual_dome_height
        
    def _generate_geodesic_profile(self):
        """Generate geodesic dome profile (simplified)"""
        # For geodesic, use a shape similar to isotensoid but with different curvature
        # This is a simplified implementation
        self._generate_isotensoid_profile()  # Use isotensoid as base
        
        # Modify the profile for geodesic characteristics
        # In practice, geodesic domes require more complex calculations
        
    def _calculate_geometric_properties(self):
        """Calculate geometric properties of the vessel"""
        if self.profile_points is None:
            return
            
        dome_height = getattr(self, 'dome_height', self.inner_radius)
        
        # Volumes
        cylinder_volume = np.pi * self.inner_radius**2 * self.cylindrical_length
        
        # Approximate dome volume (depends on dome type)
        if self.dome_type == "Hemispherical":
            single_dome_volume = (2/3) * np.pi * self.inner_radius**3
        elif self.dome_type == "Elliptical":
            single_dome_volume = (2/3) * np.pi * self.inner_radius**2 * dome_height
        else:  # Isotensoid and others
            single_dome_volume = (2/3) * np.pi * self.inner_radius**2 * dome_height
            
        total_volume = cylinder_volume + 2 * single_dome_volume
        
        # Surface areas
        cylinder_surface = 2 * np.pi * self.inner_radius * self.cylindrical_length
        
        # Approximate dome surface area
        if self.dome_type == "Hemispherical":
            single_dome_surface = 2 * np.pi * self.inner_radius**2
        else:
            # Approximate for other dome types
            single_dome_surface = 2 * np.pi * self.inner_radius**2 * (1 + dome_height/self.inner_radius) / 2
            
        total_surface_area = cylinder_surface + 2 * single_dome_surface
        
        # Overall dimensions
        overall_length = self.cylindrical_length + 2 * dome_height
        aspect_ratio = overall_length / self.outer_diameter
        
        # Estimated weight (assuming composite density ~1.6 g/cm³)
        wall_volume = total_surface_area * self.wall_thickness / 1000  # Convert to cm³
        estimated_weight = wall_volume * 1.6 / 1000  # kg
        
        self.geometric_properties = {
            'total_volume': total_volume / 1e6,  # Convert to liters
            'surface_area': total_surface_area / 1e6,  # Convert to m²
            'dome_height': dome_height,
            'overall_length': overall_length,
            'aspect_ratio': aspect_ratio,
            'estimated_weight': estimated_weight
        }
        
    def get_geometric_properties(self) -> Dict:
        """Return calculated geometric properties"""
        return self.geometric_properties or {}
        
    def get_profile_points(self) -> Dict:
        """Return profile points for visualization/export"""
        if self.profile_points is None:
            return {}
            
        return {
            'r_inner_mm': self.profile_points['r_inner_mm'],
            'r_outer_mm': self.profile_points['r_outer_mm'], 
            'z_mm': self.profile_points['z_mm']
        }
        
    def get_dome_contour_equation(self) -> str:
        """Return the mathematical equation describing the dome contour"""
        if self.dome_type == "Isotensoid":
            return f"Isotensoid (qrs): q={self.q_factor:.2f}, r={self.r_factor:.2f}, s={self.s_factor:.2f}"
        elif self.dome_type == "Elliptical":
            return f"Elliptical: r²/a² + z²/b² = 1, aspect_ratio={self.elliptical_aspect_ratio:.2f}"
        elif self.dome_type == "Hemispherical":
            return f"Hemispherical: r² + z² = R², R={self.inner_radius:.1f}mm"
        else:
            return f"{self.dome_type} dome"

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

        self.profile_points = {
            'r_inner': np.array(profile_r_inner),
            'z': np.array(profile_z_values),
            'r_outer': np.array(profile_r_inner) + self.wall_thickness,
            'dome_height': self.dome_height
        }
        self._calculate_geometric_properties()
        
    def _generate_isotensoid_profile(self, num_points_dome: int = 100):
        """Placeholder for Koussios qrs-parameterized isotensoid profile."""
        # This needs the full Koussios equations (Eq. 4.3 or 4.20 from thesis)
        # For now, returning a shape similar to hemispherical for plotting
        print("WARNING: Isotensoid profile is a placeholder.")
        R_dome = self.inner_radius
        # This is a conceptual polar opening for the dome, not necessarily vessel's final.
        # For qrs, rho_0 is the reference, could be self.inner_radius / Y_eq_from_qrs
        # Let's assume dome_height is roughly R_dome * 0.6 (typical isotensoid aspect)
        actual_dome_height = R_dome * 0.6 * self.q_factor / (self.q_factor * 0.5 + 1)  # very rough
        
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
            'r_inner_mm': self.profile_points['r_inner'],
            'r_outer_mm': self.profile_points['r_outer'],
            'z_mm': self.profile_points['z']
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

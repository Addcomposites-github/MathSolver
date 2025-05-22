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
        
    def generate_profile(self):
        """Generate the complete 2D meridian profile of the vessel"""
        if self.dome_type == "Isotensoid":
            self._generate_isotensoid_profile()
        elif self.dome_type == "Elliptical":
            self._generate_elliptical_profile()
        elif self.dome_type == "Hemispherical":
            self._generate_hemispherical_profile()
        elif self.dome_type == "Geodesic":
            self._generate_geodesic_profile()
        else:
            raise ValueError(f"Unsupported dome type: {self.dome_type}")
            
        self._calculate_geometric_properties()
        
    def _generate_isotensoid_profile(self):
        """
        Generate isotensoid dome profile using Koussios qrs-parameterization.
        
        Based on equations from Koussios thesis Chapter 4:
        - Dimensionless coordinates Y = ρ/ρ₀, Z = z/ρ₀  
        - qrs parameters define dome shape optimization
        """
        # Normalize to polar opening radius
        rho_0 = self.inner_radius  # Polar opening radius
        
        # Generate dimensionless Y coordinates from polar opening to equator
        Y_points = np.linspace(0.1, 1.0, 100)  # Avoid Y=0 for numerical stability
        Z_points = []
        
        for Y in Y_points:
            # Koussios isotensoid equation (simplified form)
            # Z'(Y) relationship for qrs-parameterized isotensoid
            if Y < 0.1:
                Z = 0.0
            else:
                # Approximate isotensoid profile based on qrs parameters
                # This is a simplified implementation - full Koussios equations are more complex
                Z = self._calculate_isotensoid_z(Y)
            Z_points.append(Z)
        
        # Convert to physical coordinates (mm)
        rho_points = [Y * rho_0 for Y in Y_points]
        z_points = [Z * rho_0 for Z in Z_points]
        
        # Build complete profile: dome + cylinder + dome
        profile_r = []
        profile_z = []
        
        # Top dome (reversed)
        dome_height = max(z_points)
        for i in range(len(rho_points)-1, -1, -1):
            profile_r.append(rho_points[i])
            profile_z.append(self.cylindrical_length/2 + dome_height - z_points[i])
        
        # Top cylinder edge
        profile_r.append(self.inner_radius)
        profile_z.append(self.cylindrical_length/2)
        
        # Cylindrical section
        profile_r.append(self.inner_radius)
        profile_z.append(-self.cylindrical_length/2)
        
        # Bottom dome
        for i in range(len(rho_points)):
            profile_r.append(rho_points[i])
            profile_z.append(-self.cylindrical_length/2 - z_points[i])
        
        self.profile_points = {
            'r_inner': profile_r,
            'z': profile_z,
            'r_outer': [r + self.wall_thickness for r in profile_r],
            'dome_height': dome_height
        }
        
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
        
    def _generate_elliptical_profile(self):
        """Generate elliptical dome profile"""
        # Ellipse parameters
        a = self.inner_radius  # Semi-major axis (radius)
        b = a * self.elliptical_aspect_ratio  # Semi-minor axis (height)
        
        # Generate elliptical dome points
        theta = np.linspace(0, np.pi/2, 50)
        rho_points = a * np.cos(theta)
        z_points = b * np.sin(theta)
        
        # Build complete profile
        profile_r = []
        profile_z = []
        
        # Top dome
        for i in range(len(rho_points)-1, -1, -1):
            profile_r.append(rho_points[i])
            profile_z.append(self.cylindrical_length/2 + z_points[i])
        
        # Cylindrical section
        profile_r.extend([self.inner_radius, self.inner_radius])
        profile_z.extend([self.cylindrical_length/2, -self.cylindrical_length/2])
        
        # Bottom dome  
        for i in range(len(rho_points)):
            profile_r.append(rho_points[i])
            profile_z.append(-self.cylindrical_length/2 - z_points[i])
            
        self.profile_points = {
            'r_inner': profile_r,
            'z': profile_z,
            'r_outer': [r + self.wall_thickness for r in profile_r],
            'dome_height': b
        }
        
    def _generate_hemispherical_profile(self):
        """Generate hemispherical dome profile"""
        # Generate hemisphere points
        theta = np.linspace(0, np.pi/2, 50)
        rho_points = self.inner_radius * np.cos(theta)
        z_points = self.inner_radius * np.sin(theta)
        
        # Build complete profile
        profile_r = []
        profile_z = []
        
        # Top dome
        for i in range(len(rho_points)-1, -1, -1):
            profile_r.append(rho_points[i])
            profile_z.append(self.cylindrical_length/2 + z_points[i])
        
        # Cylindrical section
        profile_r.extend([self.inner_radius, self.inner_radius])
        profile_z.extend([self.cylindrical_length/2, -self.cylindrical_length/2])
        
        # Bottom dome
        for i in range(len(rho_points)):
            profile_r.append(rho_points[i])
            profile_z.append(-self.cylindrical_length/2 - z_points[i])
            
        self.profile_points = {
            'r_inner': profile_r,
            'z': profile_z,
            'r_outer': [r + self.wall_thickness for r in profile_r],
            'dome_height': self.inner_radius
        }
        
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
            
        dome_height = self.profile_points['dome_height']
        
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

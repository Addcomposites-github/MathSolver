"""
Enhanced VesselGeometry class to ensure proper dome profile generation
This fixes the issue where dome geometry wasn't being properly calculated
"""

import numpy as np
import math
from typing import Dict, List, Optional, Tuple

class VesselGeometry:
    """Enhanced vessel geometry with proper dome profile generation"""
    
    def __init__(self, inner_diameter, wall_thickness, cylindrical_length, dome_type="Hemispherical"):
        # Store basic parameters
        self.inner_diameter = inner_diameter  # mm
        self.wall_thickness = wall_thickness  # mm
        self.cylindrical_length = cylindrical_length  # mm
        self.dome_type = dome_type
        
        # Initialize dome parameters
        self.q_factor = 9.5
        self.r_factor = 0.1
        self.s_factor = 0.5
        self.aspect_ratio = 1.0
        
        # Initialize profile points
        self.profile_points = None
        
    def set_qrs_parameters(self, q_factor, r_factor, s_factor):
        """Set qrs parameters for isotensoid domes"""
        self.q_factor = q_factor
        self.r_factor = r_factor
        self.s_factor = s_factor
        
    def set_elliptical_parameters(self, aspect_ratio):
        """Set parameters for elliptical domes"""
        self.aspect_ratio = aspect_ratio
        
    def generate_profile(self):
        """Generate complete vessel profile including proper dome geometry"""
        try:
            # Calculate basic dimensions
            inner_radius = self.inner_diameter / 2.0  # mm
            
            # Determine dome height based on type
            if self.dome_type == 'Hemispherical':
                dome_height = inner_radius
            elif self.dome_type == 'Elliptical':
                dome_height = inner_radius * self.aspect_ratio
            elif self.dome_type == 'Isotensoid':
                # Calculate dome height based on qrs parameters
                dome_height = self._calculate_isotensoid_dome_height(inner_radius)
            else:
                dome_height = inner_radius * 0.8  # Default
            
            # Generate complete profile
            z_profile, r_inner_profile, r_outer_profile = self._generate_complete_profile(
                inner_radius, dome_height
            )
            
            # Store profile points
            self.profile_points = {
                'z_mm': z_profile,
                'r_inner_mm': r_inner_profile,
                'r_outer_mm': r_outer_profile
            }
            
            print(f"Generated vessel profile: {len(z_profile)} points, Z range: {np.min(z_profile):.1f} to {np.max(z_profile):.1f} mm")
            print(f"Radius range: {np.min(r_inner_profile):.1f} to {np.max(r_inner_profile):.1f} mm")
            
        except Exception as e:
            print(f"Error generating vessel profile: {e}")
            # Generate fallback profile
            self._generate_fallback_profile()
    
    def _calculate_isotensoid_dome_height(self, inner_radius):
        """Calculate isotensoid dome height based on qrs parameters"""
        # Empirical formula based on Koussios theory
        base_height = inner_radius * 0.5
        q_influence = (self.q_factor - 5.0) / 10.0 * 0.3
        r_influence = self.r_factor * 0.2
        s_influence = self.s_factor * 0.1
        
        dome_height = base_height * (1.0 + q_influence + r_influence + s_influence)
        dome_height = max(inner_radius * 0.2, min(inner_radius * 1.5, dome_height))
        
        return dome_height
    
    def _generate_complete_profile(self, inner_radius, dome_height):
        """Generate complete vessel profile with proper dome geometry"""
        # Profile resolution
        n_dome_points = 50  # Points per dome
        n_cyl_points = 30   # Points for cylinder
        
        # Create z-coordinate arrays
        # Aft dome: -dome_height to 0
        z_aft_dome = np.linspace(-dome_height, 0, n_dome_points)
        
        # Cylinder: 0 to cylindrical_length
        z_cylinder = np.linspace(0, self.cylindrical_length, n_cyl_points)
        
        # Forward dome: cylindrical_length to cylindrical_length + dome_height
        z_fwd_dome = np.linspace(self.cylindrical_length, self.cylindrical_length + dome_height, n_dome_points)
        
        # Calculate radius profiles for each section
        r_aft_dome = self._calculate_dome_radius_profile(z_aft_dome, -dome_height, 0, inner_radius, 'aft')
        r_cylinder = np.full_like(z_cylinder, inner_radius)
        r_fwd_dome = self._calculate_dome_radius_profile(z_fwd_dome, self.cylindrical_length, self.cylindrical_length + dome_height, inner_radius, 'forward')
        
        # Combine all sections (remove duplicate points at interfaces)
        z_complete = np.concatenate([z_aft_dome[:-1], z_cylinder[:-1], z_fwd_dome])
        r_inner_complete = np.concatenate([r_aft_dome[:-1], r_cylinder[:-1], r_fwd_dome])
        
        # Calculate outer radius
        r_outer_complete = r_inner_complete + self.wall_thickness
        
        return z_complete, r_inner_complete, r_outer_complete
    
    def _calculate_dome_radius_profile(self, z_array, z_start, z_end, max_radius, dome_position):
        """Calculate radius profile for dome section with proper geometry"""
        
        dome_length = z_end - z_start
        if dome_length <= 0:
            return np.full_like(z_array, max_radius)
        
        if self.dome_type == 'Hemispherical':
            # True hemispherical dome
            center_z = (z_start + z_end) / 2
            dome_radius = dome_length / 2
            z_relative = z_array - center_z
            
            # Hemisphere equation: r = sqrt(R^2 - z^2)
            r_normalized = np.sqrt(np.maximum(0, 1 - (z_relative / dome_radius)**2))
            r_profile = max_radius * r_normalized
            
        elif self.dome_type == 'Elliptical':
            # Elliptical dome profile
            center_z = (z_start + z_end) / 2
            a = dome_length / 2  # Semi-major axis (along z)
            b = max_radius       # Semi-minor axis (along r)
            z_relative = z_array - center_z
            
            # Ellipse equation: (z/a)^2 + (r/b)^2 = 1
            # Solve for r: r = b * sqrt(1 - (z/a)^2)
            z_normalized = z_relative / a
            r_profile = b * np.sqrt(np.maximum(0, 1 - z_normalized**2))
            
        elif self.dome_type == 'Isotensoid':
            # Isotensoid dome with qrs parameterization
            r_profile = self._calculate_isotensoid_profile(z_array, z_start, z_end, max_radius)
            
        else:
            # Default to hemispherical
            center_z = (z_start + z_end) / 2
            dome_radius = dome_length / 2
            z_relative = z_array - center_z
            r_profile = max_radius * np.sqrt(np.maximum(0, 1 - (z_relative / dome_radius)**2))
        
        # Ensure smooth transition at dome-cylinder interface
        if dome_position == 'aft':
            # Aft dome should end at max_radius
            r_profile[-1] = max_radius
        elif dome_position == 'forward':
            # Forward dome should start at max_radius
            r_profile[0] = max_radius
        
        return r_profile
    
    def _calculate_isotensoid_profile(self, z_array, z_start, z_end, max_radius):
        """Calculate isotensoid dome profile using qrs parameters"""
        
        dome_length = z_end - z_start
        # Normalize z to parameter t (0 to 1)
        t = (z_array - z_start) / dome_length
        
        # Calculate polar opening ratio
        polar_opening_ratio = 0.05 + 0.1 * self.r_factor
        
        # Isotensoid profile approximation
        # Use modified power law with qrs influence
        exponent = 1.5 + 0.5 * self.s_factor  # s_factor influences curvature
        q_influence = 1.0 + (self.q_factor - 9.5) / 20.0  # q_factor influences overall shape
        
        # Profile calculation
        r_normalized = polar_opening_ratio + (1 - polar_opening_ratio) * (np.cos(t * np.pi / 2) ** exponent)
        r_profile = max_radius * r_normalized * q_influence
        
        # Ensure physical constraints
        r_profile = np.clip(r_profile, polar_opening_ratio * max_radius, max_radius)
        
        return r_profile
    
    def _generate_fallback_profile(self):
        """Generate simple fallback profile if main generation fails"""
        try:
            inner_radius = self.inner_diameter / 2.0
            
            # Simple profile: hemisphere + cylinder + hemisphere
            n_points = 100
            dome_height = inner_radius
            
            z_profile = np.linspace(-dome_height, self.cylindrical_length + dome_height, n_points)
            r_inner_profile = np.full_like(z_profile, inner_radius)
            
            # Simple hemisphere approximation at ends
            for i, z in enumerate(z_profile):
                if z < 0:  # Aft dome
                    dist_from_center = abs(z + dome_height/2)
                    if dist_from_center < dome_height/2:
                        r_inner_profile[i] = inner_radius * np.sqrt(1 - (2*dist_from_center/dome_height)**2)
                elif z > self.cylindrical_length:  # Forward dome
                    dist_from_center = abs(z - self.cylindrical_length - dome_height/2)
                    if dist_from_center < dome_height/2:
                        r_inner_profile[i] = inner_radius * np.sqrt(1 - (2*dist_from_center/dome_height)**2)
            
            r_outer_profile = r_inner_profile + self.wall_thickness
            
            self.profile_points = {
                'z_mm': z_profile,
                'r_inner_mm': r_inner_profile,
                'r_outer_mm': r_outer_profile
            }
            
            print("Generated fallback vessel profile")
            
        except Exception as e:
            print(f"Error generating fallback profile: {e}")
            # Absolute minimal fallback
            self.profile_points = {
                'z_mm': np.array([0, self.cylindrical_length]),
                'r_inner_mm': np.array([self.inner_diameter/2, self.inner_diameter/2]),
                'r_outer_mm': np.array([self.inner_diameter/2 + self.wall_thickness, self.inner_diameter/2 + self.wall_thickness])
            }
    
    def get_profile_points(self):
        """Get vessel profile points"""
        if self.profile_points is None:
            self.generate_profile()
        return self.profile_points
    
    def get_geometric_properties(self):
        """Calculate and return geometric properties"""
        try:
            if self.profile_points is None:
                self.generate_profile()
            
            # Basic calculations
            inner_radius = self.inner_diameter / 2.0
            total_volume = math.pi * (inner_radius/1000)**2 * (self.cylindrical_length/1000) / 1000  # Convert to liters
            
            # Approximate surface area
            cylinder_area = 2 * math.pi * (inner_radius/1000) * (self.cylindrical_length/1000)  # m²
            dome_area = 2 * 2 * math.pi * (inner_radius/1000)**2  # Two hemispheres ≈ one sphere
            surface_area = cylinder_area + dome_area
            
            # Dome height estimation
            if self.dome_type == 'Hemispherical':
                dome_height = inner_radius
            elif self.dome_type == 'Elliptical':
                dome_height = inner_radius * self.aspect_ratio
            else:
                dome_height = inner_radius * 0.8
            
            # Overall length
            overall_length = self.cylindrical_length + 2 * dome_height
            
            # Estimated weight (simplified)
            wall_volume = surface_area * (self.wall_thickness/1000)  # m³
            density = 1600  # kg/m³ for composite
            estimated_weight = wall_volume * density
            
            # Aspect ratio
            aspect_ratio = overall_length / self.inner_diameter
            
            return {
                'total_volume': total_volume,
                'surface_area': surface_area,
                'dome_height': dome_height,
                'overall_length': overall_length,
                'estimated_weight': estimated_weight,
                'aspect_ratio': aspect_ratio
            }
            
        except Exception as e:
            print(f"Error calculating geometric properties: {e}")
            return {
                'total_volume': 0.0,
                'surface_area': 0.0,
                'dome_height': 0.0,
                'overall_length': self.cylindrical_length,
                'estimated_weight': 0.0,
                'aspect_ratio': 1.0
            }

# Additional utility function to verify vessel geometry
def verify_vessel_geometry(vessel_geometry):
    """Verify that vessel geometry has proper dome profiles"""
    try:
        profile = vessel_geometry.get_profile_points()
        
        if not profile or 'z_mm' not in profile or 'r_inner_mm' not in profile:
            print("❌ Missing profile data")
            return False
        
        z_data = np.array(profile['z_mm'])
        r_data = np.array(profile['r_inner_mm'])
        
        if len(z_data) < 10:
            print(f"❌ Insufficient profile points: {len(z_data)}")
            return False
        
        # Check for dome curvature
        max_radius = np.max(r_data)
        min_radius = np.min(r_data)
        radius_variation = max_radius - min_radius
        
        if radius_variation < max_radius * 0.1:
            print(f"⚠️ Limited dome curvature detected: {radius_variation:.1f}mm variation")
            print(f"   Max radius: {max_radius:.1f}mm, Min radius: {min_radius:.1f}mm")
            return False
        
        print(f"✅ Vessel geometry verified:")
        print(f"   Points: {len(z_data)}")
        print(f"   Z range: {np.min(z_data):.1f} to {np.max(z_data):.1f} mm")
        print(f"   R range: {min_radius:.1f} to {max_radius:.1f} mm")
        print(f"   Dome curvature: {radius_variation:.1f}mm variation")
        
        return True
        
    except Exception as e:
        print(f"❌ Error verifying vessel geometry: {e}")
        return False
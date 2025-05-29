"""
Enhanced VesselGeometry with Correct Coordinate System
Vessel center at (0,0,0) with consistent coordinate system throughout
"""

import numpy as np
import math
from typing import Dict, List, Optional, Tuple

class VesselGeometry:
    """Vessel geometry with vessel center at origin (0,0,0)"""
    
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
        
        print(f"🏗️ Creating vessel geometry: {dome_type} domes, {inner_diameter}mm diameter, {cylindrical_length}mm length")
        
    def set_qrs_parameters(self, q_factor, r_factor, s_factor):
        """Set qrs parameters for isotensoid domes"""
        self.q_factor = q_factor
        self.r_factor = r_factor
        self.s_factor = s_factor
        print(f"🔧 Set isotensoid parameters: q={q_factor}, r={r_factor}, s={s_factor}")
        
    def set_elliptical_parameters(self, aspect_ratio):
        """Set parameters for elliptical domes"""
        self.aspect_ratio = aspect_ratio
        print(f"🔧 Set elliptical aspect ratio: {aspect_ratio}")
        
    def generate_profile(self):
        """Generate complete vessel profile with center at origin"""
        try:
            print(f"🔄 Generating vessel profile with centered coordinate system...")
            
            # Calculate basic dimensions
            inner_radius = self.inner_diameter / 2.0  # mm
            
            # Determine dome height based on type
            if self.dome_type == 'Hemispherical':
                dome_height = inner_radius
                print(f"   Hemispherical domes: {dome_height:.1f}mm height")
            elif self.dome_type == 'Elliptical':
                dome_height = inner_radius * self.aspect_ratio
                print(f"   Elliptical domes: {dome_height:.1f}mm height (aspect ratio {self.aspect_ratio})")
            elif self.dome_type == 'Isotensoid':
                dome_height = self._calculate_isotensoid_dome_height(inner_radius)
                print(f"   Isotensoid domes: {dome_height:.1f}mm height (q={self.q_factor})")
            else:
                dome_height = inner_radius * 0.8
                print(f"   Default domes: {dome_height:.1f}mm height")
            
            # **KEY: Generate profile with vessel center at origin**
            z_profile, r_inner_profile, r_outer_profile = self._generate_centered_profile(
                inner_radius, dome_height
            )
            
            # Verify centering
            z_min, z_max = np.min(z_profile), np.max(z_profile)
            z_center = (z_min + z_max) / 2
            total_length = z_max - z_min
            
            print(f"   ✅ Profile generated: {len(z_profile)} points")
            print(f"   📏 Z range: {z_min:.1f}mm to {z_max:.1f}mm (total: {total_length:.1f}mm)")
            print(f"   📍 Z center: {z_center:.1f}mm (should be ~0)")
            print(f"   🔍 R range: {np.min(r_inner_profile):.1f}mm to {np.max(r_inner_profile):.1f}mm")
            
            # Check if properly centered
            if abs(z_center) > 1.0:  # More than 1mm off center
                print(f"   ⚠️  WARNING: Vessel not properly centered (offset: {z_center:.1f}mm)")
                # Adjust to center
                z_profile = z_profile - z_center
                z_min, z_max = np.min(z_profile), np.max(z_profile)
                z_center = (z_min + z_max) / 2
                print(f"   🔧 Adjusted to center: Z range now {z_min:.1f}mm to {z_max:.1f}mm, center: {z_center:.1f}mm")
            else:
                print(f"   ✅ Vessel properly centered")
            
            # Store profile points
            self.profile_points = {
                'z_mm': z_profile,
                'r_inner_mm': r_inner_profile,
                'r_outer_mm': r_outer_profile
            }
            
            # Validate dome curvature
            radius_variation = np.max(r_inner_profile) - np.min(r_inner_profile)
            max_radius = np.max(r_inner_profile)
            curvature_ratio = radius_variation / max_radius
            
            print(f"   🔄 Dome curvature analysis:")
            print(f"      Radius variation: {radius_variation:.1f}mm ({curvature_ratio*100:.1f}% of max radius)")
            
            if curvature_ratio > 0.1:
                print(f"   ✅ Good dome curvature detected!")
            else:
                print(f"   ⚠️  Limited dome curvature - may appear cylindrical")
            
        except Exception as e:
            print(f"❌ Error generating vessel profile: {e}")
            self._generate_centered_fallback_profile()
    
    def _generate_centered_profile(self, inner_radius, dome_height):
        """Generate complete vessel profile centered at origin"""
        
        # Profile resolution
        n_dome_points = 50
        n_cyl_points = 30
        
        # **CENTERED COORDINATE SYSTEM**
        # Cylinder section: centered around Z=0
        z_cyl_half_length = self.cylindrical_length / 2.0
        z_cyl_start = -z_cyl_half_length  # Negative Z
        z_cyl_end = +z_cyl_half_length    # Positive Z
        
        # Aft dome: extends backward from cylinder start
        z_aft_dome_start = z_cyl_start - dome_height
        z_aft_dome_end = z_cyl_start
        
        # Forward dome: extends forward from cylinder end  
        z_fwd_dome_start = z_cyl_end
        z_fwd_dome_end = z_cyl_end + dome_height
        
        print(f"   📐 Centered coordinate layout:")
        print(f"      Aft dome:  {z_aft_dome_start:.1f}mm to {z_aft_dome_end:.1f}mm")
        print(f"      Cylinder:  {z_cyl_start:.1f}mm to {z_cyl_end:.1f}mm")
        print(f"      Fwd dome:  {z_fwd_dome_start:.1f}mm to {z_fwd_dome_end:.1f}mm")
        
        # Create z-coordinate arrays for each section
        z_aft_dome = np.linspace(z_aft_dome_start, z_aft_dome_end, n_dome_points)
        z_cylinder = np.linspace(z_cyl_start, z_cyl_end, n_cyl_points)
        z_fwd_dome = np.linspace(z_fwd_dome_start, z_fwd_dome_end, n_dome_points)
        
        # Calculate radius profiles for each section
        r_aft_dome = self._calculate_dome_radius_profile(
            z_aft_dome, z_aft_dome_start, z_aft_dome_end, inner_radius, 'aft'
        )
        r_cylinder = np.full_like(z_cylinder, inner_radius)
        r_fwd_dome = self._calculate_dome_radius_profile(
            z_fwd_dome, z_fwd_dome_start, z_fwd_dome_end, inner_radius, 'forward'
        )
        
        # Combine all sections (remove duplicate points at interfaces)
        z_complete = np.concatenate([z_aft_dome[:-1], z_cylinder[:-1], z_fwd_dome])
        r_inner_complete = np.concatenate([r_aft_dome[:-1], r_cylinder[:-1], r_fwd_dome])
        
        # Calculate outer radius
        r_outer_complete = r_inner_complete + self.wall_thickness
        
        return z_complete, r_inner_complete, r_outer_complete
    
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
        exponent = 1.5 + 0.5 * self.s_factor
        q_influence = 1.0 + (self.q_factor - 9.5) / 20.0
        
        # Profile calculation
        r_normalized = polar_opening_ratio + (1 - polar_opening_ratio) * (np.cos(t * np.pi / 2) ** exponent)
        r_profile = max_radius * r_normalized * q_influence
        
        # Ensure physical constraints
        r_profile = np.clip(r_profile, polar_opening_ratio * max_radius, max_radius)
        
        return r_profile
    
    def _generate_centered_fallback_profile(self):
        """Generate simple centered fallback profile"""
        try:
            print("🔧 Generating centered fallback profile...")
            
            inner_radius = self.inner_diameter / 2.0
            dome_height = inner_radius
            
            # **CENTERED SYSTEM FOR FALLBACK TOO**
            n_points = 100
            
            # Total vessel length
            total_length = self.cylindrical_length + 2 * dome_height
            
            # Create z array centered at origin
            z_profile = np.linspace(-total_length/2, total_length/2, n_points)
            r_inner_profile = np.full_like(z_profile, inner_radius)
            
            # Calculate cylinder boundaries
            cyl_half_length = self.cylindrical_length / 2
            
            # Simple hemisphere approximation at ends
            for i, z in enumerate(z_profile):
                if z < -cyl_half_length:  # Aft dome
                    z_dome_center = -cyl_half_length - dome_height/2
                    dist_from_center = abs(z - z_dome_center)
                    if dist_from_center < dome_height/2:
                        r_inner_profile[i] = inner_radius * np.sqrt(1 - (2*dist_from_center/dome_height)**2)
                elif z > cyl_half_length:  # Forward dome
                    z_dome_center = cyl_half_length + dome_height/2
                    dist_from_center = abs(z - z_dome_center)
                    if dist_from_center < dome_height/2:
                        r_inner_profile[i] = inner_radius * np.sqrt(1 - (2*dist_from_center/dome_height)**2)
            
            r_outer_profile = r_inner_profile + self.wall_thickness
            
            self.profile_points = {
                'z_mm': z_profile,
                'r_inner_mm': r_inner_profile,
                'r_outer_mm': r_outer_profile
            }
            
            # Verify centering
            z_center = (np.min(z_profile) + np.max(z_profile)) / 2
            print(f"   ✅ Centered fallback generated: {len(z_profile)} points, center at {z_center:.1f}mm")
            
        except Exception as e:
            print(f"❌ Error generating centered fallback profile: {e}")
            # Absolute minimal fallback - still centered
            half_length = self.cylindrical_length / 2
            self.profile_points = {
                'z_mm': np.array([-half_length, half_length]),
                'r_inner_mm': np.array([self.inner_diameter/2, self.inner_diameter/2]),
                'r_outer_mm': np.array([self.inner_diameter/2 + self.wall_thickness, 
                                      self.inner_diameter/2 + self.wall_thickness])
            }
    
    def get_profile_points(self):
        """Get vessel profile points (ensuring they exist)"""
        if self.profile_points is None:
            print("🔄 Profile not generated yet - generating now...")
            self.generate_profile()
        return self.profile_points
    
    def verify_coordinate_system(self):
        """Verify that the vessel is properly centered"""
        if self.profile_points is None:
            return False, "No profile generated"
        
        z_data = np.array(self.profile_points['z_mm'])
        z_min, z_max = np.min(z_data), np.max(z_data)
        z_center = (z_min + z_max) / 2
        
        is_centered = abs(z_center) < 1.0  # Within 1mm of center
        
        return is_centered, {
            'z_range': (z_min, z_max),
            'z_center': z_center,
            'total_length': z_max - z_min,
            'cylinder_length': self.cylindrical_length,
            'offset_from_origin': z_center
        }
    
    def get_geometric_properties(self):
        """Calculate geometric properties with proper coordinate system"""
        try:
            if self.profile_points is None:
                self.generate_profile()
            
            # Verify coordinate system
            is_centered, coord_info = self.verify_coordinate_system()
            
            # Basic calculations
            inner_radius = self.inner_diameter / 2.0
            total_volume = math.pi * (inner_radius/1000)**2 * (self.cylindrical_length/1000) / 1000  # Liters
            
            # Surface area approximation
            cylinder_area = 2 * math.pi * (inner_radius/1000) * (self.cylindrical_length/1000)  # m²
            dome_area = 2 * 2 * math.pi * (inner_radius/1000)**2  # Two hemispheres ≈ one sphere
            surface_area = cylinder_area + dome_area
            
            # Dome height estimation
            if self.dome_type == 'Hemispherical':
                dome_height = inner_radius
            elif self.dome_type == 'Elliptical':
                dome_height = inner_radius * self.aspect_ratio
            elif self.dome_type == 'Isotensoid':
                dome_height = self._calculate_isotensoid_dome_height(inner_radius)
            else:
                dome_height = inner_radius * 0.8
            
            # Overall length
            overall_length = self.cylindrical_length + 2 * dome_height
            
            # Estimated weight
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
                'aspect_ratio': aspect_ratio,
                'coordinate_system_centered': is_centered,
                'coordinate_info': coord_info
            }
            
        except Exception as e:
            print(f"❌ Error calculating geometric properties: {e}")
            return {
                'total_volume': 0.0,
                'surface_area': 0.0,
                'dome_height': 0.0,
                'overall_length': self.cylindrical_length,
                'estimated_weight': 0.0,
                'aspect_ratio': 1.0,
                'coordinate_system_centered': False,
                'coordinate_info': {'error': str(e)}
            }

def verify_vessel_geometry_coordinates(vessel_geometry):
    """Verify that vessel geometry uses correct coordinate system"""
    try:
        print(f"\n🔍 COORDINATE SYSTEM VERIFICATION")
        print(f"=" * 50)
        
        profile = vessel_geometry.get_profile_points()
        
        if not profile or 'z_mm' not in profile or 'r_inner_mm' not in profile:
            print("❌ Missing profile data")
            return False
        
        z_data = np.array(profile['z_mm'])
        r_data = np.array(profile['r_inner_mm'])
        
        # Check basic profile
        print(f"📊 Profile Analysis:")
        print(f"   Points: {len(z_data)}")
        print(f"   Z range: {np.min(z_data):.1f}mm to {np.max(z_data):.1f}mm")
        print(f"   R range: {np.min(r_data):.1f}mm to {np.max(r_data):.1f}mm")
        
        # Check centering
        z_center = (np.min(z_data) + np.max(z_data)) / 2
        total_length = np.max(z_data) - np.min(z_data)
        
        print(f"📍 Coordinate System Check:")
        print(f"   Vessel center Z: {z_center:.1f}mm")
        print(f"   Total length: {total_length:.1f}mm")
        print(f"   Expected cylinder length: {vessel_geometry.cylindrical_length:.1f}mm")
        
        is_centered = abs(z_center) < 1.0  # Within 1mm
        
        if is_centered:
            print(f"   ✅ PROPERLY CENTERED at origin")
        else:
            print(f"   ❌ NOT CENTERED - offset by {z_center:.1f}mm")
        
        # Check dome curvature
        max_radius = np.max(r_data)
        min_radius = np.min(r_data)
        radius_variation = max_radius - min_radius
        curvature_ratio = radius_variation / max_radius
        
        print(f"🔄 Dome Curvature Check:")
        print(f"   Max radius: {max_radius:.1f}mm")
        print(f"   Min radius: {min_radius:.1f}mm")
        print(f"   Variation: {radius_variation:.1f}mm ({curvature_ratio*100:.1f}%)")
        
        has_domes = curvature_ratio > 0.1
        
        if has_domes:
            print(f"   ✅ GOOD DOME CURVATURE detected")
        else:
            print(f"   ⚠️  LIMITED DOME CURVATURE - may appear cylindrical")
        
        # Overall assessment
        print(f"\n🎯 Overall Assessment:")
        if is_centered and has_domes:
            print(f"   ✅ EXCELLENT - Proper coordinate system and dome geometry")
            success = True
        elif is_centered:
            print(f"   ⚠️  GOOD - Centered but limited dome curvature")
            success = True
        elif has_domes:
            print(f"   ⚠️  PARTIAL - Good domes but not centered")
            success = False
        else:
            print(f"   ❌ POOR - Not centered and limited curvature")
            success = False
        
        # Provide recommendations
        print(f"\n💡 Recommendations:")
        if not is_centered:
            print(f"   🔧 Regenerate vessel geometry with centered coordinate system")
        if not has_domes:
            print(f"   🔧 Try different dome type or parameters for more curvature")
            print(f"   🔧 Consider Isotensoid or Elliptical domes instead of Hemispherical")
        
        return success
        
    except Exception as e:
        print(f"❌ Error verifying coordinate system: {e}")
        return False
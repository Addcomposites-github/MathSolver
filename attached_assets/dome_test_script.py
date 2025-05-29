"""
Quick Dome Geometry Test Script
Run this to verify that your vessel geometry is generating proper dome profiles
"""

import numpy as np
import matplotlib.pyplot as plt

def test_dome_geometry():
    """Test dome geometry generation independently"""
    
    print("🔍 Testing Dome Geometry Generation...")
    print("=" * 50)
    
    # Test different dome types
    dome_types = ['Hemispherical', 'Elliptical', 'Isotensoid']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, dome_type in enumerate(dome_types):
        print(f"\n📐 Testing {dome_type} Dome:")
        
        try:
            # Create test vessel (you'll need to import your VesselGeometry class)
            from modules.geometry import VesselGeometry  # Adjust import as needed
            
            test_vessel = VesselGeometry(
                inner_diameter=200.0,  # 200mm
                wall_thickness=5.0,    # 5mm
                cylindrical_length=300.0,  # 300mm
                dome_type=dome_type
            )
            
            # Set specific parameters for each type
            if dome_type == 'Elliptical':
                test_vessel.set_elliptical_parameters(1.2)  # Aspect ratio
            elif dome_type == 'Isotensoid':
                test_vessel.set_qrs_parameters(9.5, 0.1, 0.5)
            
            # Generate profile
            test_vessel.generate_profile()
            profile = test_vessel.get_profile_points()
            
            if profile and 'z_mm' in profile:
                z_data = np.array(profile['z_mm'])
                r_inner_data = np.array(profile['r_inner_mm'])
                
                # Analyze the profile
                max_radius = np.max(r_inner_data)
                min_radius = np.min(r_inner_data)
                radius_variation = max_radius - min_radius
                
                print(f"   ✅ Profile generated: {len(z_data)} points")
                print(f"   📏 Z range: {np.min(z_data):.1f} to {np.max(z_data):.1f} mm")
                print(f"   📏 R range: {min_radius:.1f} to {max_radius:.1f} mm")
                print(f"   🔄 Radius variation: {radius_variation:.1f} mm")
                
                # Check if this looks like a proper dome
                if radius_variation > max_radius * 0.1:
                    print(f"   ✅ Good dome curvature detected!")
                else:
                    print(f"   ⚠️  Limited dome curvature (mostly cylindrical)")
                
                # Plot the profile
                ax = axes[i]
                ax.plot(z_data, r_inner_data, 'b-', linewidth=2, label='Inner Surface')
                ax.plot(z_data, r_inner_data + test_vessel.wall_thickness, 'r-', linewidth=1, label='Outer Surface')
                ax.set_title(f'{dome_type} Dome')
                ax.set_xlabel('Z Position (mm)')
                ax.set_ylabel('Radius (mm)')
                ax.grid(True, alpha=0.3)
                ax.legend()
                ax.set_aspect('equal')
                
            else:
                print(f"   ❌ Failed to generate profile for {dome_type}")
                
        except Exception as e:
            print(f"   ❌ Error testing {dome_type}: {str(e)}")
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 50)
    print("🎯 Dome Geometry Test Complete!")
    print("\nWhat to look for:")
    print("✅ Good: Radius varies significantly (dome shape visible)")
    print("❌ Bad: Radius mostly constant (cylindrical shape)")
    print("🔄 Expected: Radius should taper toward dome ends")

def test_3d_surface_generation():
    """Test 3D surface generation for visualization"""
    
    print("\n🎨 Testing 3D Surface Generation...")
    print("=" * 50)
    
    try:
        # Test the advanced 3D visualization surface generation
        from modules.advanced_3d_visualization import Advanced3DVisualizer
        from modules.geometry import VesselGeometry
        
        # Create test vessel
        test_vessel = VesselGeometry(200.0, 5.0, 300.0, "Isotensoid")
        test_vessel.set_qrs_parameters(9.5, 0.1, 0.5)
        test_vessel.generate_profile()
        
        # Create visualizer
        visualizer = Advanced3DVisualizer()
        
        # Test surface generation
        print("   🔧 Testing surface mesh generation...")
        
        # This would be called internally by the visualizer
        inner_radius = test_vessel.inner_diameter / 2000  # Convert to meters
        cyl_length = test_vessel.cylindrical_length / 1000
        
        # Test profile generation
        z_profile, r_profile = visualizer._generate_complete_vessel_profile(
            inner_radius, cyl_length, test_vessel.dome_type, test_vessel
        )
        
        print(f"   ✅ 3D profile generated: {len(z_profile)} points")
        print(f"   📏 Z range: {np.min(z_profile):.3f} to {np.max(z_profile):.3f} m")
        print(f"   📏 R range: {np.min(r_profile):.3f} to {np.max(r_profile):.3f} m")
        
        # Check for dome curvature
        radius_variation = np.max(r_profile) - np.min(r_profile)
        if radius_variation > np.max(r_profile) * 0.1:
            print(f"   ✅ 3D surface will show proper dome geometry!")
        else:
            print(f"   ⚠️  3D surface may appear cylindrical")
        
    except Exception as e:
        print(f"   ❌ Error testing 3D surface generation: {str(e)}")

def quick_vessel_check():
    """Quick check of current vessel geometry in session state"""
    
    print("\n🔍 Quick Session State Check...")
    print("=" * 30)
    
    try:
        import streamlit as st
        
        if 'vessel_geometry' in st.session_state and st.session_state.vessel_geometry:
            vessel = st.session_state.vessel_geometry
            print(f"   ✅ Vessel geometry exists")
            print(f"   📐 Dome type: {vessel.dome_type}")
            print(f"   📏 Diameter: {vessel.inner_diameter}mm")
            print(f"   📏 Length: {vessel.cylindrical_length}mm")
            
            # Check profile
            if hasattr(vessel, 'profile_points') and vessel.profile_points:
                profile = vessel.profile_points
                z_data = np.array(profile['z_mm'])
                r_data = np.array(profile['r_inner_mm'])
                
                radius_variation = np.max(r_data) - np.min(r_data)
                max_radius = np.max(r_data)
                
                print(f"   📊 Profile points: {len(z_data)}")
                print(f"   🔄 Radius variation: {radius_variation:.1f}mm ({radius_variation/max_radius*100:.1f}%)")
                
                if radius_variation > max_radius * 0.1:
                    print(f"   ✅ Current vessel has good dome curvature!")
                else:
                    print(f"   ⚠️  Current vessel appears mostly cylindrical")
                    print(f"   💡 Try regenerating with different dome parameters")
            else:
                print(f"   ⚠️  No profile data found - may need to regenerate")
        else:
            print(f"   ❌ No vessel geometry in session state")
            
    except Exception as e:
        print(f"   ❌ Error checking session state: {str(e)}")

# Main test function
def run_all_tests():
    """Run all dome geometry tests"""
    
    print("🧪 DOME GEOMETRY DIAGNOSTIC SUITE")
    print("=" * 60)
    
    # Test 1: Basic dome geometry generation
    test_dome_geometry()
    
    # Test 2: 3D surface generation
    test_3d_surface_generation()
    
    # Test 3: Current session state
    quick_vessel_check()
    
    print("\n🎯 DIAGNOSIS COMPLETE!")
    print("=" * 60)
    print("If all tests show good dome curvature but visualization")
    print("still appears cylindrical, the issue is in the 3D")
    print("visualization surface mesh generation.")

if __name__ == "__main__":
    run_all_tests()
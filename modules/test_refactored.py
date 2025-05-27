"""
Test interface for the refactored trajectory planner.
This allows us to validate the new architecture before full integration.
"""

from .trajectories_refactored import TrajectoryPlannerRefactored
from .geometry import VesselGeometry
import numpy as np


def test_refactored_geodesic_engine(vessel_geometry):
    """Test the refactored geodesic engine with a simple case."""
    print("🔧 Testing Refactored Geodesic Engine")
    
    # Create refactored planner
    planner = TrajectoryPlannerRefactored(
        vessel_geometry=vessel_geometry,
        dry_roving_width_m=0.003,  # 3mm
        dry_roving_thickness_m=0.0002,  # 0.2mm
        target_cylinder_angle_deg=45.0,
        mu_friction_coefficient=0.0
    )
    
    # Test validation
    validation = planner.get_validation_results()
    print(f"✅ Validation: {validation}")
    
    # Test single circuit generation
    print("\n🎯 Generating single circuit...")
    result = planner.generate_trajectory(
        pattern_name='geodesic_spiral',
        coverage_option='single_circuit',
        user_circuits=1
    )
    
    if result and result.get('success'):
        print(f"✅ Single circuit: {result['total_points']} points generated")
        print(f"   Pattern type: {result['pattern_type']}")
        print(f"   Circuits: {result['total_circuits_legs']}")
        return True
    else:
        print(f"❌ Single circuit failed: {result}")
        return False


def test_refactored_multi_circuit(vessel_geometry):
    """Test multi-circuit generation with the refactored engine."""
    print("\n🔄 Testing Multi-Circuit Generation")
    
    planner = TrajectoryPlannerRefactored(
        vessel_geometry=vessel_geometry,
        dry_roving_width_m=0.003,
        target_cylinder_angle_deg=45.0
    )
    
    # Test user-defined circuits
    result = planner.generate_trajectory(
        pattern_name='geodesic_spiral',
        coverage_option='user_defined',
        user_circuits=2
    )
    
    if result and result.get('success'):
        print(f"✅ Multi-circuit: {result['total_points']} points generated")
        print(f"   Circuits: {result['total_circuits_legs']}")
        print(f"   Final angle: {result['final_turn_around_angle_deg']:.1f}°")
        return True
    else:
        print(f"❌ Multi-circuit failed: {result}")
        return False


def compare_engines(vessel_geometry):
    """Compare old vs new engine outputs."""
    print("\n⚖️ Comparing Engine Outputs")
    
    # Test refactored engine
    planner_new = TrajectoryPlannerRefactored(
        vessel_geometry=vessel_geometry,
        target_cylinder_angle_deg=45.0
    )
    
    result_new = planner_new.generate_trajectory(
        pattern_name='geodesic_spiral',
        coverage_option='single_circuit'
    )
    
    if result_new and result_new.get('success'):
        print(f"📊 Refactored Engine:")
        print(f"   Points: {result_new['total_points']}")
        print(f"   Clairaut's C: {result_new['clairauts_constant_used_m']*1000:.2f}mm")
        print(f"   Final φ: {result_new['final_turn_around_angle_deg']:.1f}°")
        
        # Check data quality
        x_data = result_new.get('x_points_m', [])
        y_data = result_new.get('y_points_m', [])
        z_data = result_new.get('z_points_m', [])
        
        if len(x_data) > 0:
            print(f"   X range: {np.min(x_data):.3f} to {np.max(x_data):.3f} m")
            print(f"   Y range: {np.min(y_data):.3f} to {np.max(y_data):.3f} m") 
            print(f"   Z range: {np.min(z_data):.3f} to {np.max(z_data):.3f} m")
            return True
        
    print("❌ Refactored engine comparison failed")
    return False
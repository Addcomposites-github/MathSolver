# Add this test function to your app.py to verify the fix

def test_trajectory_conversion():
    """Test function to verify trajectory data conversion is working"""
    st.markdown("### 🧪 Trajectory Conversion Test")
    
    try:
        from modules.trajectory_data_converter import TrajectoryDataConverter
        converter = TrajectoryDataConverter()
        
        # Test with sample unified trajectory data
        sample_unified_data = {
            'points': [
                type('TrajectoryPoint', (), {
                    'rho': 0.1, 'z': 0.0, 'phi': 0.0, 'alpha_deg': 45.0
                })(),
                type('TrajectoryPoint', (), {
                    'rho': 0.1, 'z': 0.1, 'phi': 0.1, 'alpha_deg': 45.0
                })(),
                type('TrajectoryPoint', (), {
                    'rho': 0.1, 'z': 0.2, 'phi': 0.2, 'alpha_deg': 45.0
                })()
            ],
            'metadata': {'pattern_type': 'test'},
            'quality_metrics': {'success': True}
        }
        
        # Test conversion
        converted = converter.convert_unified_trajectory_to_visualization_format(sample_unified_data)
        
        if converted and converted.get('success'):
            st.success("✅ Trajectory conversion test PASSED")
            st.write(f"Converted {converted['total_points']} points")
            st.write(f"X range: {min(converted['x_points_m']):.3f} to {max(converted['x_points_m']):.3f}")
            st.write(f"Y range: {min(converted['y_points_m']):.3f} to {max(converted['y_points_m']):.3f}")
            return True
        else:
            st.error("❌ Trajectory conversion test FAILED")
            return False
            
    except Exception as e:
        st.error(f"❌ Test error: {e}")
        return False

# Add this to your sidebar for testing
if st.sidebar.button("🧪 Test Trajectory Fix"):
    test_trajectory_conversion()

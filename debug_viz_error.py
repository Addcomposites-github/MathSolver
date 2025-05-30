"""
Debug script to isolate the array comparison error in visualization
"""
import streamlit as st
import numpy as np

def test_array_comparisons():
    """Test various array comparison patterns to find the issue"""
    st.write("Testing array comparison patterns...")
    
    # Create test array
    test_array = np.array([1, 2, 3, 4, 5])
    
    try:
        # Test 1: Direct equality comparison (problematic)
        if test_array == 0:  # This will cause the error
            st.write("Array equals zero")
    except ValueError as e:
        st.error(f"Error in direct array comparison: {e}")
    
    try:
        # Test 2: Correct way using numpy functions
        if np.all(test_array == 0):
            st.write("All array elements equal zero")
        else:
            st.write("Not all array elements equal zero")
    except Exception as e:
        st.error(f"Error in numpy.all comparison: {e}")
    
    try:
        # Test 3: Check for problematic patterns in our code
        z_test = np.array([0.1, 0.2, 0.3])
        
        # This pattern might be problematic
        if z_test == None:  # This could cause issues
            st.write("Array is None")
    except ValueError as e:
        st.error(f"Error in None comparison: {e}")
    
    try:
        # Correct way to check for None
        if z_test is None:
            st.write("Array is None")
        else:
            st.write("Array is not None")
    except Exception as e:
        st.error(f"Error in is None comparison: {e}")

if __name__ == "__main__":
    test_array_comparisons()
# Add this method to UnifiedTrajectoryPlanner class

def _validate_trajectory_inputs(self, **params) -> Tuple[bool, str]:
    """
    Validate all trajectory generation inputs before processing
    
    Returns:
        (is_valid, error_message)
    """
    try:
        # Check vessel geometry
        if not self.vessel_geometry:
            return False, "Vessel geometry is None"
        
        # Check roving width
        if self.roving_width_m <= 0:
            return False, f"Invalid roving width: {self.roving_width_m}m"
        
        # Check target parameters
        target_params = params.get('target_params', {})
        if target_params:
            angle_deg = target_params.get('winding_angle_deg')
            if angle_deg and (angle_deg <= 0 or angle_deg >= 90):
                return False, f"Invalid winding angle: {angle_deg}°"
        
        # Check vessel geometry has required attributes
        vessel_radius = self._get_vessel_radius()
        if vessel_radius <= 0:
            return False, f"Invalid vessel radius: {vessel_radius}m"
        
        return True, ""
        
    except Exception as e:
        return False, f"Input validation error: {str(e)}"

# Modify the generate_trajectory method to include validation:
def generate_trajectory(self, **kwargs) -> TrajectoryResult:
    """Enhanced trajectory generation with robust validation"""
    
    # Add input validation at the start
    valid, error_msg = self._validate_trajectory_inputs(**kwargs)
    if not valid:
        self._log_message(f"Input validation failed: {error_msg}")
        return TrajectoryResult(
            points=[], 
            metadata={'validation_error': error_msg}, 
            quality_metrics={'error': error_msg, 'success': False}
        )
    
    # Rest of existing method...
    self._log_message(f"Generating trajectory: {kwargs.get('pattern_type', 'unknown')}")
    # ... existing code ...
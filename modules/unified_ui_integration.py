"""
Unified UI Integration
Complete integration layer that routes all trajectory requests through the unified system
"""

import streamlit as st
from typing import Dict, Any, Optional
from .unified_trajectory_planner import UnifiedTrajectoryPlanner
from .legacy_trajectory_adapter import LegacyTrajectoryAdapter
from .ui_parameter_mapper import UIParameterMapper
from .unified_visualization_adapter import UnifiedVisualizationAdapter

class UnifiedTrajectoryHandler:
    """
    Handles all trajectory generation requests through the unified system.
    Provides backward compatibility while using the new architecture.
    """
    
    def __init__(self):
        self.mapper = UIParameterMapper()
        self.planner = None
        self.adapter = None
        self.viz_adapter = UnifiedVisualizationAdapter()
    
    def initialize_planner(self, vessel_geometry, roving_width_mm: float = 3.0):
        """Initialize the unified planner with vessel geometry"""
        self.planner = UnifiedTrajectoryPlanner(
            vessel_geometry=vessel_geometry,
            roving_width_m=roving_width_mm / 1000,  # Convert mm to m
            payout_length_m=0.5,  # Default 500mm payout
            default_friction_coeff=0.1
        )
        self.adapter = LegacyTrajectoryAdapter(self.planner)
    
    def generate_trajectory_unified(self, pattern_type: str, ui_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trajectory using unified system with automatic parameter mapping.
        
        Args:
            pattern_type: Pattern type selected in UI
            ui_params: Dictionary of UI parameters
            
        Returns:
            Trajectory data in legacy format for visualization compatibility
        """
        
        if not self.planner:
            raise RuntimeError("Planner not initialized. Call initialize_planner() first.")
        
        try:
            # Handle the new unified system interface directly
            if pattern_type == "🚀 Unified Trajectory System (New)":
                # Parameters already in correct format from UI
                result = self.planner.generate_trajectory(**ui_params)
                return self.viz_adapter.convert_trajectory_result_for_viz(result, pattern_type)
            
            # Map legacy UI parameters to unified format
            unified_params = self.mapper.map_streamlit_ui_to_unified(pattern_type, ui_params)
            
            # Generate trajectory using unified system
            result = self.planner.generate_trajectory(**unified_params)
            
            # Convert to visualization-compatible format using enhanced adapter
            viz_output = self.viz_adapter.convert_trajectory_result_for_viz(result, pattern_type)
            
            # Add unified system indicators
            viz_output['unified_system_used'] = True
            viz_output['enhanced_quality_metrics'] = True
            
            return viz_output
            
        except Exception as e:
            st.error(f"Trajectory generation failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'total_points': 0,
                'pattern_type': pattern_type
            }

def create_unified_trajectory_interface():
    """
    Create a unified trajectory generation interface that works with all pattern types.
    This replaces the scattered trajectory generation calls throughout the app.
    """
    
    # Initialize the handler in session state if not present
    if 'unified_handler' not in st.session_state:
        st.session_state.unified_handler = UnifiedTrajectoryHandler()
    
    return st.session_state.unified_handler

def unified_trajectory_generator(pattern_type: str, **ui_params) -> Optional[Dict[str, Any]]:
    """
    Universal trajectory generation function that works with any pattern type.
    
    Args:
        pattern_type: The selected pattern type from UI
        **ui_params: All UI parameters as keyword arguments
        
    Returns:
        Trajectory data dictionary or None if generation fails
    """
    
    # Get or create unified handler
    handler = create_unified_trajectory_interface()
    
    # Initialize planner if needed
    if not handler.planner and st.session_state.vessel_geometry:
        roving_width = ui_params.get('roving_width', ui_params.get('unified_roving_width', 3.0))
        handler.initialize_planner(st.session_state.vessel_geometry, roving_width)
    
    if not handler.planner:
        st.error("Please generate vessel geometry first")
        return None
    
    # Generate trajectory
    with st.spinner(f"Generating {pattern_type} trajectory using unified system..."):
        trajectory_data = handler.generate_trajectory_unified(pattern_type, ui_params)
    
    if trajectory_data.get('success', True) and trajectory_data.get('total_points', 0) > 0:
        st.success(f"✅ Generated {trajectory_data['total_points']} trajectory points!")
        
        # Show unified system indicator
        if trajectory_data.get('unified_system_used'):
            st.info("🚀 **Powered by Unified Trajectory System** - Enhanced accuracy and reliability")
        
        return trajectory_data
    else:
        st.error(f"❌ Failed to generate {pattern_type} trajectory")
        return None

def get_ui_parameters_for_pattern(pattern_type: str) -> Dict[str, Any]:
    """
    Extract current UI parameters based on pattern type.
    This function should be called within Streamlit context to access current form values.
    """
    
    # This function would extract actual parameters from the Streamlit UI
    # For now, return empty dict - actual implementation would access live UI state
    
    return {}

def show_trajectory_quality_metrics(trajectory_data: Dict[str, Any]):
    """Display enhanced quality metrics from unified system"""
    
    if not trajectory_data or not trajectory_data.get('quality_metrics'):
        return
    
    st.markdown("### 📊 Trajectory Quality Metrics")
    
    quality = trajectory_data['quality_metrics']
    
    # Create metrics columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'max_c0_gap_mm' in quality:
            st.metric("Max Gap", f"{quality['max_c0_gap_mm']:.3f}mm", 
                     help="Maximum position discontinuity")
    
    with col2:
        if 'max_c1_velocity_jump_mps' in quality:
            st.metric("Max Vel Jump", f"{quality['max_c1_velocity_jump_mps']:.2f}m/s",
                     help="Maximum velocity discontinuity")
    
    with col3:
        if 'total_length_m' in quality:
            st.metric("Path Length", f"{quality['total_length_m']:.2f}m",
                     help="Total trajectory length")
    
    with col4:
        smoothness = "High" if quality.get('is_smooth_c2') else "Medium" if quality.get('is_smooth_c1') else "Basic"
        st.metric("Smoothness", smoothness, help="Mathematical continuity level")
    
    # Detailed quality report
    if st.expander("🔬 Detailed Quality Analysis", expanded=False):
        quality_df_data = []
        for key, value in quality.items():
            if isinstance(value, (int, float)):
                quality_df_data.append({
                    "Property": key.replace('_', ' ').title(),
                    "Value": f"{value:.4f}" if isinstance(value, float) else str(value),
                    "Status": "✅ Pass" if value < 0.1 else "⚠️ Check" if value < 1.0 else "❌ Issue"
                })
            else:
                quality_df_data.append({
                    "Property": key.replace('_', ' ').title(),
                    "Value": str(value),
                    "Status": "✅ Pass" if value else "❌ Fail"
                })
        
        if quality_df_data:
            st.dataframe(quality_df_data, use_container_width=True, hide_index=True)
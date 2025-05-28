"""
UI Parameter Mapper for Unified Trajectory System
Converts Streamlit UI parameters to unified trajectory system format
"""

from typing import Dict, Any, Optional

class UIParameterMapper:
    """
    Maps UI parameters from various Streamlit interfaces to the unified trajectory system.
    Handles backward compatibility and parameter translation.
    """
    
    def __init__(self):
        self.pattern_type_mapping = {
            'Geodesic': 'geodesic',
            'Non-Geodesic': 'non_geodesic', 
            'Helical': 'helical',
            'Hoop': 'hoop',
            'Multi-Circuit Pattern': 'geodesic',
            'Transitional': 'helical',
            'Polar': 'hoop'
        }
        
        self.physics_model_mapping = {
            'geodesic': 'clairaut',
            'non_geodesic': 'friction',
            'helical': 'constant_angle',
            'hoop': 'constant_angle'
        }
    
    def map_legacy_geodesic_params(self, ui_params: Dict[str, Any]) -> Dict[str, Any]:
        """Map legacy geodesic UI parameters to unified format"""
        return {
            'pattern_type': 'geodesic',
            'physics_model': 'clairaut',
            'coverage_mode': 'full_coverage' if ui_params.get('num_circuits', 1) > 1 else 'single_pass',
            'continuity_level': 1,  # Default smooth continuity
            'num_layers_desired': ui_params.get('number_of_passes', 1),
            'target_params': {
                'winding_angle_deg': ui_params.get('target_angle', 45.0)
            },
            'options': {
                'num_points': ui_params.get('dome_points', 150) + ui_params.get('cylinder_points', 20),
                'friction_coefficient': 0.0  # Pure geodesic
            }
        }
    
    def map_legacy_non_geodesic_params(self, ui_params: Dict[str, Any]) -> Dict[str, Any]:
        """Map legacy non-geodesic UI parameters to unified format"""
        return {
            'pattern_type': 'non_geodesic',
            'physics_model': 'friction', 
            'coverage_mode': 'full_coverage' if ui_params.get('num_circuits', 1) > 1 else 'single_pass',
            'continuity_level': 1,
            'num_layers_desired': ui_params.get('num_circuits', 1),
            'target_params': {
                'winding_angle_deg': ui_params.get('target_cylinder_angle_deg', 45.0)
            },
            'options': {
                'num_points': ui_params.get('dome_points', 150) + ui_params.get('cylinder_points', 20),
                'friction_coefficient': ui_params.get('friction_coefficient', 0.3)
            }
        }
    
    def map_legacy_multi_circuit_params(self, ui_params: Dict[str, Any]) -> Dict[str, Any]:
        """Map legacy multi-circuit UI parameters to unified format"""
        pattern_type = ui_params.get('pattern_type', 'geodesic').lower()
        if 'geodesic' in pattern_type:
            base_pattern = 'geodesic'
            physics_model = 'clairaut'
        else:
            base_pattern = 'non_geodesic'
            physics_model = 'friction'
            
        return {
            'pattern_type': base_pattern,
            'physics_model': physics_model,
            'coverage_mode': 'full_coverage',
            'continuity_level': 2,  # High continuity for multi-circuit
            'num_layers_desired': ui_params.get('num_circuits_for_vis', 5),
            'target_params': {
                'winding_angle_deg': ui_params.get('target_angle', 45.0)
            },
            'options': {
                'num_points': ui_params.get('dome_points', 50) + ui_params.get('cylinder_points', 10),
                'friction_coefficient': ui_params.get('friction_coefficient', 0.0 if base_pattern == 'geodesic' else 0.3)
            }
        }
    
    def map_helical_params(self, ui_params: Dict[str, Any]) -> Dict[str, Any]:
        """Map helical pattern UI parameters to unified format"""
        return {
            'pattern_type': 'helical',
            'physics_model': 'constant_angle',
            'coverage_mode': 'full_coverage' if ui_params.get('circuits_to_close', 1) > 1 else 'single_pass',
            'continuity_level': 1,
            'num_layers_desired': ui_params.get('circuits_to_close', 1),
            'target_params': {
                'winding_angle_deg': ui_params.get('winding_angle', 55.0)
            },
            'options': {
                'num_points': 100,
                'overlap_allowance': ui_params.get('overlap_allowance', 10.0)
            }
        }
    
    def map_refactored_engine_params(self, ui_params: Dict[str, Any]) -> Dict[str, Any]:
        """Map refactored engine UI parameters to unified format"""
        pattern_name = ui_params.get('refactored_pattern', 'geodesic_spiral')
        coverage_option = ui_params.get('coverage_option', 'single_circuit')
        
        # Determine pattern type and physics model
        if 'geodesic' in pattern_name:
            pattern_type = 'geodesic'
            physics_model = 'clairaut'
        elif 'non_geodesic' in pattern_name:
            pattern_type = 'non_geodesic'
            physics_model = 'friction'
        else:
            pattern_type = 'helical'
            physics_model = 'constant_angle'
        
        # Map coverage option
        coverage_mapping = {
            'single_circuit': 'single_pass',
            'full_coverage': 'full_coverage',
            'user_defined': 'custom'
        }
        
        return {
            'pattern_type': pattern_type,
            'physics_model': physics_model,
            'coverage_mode': coverage_mapping.get(coverage_option, 'single_pass'),
            'continuity_level': 1,
            'num_layers_desired': ui_params.get('user_circuits', 1),
            'target_params': {
                'winding_angle_deg': ui_params.get('target_angle', 45.0)
            },
            'options': {
                'num_points': 100,
                'friction_coefficient': 0.0 if pattern_type == 'geodesic' else 0.3
            }
        }
    
    def map_streamlit_ui_to_unified(self, pattern_type: str, ui_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main mapping function that routes UI parameters based on pattern type.
        
        Args:
            pattern_type: The selected pattern type from Streamlit UI
            ui_params: Dictionary of UI parameters from Streamlit
            
        Returns:
            Dictionary formatted for UnifiedTrajectoryPlanner.generate_trajectory()
        """
        
        # Handle the new unified system (pass through)
        if pattern_type == "🚀 Unified Trajectory System (New)":
            return ui_params  # Already in correct format
        
        # Map legacy pattern types
        if pattern_type == "Geodesic":
            return self.map_legacy_geodesic_params(ui_params)
        elif pattern_type == "Non-Geodesic":
            return self.map_legacy_non_geodesic_params(ui_params)
        elif pattern_type == "Multi-Circuit Pattern":
            return self.map_legacy_multi_circuit_params(ui_params)
        elif pattern_type in ["Helical", "Transitional"]:
            return self.map_helical_params(ui_params)
        elif pattern_type == "🔬 Refactored Engine (Test)":
            return self.map_refactored_engine_params(ui_params)
        elif pattern_type == "Hoop":
            return {
                'pattern_type': 'hoop',
                'physics_model': 'constant_angle',
                'coverage_mode': 'single_pass',
                'continuity_level': 1,
                'num_layers_desired': 1,
                'target_params': {},
                'options': {'num_points': 100}
            }
        elif pattern_type == "Polar":
            return {
                'pattern_type': 'hoop',
                'physics_model': 'constant_angle', 
                'coverage_mode': 'single_pass',
                'continuity_level': 1,
                'num_layers_desired': 1,
                'target_params': {},
                'options': {'num_points': 50}
            }
        else:
            # Default fallback
            return {
                'pattern_type': 'geodesic',
                'physics_model': 'clairaut',
                'coverage_mode': 'single_pass',
                'continuity_level': 1,
                'num_layers_desired': 1,
                'target_params': {'winding_angle_deg': 45.0},
                'options': {'num_points': 100}
            }
    
    def extract_ui_parameters(self, streamlit_session_state, pattern_type: str) -> Dict[str, Any]:
        """
        Extract relevant UI parameters from Streamlit session state and current form values.
        This method should be called from within the Streamlit app context.
        """
        # This will be populated with actual UI values when called from Streamlit
        ui_params = {}
        
        # Note: This method would typically access st.session_state and current form values
        # For now, return empty dict - actual implementation would extract live UI values
        
        return ui_params
    
    def get_parameter_validation_rules(self, pattern_type: str) -> Dict[str, Any]:
        """Get validation rules for UI parameters based on pattern type"""
        
        base_rules = {
            'winding_angle_min': 10.0,
            'winding_angle_max': 85.0,
            'roving_width_min': 0.1,
            'roving_width_max': 10.0,
            'num_points_min': 50,
            'num_points_max': 500
        }
        
        pattern_specific_rules = {
            'geodesic': {
                'friction_coefficient': 0.0,
                'requires_clairaut_validation': True
            },
            'non_geodesic': {
                'friction_coefficient_min': 0.1,
                'friction_coefficient_max': 1.0,
                'allows_extreme_angles': True
            },
            'helical': {
                'circuits_min': 1,
                'circuits_max': 20,
                'overlap_allowance_range': (-10.0, 50.0)
            },
            'hoop': {
                'fixed_z_position': True,
                'angular_coverage_only': True
            }
        }
        
        mapped_pattern = self.pattern_type_mapping.get(pattern_type, pattern_type.lower())
        rules = base_rules.copy()
        rules.update(pattern_specific_rules.get(mapped_pattern, {}))
        
        return rules
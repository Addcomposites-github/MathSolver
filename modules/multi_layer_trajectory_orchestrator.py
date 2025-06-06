"""
Multi-Layer Trajectory Orchestrator
Robust integration between LayerStackManager and StreamlinedTrajectoryPlanner
Ensures each layer's trajectory is planned on the correct evolved mandrel surface
"""

import numpy as np
import streamlit as st
from typing import Dict, List, Tuple, Optional
from modules.geometry import VesselGeometry
from modules.unified_pattern_calculator import PatternCalculator
from modules.trajectory_visualization import create_3d_trajectory_visualization

# Import unified trajectory system
try:
    from modules.unified_trajectory_planner import UnifiedTrajectoryPlanner
    from modules.unified_visualization_adapter import UnifiedVisualizationAdapter
    UNIFIED_SYSTEM_AVAILABLE = True
except ImportError:
    # Fallback to old system if unified not available
    from modules.trajectories_streamlined import StreamlinedTrajectoryPlanner
    UNIFIED_SYSTEM_AVAILABLE = False


class MultiLayerTrajectoryOrchestrator:
    """
    Orchestrates trajectory planning for multi-layer composite vessels.
    Ensures each layer uses the correct evolved mandrel surface.
    """
    
    def __init__(self, layer_manager):
        """Initialize with LayerStackManager instance"""
        self.layer_manager = layer_manager
        self.pattern_calculator = PatternCalculator(resin_factor=1.0)
        self.generated_trajectories = []
        
        # Initialize unified trajectory system if available
        if UNIFIED_SYSTEM_AVAILABLE:
            self.unified_planner = None  # Will be initialized per layer
            self.viz_adapter = UnifiedVisualizationAdapter()
            self.using_unified_system = True
        else:
            self.using_unified_system = False
        
    def generate_all_layer_trajectories(self, roving_width_mm: float = 3.0, 
                                      roving_thickness_mm: float = 0.125) -> List[Dict]:
        """
        Generate trajectories for all layers with proper mandrel evolution.
        
        Parameters:
        -----------
        roving_width_mm : float
            Roving width in millimeters
        roving_thickness_mm : float
            Roving thickness in millimeters
            
        Returns:
        --------
        List[Dict] : Generated trajectory data for all layers
        """
        all_trajectories = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for layer_index, layer_def in enumerate(self.layer_manager.layer_stack):
            status_text.text(f"Planning trajectory for Layer {layer_def.layer_set_id} "
                           f"({layer_def.layer_type} at {layer_def.winding_angle_deg}°)...")
            
            try:
                # Apply previous layers to mandrel BEFORE generating this layer's trajectory
                # This ensures each layer uses the correct evolved mandrel surface
                while len(self.layer_manager.mandrel.layers_applied) < layer_index:
                    layer_to_apply_index = len(self.layer_manager.mandrel.layers_applied)
                    print(f"[DEBUG] Applying layer {layer_to_apply_index} to mandrel before generating trajectory for layer {layer_index}")
                    self.layer_manager.apply_layer_to_mandrel(layer_to_apply_index)
                
                # Get current mandrel surface for this layer (after previous layers applied)
                current_mandrel = self.layer_manager.get_current_mandrel_for_trajectory()
                print(f"[DEBUG] Generating trajectory for layer {layer_index} using mandrel with {len(self.layer_manager.mandrel.layers_applied)} previous layers applied")
                print(f"[DEBUG] Current mandrel equatorial radius: {current_mandrel['equatorial_radius_mm']:.1f}mm")
                print(f"[DEBUG] Current mandrel polar radius: {current_mandrel['polar_opening_radius_mm']:.1f}mm")
                
                trajectory_data = self._generate_single_layer_trajectory(
                    layer_index, layer_def, roving_width_mm, roving_thickness_mm
                )
                
                if trajectory_data:
                    # Wrap in expected format for visualization
                    wrapped_trajectory = {
                        'layer_id': layer_def.layer_set_id,
                        'layer_type': layer_def.layer_type,
                        'winding_angle': layer_def.winding_angle_deg,
                        'trajectory_data': trajectory_data,
                        'mandrel_state': current_mandrel
                    }
                    all_trajectories.append(wrapped_trajectory)
                    st.success(f"✅ Layer {layer_def.layer_set_id} trajectory generated using evolved mandrel")
                
            except Exception as e:
                st.error(f"❌ Error generating trajectory for Layer {layer_def.layer_set_id}: {str(e)}")
                continue
            
            progress_bar.progress((layer_index + 1) / len(self.layer_manager.layer_stack))
        
        progress_bar.progress(1.0)
        status_text.text("✅ All layer trajectories generated!")
        
        # Store trajectories in the correct session state for visualization
        st.session_state.all_layer_trajectories = all_trajectories
        
        # Debug: Show what we're storing
        print(f"DEBUG: Stored {len(all_trajectories)} trajectories in session state")
        for i, traj in enumerate(all_trajectories):
            print(f"DEBUG: Trajectory {i+1} - Layer ID: {traj['layer_id']}, Path points: {len(traj['trajectory_data'].get('path_points', []))}")
        
        self.generated_trajectories = all_trajectories
        return all_trajectories
    
    def _generate_single_layer_trajectory(self, layer_index: int, layer_def, 
                                        roving_width_mm: float, roving_thickness_mm: float) -> Dict:
        """
        Generate trajectory for a single layer using current mandrel surface.
        Uses unified trajectory system for enhanced quality and consistency.
        
        Parameters:
        -----------
        layer_index : int
            Index of layer in stack
        layer_def : LayerDefinition
            Layer definition object
        roving_width_mm : float
            Roving width in millimeters
        roving_thickness_mm : float
            Roving thickness in millimeters
            
        Returns:
        --------
        Dict : Trajectory data for the layer
        """
        # Get current mandrel surface
        mandrel_data = self.layer_manager.get_current_mandrel_for_trajectory()
        
        # Debug mandrel data being passed
        print(f"[DEBUG] Mandrel data keys: {list(mandrel_data.keys())}")
        if 'profile_points' in mandrel_data:
            profile = mandrel_data['profile_points']
            if 'z_mm' in profile:
                z_range = profile['z_mm']
                print(f"[DEBUG] Mandrel profile Z range: {z_range.min():.1f} to {z_range.max():.1f}mm")
            if 'r_inner_mm' in profile:
                r_range = profile['r_inner_mm']
                print(f"[DEBUG] Mandrel profile R range: {r_range.min():.1f} to {r_range.max():.1f}mm")
        print(f"[DEBUG] Mandrel equatorial radius: {mandrel_data.get('equatorial_radius_mm', 'N/A')}mm")
        
        # Create temporary VesselGeometry for this layer's winding surface
        temp_vessel = self._create_layer_vessel_geometry(mandrel_data, layer_def)
        
        if self.using_unified_system:
            # Use unified trajectory system for enhanced quality
            return self._generate_unified_layer_trajectory(
                temp_vessel, layer_def, roving_width_mm, roving_thickness_mm
            )
        else:
            # Fallback to legacy system
            return self._generate_legacy_layer_trajectory(
                temp_vessel, layer_def, roving_width_mm, roving_thickness_mm
            )
    
    def _generate_unified_layer_trajectory(self, temp_vessel, layer_def, 
                                         roving_width_mm: float, roving_thickness_mm: float) -> Dict:
        """Generate trajectory using unified system with layer-specific parameters"""
        
        # Initialize unified planner for this layer
        print(f"[DEBUG] Creating UnifiedTrajectoryPlanner with temp_vessel type: {type(temp_vessel)}")
        
        # Check vessel geometry bounds before passing to planner
        if hasattr(temp_vessel, 'get_profile_points'):
            profile = temp_vessel.get_profile_points()
            if 'z_mm' in profile:
                z_range = profile['z_mm']
                print(f"[DEBUG] temp_vessel Z range (mm): {z_range.min():.1f} to {z_range.max():.1f}")
                print(f"[DEBUG] temp_vessel Z range (m): {z_range.min()/1000:.3f} to {z_range.max()/1000:.3f}")
        
        self.unified_planner = UnifiedTrajectoryPlanner(
            vessel_geometry=temp_vessel,
            roving_width_m=roving_width_mm / 1000,  # Convert mm to m
            payout_length_m=0.5,  # Default 500mm payout
            default_friction_coeff=0.1
        )
        
        # Determine optimal parameters for this layer type
        layer_params = self._determine_layer_parameters(layer_def, roving_width_mm, roving_thickness_mm)
        
        print(f"=== UNIFIED LAYER TRAJECTORY GENERATION ===")
        print(f"Layer {layer_def.layer_set_id}: {layer_def.layer_type} at {layer_def.winding_angle_deg}°")
        print(f"Pattern: {layer_params['pattern_type']}")
        print(f"Physics: {layer_params['physics_model']}")
        print(f"Coverage: {layer_params['coverage_mode']}")
        
        try:
            # Extract target angle and create proper target_params
            target_angle = layer_params.pop('target_angle_deg', 45.0)
            
            # Generate trajectory using unified system with proper parameter format
            result = self.unified_planner.generate_trajectory(
                pattern_type=layer_params['pattern_type'],
                coverage_mode=layer_params['coverage_mode'],
                physics_model=layer_params['physics_model'],
                continuity_level=layer_params['continuity_level'],
                num_layers_desired=layer_params['num_layers_desired'],
                target_params={'winding_angle_deg': target_angle}
            )
            
            # Debug the result before conversion
            print(f"[DEBUG] UnifiedPlanner result type: {type(result)}")
            if hasattr(result, 'points'):
                print(f"[DEBUG] UnifiedPlanner generated {len(result.points)} points")
            else:
                print(f"[DEBUG] UnifiedPlanner result has no 'points' attribute")
                print(f"[DEBUG] Result attributes: {dir(result)}")
            
            # Convert to visualization-compatible format
            try:
                trajectory_data = self.viz_adapter.convert_trajectory_result_for_viz(
                    result, f"{layer_def.layer_type}_{layer_def.winding_angle_deg}deg"
                )
                print(f"[DEBUG] Conversion successful, trajectory_data keys: {list(trajectory_data.keys())}")
            except Exception as conv_error:
                print(f"[ERROR] Trajectory conversion failed: {conv_error}")
                import traceback
                print(f"[ERROR] Conversion traceback:")
                print(traceback.format_exc())
                # Return error data to prevent fallback
                trajectory_data = {
                    'success': False,
                    'error': f'Conversion failed: {conv_error}',
                    'path_points': []
                }
            
            # Add layer-specific metadata
            trajectory_data.update({
                'layer_system_used': 'unified',
                'layer_id': layer_def.layer_set_id,
                'layer_type': layer_def.layer_type,
                'target_angle': layer_def.winding_angle_deg,
                'enhanced_quality': True
            })
            
            # Debug: Print what data format we're returning
            print(f"DEBUG: Returning trajectory data with keys: {list(trajectory_data.keys())}")
            print(f"DEBUG: Path points available: {len(trajectory_data.get('path_points', []))}")
            
            return trajectory_data
            
        except Exception as e:
            st.error(f"Unified trajectory generation failed for layer {layer_def.layer_set_id}: {str(e)}")
            # Fallback to legacy system
            return self._generate_legacy_layer_trajectory(
                temp_vessel, layer_def, roving_width_mm, roving_thickness_mm
            )
    
    def _determine_layer_parameters(self, layer_def, roving_width_mm: float, roving_thickness_mm: float) -> Dict:
        """
        Determine optimal unified system parameters for specific layer type.
        Maps layer characteristics to unified trajectory parameters.
        """
        base_params = {
            'roving_width_mm': roving_width_mm,
            'roving_thickness_mm': roving_thickness_mm,
            'target_angle_deg': layer_def.winding_angle_deg,
            'continuity_level': 2,  # C2 continuity for manufacturing quality
            'num_layers_desired': 1  # Single layer at a time
        }
        
        # Determine pattern type and physics model based on layer characteristics
        if layer_def.layer_type == 'hoop':
            # Hoop layers: nearly circumferential winding
            base_params.update({
                'pattern_type': 'hoop',
                'physics_model': 'constant_angle',
                'coverage_mode': 'full_coverage',
                'num_layers_desired': min(8, max(3, int(360 / max(layer_def.winding_angle_deg, 5))))
            })
        elif layer_def.winding_angle_deg < 25:
            # Low-angle layers: geodesic-dominant
            base_params.update({
                'pattern_type': 'geodesic',
                'physics_model': 'clairaut',
                'coverage_mode': 'optimized_coverage',
                'num_layers_desired': max(4, int(20 / max(layer_def.winding_angle_deg, 5)))
            })
        elif layer_def.winding_angle_deg <= 75:
            # Mid to high-angle layers: geodesic-based helical with full coverage for realistic fiber behavior
            base_params.update({
                'pattern_type': 'helical',
                'physics_model': 'clairaut',
                'coverage_mode': 'full_coverage',
                'num_layers_desired': max(6, int(30 / max(layer_def.winding_angle_deg, 10)))
            })
        else:
            # Near-hoop layers: treat as hoop with friction
            base_params.update({
                'pattern_type': 'non_geodesic',
                'physics_model': 'friction',
                'coverage_mode': 'full_coverage',
                'friction_coefficient': 0.3,
                'num_layers_desired': min(12, max(6, int(60 / max(90 - layer_def.winding_angle_deg, 5))))
            })
        
        return base_params
    
    def _generate_legacy_layer_trajectory(self, temp_vessel, layer_def, 
                                        roving_width_mm: float, roving_thickness_mm: float) -> Dict:
        """Fallback trajectory generation using legacy system"""
        
        # Set up trajectory planner for this specific layer
        # Use unified trajectory planner with correct parameters
        layer_planner = UnifiedTrajectoryPlanner(
            vessel_geometry=temp_vessel,
            roving_width_m=roving_width_mm / 1000.0,
            payout_length_m=0.5,  # Default payout length
            default_friction_coeff=0.1
        )
        
        # Calculate winding pattern for this layer
        pattern_params = self._calculate_layer_pattern(temp_vessel, layer_def, roving_width_mm)
        
        # Determine pattern name based on layer type
        if layer_def.layer_type == 'hoop':
            pattern_name = "helical_unified"  # Near-hoop pattern
        elif layer_def.winding_angle_deg < 30:
            pattern_name = "geodesic_spiral"  # Low-angle helical
        else:
            pattern_name = "non_geodesic_spiral"  # Standard helical
        
        # Generate trajectory with practical circuit count
        calculated_circuits = pattern_params.get('num_passes', 10)
        practical_circuits = min(calculated_circuits, 8)  # Practical limit
        
        trajectory_data = layer_planner.generate_trajectory(
            pattern_name=pattern_name,
            coverage_option="user_defined_circuits",
            user_circuits=practical_circuits
        )
        
        # Add legacy system indicator
        if trajectory_data:
            trajectory_data['layer_system_used'] = 'legacy'
        
        return trajectory_data
    
    def _create_layer_vessel_geometry(self, mandrel_data: Dict, layer_def) -> VesselGeometry:
        """
        Create VesselGeometry object representing current winding surface.
        
        Parameters:
        -----------
        mandrel_data : Dict
            Current mandrel geometry data
        layer_def : LayerDefinition
            Layer definition object
            
        Returns:
        --------
        VesselGeometry : Vessel geometry for current winding surface
        """
        current_surface_profile = mandrel_data['profile_points']
        equatorial_radius = mandrel_data['equatorial_radius_mm']
        
        # Use original vessel's cylindrical length instead of total height
        # (total height includes domes, but cylindrical_length parameter is just the cylinder section)
        original_vessel = st.session_state.get('vessel_geometry')
        if original_vessel and hasattr(original_vessel, 'cylindrical_length'):
            cylinder_length = original_vessel.cylindrical_length
        else:
            # Fallback: estimate cylinder length (total height minus estimated dome heights)
            z_mm = current_surface_profile['z_mm']
            total_height = np.max(z_mm) - np.min(z_mm)
            estimated_dome_height = equatorial_radius * 0.7  # Conservative estimate
            cylinder_length = max(50, total_height - 2 * estimated_dome_height)  # Ensure minimum length
        
        print(f"[DEBUG] Using cylindrical_length: {cylinder_length:.1f}mm for temp_vessel")
        
        # Create temporary vessel geometry using original vessel parameters
        if original_vessel:
            # Copy dome parameters from original vessel to ensure consistency
            temp_vessel = VesselGeometry(
                inner_diameter=equatorial_radius * 2,
                wall_thickness=0.1,  # Nominal value
                cylindrical_length=cylinder_length,
                dome_type=getattr(original_vessel, 'dome_type', "Isotensoid")
            )
            
            # Copy dome-specific parameters if they exist
            if hasattr(original_vessel, 'q_factor'):
                temp_vessel.set_qrs_parameters(
                    original_vessel.q_factor,
                    original_vessel.r_factor, 
                    original_vessel.s_factor
                )
                print(f"[DEBUG] Copied QRS parameters: q={original_vessel.q_factor}, r={original_vessel.r_factor}, s={original_vessel.s_factor}")
        else:
            # Fallback without original vessel
            temp_vessel = VesselGeometry(
                inner_diameter=equatorial_radius * 2,
                wall_thickness=0.1,
                cylindrical_length=cylinder_length,
                dome_type="Isotensoid"
            )
        
        # Check temp_vessel bounds before profile override
        if hasattr(temp_vessel, 'get_profile_points'):
            pre_profile = temp_vessel.get_profile_points()
            if 'z_mm' in pre_profile:
                z_pre = pre_profile['z_mm']
                print(f"[DEBUG] temp_vessel Z range BEFORE profile override: {z_pre.min():.1f} to {z_pre.max():.1f}mm")
        
        # Use the actual mandrel profile directly instead of regenerating dome geometry
        # This preserves the correct dome shape that has evolved with layer buildup
        temp_vessel.profile_points = {
            'z_mm': np.array(current_surface_profile['z_mm']),
            'r_inner_mm': np.array(current_surface_profile['r_inner_mm']),
            'r_outer_mm': np.array(current_surface_profile['r_inner_mm']) + layer_def.calculated_set_thickness_mm,
            'dome_height_mm': current_surface_profile.get('dome_height_mm', 70.0)
        }
        
        # Force the physics engine to use the actual profile instead of generated dome geometry
        temp_vessel._use_profile_override = True
        print(f"[DEBUG] Set profile override flag: {hasattr(temp_vessel, '_use_profile_override')} = {getattr(temp_vessel, '_use_profile_override', 'NOT_SET')}")
        print(f"[DEBUG] temp_vessel profile_points keys: {temp_vessel.profile_points.keys()}")
        print(f"[DEBUG] temp_vessel r_inner_mm range: {temp_vessel.profile_points['r_inner_mm'].min():.1f} to {temp_vessel.profile_points['r_inner_mm'].max():.1f}mm")
        
        # Check temp_vessel bounds after profile override
        if hasattr(temp_vessel, 'get_profile_points'):
            post_profile = temp_vessel.get_profile_points()
            if 'z_mm' in post_profile:
                z_post = post_profile['z_mm']
                print(f"[DEBUG] temp_vessel Z range AFTER profile override: {z_post.min():.1f} to {z_post.max():.1f}mm")
        
        print(f"[DEBUG] current_surface_profile Z range: {current_surface_profile['z_mm'].min():.1f} to {current_surface_profile['z_mm'].max():.1f}mm")
        
        return temp_vessel
    
    def _calculate_layer_pattern(self, vessel_geometry, layer_def, roving_width_mm: float) -> Dict:
        """
        Calculate winding pattern parameters for the specific layer.
        
        Parameters:
        -----------
        mandrel_data : Dict
            Current mandrel geometry data
        layer_def : LayerDefinition
            Layer definition object
        roving_width_mm : float
            Roving width in millimeters
            
        Returns:
        --------
        Dict : Pattern parameters for the layer
        """
        try:
            # Use pattern calculator with current mandrel geometry
            pattern_results = self.pattern_calculator.calculate_pattern_parameters(
                current_mandrel_geometry=mandrel_data,
                roving_width_mm=roving_width_mm,
                target_angle_deg=layer_def.winding_angle_deg,
                num_layers=1
            )
            
            # Extract number of passes from pattern results with reasonable limits
            calculated_passes = pattern_results.nd_windings
            
            # Apply reasonable limits based on winding angle and coverage theory
            if layer_def.layer_type == 'hoop':
                # Hoop layers: 10-20 circuits typically sufficient
                num_passes = min(calculated_passes, 15)
            elif layer_def.winding_angle_deg >= 70:
                # Near-hoop patterns: 12-18 circuits
                num_passes = min(calculated_passes, 18)
            elif layer_def.winding_angle_deg >= 30:
                # Helical patterns (30-70°): 8-15 circuits based on Koussios theory
                num_passes = min(calculated_passes, 15)
            else:
                # Low angle patterns: fewer circuits but longer paths
                num_passes = min(calculated_passes, 12)
            
            return {
                'num_passes': num_passes,
                'delta_phi_pattern_rad': pattern_results.delta_phi_pattern_rad,
                'pattern_results': pattern_results
            }
            
        except Exception as e:
            st.warning(f"Pattern calculation failed for layer {layer_def.layer_set_id}: {str(e)}")
            # Fallback pattern parameters
            if layer_def.layer_type == 'hoop':
                return {'num_passes': 20, 'delta_phi_pattern_rad': 0.314}  # Multiple wraps
            else:
                return {'num_passes': 8, 'delta_phi_pattern_rad': 0.785}   # Helical pattern
    
    def get_trajectory_summary(self) -> List[Dict]:
        """Get summary of all generated trajectories"""
        summary = []
        for traj in self.generated_trajectories:
            summary.append({
                "Layer": f"Layer {traj['layer_id']}",
                "Type": traj['layer_type'],
                "Angle": f"{traj['winding_angle']}°",
                "Points": len(traj['trajectory_data'].get('path_points', [])),
                "Status": traj['trajectory_data'].get('status', 'Unknown')
            })
        return summary
    
    def visualize_layer_trajectory(self, layer_index: int, vessel_geometry, 
                                 decimation_factor: int = 10, surface_segments: int = 30) -> Optional:
        """
        Visualize specific layer trajectory with correct mandrel surface and performance optimization.
        
        Parameters:
        -----------
        layer_index : int
            Index of layer to visualize
        vessel_geometry : VesselGeometry
            Base vessel geometry for reference
        decimation_factor : int
            Plot every Nth point for performance (default: 10)
        surface_segments : int
            Number of segments for mandrel surface (default: 30)
            
        Returns:
        --------
        Plotly figure or None
        """
        if layer_index >= len(self.generated_trajectories):
            return None
        
        selected_traj = self.generated_trajectories[layer_index]
        
        try:
            # Create layer info for visualization
            layer_info = {
                'layer_type': selected_traj['layer_type'],
                'winding_angle': selected_traj['winding_angle'],
                'layer_id': selected_traj['layer_id']
            }
            
            # Use mandrel state at time of trajectory generation
            mandrel_state = selected_traj['mandrel_state']
            if mandrel_state:
                # Create vessel geometry representing the mandrel surface this layer was planned on
                viz_vessel = self._create_layer_vessel_geometry(mandrel_state, None)
                return create_3d_trajectory_visualization(
                    selected_traj['trajectory_data'],
                    viz_vessel,
                    layer_info,
                    decimation_factor=decimation_factor,
                    surface_segments=surface_segments
                )
            else:
                # Fallback to base vessel geometry
                return create_3d_trajectory_visualization(
                    selected_traj['trajectory_data'],
                    vessel_geometry,
                    layer_info,
                    decimation_factor=decimation_factor,
                    surface_segments=surface_segments
                )
                
        except Exception as e:
            st.error(f"Visualization error for layer {layer_index}: {str(e)}")
            return None
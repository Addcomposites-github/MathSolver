"""
Comprehensive Multi-Layer Definition & Geometry Module for COPV Design
Handles layer stacking, mandrel profile updates, and thickness calculations
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class LayerDefinition:
    """
    Definition of a single composite layer set in the laminate stack.
    Based on practical filament winding requirements and literature.
    """
    layer_set_id: int
    layer_type: str  # "helical", "hoop", "polar"
    fiber_material: str  # Link to material database
    resin_material: str  # Link to material database
    winding_angle_deg: float
    num_plies: int
    single_ply_thickness_mm: float
    coverage_percentage: float = 100.0  # % coverage for non-continuous patterns
    
    def __post_init__(self):
        """Calculate derived properties after initialization."""
        self.calculated_set_thickness_mm = self.num_plies * self.single_ply_thickness_mm
        self.composite_properties = None  # To be filled by MaterialDatabase
    
    def get_layer_set_thickness(self) -> float:
        """Get total thickness of this layer set."""
        return self.calculated_set_thickness_mm
    
    def is_structural_layer(self) -> bool:
        """Check if this is a structural (load-bearing) layer."""
        return self.layer_type in ['helical', 'hoop'] and self.num_plies > 0
    
    def get_effective_angle_rad(self) -> float:
        """Get winding angle in radians."""
        return math.radians(self.winding_angle_deg)


class MandrelGeometry:
    """
    Manages the evolving mandrel geometry as layers are added.
    Tracks the current winding surface for subsequent layer placement.
    """
    
    def __init__(self, initial_profile_points: Dict):
        """
        Initialize with the bare mandrel (liner + boss) profile.
        
        Parameters:
        -----------
        initial_profile_points : Dict
            Contains 'z_mm', 'r_inner_mm', 'r_outer_mm' arrays
        """
        self.initial_profile = initial_profile_points.copy()
        self.current_profile = initial_profile_points.copy()
        self.layers_applied = []
        self.total_buildup_mm = np.zeros_like(initial_profile_points['r_inner_mm'])
        
        print(f"Mandrel initialized with {len(self.current_profile['z_mm'])} profile points")
    
    def get_current_winding_surface(self) -> Dict:
        """
        Get the current outer surface available for winding the next layer.
        
        Returns:
        --------
        Dict with 'z_mm', 'r_current_mm' representing current mandrel surface
        """
        return {
            'z_mm': self.current_profile['z_mm'].copy(),
            'r_current_mm': self.current_profile['r_outer_mm'].copy()
        }
    
    def get_current_polar_opening_radius_mm(self) -> float:
        """Get current polar opening radius (minimum radius on current surface)."""
        return float(np.min(self.current_profile['r_outer_mm']))
    
    def get_current_equatorial_radius_mm(self) -> float:
        """Get current equatorial radius (maximum radius on current surface)."""
        return float(np.max(self.current_profile['r_outer_mm']))
    
    def apply_layer_buildup(self, layer: LayerDefinition) -> bool:
        """
        Apply thickness buildup from a new layer and update mandrel geometry.
        
        Uses Koussios thickness distribution theory for domes.
        """
        try:
            print(f"\nApplying layer {layer.layer_set_id}: {layer.layer_type} "
                  f"({layer.num_plies} plies × {layer.single_ply_thickness_mm}mm)")
            
            # Calculate thickness distribution for this layer
            thickness_distribution = self._calculate_layer_thickness_distribution(layer)
            
            # Update current profile with new thickness
            self.current_profile['r_outer_mm'] = (
                self.current_profile['r_outer_mm'] + thickness_distribution
            )
            
            # Track total buildup
            self.total_buildup_mm += thickness_distribution
            
            # Record layer application
            self.layers_applied.append({
                'layer': layer,
                'thickness_added_mm': thickness_distribution.copy(),
                'applied_at_step': len(self.layers_applied)
            })
            
            print(f"  Layer applied. New polar opening: {self.get_current_polar_opening_radius_mm():.2f}mm")
            print(f"  New equatorial radius: {self.get_current_equatorial_radius_mm():.2f}mm")
            
            return True
            
        except Exception as e:
            print(f"Error applying layer buildup: {e}")
            return False
    
    def reset_to_base_geometry(self):
        """Reset mandrel to initial geometry, removing all applied layers."""
        self.current_profile = self.initial_profile.copy()
        self.layers_applied = []
        self.total_buildup_mm = np.zeros_like(self.initial_profile['r_inner_mm'])
        print("Mandrel reset to base geometry")
    
    def _calculate_layer_thickness_distribution(self, layer: LayerDefinition) -> np.ndarray:
        """
        Calculate thickness distribution for a layer using Koussios theory.
        
        Based on Koussios Eq. 3.46 (classical smeared thickness):
        T_sm(Y) = T_eq * sqrt((Y_eq^2 - 1) / (Y^2 - 1))
        where Y = ρ/c (dimensionless radius)
        """
        try:
            z_mm = self.current_profile['z_mm']
            r_current_mm = self.current_profile['r_outer_mm']
            
            # Base thickness for this layer
            T_eq_layer = layer.get_layer_set_thickness()
            
            # Current mandrel parameters
            c_eff_mm = self.get_current_polar_opening_radius_mm()
            r_eq_mm = self.get_current_equatorial_radius_mm()
            
            # Avoid division issues
            if c_eff_mm < 1e-3:
                c_eff_mm = 1e-3
            
            # Dimensionless parameters
            Y_eq = r_eq_mm / c_eff_mm
            if Y_eq <= 1.0:
                Y_eq = 1.01  # Ensure Y_eq > 1 for physical validity
            
            # Calculate thickness at each profile point
            thickness_distribution = np.zeros_like(r_current_mm)
            
            for i, (z_val, r_val) in enumerate(zip(z_mm, r_current_mm)):
                thickness_distribution[i] = self._calculate_local_thickness(
                    r_val, c_eff_mm, Y_eq, T_eq_layer, layer
                )
            
            return thickness_distribution
            
        except Exception as e:
            print(f"Error in thickness distribution calculation: {e}")
            # Fallback to uniform thickness
            return np.full_like(self.current_profile['r_outer_mm'], 
                              layer.get_layer_set_thickness())
    
    def _calculate_local_thickness(self, r_local_mm: float, c_eff_mm: float, 
                                 Y_eq: float, T_eq_layer: float, 
                                 layer: LayerDefinition) -> float:
        """
        Calculate local thickness at a specific radius using Koussios theory.
        """
        try:
            # Handle different regions
            if r_local_mm < c_eff_mm * 1.05:  # Near polar region
                # Use simplified model near pole to avoid singularities
                return T_eq_layer * Y_eq * 0.8  # Reduced thickness near pole
            
            # Dimensionless radius
            Y = r_local_mm / c_eff_mm
            
            if Y <= 1.001:  # Very close to polar opening
                return T_eq_layer * Y_eq  # Maximum buildup
            
            # Koussios classical smeared thickness (Eq. 3.46)
            term_in_sqrt = (Y_eq**2 - 1) / (Y**2 - 1)
            
            if term_in_sqrt <= 0:
                return T_eq_layer  # Fallback
            
            thickness = T_eq_layer * math.sqrt(term_in_sqrt)
            
            # Apply layer-specific adjustments
            if layer.layer_type == 'hoop':
                # Hoop layers have more uniform thickness
                thickness *= 0.9
            elif layer.layer_type == 'polar':
                # Polar layers concentrate near poles
                thickness *= 1.2 if Y < 2.0 else 0.8
            
            # Reasonable bounds
            return max(0.1 * T_eq_layer, min(thickness, T_eq_layer * Y_eq))
            
        except Exception:
            return T_eq_layer  # Safe fallback


class LayerStackManager:
    """
    Manages the complete composite layer stack for COPV design.
    Coordinates layer definitions, mandrel updates, and trajectory planning.
    """
    
    def __init__(self, initial_mandrel_profile: Dict):
        """Initialize with bare mandrel geometry."""
        self.mandrel = MandrelGeometry(initial_mandrel_profile)
        self.layer_stack: List[LayerDefinition] = []
        self.winding_sequence: List[Dict] = []
        
    def add_layer(self, layer_type: str, fiber_material: str, resin_material: str,
                  winding_angle_deg: float, num_plies: int, 
                  single_ply_thickness_mm: float, coverage_percentage: float = 100.0) -> LayerDefinition:
        """
        Add a new layer to the stack.
        
        Returns the created LayerDefinition for further use.
        """
        layer_id = len(self.layer_stack) + 1
        
        new_layer = LayerDefinition(
            layer_set_id=layer_id,
            layer_type=layer_type,
            fiber_material=fiber_material,
            resin_material=resin_material,
            winding_angle_deg=winding_angle_deg,
            num_plies=num_plies,
            single_ply_thickness_mm=single_ply_thickness_mm,
            coverage_percentage=coverage_percentage
        )
        
        self.layer_stack.append(new_layer)
        
        print(f"Added layer {layer_id}: {layer_type} at {winding_angle_deg}° "
              f"({num_plies} plies, {single_ply_thickness_mm}mm each)")
        
        return new_layer
    
    def apply_layer_to_mandrel(self, layer_index: int) -> bool:
        """
        Apply a specific layer to the mandrel geometry.
        This simulates the physical winding process.
        """
        if not (0 <= layer_index < len(self.layer_stack)):
            print(f"Invalid layer index: {layer_index}")
            return False
        
        layer = self.layer_stack[layer_index]
        success = self.mandrel.apply_layer_buildup(layer)
        
        if success:
            self.winding_sequence.append({
                'step': len(self.winding_sequence) + 1,
                'layer_id': layer.layer_set_id,
                'layer_type': layer.layer_type,
                'mandrel_state': self.mandrel.get_current_winding_surface()
            })
        
        return success
    
    def get_current_mandrel_for_trajectory(self) -> Dict:
        """
        Get current mandrel geometry formatted for trajectory planning.
        
        Returns geometry compatible with TrajectoryPlanner requirements.
        """
        surface = self.mandrel.get_current_winding_surface()
        
        return {
            'profile_points': {
                'z_mm': surface['z_mm'],
                'r_inner_mm': surface['r_current_mm'],  # Current surface is "inner" for next layer
                'r_outer_mm': surface['r_current_mm']   # Will be updated after next layer
            },
            'polar_opening_radius_mm': self.mandrel.get_current_polar_opening_radius_mm(),
            'equatorial_radius_mm': self.mandrel.get_current_equatorial_radius_mm()
        }
    
    @property
    def total_thickness_mm(self) -> float:
        """Get total thickness of all layers in the stack."""
        return sum(layer.get_layer_set_thickness() for layer in self.layer_stack)
    
    def get_layer_stack_summary(self) -> Dict:
        """Get comprehensive summary of the current layer stack."""
        total_thickness = sum(layer.get_layer_set_thickness() for layer in self.layer_stack)
        structural_layers = [layer for layer in self.layer_stack if layer.is_structural_layer()]
        
        return {
            'total_layers': len(self.layer_stack),
            'structural_layers': len(structural_layers),
            'total_thickness_mm': total_thickness,
            'layers_applied_to_mandrel': len(self.mandrel.layers_applied),
            'current_polar_radius_mm': self.mandrel.get_current_polar_opening_radius_mm(),
            'current_equatorial_radius_mm': self.mandrel.get_current_equatorial_radius_mm(),
            'layer_details': [
                {
                    'id': layer.layer_set_id,
                    'type': layer.layer_type,
                    'angle': layer.winding_angle_deg,
                    'plies': layer.num_plies,
                    'thickness_mm': layer.get_layer_set_thickness()
                }
                for layer in self.layer_stack
            ]
        }
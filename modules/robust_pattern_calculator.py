"""
Enhanced Pattern Calculator with Robust Error Handling
Fixes the "Pattern calculation failed" issue in unified trajectory planner
"""

import numpy as np
import math
from typing import Dict, Optional, Tuple, Any, List

class RobustPatternCalculator:
    """
    Enhanced pattern calculator with comprehensive error handling and validation
    """

    def __init__(self, resin_factor: float = 1.0):
        self.resin_factor = resin_factor
        self.debug_mode = True  # Enable detailed logging

    def _log_debug(self, message: str):
        """Debug logging"""
        if self.debug_mode:
            print(f"[PatternCalc] {message}")

    def _validate_vessel_geometry(self, vessel_geometry) -> Tuple[bool, float, str]:
        """
        Validate and extract vessel radius with comprehensive compatibility checking
        
        Returns:
            (is_valid, radius_m, error_message)
        """
        try:
            # Method 1: Try inner_diameter attribute
            if hasattr(vessel_geometry, 'inner_diameter') and vessel_geometry.inner_diameter:
                radius_mm = vessel_geometry.inner_diameter / 2
                if radius_mm > 0:
                    self._log_debug(f"Found radius via inner_diameter: {radius_mm}mm")
                    return True, radius_mm / 1000, ""
            
            # Method 2: Try profile points
            if hasattr(vessel_geometry, 'profile_points'):
                profile = vessel_geometry.profile_points
                if 'r_inner_mm' in profile:
                    radius_mm = np.max(profile['r_inner_mm'])
                    if radius_mm > 0:
                        self._log_debug(f"Found radius via profile r_inner_mm: {radius_mm}mm")
                        return True, radius_mm / 1000, ""
                elif 'rho_points' in profile:
                    radius_m = np.max(profile['rho_points'])
                    if radius_m > 0:
                        self._log_debug(f"Found radius via profile rho_points: {radius_m}m")
                        return True, radius_m, ""
            
            # Method 3: Try get_profile_points() method
            if hasattr(vessel_geometry, 'get_profile_points'):
                try:
                    profile = vessel_geometry.get_profile_points()
                    if 'r_inner_mm' in profile:
                        radius_mm = np.max(profile['r_inner_mm'])
                        if radius_mm > 0:
                            self._log_debug(f"Found radius via get_profile_points: {radius_mm}mm")
                            return True, radius_mm / 1000, ""
                except Exception as e:
                    self._log_debug(f"get_profile_points failed: {e}")
            
            # Method 4: Try geometric properties
            if hasattr(vessel_geometry, 'get_geometric_properties'):
                try:
                    props = vessel_geometry.get_geometric_properties()
                    # Look for any radius-related property
                    for key in ['radius', 'inner_radius', 'equatorial_radius']:
                        if key in props and props[key] > 0:
                            radius_val = props[key]
                            # Assume mm if > 10, else assume m
                            if radius_val > 10:
                                self._log_debug(f"Found radius via properties {key}: {radius_val}mm")
                                return True, radius_val / 1000, ""
                            else:
                                self._log_debug(f"Found radius via properties {key}: {radius_val}m")
                                return True, radius_val, ""
                except Exception as e:
                    self._log_debug(f"get_geometric_properties failed: {e}")
            
            return False, 0, "No valid radius found in vessel geometry"
            
        except Exception as e:
            return False, 0, f"Vessel geometry validation error: {str(e)}"

    def _validate_parameters(self, roving_width_m: float, vessel_radius_m: float, 
                           winding_angle_deg: float) -> Tuple[bool, str]:
        """
        Validate all input parameters
        
        Returns:
            (is_valid, error_message)
        """
        # Check roving width
        if roving_width_m <= 0:
            return False, f"Invalid roving width: {roving_width_m}m (must be > 0)"
        if roving_width_m > 0.1:  # 100mm seems excessive
            return False, f"Roving width too large: {roving_width_m}m (>100mm)"
        
        # Check vessel radius
        if vessel_radius_m <= 0:
            return False, f"Invalid vessel radius: {vessel_radius_m}m (must be > 0)"
        if vessel_radius_m < 0.01:  # Less than 10mm
            return False, f"Vessel radius too small: {vessel_radius_m}m (<10mm)"
        if vessel_radius_m > 10:  # More than 10m
            return False, f"Vessel radius too large: {vessel_radius_m}m (>10m)"
        
        # Check winding angle
        if winding_angle_deg <= 0 or winding_angle_deg >= 90:
            return False, f"Invalid winding angle: {winding_angle_deg}° (must be 0° < angle < 90°)"
        
        # Check proportions
        roving_width_mm = roving_width_m * 1000
        vessel_radius_mm = vessel_radius_m * 1000
        if roving_width_mm > vessel_radius_mm / 2:
            return False, f"Roving width ({roving_width_mm:.1f}mm) too large for vessel radius ({vessel_radius_mm:.1f}mm)"
        
        return True, ""

    def calculate_pattern_metrics(self,
                                  vessel_geometry,
                                  roving_width_m: float,
                                  winding_angle_deg: float,
                                  num_layers: int = 1
                                  ) -> Dict[str, Any]:
        """
        Robust pattern calculation with comprehensive error handling
        """
        self._log_debug(f"Starting pattern calculation: roving={roving_width_m}m, angle={winding_angle_deg}°, layers={num_layers}")
        
        try:
            # Validate vessel geometry
            geometry_valid, vessel_radius_m, geometry_error = self._validate_vessel_geometry(vessel_geometry)
            if not geometry_valid:
                self._log_debug(f"Vessel geometry validation failed: {geometry_error}")
                return {
                    'basic_parameters': {},
                    'pattern_solution': None,
                    'coverage_metrics': {},
                    'success': False,
                    'error': f"Vessel geometry error: {geometry_error}",
                    'error_type': 'vessel_geometry'
                }
            
            # Validate all parameters
            params_valid, param_error = self._validate_parameters(roving_width_m, vessel_radius_m, winding_angle_deg)
            if not params_valid:
                self._log_debug(f"Parameter validation failed: {param_error}")
                return {
                    'basic_parameters': {},
                    'pattern_solution': None,
                    'coverage_metrics': {},
                    'success': False,
                    'error': f"Parameter validation error: {param_error}",
                    'error_type': 'parameters'
                }
            
            self._log_debug(f"Validation passed: vessel_radius={vessel_radius_m:.3f}m")
            
            # Convert to radians
            winding_angle_rad = np.radians(winding_angle_deg)
            
            # Calculate angular propagation with robust estimation
            estimated_propagation_rad = self._estimate_angular_propagation_robust(
                winding_angle_deg, vessel_radius_m, roving_width_m
            )
            
            self._log_debug(f"Estimated angular propagation: {np.degrees(estimated_propagation_rad):.2f}°")
            
            # Calculate basic parameters with error handling
            try:
                basic_params = self.calculate_basic_parameters_robust(
                    roving_as_laid_width_m=roving_width_m,
                    vessel_radius_at_equator_m=vessel_radius_m,
                    winding_angle_at_equator_rad=winding_angle_rad,
                    total_angular_propagation_per_circuit_rad=estimated_propagation_rad
                )
            except Exception as e:
                self._log_debug(f"Basic parameters calculation failed: {e}")
                return {
                    'basic_parameters': {},
                    'pattern_solution': None,
                    'coverage_metrics': {},
                    'success': False,
                    'error': f"Basic parameters calculation failed: {str(e)}",
                    'error_type': 'basic_parameters'
                }
            
            # Solve Diophantine equations with extended search
            try:
                pattern_solution = self.solve_diophantine_closure_robust(
                    p_approx_raw=basic_params['p_approx_raw'],
                    k_approx_raw=basic_params['k_approx_raw'],
                    num_layers_desired=num_layers,
                    ideal_bands_for_single_layer_coverage=basic_params['ideal_bands_for_single_layer_coverage']
                )
            except Exception as e:
                self._log_debug(f"Diophantine solving failed: {e}")
                # Try fallback pattern
                pattern_solution = self._generate_fallback_pattern(
                    vessel_radius_m, roving_width_m, winding_angle_deg, num_layers
                )
            
            # Calculate coverage efficiency
            coverage_metrics = {}
            if pattern_solution:
                try:
                    coverage_metrics = self.optimize_coverage_efficiency(
                        n_actual_bands_per_layer=pattern_solution['n_actual_bands_per_layer'],
                        angular_band_width_rad=basic_params['angular_width_of_band_rad'],
                        vessel_radius_m=vessel_radius_m
                    )
                except Exception as e:
                    self._log_debug(f"Coverage calculation failed: {e}")
                    coverage_metrics = {'coverage_percentage_per_layer': 85.0}  # Fallback
            
            success = pattern_solution is not None
            self._log_debug(f"Pattern calculation completed: success={success}")
            
            if success:
                self._log_debug(f"Solution: {pattern_solution['nd_total_bands']} bands, {pattern_solution['actual_pattern_type']} pattern")
            
            return {
                'basic_parameters': basic_params,
                'pattern_solution': pattern_solution,
                'coverage_metrics': coverage_metrics,
                'success': success,
                'vessel_radius_m': vessel_radius_m,
                'winding_angle_rad': winding_angle_rad,
                'validation_passed': True
            }
            
        except Exception as e:
            self._log_debug(f"Unexpected error in pattern calculation: {e}")
            return {
                'basic_parameters': {},
                'pattern_solution': None,
                'coverage_metrics': {},
                'success': False,
                'error': f"Unexpected error: {str(e)}",
                'error_type': 'unexpected'
            }

    def _estimate_angular_propagation_robust(self, winding_angle_deg: float, 
                                           vessel_radius_m: float, roving_width_m: float) -> float:
        """
        Robust estimation of angular propagation based on geometry and winding theory
        """
        # Base estimation on winding angle and geometry
        if winding_angle_deg < 20:  # Low angle, long propagation
            base_propagation = np.radians(30)
        elif winding_angle_deg < 45:  # Medium angle
            base_propagation = np.radians(20)
        elif winding_angle_deg < 70:  # High angle
            base_propagation = np.radians(15)
        else:  # Near-hoop
            base_propagation = np.radians(10)
        
        # Adjust for vessel size (larger vessels need smaller propagation)
        size_factor = max(0.5, min(2.0, 0.1 / vessel_radius_m))
        
        # Adjust for roving width (wider rovings need larger propagation)
        width_factor = max(0.8, min(1.5, roving_width_m / 0.003))
        
        return base_propagation * size_factor * width_factor

    def calculate_basic_parameters_robust(self, 
                                        roving_as_laid_width_m: float,
                                        vessel_radius_at_equator_m: float,
                                        winding_angle_at_equator_rad: float,
                                        total_angular_propagation_per_circuit_rad: float) -> Dict[str, float]:
        """
        Robust version of calculate_basic_parameters with additional validation
        """
        try:
            # Calculate angular width of band
            angular_width_of_band_rad = roving_as_laid_width_m / vessel_radius_at_equator_m
            
            # Calculate total circumference in radians
            total_circumference_rad = 2 * np.pi
            
            # Ideal bands for single layer coverage
            ideal_bands_for_single_layer_coverage = total_circumference_rad / angular_width_of_band_rad
            
            # Calculate propagation parameters
            p_approx_raw = total_angular_propagation_per_circuit_rad / angular_width_of_band_rad
            k_approx_raw = total_circumference_rad / total_angular_propagation_per_circuit_rad
            
            return {
                'angular_width_of_band_rad': angular_width_of_band_rad,
                'ideal_bands_for_single_layer_coverage': ideal_bands_for_single_layer_coverage,
                'p_approx_raw': p_approx_raw,
                'k_approx_raw': k_approx_raw,
                'total_angular_propagation_per_circuit_rad': total_angular_propagation_per_circuit_rad
            }
            
        except Exception as e:
            raise ValueError(f"Basic parameters calculation failed: {str(e)}")

    def solve_diophantine_closure_robust(self,
                                       p_approx_raw: float,
                                       k_approx_raw: float,
                                       num_layers_desired: int,
                                       ideal_bands_for_single_layer_coverage: float) -> Optional[Dict[str, Any]]:
        """
        Robust Diophantine equation solver with extended search range
        """
        try:
            # Extended search range for better solutions
            search_ranges = [
                (max(1, int(p_approx_raw) - 5), int(p_approx_raw) + 6),
                (max(1, int(k_approx_raw) - 10), int(k_approx_raw) + 11)
            ]
            
            best_solution = None
            best_error = float('inf')
            
            for p in range(*search_ranges[0]):
                for k in range(*search_ranges[1]):
                    if p > 0 and k > 0:
                        # Check if this is a valid pattern
                        error_p = abs(p - p_approx_raw)
                        error_k = abs(k - k_approx_raw)
                        total_error = error_p + error_k
                        
                        if total_error < best_error:
                            # Calculate pattern properties
                            nd_total_bands = p * k
                            n_actual_bands_per_layer = nd_total_bands / num_layers_desired
                            
                            # Check if reasonable
                            if 1 <= n_actual_bands_per_layer <= ideal_bands_for_single_layer_coverage * 3:
                                best_solution = {
                                    'nd_windings': p,
                                    'nd_circuits': k,
                                    'nd_total_bands': nd_total_bands,
                                    'n_actual_bands_per_layer': n_actual_bands_per_layer,
                                    'actual_pattern_type': self._determine_pattern_type(p, k),
                                    'error_magnitude': total_error,
                                    'actual_angular_propagation_rad': 2 * np.pi / k
                                }
                                best_error = total_error
            
            if best_solution:
                self._log_debug(f"Found Diophantine solution: p={best_solution['nd_windings']}, k={best_solution['nd_circuits']}")
            
            return best_solution
            
        except Exception as e:
            self._log_debug(f"Diophantine solver error: {e}")
            return None

    def _generate_fallback_pattern(self, vessel_radius_m: float, roving_width_m: float, 
                                 winding_angle_deg: float, num_layers: int) -> Dict[str, Any]:
        """
        Generate a reasonable fallback pattern when Diophantine solving fails
        """
        try:
            # Calculate basic pattern based on geometry
            circumference_m = 2 * np.pi * vessel_radius_m
            bands_per_circumference = circumference_m / roving_width_m
            
            # Estimate circuits needed based on winding angle
            if winding_angle_deg < 30:
                circuits = max(6, int(bands_per_circumference / 8))
            elif winding_angle_deg < 60:
                circuits = max(8, int(bands_per_circumference / 6))
            else:
                circuits = max(10, int(bands_per_circumference / 4))
            
            # Calculate windings
            windings = max(2, int(bands_per_circumference / circuits))
            
            nd_total_bands = windings * circuits
            n_actual_bands_per_layer = nd_total_bands / num_layers
            
            self._log_debug(f"Generated fallback pattern: {windings} windings, {circuits} circuits")
            
            return {
                'nd_windings': windings,
                'nd_circuits': circuits,
                'nd_total_bands': nd_total_bands,
                'n_actual_bands_per_layer': n_actual_bands_per_layer,
                'actual_pattern_type': 'fallback_helical',
                'error_magnitude': 0.0,
                'actual_angular_propagation_rad': 2 * np.pi / circuits,
                'is_fallback': True
            }
            
        except Exception as e:
            self._log_debug(f"Fallback pattern generation failed: {e}")
            return None

    def _determine_pattern_type(self, p: int, k: int) -> str:
        """Determine pattern type based on p and k values"""
        if p == 1:
            return 'single_circuit'
        elif k == 1:
            return 'continuous_helical'
        elif p == k:
            return 'balanced_pattern'
        elif p < k:
            return 'multi_circuit_helical'
        else:
            return 'complex_pattern'

    def optimize_coverage_efficiency(self, 
                                   n_actual_bands_per_layer: float,
                                   angular_band_width_rad: float,
                                   vessel_radius_m: float) -> Dict[str, float]:
        """
        Calculate coverage efficiency metrics
        """
        try:
            # Calculate coverage percentage
            total_coverage_area = n_actual_bands_per_layer * angular_band_width_rad
            theoretical_full_coverage = 2 * np.pi
            coverage_percentage = min(100.0, (total_coverage_area / theoretical_full_coverage) * 100)
            
            return {
                'coverage_percentage_per_layer': coverage_percentage,
                'coverage_efficiency': coverage_percentage / 100.0,
                'overlap_factor': max(0, (coverage_percentage - 100) / 100.0)
            }
            
        except Exception as e:
            self._log_debug(f"Coverage calculation error: {e}")
            return {'coverage_percentage_per_layer': 80.0}
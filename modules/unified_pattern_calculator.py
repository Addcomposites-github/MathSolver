"""
Unified Trajectory Planner - Pattern Calculator
Step 3: Koussios pattern theory for optimal coverage and Diophantine equation solving
"""

import numpy as np
import math
from typing import Dict, Optional, Tuple, Any

class PatternCalculator:
    """
    Calculates filament winding pattern parameters based on Koussios's pattern theory
    for optimal coverage on axisymmetric mandrels.
    """

    def __init__(self, resin_factor: float = 1.0):
        """
        Initializes the PatternCalculator.

        Args:
            resin_factor (w): Factor to account for resin inclusion in wet winding,
                              affecting the effective cross-sectional area or thickness
                              of the laid band. Typically w > 1. For dry winding, w = 1.
                              Koussios uses 'w' such that effective thickness might be
                              T_roving * w or area BT * w (Eq 3.33 Koussios [cite: 959]).
        """
        self.resin_factor = resin_factor

    def calculate_basic_parameters(self,
                                   roving_as_laid_width_m: float,
                                   vessel_radius_at_equator_m: float,
                                   winding_angle_at_equator_rad: float,
                                   total_angular_propagation_per_circuit_rad: float
                                   ) -> Dict[str, float]:
        """
        Calculates fundamental parameters needed for pattern determination.

        Args:
            roving_as_laid_width_m (b): The width of the fiber band as it is laid on the mandrel, in meters.
            vessel_radius_at_equator_m (R_eq): The radius of the vessel at the equator (or reference diameter), in meters.
            winding_angle_at_equator_rad (alpha_eq): The winding angle (with the meridian/axis)
                                                      at the equator, in radians.
            total_angular_propagation_per_circuit_rad (Delta_Phi_tot): The net angular advancement of the pattern
                                                                      around the mandrel after one complete circuit
                                                                      (e.g., pole-to-pole-to-pole), in radians.

        Returns:
            Dict[str, float]: A dictionary containing fundamental pattern parameters
        """
        if not (0 < winding_angle_at_equator_rad < np.pi / 2):
            # Handle special cases
            if abs(winding_angle_at_equator_rad - np.pi/2) < 1e-6:  # Hoop
                effective_band_width_m = roving_as_laid_width_m
            elif abs(winding_angle_at_equator_rad) < 1e-6:  # Axial
                effective_band_width_m = np.inf
            else:  # Helical
                effective_band_width_m = roving_as_laid_width_m / np.cos(winding_angle_at_equator_rad)
        else:  # Helical as default
            # Koussios Eq 3.32 [cite: 953]
            effective_band_width_m = roving_as_laid_width_m / np.cos(winding_angle_at_equator_rad)

        if vessel_radius_at_equator_m < 1e-6:
            raise ValueError("Vessel radius must be positive.")

        # Angular width of a single band on the equator (delta_phi)
        # Koussios: delta_phi = B_eff / R_eq [cite: 977]
        angular_width_of_band_rad = effective_band_width_m / vessel_radius_at_equator_m

        if angular_width_of_band_rad == 0:
            raise ValueError("Angular width of band cannot be zero. Check roving width and vessel radius.")

        # Ideal number of bands for single layer (n)
        # Koussios: n = 2*pi / delta_phi [cite: 983]
        ideal_bands_for_single_layer_coverage = 2 * np.pi / angular_width_of_band_rad

        # Approximate p and k from Koussios Eq 3.40 [cite: 982]
        if abs(total_angular_propagation_per_circuit_rad) < 1e-9:
            p_approx_raw = np.inf
            k_approx_raw = 0
        else:
            p_approx_raw = (2 * np.pi) / total_angular_propagation_per_circuit_rad
            k_approx_raw = total_angular_propagation_per_circuit_rad / angular_width_of_band_rad
            
        return {
            'effective_band_width_at_equator_m': effective_band_width_m,
            'angular_width_of_band_rad': angular_width_of_band_rad,
            'ideal_bands_for_single_layer_coverage': ideal_bands_for_single_layer_coverage,
            'p_approx_raw': p_approx_raw,
            'k_approx_raw': k_approx_raw
        }

    def solve_diophantine_closure(self,
                                  p_approx_raw: float,
                                  k_approx_raw: float,
                                  num_layers_desired: int,
                                  ideal_bands_for_single_layer_coverage: float,
                                  pattern_type: str = 'auto'
                                 ) -> Optional[Dict[str, Any]]:
        """
        Solves the Diophantine equations for pattern closure to find integer pattern
        parameters (p, k, nd) that provide full coverage for the desired number of layers.

        Args:
            p_approx_raw: Approximate p value (e.g., 2*pi / Delta_Phi_tot).
            k_approx_raw: Approximate k value (e.g., Delta_Phi_tot / delta_phi).
            num_layers_desired (d): The desired number of complete layers.
            ideal_bands_for_single_layer_coverage: Ideal number of bands for one layer.
            pattern_type: 'leading', 'lagging', or 'auto' to try both.

        Returns:
            Optional[Dict[str, Any]]: Dictionary with pattern solution or None if not found
        """
        if num_layers_desired <= 0:
            raise ValueError("Number of layers desired must be positive.")

        # Search range around approximate values
        p_search_range = [max(1, math.floor(p_approx_raw) - 2), math.ceil(p_approx_raw) + 2]
        k_search_range = [max(1, math.floor(k_approx_raw) - 2), math.ceil(k_approx_raw) + 2]

        best_solution = None
        min_coverage_error = float('inf')

        for p_actual in range(p_search_range[0], p_search_range[1] + 1):
            if p_actual == 0:
                continue
            for k_actual in range(k_search_range[0], k_search_range[1] + 1):
                if k_actual == 0:
                    continue

                # Koussios Diophantine equations
                nd_leading = (p_actual + 1) * k_actual * num_layers_desired - 1  # Eq 3.34 [cite: 974]
                nd_lagging = p_actual * k_actual * num_layers_desired + 1        # Eq 3.35 [cite: 976]

                solutions_to_check = []
                if pattern_type == 'leading' or pattern_type == 'auto':
                    solutions_to_check.append({'nd': nd_leading, 'type': 'leading', 'p_eff': p_actual + 1})
                if pattern_type == 'lagging' or pattern_type == 'auto':
                    solutions_to_check.append({'nd': nd_lagging, 'type': 'lagging', 'p_eff': p_actual})
                
                for sol_candidate in solutions_to_check:
                    nd_total = sol_candidate['nd']
                    p_effective = sol_candidate['p_eff']
                    
                    if nd_total <= 0:
                        continue

                    # Actual number of bands per layer
                    n_actual_one_layer = nd_total / num_layers_desired
                    
                    # Coverage error: deviation from ideal
                    coverage_error = abs(n_actual_one_layer - ideal_bands_for_single_layer_coverage)

                    # Calculate actual angular advancement for this integer solution
                    # From Koussios Eq 3.43 [cite: 984]
                    if sol_candidate['type'] == 'leading':
                        actual_advancement_rad = (2 * np.pi / p_effective) * (1 + 1 / n_actual_one_layer)
                    else:  # lagging
                        actual_advancement_rad = (2 * np.pi / p_effective) * (1 - 1 / n_actual_one_layer)
                    
                    # GCD check to avoid pattern repetition within layers
                    common_divisor = math.gcd(p_effective * k_actual, round(n_actual_one_layer))
                    
                    if common_divisor == 1:  # Good, non-repeating basic pattern
                        if coverage_error < min_coverage_error:
                            min_coverage_error = coverage_error
                            best_solution = {
                                'p_actual': p_actual,
                                'k_actual': k_actual,
                                'nd_total_bands': nd_total,
                                'num_actual_layers': num_layers_desired,
                                'n_actual_bands_per_layer': n_actual_one_layer,
                                'actual_pattern_type': sol_candidate['type'],
                                'actual_angular_propagation_rad': actual_advancement_rad,
                                'coverage_error_metric': coverage_error,
                                'comment': 'Solution found.'
                            }

        return best_solution

    def optimize_coverage_efficiency(self,
                                     n_actual_bands_per_layer: float,
                                     angular_band_width_rad: float,
                                     vessel_radius_m: float
                                    ) -> Dict[str, float]:
        """
        Calculates the coverage efficiency, overlap, or gap for a given pattern.

        Args:
            n_actual_bands_per_layer: Actual number of unique band positions per layer from Diophantine solution.
            angular_band_width_rad (delta_phi): Angular width of one fiber band.
            vessel_radius_m: Radius of the vessel where coverage is being assessed.

        Returns:
            Dict[str, float]: Dictionary with overlap, gap, and coverage metrics
        """
        circumference_m = 2 * np.pi * vessel_radius_m
        covered_circumference_m = n_actual_bands_per_layer * angular_band_width_rad * vessel_radius_m
        
        # Total width of all bands if laid side-by-side ideally for one layer
        total_band_width_m = n_actual_bands_per_layer * (angular_band_width_rad * vessel_radius_m)

        # The difference per band position
        discrepancy_m = total_band_width_m - circumference_m
        
        overlap_m = 0.0
        gap_m = 0.0

        if discrepancy_m > 1e-9:  # Overall overlap
            # Koussios Eq 8.20: overlap calculation
            overlap_per_band_m = (angular_band_width_rad * vessel_radius_m) - (circumference_m / n_actual_bands_per_layer)
            if overlap_per_band_m > 0:
                overlap_m = overlap_per_band_m
            else:
                gap_m = -overlap_per_band_m

        elif discrepancy_m < -1e-9:  # Overall gap
            gap_m = -discrepancy_m / n_actual_bands_per_layer

        # Coverage percentage
        coverage_percentage = (total_band_width_m / circumference_m) * 100 if circumference_m > 0 else 0
        
        return {
            'overlap_mm': overlap_m * 1000,  # convert to mm
            'gap_mm': gap_m * 1000,          # convert to mm
            'coverage_percentage_per_layer': coverage_percentage
        }

    def calculate_pattern_metrics(self,
                                  vessel_geometry,
                                  roving_width_m: float,
                                  winding_angle_deg: float,
                                  num_layers: int = 1
                                  ) -> Dict[str, Any]:
        """
        Complete pattern calculation workflow combining all methods.

        Args:
            vessel_geometry: Vessel geometry object with radius information
            roving_width_m: Width of the roving in meters
            winding_angle_deg: Winding angle in degrees
            num_layers: Desired number of layers

        Returns:
            Dict[str, Any]: Complete pattern analysis results
        """
        try:
            # Get vessel radius at equator (cylinder section)
            vessel_radius_m = vessel_geometry.inner_diameter / 2000  # Convert mm to m
            winding_angle_rad = np.radians(winding_angle_deg)
            
            # Estimate angular propagation (simplified for now)
            # In practice, this would come from PhysicsEngine trajectory simulation
            estimated_propagation_rad = 2 * np.pi / 10  # Rough estimate
            
            # Calculate basic parameters
            basic_params = self.calculate_basic_parameters(
                roving_as_laid_width_m=roving_width_m,
                vessel_radius_at_equator_m=vessel_radius_m,
                winding_angle_at_equator_rad=winding_angle_rad,
                total_angular_propagation_per_circuit_rad=estimated_propagation_rad
            )
            
            # Solve Diophantine equations
            pattern_solution = self.solve_diophantine_closure(
                p_approx_raw=basic_params['p_approx_raw'],
                k_approx_raw=basic_params['k_approx_raw'],
                num_layers_desired=num_layers,
                ideal_bands_for_single_layer_coverage=basic_params['ideal_bands_for_single_layer_coverage']
            )
            
            # Calculate coverage efficiency
            coverage_metrics = {}
            if pattern_solution:
                coverage_metrics = self.optimize_coverage_efficiency(
                    n_actual_bands_per_layer=pattern_solution['n_actual_bands_per_layer'],
                    angular_band_width_rad=basic_params['angular_width_of_band_rad'],
                    vessel_radius_m=vessel_radius_m
                )
            
            return {
                'basic_parameters': basic_params,
                'pattern_solution': pattern_solution,
                'coverage_metrics': coverage_metrics,
                'success': pattern_solution is not None,
                'vessel_radius_m': vessel_radius_m,
                'winding_angle_rad': winding_angle_rad
            }
            
        except Exception as e:
            return {
                'basic_parameters': {},
                'pattern_solution': None,
                'coverage_metrics': {},
                'success': False,
                'error': str(e)
            }

    def get_pattern_recommendations(self, pattern_metrics: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on pattern analysis results.

        Args:
            pattern_metrics: Results from calculate_pattern_metrics

        Returns:
            List[str]: List of recommendation strings
        """
        recommendations = []
        
        if not pattern_metrics['success']:
            recommendations.append("❌ Pattern calculation failed - check input parameters")
            return recommendations
        
        pattern_solution = pattern_metrics['pattern_solution']
        coverage_metrics = pattern_metrics['coverage_metrics']
        
        if pattern_solution:
            recommendations.append(f"✅ Found {pattern_solution['actual_pattern_type']} pattern solution")
            recommendations.append(f"📊 Pattern parameters: p={pattern_solution['p_actual']}, k={pattern_solution['k_actual']}")
            
            # Coverage analysis
            if coverage_metrics:
                overlap_mm = coverage_metrics['gap_mm']
                gap_mm = coverage_metrics['gap_mm']
                coverage_pct = coverage_metrics['coverage_percentage_per_layer']
                
                if overlap_mm > 0.1:
                    recommendations.append(f"⚠️ Significant overlap: {overlap_mm:.1f}mm - consider wider band spacing")
                elif gap_mm > 0.1:
                    recommendations.append(f"⚠️ Coverage gaps: {gap_mm:.1f}mm - consider narrower band spacing")
                else:
                    recommendations.append(f"✅ Good coverage: {coverage_pct:.1f}% with minimal gaps/overlap")
        
        return recommendations
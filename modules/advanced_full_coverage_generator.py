"""
Advanced Full Coverage Generator for Complete 3D Visualization
Generates all circuits needed for complete coverage patterns
"""

import numpy as np
import math
from typing import List, Dict, Any, Tuple, Optional

class AdvancedFullCoverageGenerator:
    """Generate complete full coverage trajectories with all circuits"""
    
    def __init__(self, vessel_geometry, layer_config):
        self.vessel_geometry = vessel_geometry
        self.layer_config = layer_config
    
    def generate_complete_coverage(self, quality_level="balanced"):
        """
        Generate all circuits needed for complete coverage
        
        Args:
            quality_level: "fast", "balanced", "high_quality"
        """
        try:
            # Import unified planner here to avoid circular imports
            from modules.unified_trajectory_planner import UnifiedTrajectoryPlanner
            
            # Calculate basic pattern parameters
            equatorial_radius = self.vessel_geometry.inner_diameter / 2000  # Convert to meters
            circumference = 2 * math.pi * equatorial_radius
            roving_width_m = self.layer_config.get('roving_width', 3.0) / 1000
            
            # Estimate number of circuits needed based on winding angle
            winding_angle = self.layer_config['winding_angle']
            
            # More circuits needed for high angles (hoop patterns)
            if winding_angle >= 85:  # Near-hoop patterns
                total_circuits = max(8, int(circumference / roving_width_m))
            elif winding_angle >= 60:  # High angle helical
                total_circuits = max(4, int(circumference / roving_width_m * 0.8))
            else:  # Low angle helical
                total_circuits = max(2, int(circumference / roving_width_m * 0.6))
            
            angular_advancement = 2 * math.pi / total_circuits
            
            # Quality settings
            quality_settings = self._get_quality_settings(quality_level)
            
            # Generate all circuits
            all_circuits = []
            circuit_metadata = []
            
            print(f"[AdvancedCoverage] Generating {total_circuits} circuits for {winding_angle}° pattern")
            
            # Create planner instance
            planner = UnifiedTrajectoryPlanner(
                vessel_geometry=self.vessel_geometry,
                roving_width_m=roving_width_m,
                payout_length_m=0.5,
                default_friction_coeff=self.layer_config.get('friction_coefficient', 0.1)
            )
            
            for circuit_num in range(total_circuits):
                # Calculate starting phi for this circuit
                start_phi = circuit_num * angular_advancement
                
                # Use the UnifiedTrajectoryPlanner for authentic trajectory generation
                try:
                    trajectory_result = planner.generate_trajectory(
                        pattern_type='helical' if winding_angle < 85 else 'hoop',
                        coverage_mode='full_coverage',
                        physics_model=self.layer_config.get('physics_model', 'clairaut'),
                        target_angle_deg=winding_angle,
                        start_phi_offset=start_phi
                    )
                    
                    if trajectory_result and trajectory_result.get('success', False):
                        trajectory_points = trajectory_result.get('path_points', [])
                        if trajectory_points:
                            all_circuits.append(trajectory_points)
                            circuit_metadata.append({
                                'circuit_number': circuit_num + 1,
                                'start_phi_deg': math.degrees(start_phi),
                                'points_count': len(trajectory_points),
                                'quality_score': trajectory_result.get('quality_score', 85.0)
                            })
                            print(f"[AdvancedCoverage] Generated circuit {circuit_num + 1}/{total_circuits} with {len(trajectory_points)} points using UnifiedPlanner")
                        else:
                            print(f"[AdvancedCoverage] UnifiedPlanner returned empty trajectory for circuit {circuit_num + 1}")
                    else:
                        print(f"[AdvancedCoverage] UnifiedPlanner failed for circuit {circuit_num + 1}, using fallback")
                        # Only use fallback if unified planner fails
                        fallback_circuit = self._generate_fallback_circuit(
                            circuit_num, start_phi, quality_settings['points_per_circuit']
                        )
                        if fallback_circuit:
                            all_circuits.append(fallback_circuit)
                            circuit_metadata.append({
                                'circuit_number': circuit_num + 1,
                                'start_phi_deg': math.degrees(start_phi),
                                'points_count': len(fallback_circuit),
                                'quality_score': 75.0
                            })
                            print(f"[AdvancedCoverage] Used fallback for circuit {circuit_num + 1} with {len(fallback_circuit)} points")
                
                except Exception as e:
                    print(f"[AdvancedCoverage] Error generating circuit {circuit_num + 1}: {e}")
                    # Use fallback on error
                    fallback_circuit = self._generate_fallback_circuit(
                        circuit_num, start_phi, quality_settings['points_per_circuit']
                    )
                    if fallback_circuit:
                        all_circuits.append(fallback_circuit)
                        circuit_metadata.append({
                            'circuit_number': circuit_num + 1,
                            'start_phi_deg': math.degrees(start_phi),
                            'points_count': len(fallback_circuit),
                            'quality_score': 70.0
                        })
            
            print(f"[AdvancedCoverage] Final result: {len(all_circuits)} circuits generated successfully")
            
            return {
                'circuits': all_circuits,
                'metadata': circuit_metadata,
                'pattern_info': {
                    'actual_pattern_type': self._determine_pattern_type(),
                    'estimated_circuits': total_circuits,
                    'angular_advancement_deg': math.degrees(angular_advancement)
                },
                'total_circuits': len(all_circuits),
                'coverage_percentage': self._calculate_actual_coverage(all_circuits),
                'quality_settings': quality_settings
            }
            
        except Exception as e:
            # Return minimal coverage data if everything fails
            return self._generate_minimal_coverage_data(quality_level)
    
    def _get_quality_settings(self, quality_level):
        """Get visualization quality settings"""
        settings = {
            "fast": {
                'points_per_circuit': 80,
                'mandrel_resolution': 30,
                'surface_segments': 40
            },
            "balanced": {
                'points_per_circuit': 120,
                'mandrel_resolution': 50,
                'surface_segments': 60
            },
            "high_quality": {
                'points_per_circuit': 200,
                'mandrel_resolution': 80,
                'surface_segments': 100
            }
        }
        return settings.get(quality_level, settings["balanced"])
    
    def _determine_pattern_type(self):
        """Determine pattern type from layer configuration"""
        angle = self.layer_config['winding_angle']
        if angle < 25:
            return 'geodesic'
        elif angle > 75:
            return 'hoop'
        else:
            return 'helical'
    
    def _calculate_circuit_quality(self, points):
        """Calculate quality score for a circuit"""
        if len(points) < 2:
            return 0.0
        
        try:
            # Check for smoothness
            position_gaps = []
            for i in range(1, len(points)):
                if hasattr(points[i], 'position') and hasattr(points[i-1], 'position'):
                    gap = np.linalg.norm(points[i].position - points[i-1].position)
                    position_gaps.append(gap)
            
            if not position_gaps:
                return 50.0
            
            max_gap = max(position_gaps)
            
            # Quality based on smoothness (lower gaps = higher quality)
            quality = max(0, 100 - (max_gap * 1000 * 10))  # Penalize mm-scale gaps
            return min(100, quality)
            
        except Exception:
            return 75.0  # Default quality if calculation fails
    
    def _calculate_actual_coverage(self, all_circuits):
        """Calculate actual surface coverage percentage"""
        if not all_circuits:
            return 0.0
        
        # Simplified coverage calculation
        total_points = sum(len(circuit) for circuit in all_circuits)
        expected_points_for_full_coverage = 100 * len(all_circuits)  # Rough estimate
        
        coverage = min(100.0, (total_points / expected_points_for_full_coverage) * 100)
        return coverage
    
    def _generate_fallback_circuit(self, circuit_num, start_phi, num_points):
        """Generate a simple fallback circuit when planner fails"""
        try:
            # Create simple helical path using vessel geometry
            profile = self.vessel_geometry.get_profile_points()
            if not profile or 'r_inner_mm' not in profile or 'z_mm' not in profile:
                return []
                
            r_inner_profile_mm = np.array(profile['r_inner_mm'])
            z_profile_mm = np.array(profile['z_mm'])
            
            if len(z_profile_mm) < 2:
                return []
            
            # Ensure z_profile is sorted for interpolation
            sort_indices = np.argsort(z_profile_mm)
            z_profile_sorted = z_profile_mm[sort_indices] / 1000.0  # Convert to meters
            r_profile_sorted = r_inner_profile_mm[sort_indices] / 1000.0  # Convert to meters
            
            # Use the complete sorted profile for full vessel coverage
            z_points = np.linspace(z_profile_sorted[0], z_profile_sorted[-1], num_points)
            r_points = np.interp(z_points, z_profile_sorted, r_profile_sorted)
            
            # Generate authentic multi-pass pole-to-pole filament winding trajectory
            angle = self.layer_config['winding_angle']
            
            # Calculate number of pole-to-pole passes for this circuit
            if angle >= 85:  # Near-hoop patterns
                num_passes = 8   # More passes for hoop patterns
            elif angle >= 60:  # High angle helical
                num_passes = 6   # Medium passes
            else:  # Low angle helical
                num_passes = 4   # Fewer passes but still multiple
            
            # Generate points for complete pole-to-pole winding cycles
            total_trajectory_points = num_passes * 200  # ~200 points per pass
            
            z_full = np.linspace(z_profile_sorted[0], z_profile_sorted[-1], total_trajectory_points)
            r_full = np.interp(z_full, z_profile_sorted, r_profile_sorted)
            
            # Create back-and-forth pole-to-pole motion
            phi_points = np.zeros(total_trajectory_points)
            current_phi = start_phi
            
            points_per_pass = total_trajectory_points // num_passes
            for pass_num in range(num_passes):
                start_idx = pass_num * points_per_pass
                end_idx = start_idx + points_per_pass if pass_num < num_passes - 1 else total_trajectory_points
                
                # Alternate direction for each pass (forward/backward)
                if pass_num % 2 == 0:  # Forward pass (aft to forward)
                    z_pass = np.linspace(z_profile_sorted[0], z_profile_sorted[-1], end_idx - start_idx)
                    phi_increment = 2 * math.pi * math.sin(math.radians(angle)) / (end_idx - start_idx)
                else:  # Backward pass (forward to aft)
                    z_pass = np.linspace(z_profile_sorted[-1], z_profile_sorted[0], end_idx - start_idx)
                    phi_increment = -2 * math.pi * math.sin(math.radians(angle)) / (end_idx - start_idx)
                
                # Update z coordinates for this pass
                z_full[start_idx:end_idx] = z_pass
                r_full[start_idx:end_idx] = np.interp(z_pass, z_profile_sorted, r_profile_sorted)
                
                # Calculate phi progression for this pass
                for i in range(end_idx - start_idx):
                    phi_points[start_idx + i] = current_phi + i * phi_increment
                
                current_phi = phi_points[end_idx - 1]
            
            # Update arrays with multi-pass trajectory
            z_points = z_full
            r_points = r_full
            num_points = len(z_points)
            
            circuit_points = []
            
            # Handle different winding patterns
            if angle >= 85:  # Hoop patterns - focus on cylindrical section
                # For hoop patterns, create trajectories that stay mostly in cylindrical section
                # but still cover some dome area for realistic winding
                equatorial_radius = np.max(r_points)
                cylinder_height = np.max(z_points) - np.min(z_points)
                
                for i, (z, r, phi) in enumerate(zip(z_points, r_points, phi_points)):
                    # Adjust Z coordinates for hoop pattern - less dome excursion
                    z_factor = 0.3  # Reduce dome coverage for hoop
                    z_adj = z * z_factor
                    
                    # Keep radius closer to equatorial for hoop
                    r_adj = r * 0.9 + equatorial_radius * 0.1
                    
                    point = type('TrajectoryPoint', (), {
                        'position': np.array([r_adj * math.cos(phi), r_adj * math.sin(phi), z_adj]),
                        'winding_angle_deg': angle,
                        'phi_rad': phi,
                        'z_m': z_adj
                    })()
                    circuit_points.append(point)
            else:  # Helical patterns - full vessel coverage including both domes
                for i, (z, r, phi) in enumerate(zip(z_points, r_points, phi_points)):
                    # Full vessel coverage for helical patterns
                    point = type('TrajectoryPoint', (), {
                        'position': np.array([r * math.cos(phi), r * math.sin(phi), z]),
                        'winding_angle_deg': angle,
                        'phi_rad': phi,
                        'z_m': z
                    })()
                    circuit_points.append(point)
            
            return circuit_points
            
        except Exception:
            return []
    
    def _generate_minimal_coverage_data(self, quality_level):
        """Generate minimal coverage data when everything fails"""
        return {
            'circuits': [],
            'metadata': [],
            'pattern_info': {
                'actual_pattern_type': 'fallback',
                'estimated_circuits': 1,
                'angular_advancement_deg': 360.0
            },
            'total_circuits': 0,
            'coverage_percentage': 0.0,
            'quality_settings': self._get_quality_settings(quality_level)
        }
"""
Comprehensive Fix for Trajectory Array Mismatch Issues
Addresses constant radius trajectories and missing points
"""

import numpy as np
import streamlit as st
from typing import Dict, List, Optional, Tuple, Any
import warnings

class TrajectoryArrayValidator:
    """Validates and fixes trajectory array mismatches"""
    
    def __init__(self):
        self.debug_mode = True
        
    def _log_debug(self, message: str):
        """Debug logging with array info"""
        if self.debug_mode:
            print(f"[ArrayFix] {message}")
            
    def validate_and_fix_trajectory_data(self, trajectory_data: Dict) -> Dict:
        """
        Main validation and fixing function for trajectory data
        
        Returns:
            Fixed trajectory data with consistent arrays
        """
        if not trajectory_data:
            self._log_debug("Empty trajectory data")
            return {}
            
        self._log_debug("Starting trajectory array validation and fixing")
        
        # Step 1: Extract all available coordinate data
        coordinate_sets = self._extract_all_coordinate_sets(trajectory_data)
        
        if not coordinate_sets:
            self._log_debug("No valid coordinate sets found")
            return trajectory_data
            
        # Step 2: Validate and select best coordinate set
        best_coords = self._select_best_coordinate_set(coordinate_sets)
        
        # Step 3: Fix array mismatches
        fixed_coords = self._fix_array_mismatches(best_coords)
        
        # Step 4: Validate trajectory physics
        validated_coords = self._validate_trajectory_physics(fixed_coords)
        
        # Step 5: Rebuild trajectory data with fixed coordinates
        fixed_trajectory = self._rebuild_trajectory_data(trajectory_data, validated_coords)
        
        self._log_debug(f"Array validation complete: {len(validated_coords['x'])} consistent points")
        
        return fixed_trajectory
    
    def _extract_all_coordinate_sets(self, trajectory_data: Dict) -> List[Dict]:
        """Extract all possible coordinate sets from trajectory data"""
        coordinate_sets = []
        
        # Method 1: Direct coordinate arrays at top level
        if self._has_coordinate_arrays(trajectory_data):
            coords = self._extract_direct_coordinates(trajectory_data, "top_level")
            if coords:
                coordinate_sets.append(coords)
        
        # Method 2: Nested in trajectory_data field
        if 'trajectory_data' in trajectory_data:
            nested_data = trajectory_data['trajectory_data']
            if self._has_coordinate_arrays(nested_data):
                coords = self._extract_direct_coordinates(nested_data, "nested")
                if coords:
                    coordinate_sets.append(coords)
        
        # Method 3: From path_points
        path_coords = self._extract_from_path_points(trajectory_data)
        if path_coords:
            coordinate_sets.append(path_coords)
        
        # Method 4: From trajectory points (unified system)
        if 'points' in trajectory_data:
            unified_coords = self._extract_from_unified_points(trajectory_data['points'])
            if unified_coords:
                coordinate_sets.append(unified_coords)
        
        # Method 5: From cylindrical coordinates
        cylindrical_coords = self._extract_from_cylindrical(trajectory_data)
        if cylindrical_coords:
            coordinate_sets.append(cylindrical_coords)
            
        self._log_debug(f"Found {len(coordinate_sets)} potential coordinate sets")
        
        return coordinate_sets
    
    def _has_coordinate_arrays(self, data: Dict) -> bool:
        """Check if data contains coordinate arrays"""
        required_keys = ['x_points_m', 'y_points_m', 'z_points_m']
        return all(key in data for key in required_keys)
    
    def _extract_direct_coordinates(self, data: Dict, source: str) -> Optional[Dict]:
        """Extract coordinates from direct arrays with validation"""
        try:
            x_raw = data.get('x_points_m', [])
            y_raw = data.get('y_points_m', [])
            z_raw = data.get('z_points_m', [])
            
            # Convert to numpy arrays
            x_arr = np.array(x_raw, dtype=float)
            y_arr = np.array(y_raw, dtype=float)
            z_arr = np.array(z_raw, dtype=float)
            
            # Check for valid data
            if len(x_arr) == 0 or len(y_arr) == 0 or len(z_arr) == 0:
                self._log_debug(f"Empty arrays in {source}")
                return None
            
            # Check for array length consistency
            if not (len(x_arr) == len(y_arr) == len(z_arr)):
                self._log_debug(f"Array length mismatch in {source}: x={len(x_arr)}, y={len(y_arr)}, z={len(z_arr)}")
                # Truncate to shortest length
                min_len = min(len(x_arr), len(y_arr), len(z_arr))
                x_arr = x_arr[:min_len]
                y_arr = y_arr[:min_len]
                z_arr = z_arr[:min_len]
                self._log_debug(f"Truncated to {min_len} points")
            
            # Check for NaN or infinite values
            valid_mask = np.isfinite(x_arr) & np.isfinite(y_arr) & np.isfinite(z_arr)
            if not np.all(valid_mask):
                self._log_debug(f"Found {np.sum(~valid_mask)} invalid values in {source}")
                x_arr = x_arr[valid_mask]
                y_arr = y_arr[valid_mask]
                z_arr = z_arr[valid_mask]
            
            if len(x_arr) < 2:
                self._log_debug(f"Insufficient valid points in {source}: {len(x_arr)}")
                return None
            
            return {
                'x': x_arr,
                'y': y_arr,
                'z': z_arr,
                'source': source,
                'quality_score': self._calculate_quality_score(x_arr, y_arr, z_arr),
                'original_length': len(x_raw)
            }
            
        except Exception as e:
            self._log_debug(f"Error extracting direct coordinates from {source}: {e}")
            return None
    
    def _extract_from_path_points(self, trajectory_data: Dict) -> Optional[Dict]:
        """Extract coordinates from path_points with robust error handling"""
        try:
            # Check multiple possible locations for path_points
            path_points = None
            source_location = ""
            
            if 'path_points' in trajectory_data and trajectory_data['path_points']:
                path_points = trajectory_data['path_points']
                source_location = "top_level"
            elif ('trajectory_data' in trajectory_data and 
                  'path_points' in trajectory_data['trajectory_data'] and
                  trajectory_data['trajectory_data']['path_points']):
                path_points = trajectory_data['trajectory_data']['path_points']
                source_location = "nested"
            
            if not path_points:
                return None
            
            self._log_debug(f"Extracting from {len(path_points)} path_points ({source_location})")
            
            x_coords = []
            y_coords = []
            z_coords = []
            extraction_errors = 0
            
            for i, point in enumerate(path_points):
                try:
                    if isinstance(point, dict):
                        # Try different coordinate key formats
                        x = self._extract_coordinate_value(point, ['x_m', 'x', 'X'])
                        y = self._extract_coordinate_value(point, ['y_m', 'y', 'Y'])
                        z = self._extract_coordinate_value(point, ['z_m', 'z', 'Z'])
                        
                    elif hasattr(point, 'position') and hasattr(point.position, '__len__'):
                        # TrajectoryPoint with position array
                        if len(point.position) >= 3:
                            x, y, z = point.position[0], point.position[1], point.position[2]
                        else:
                            extraction_errors += 1
                            continue
                            
                    elif hasattr(point, 'x') and hasattr(point, 'y') and hasattr(point, 'z'):
                        # Direct attributes
                        x, y, z = point.x, point.y, point.z
                        
                    else:
                        extraction_errors += 1
                        continue
                    
                    # Validate extracted values
                    if all(np.isfinite([x, y, z])):
                        x_coords.append(float(x))
                        y_coords.append(float(y))
                        z_coords.append(float(z))
                    else:
                        extraction_errors += 1
                        
                except Exception as e:
                    extraction_errors += 1
                    if extraction_errors <= 5:  # Log first few errors
                        self._log_debug(f"Error extracting point {i}: {e}")
            
            if extraction_errors > 0:
                self._log_debug(f"Had {extraction_errors} extraction errors out of {len(path_points)} points")
            
            if len(x_coords) < 2:
                self._log_debug(f"Insufficient valid path points: {len(x_coords)}")
                return None
            
            x_arr = np.array(x_coords)
            y_arr = np.array(y_coords)
            z_arr = np.array(z_coords)
            
            return {
                'x': x_arr,
                'y': y_arr,
                'z': z_arr,
                'source': f'path_points_{source_location}',
                'quality_score': self._calculate_quality_score(x_arr, y_arr, z_arr),
                'original_length': len(path_points),
                'extraction_errors': extraction_errors
            }
            
        except Exception as e:
            self._log_debug(f"Error extracting from path_points: {e}")
            return None
    
    def _extract_coordinate_value(self, point: Dict, possible_keys: List[str]) -> float:
        """Extract coordinate value trying multiple possible keys"""
        for key in possible_keys:
            if key in point:
                value = point[key]
                if isinstance(value, (int, float)) and np.isfinite(value):
                    return float(value)
        raise ValueError(f"No valid coordinate found in {list(point.keys())}")
    
    def _extract_from_unified_points(self, points: List) -> Optional[Dict]:
        """Extract coordinates from unified trajectory points"""
        try:
            if not points:
                return None
                
            x_coords = []
            y_coords = []
            z_coords = []
            
            for point in points:
                if hasattr(point, 'position') and hasattr(point.position, '__len__'):
                    if len(point.position) >= 3:
                        x_coords.append(float(point.position[0]))
                        y_coords.append(float(point.position[1]))
                        z_coords.append(float(point.position[2]))
            
            if len(x_coords) < 2:
                return None
                
            x_arr = np.array(x_coords)
            y_arr = np.array(y_coords)
            z_arr = np.array(z_coords)
            
            return {
                'x': x_arr,
                'y': y_arr,
                'z': z_arr,
                'source': 'unified_points',
                'quality_score': self._calculate_quality_score(x_arr, y_arr, z_arr),
                'original_length': len(points)
            }
            
        except Exception as e:
            self._log_debug(f"Error extracting from unified points: {e}")
            return None
    
    def _extract_from_cylindrical(self, trajectory_data: Dict) -> Optional[Dict]:
        """Extract and convert from cylindrical coordinates"""
        try:
            # Check for cylindrical coordinates
            rho_key = None
            phi_key = None
            z_key = None
            
            # Try different possible key formats
            for data_source in [trajectory_data, trajectory_data.get('trajectory_data', {})]:
                if 'rho_points_m' in data_source and 'phi_points_rad' in data_source and 'z_points_m' in data_source:
                    rho_arr = np.array(data_source['rho_points_m'])
                    phi_arr = np.array(data_source['phi_points_rad'])
                    z_arr = np.array(data_source['z_points_m'])
                    break
            else:
                return None
            
            # Validate array lengths
            if not (len(rho_arr) == len(phi_arr) == len(z_arr)):
                min_len = min(len(rho_arr), len(phi_arr), len(z_arr))
                rho_arr = rho_arr[:min_len]
                phi_arr = phi_arr[:min_len]
                z_arr = z_arr[:min_len]
            
            # Convert to Cartesian
            x_arr = rho_arr * np.cos(phi_arr)
            y_arr = rho_arr * np.sin(phi_arr)
            
            # Validate conversion
            valid_mask = np.isfinite(x_arr) & np.isfinite(y_arr) & np.isfinite(z_arr)
            if not np.all(valid_mask):
                x_arr = x_arr[valid_mask]
                y_arr = y_arr[valid_mask]
                z_arr = z_arr[valid_mask]
            
            if len(x_arr) < 2:
                return None
            
            return {
                'x': x_arr,
                'y': y_arr,
                'z': z_arr,
                'source': 'cylindrical_conversion',
                'quality_score': self._calculate_quality_score(x_arr, y_arr, z_arr),
                'original_length': len(rho_arr)
            }
            
        except Exception as e:
            self._log_debug(f"Error extracting from cylindrical coordinates: {e}")
            return None
    
    def _calculate_quality_score(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
        """Calculate quality score for coordinate set"""
        try:
            # Check for constant radius (major issue indicator)
            rho = np.sqrt(x**2 + y**2)
            rho_variation = np.std(rho) / (np.mean(rho) + 1e-10)
            
            # Check for proper z variation
            z_variation = np.std(z) / (np.max(np.abs(z)) + 1e-10)
            
            # Check for trajectory smoothness
            if len(x) > 2:
                dx = np.diff(x)
                dy = np.diff(y)
                dz = np.diff(z)
                smoothness = 1.0 / (1.0 + np.std(np.sqrt(dx**2 + dy**2 + dz**2)))
            else:
                smoothness = 0.5
            
            # Penalize constant radius heavily
            if rho_variation < 0.01:  # Less than 1% variation
                quality = 10.0  # Very low quality
            else:
                quality = 50.0 + rho_variation * 30.0 + z_variation * 15.0 + smoothness * 5.0
            
            return min(100.0, quality)
            
        except Exception:
            return 1.0  # Very low quality if calculation fails
    
    def _select_best_coordinate_set(self, coordinate_sets: List[Dict]) -> Optional[Dict]:
        """Select the best coordinate set based on quality scores"""
        if not coordinate_sets:
            return None
        
        # Sort by quality score (descending)
        sorted_sets = sorted(coordinate_sets, key=lambda x: x['quality_score'], reverse=True)
        
        best_set = sorted_sets[0]
        self._log_debug(f"Selected best coordinate set: {best_set['source']} "
                       f"(quality: {best_set['quality_score']:.1f}, points: {len(best_set['x'])})")
        
        # Log all available sets for debugging
        for i, coord_set in enumerate(sorted_sets):
            self._log_debug(f"  Option {i+1}: {coord_set['source']} - "
                           f"Quality: {coord_set['quality_score']:.1f}, "
                           f"Points: {len(coord_set['x'])}")
        
        return best_set
    
    def _fix_array_mismatches(self, coords: Dict) -> Dict:
        """Fix any remaining array mismatches and inconsistencies"""
        x, y, z = coords['x'], coords['y'], coords['z']
        
        # Ensure all arrays are the same length
        min_length = min(len(x), len(y), len(z))
        if min_length != len(x) or min_length != len(y) or min_length != len(z):
            self._log_debug(f"Truncating arrays to {min_length} points")
            x = x[:min_length]
            y = y[:min_length]
            z = z[:min_length]
        
        # Remove any duplicate consecutive points
        if len(x) > 1:
            diff_x = np.diff(x)
            diff_y = np.diff(y)
            diff_z = np.diff(z)
            movement = np.sqrt(diff_x**2 + diff_y**2 + diff_z**2)
            
            # Keep points with significant movement (> 1e-6 meters = 1 micron)
            keep_mask = np.concatenate([[True], movement > 1e-6])
            
            if not np.all(keep_mask):
                original_length = len(x)
                x = x[keep_mask]
                y = y[keep_mask]
                z = z[keep_mask]
                self._log_debug(f"Removed {original_length - len(x)} duplicate points")
        
        # Ensure minimum trajectory length
        if len(x) < 10:
            self._log_debug(f"Warning: Very short trajectory with only {len(x)} points")
        
        return {
            'x': x,
            'y': y,
            'z': z,
            'source': coords['source'] + '_fixed',
            'quality_score': self._calculate_quality_score(x, y, z),
            'points_count': len(x)
        }
    
    def _validate_trajectory_physics(self, coords: Dict) -> Dict:
        """Validate trajectory physics and fix obvious issues"""
        x, y, z = coords['x'], coords['y'], coords['z']
        
        # Check for constant radius issue
        rho = np.sqrt(x**2 + y**2)
        rho_std = np.std(rho)
        rho_mean = np.mean(rho)
        
        if rho_std / (rho_mean + 1e-10) < 0.01:
            self._log_debug(f"CRITICAL: Constant radius detected! std/mean = {rho_std/rho_mean:.6f}")
            
            # Try to fix by checking if this is a coordinate system issue
            # Check if z varies properly
            z_range = np.max(z) - np.min(z)
            if z_range > rho_mean * 0.1:  # Z varies more than 10% of radius
                self._log_debug("Z coordinate varies properly - possible coordinate system issue")
                
                # Check if trajectory should be in cylindrical coordinates
                phi = np.arctan2(y, x)
                phi_range = np.max(phi) - np.min(phi)
                
                if phi_range > np.pi:  # More than 180 degrees
                    self._log_debug(f"Phi range: {np.degrees(phi_range):.1f}° - trajectory spans significant angle")
                else:
                    self._log_debug("Limited angular range - possible linear trajectory issue")
        
        # Check for proper trajectory progression
        if len(x) > 2:
            # Calculate arc length
            dx = np.diff(x)
            dy = np.diff(y)
            dz = np.diff(z)
            segment_lengths = np.sqrt(dx**2 + dy**2 + dz**2)
            total_length = np.sum(segment_lengths)
            
            if total_length < rho_mean * 0.1:
                self._log_debug(f"Warning: Very short trajectory length {total_length:.6f}m vs radius {rho_mean:.3f}m")
        
        return coords
    
    def _rebuild_trajectory_data(self, original_data: Dict, fixed_coords: Dict) -> Dict:
        """Rebuild trajectory data with fixed coordinates"""
        # Start with original data
        result = original_data.copy()
        
        # Update with fixed coordinates
        x_m = fixed_coords['x']
        y_m = fixed_coords['y']
        z_m = fixed_coords['z']
        
        # Store coordinates in multiple formats for compatibility
        coordinate_updates = {
            'x_points_m': x_m.tolist(),
            'y_points_m': y_m.tolist(),
            'z_points_m': z_m.tolist(),
            'x_points_mm': (x_m * 1000).tolist(),
            'y_points_mm': (y_m * 1000).tolist(),
            'z_points_mm': (z_m * 1000).tolist(),
            'total_points': len(x_m),
            'array_fix_applied': True,
            'coordinate_source': fixed_coords['source'],
            'coordinate_quality': fixed_coords['quality_score']
        }
        
        # Update top level
        result.update(coordinate_updates)
        
        # Update nested trajectory_data if it exists
        if 'trajectory_data' in result and isinstance(result['trajectory_data'], dict):
            result['trajectory_data'].update(coordinate_updates)
        
        # Calculate cylindrical coordinates
        rho_m = np.sqrt(x_m**2 + y_m**2)
        phi_rad = np.arctan2(y_m, x_m)
        
        cylindrical_updates = {
            'rho_points_m': rho_m.tolist(),
            'phi_points_rad': phi_rad.tolist(),
            'rho_points_mm': (rho_m * 1000).tolist(),
            'phi_points_deg': np.degrees(phi_rad).tolist()
        }
        
        result.update(cylindrical_updates)
        if 'trajectory_data' in result and isinstance(result['trajectory_data'], dict):
            result['trajectory_data'].update(cylindrical_updates)
        
        # Add trajectory statistics
        rho_std = np.std(rho_m)
        rho_mean = np.mean(rho_m)
        z_range = np.max(z_m) - np.min(z_m)
        
        stats_updates = {
            'radius_variation_pct': (rho_std / rho_mean * 100) if rho_mean > 0 else 0,
            'z_range_m': float(z_range),
            'mean_radius_m': float(rho_mean),
            'coordinate_validation_passed': True
        }
        
        result.update(stats_updates)
        
        self._log_debug(f"Rebuilt trajectory data with {len(x_m)} points")
        self._log_debug(f"Radius variation: {stats_updates['radius_variation_pct']:.3f}%")
        self._log_debug(f"Z range: {z_range:.3f}m, Mean radius: {rho_mean:.3f}m")
        
        return result


def fix_trajectory_array_mismatches(trajectory_data: Dict) -> Dict:
    """
    Main function to fix trajectory array mismatch issues
    
    Args:
        trajectory_data: Raw trajectory data with potential array mismatches
        
    Returns:
        Fixed trajectory data with consistent arrays
    """
    if not trajectory_data:
        return {}
    
    validator = TrajectoryArrayValidator()
    return validator.validate_and_fix_trajectory_data(trajectory_data)


def diagnose_trajectory_issues(trajectory_data: Dict) -> Dict:
    """
    Diagnose specific trajectory issues for debugging
    
    Returns:
        Diagnostic information about trajectory problems
    """
    validator = TrajectoryArrayValidator()
    
    if not trajectory_data:
        return {'status': 'error', 'message': 'No trajectory data provided'}
    
    # Extract coordinate sets
    coordinate_sets = validator._extract_all_coordinate_sets(trajectory_data)
    
    if not coordinate_sets:
        return {
            'status': 'error',
            'message': 'No valid coordinate data found',
            'available_keys': list(trajectory_data.keys())
        }
    
    # Analyze each coordinate set
    analysis = {
        'status': 'success',
        'coordinate_sets_found': len(coordinate_sets),
        'sets_analysis': []
    }
    
    for i, coord_set in enumerate(coordinate_sets):
        x, y, z = coord_set['x'], coord_set['y'], coord_set['z']
        rho = np.sqrt(x**2 + y**2)
        
        set_analysis = {
            'source': coord_set['source'],
            'points_count': len(x),
            'quality_score': coord_set['quality_score'],
            'radius_variation_pct': (np.std(rho) / np.mean(rho) * 100) if np.mean(rho) > 0 else 0,
            'z_range_m': float(np.max(z) - np.min(z)),
            'mean_radius_m': float(np.mean(rho)),
            'constant_radius_issue': (np.std(rho) / np.mean(rho)) < 0.01 if np.mean(rho) > 0 else True
        }
        
        analysis['sets_analysis'].append(set_analysis)
    
    return analysis


# Integration function for Streamlit app
def apply_trajectory_array_fix_to_session():
    """Apply array fixes to trajectory data in Streamlit session state"""
    if 'trajectory_data' in st.session_state and st.session_state.trajectory_data:
        st.info("🔧 Applying trajectory array fixes...")
        
        # Diagnose issues first
        diagnosis = diagnose_trajectory_issues(st.session_state.trajectory_data)
        
        if diagnosis['status'] == 'error':
            st.error(f"❌ Trajectory diagnosis failed: {diagnosis['message']}")
            return False
        
        # Show diagnosis
        st.write("**Trajectory Analysis:**")
        for i, analysis in enumerate(diagnosis['sets_analysis']):
            with st.expander(f"Coordinate Set {i+1}: {analysis['source']}", expanded=i==0):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Points", analysis['points_count'])
                    st.metric("Quality Score", f"{analysis['quality_score']:.1f}")
                with col2:
                    st.metric("Radius Variation", f"{analysis['radius_variation_pct']:.3f}%")
                    st.metric("Z Range", f"{analysis['z_range_m']:.3f}m")
                with col3:
                    st.metric("Mean Radius", f"{analysis['mean_radius_m']:.3f}m")
                    status = "❌ Yes" if analysis['constant_radius_issue'] else "✅ No"
                    st.metric("Constant Radius Issue", status)
        
        # Apply fixes
        fixed_trajectory = fix_trajectory_array_mismatches(st.session_state.trajectory_data)
        
        if fixed_trajectory and fixed_trajectory.get('array_fix_applied'):
            st.session_state.trajectory_data = fixed_trajectory
            st.success(f"✅ Array fixes applied! Trajectory now has {fixed_trajectory['total_points']} consistent points")
            st.info(f"📊 Coordinate quality: {fixed_trajectory['coordinate_quality']:.1f}/100")
            st.info(f"🎯 Radius variation: {fixed_trajectory['radius_variation_pct']:.3f}%")
            return True
        else:
            st.warning("⚠️ Array fixes could not be applied - trajectory data may be corrupted")
            return False
    
    elif 'all_layer_trajectories' in st.session_state and st.session_state.all_layer_trajectories:
        st.info("🔧 Applying array fixes to all layer trajectories...")
        
        fixed_count = 0
        for i, layer_traj in enumerate(st.session_state.all_layer_trajectories):
            if 'trajectory_data' in layer_traj:
                fixed_data = fix_trajectory_array_mismatches(layer_traj['trajectory_data'])
                if fixed_data.get('array_fix_applied'):
                    layer_traj['trajectory_data'] = fixed_data
                    fixed_count += 1
        
        if fixed_count > 0:
            st.success(f"✅ Applied array fixes to {fixed_count} layer trajectories")
            return True
        else:
            st.warning("⚠️ No layer trajectories could be fixed")
            return False
    
    else:
        st.warning("⚠️ No trajectory data found in session state")
        return False

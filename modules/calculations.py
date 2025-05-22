import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from modules.geometry import VesselGeometry

class VesselCalculations:
    """
    Engineering calculations for composite pressure vessel analysis.
    Includes stress analysis, failure criteria, and performance metrics.
    """
    
    def __init__(self):
        self.failure_criteria = ['Max Stress', 'Max Strain', 'Tsai-Hill', 'Tsai-Wu']
        
    def calculate_vessel_stresses(self, vessel: VesselGeometry, pressure: float, 
                                axial_load: float = 0.0, material_props: Dict = None) -> Dict:
        """
        Calculate stresses in pressure vessel using classical theory.
        
        Parameters:
        -----------
        vessel : VesselGeometry
            Vessel geometry object
        pressure : float
            Internal pressure in MPa
        axial_load : float
            External axial load in N
        material_props : Dict
            Material properties (optional)
            
        Returns:
        --------
        Dict : Stress analysis results
        """
        # Basic vessel dimensions
        r_inner = vessel.inner_radius  # mm
        r_outer = vessel.outer_radius  # mm
        t = vessel.wall_thickness  # mm
        
        # Convert pressure to N/mm²
        p = pressure  # MPa = N/mm²
        
        # Cylindrical section stresses (thin-wall theory)
        # Hoop stress: σ_θ = p*r/t
        sigma_hoop_cyl = p * r_inner / t
        
        # Axial stress: σ_z = p*r/(2*t) + F/(π*r²)
        pressure_axial = p * r_inner / (2 * t)
        applied_axial = axial_load / (np.pi * r_inner**2)
        sigma_axial_cyl = pressure_axial + applied_axial
        
        # Radial stress (approximately zero for thin walls)
        sigma_radial_cyl = 0.0
        
        # For thick walls, use exact solution
        if t / r_inner > 0.1:  # Thick wall criterion
            sigma_hoop_cyl, sigma_axial_cyl, sigma_radial_cyl = self._thick_wall_stresses(
                r_inner, r_outer, p, axial_load
            )
        
        # Dome stresses (more complex - simplified here)
        dome_stress_results = self._calculate_dome_stresses(vessel, pressure)
        
        # Compile results
        stress_results = {
            'hoop_stress_cyl': round(sigma_hoop_cyl, 2),
            'axial_stress_cyl': round(sigma_axial_cyl, 2),
            'radial_stress_cyl': round(sigma_radial_cyl, 2),
            'pressure_mpa': pressure,
            'axial_load_n': axial_load
        }
        
        # Add dome stress results
        stress_results.update(dome_stress_results)
        
        # Calculate safety factors if material properties available
        if material_props:
            safety_factors = self._calculate_safety_factors(stress_results, material_props)
            stress_results.update(safety_factors)
        
        return stress_results
        
    def _thick_wall_stresses(self, r_inner: float, r_outer: float, 
                           pressure: float, axial_load: float) -> Tuple[float, float, float]:
        """Calculate stresses using thick wall cylinder theory (Lamé equations)"""
        
        # At inner surface (r = r_inner)
        k = (r_outer / r_inner)**2
        
        # Hoop stress at inner surface
        sigma_hoop = pressure * (k + 1) / (k - 1)
        
        # Radial stress at inner surface
        sigma_radial = -pressure
        
        # Axial stress (assuming plane strain)
        sigma_axial = pressure * (2 * r_inner**2) / (r_outer**2 - r_inner**2)
        
        # Add applied axial load
        cross_sectional_area = np.pi * (r_outer**2 - r_inner**2)
        sigma_axial += axial_load / cross_sectional_area
        
        return sigma_hoop, sigma_axial, sigma_radial
        
    def _calculate_dome_stresses(self, vessel: VesselGeometry, pressure: float) -> Dict:
        """
        Calculate dome stresses based on dome type and geometry.
        """
        r_inner = vessel.inner_radius
        t = vessel.wall_thickness
        p = pressure
        
        dome_results = {}
        
        if vessel.dome_type == "Hemispherical":
            # Hemispherical dome: σ = p*r/(2*t) (both hoop and meridional)
            sigma_dome = p * r_inner / (2 * t)
            dome_results['dome_stress_hoop'] = round(sigma_dome, 2)
            dome_results['dome_stress_meridional'] = round(sigma_dome, 2)
            dome_results['dome_stress_max'] = round(sigma_dome, 2)
            
        elif vessel.dome_type == "Elliptical":
            # Elliptical dome - varies with position
            aspect_ratio = vessel.elliptical_aspect_ratio
            # Maximum stress at crown (simplified)
            sigma_max = p * r_inner / (2 * t) * (2 - aspect_ratio**2) / aspect_ratio**2
            dome_results['dome_stress_max'] = round(sigma_max, 2)
            
        elif vessel.dome_type == "Isotensoid":
            # Isotensoid dome - designed for equal stress
            # Stress should be approximately equal to hoop stress in cylinder
            sigma_isotensoid = p * r_inner / t  # Target stress
            dome_results['dome_stress_isotensoid'] = round(sigma_isotensoid, 2)
            dome_results['dome_stress_max'] = round(sigma_isotensoid, 2)
            
        else:
            # Generic dome approximation
            sigma_dome = p * r_inner / (2 * t) * 1.2  # Conservative factor
            dome_results['dome_stress_max'] = round(sigma_dome, 2)
        
        return dome_results
        
    def _calculate_safety_factors(self, stress_results: Dict, material_props: Dict) -> Dict:
        """Calculate safety factors based on material allowables"""
        
        # Extract material strengths
        F_1t = material_props.get('F_1t_longitudinal_tensile_mpa', 1000.0)
        F_2t = material_props.get('F_2t_transverse_tensile_mpa', 50.0)
        F_12s = material_props.get('F_12s_inplane_shear_mpa', 70.0)
        
        # Calculate safety factors
        safety_factors = {}
        
        # Hoop stress safety factor (assuming fiber direction)
        if stress_results['hoop_stress_cyl'] > 0:
            safety_factors['safety_factor_hoop'] = round(F_1t / stress_results['hoop_stress_cyl'], 2)
        
        # Axial stress safety factor
        if stress_results['axial_stress_cyl'] > 0:
            safety_factors['safety_factor_axial'] = round(F_1t / stress_results['axial_stress_cyl'], 2)
        
        # Overall minimum safety factor
        sf_values = [sf for sf in safety_factors.values() if sf > 0]
        if sf_values:
            safety_factors['safety_factor_min'] = round(min(sf_values), 2)
        
        return safety_factors
        
    def calculate_burst_pressure(self, vessel: VesselGeometry, material_props: Dict) -> Dict:
        """
        Calculate burst pressure using various failure criteria.
        """
        r_inner = vessel.inner_radius
        t = vessel.wall_thickness
        
        # Extract material properties
        F_1t = material_props.get('F_1t_longitudinal_tensile_mpa', 1000.0)
        F_2t = material_props.get('F_2t_transverse_tensile_mpa', 50.0)
        F_1c = material_props.get('F_1c_longitudinal_compressive_mpa', 600.0)
        F_2c = material_props.get('F_2c_transverse_compressive_mpa', 100.0)
        F_12s = material_props.get('F_12s_inplane_shear_mpa', 70.0)
        
        burst_results = {}
        
        # Maximum stress criterion (hoop direction)
        # σ_hoop = p*r/t = F_1t → p_burst = F_1t * t / r
        p_burst_max_stress = F_1t * t / r_inner
        burst_results['burst_pressure_max_stress_mpa'] = round(p_burst_max_stress, 2)
        
        # For composite laminates, consider fiber direction and layup
        # This is simplified - full analysis requires laminate theory
        
        # Typical burst pressure for well-designed COPV
        # Accounts for stress concentration and safety margins
        p_burst_design = p_burst_max_stress * 0.8  # Design margin
        burst_results['burst_pressure_design_mpa'] = round(p_burst_design, 2)
        
        return burst_results
        
    def calculate_performance_metrics(self, vessel: VesselGeometry, 
                                    material_props: Dict = None) -> Dict:
        """
        Calculate key performance metrics for the pressure vessel.
        """
        # Get geometric properties
        geom_props = vessel.get_geometric_properties()
        
        volume_liters = geom_props.get('total_volume', 0)
        weight_kg = geom_props.get('estimated_weight', 1)
        
        performance = {}
        
        # Volume to weight ratio
        performance['volume_weight_ratio'] = round(volume_liters / weight_kg, 2)
        
        # Estimated burst pressure (simplified)
        if material_props:
            burst_data = self.calculate_burst_pressure(vessel, material_props)
            performance['estimated_burst_pressure'] = burst_data.get('burst_pressure_design_mpa', 0)
        else:
            # Use typical composite strength
            typical_strength = 1000  # MPa
            r_inner = vessel.inner_radius
            t = vessel.wall_thickness
            performance['estimated_burst_pressure'] = round(typical_strength * t / r_inner * 0.8, 1)
        
        # PV/W ratio (performance index)
        # Pressure × Volume / Weight (energy storage capability)
        operating_pressure = 30.0  # Default assumption, MPa
        energy_density = (operating_pressure * volume_liters) / weight_kg
        performance['pv_w_ratio'] = round(energy_density, 2)
        
        # Structural efficiency
        r_inner = vessel.inner_radius
        t = vessel.wall_thickness
        performance['thickness_ratio'] = round(t / r_inner, 3)
        
        return performance
        
    def apply_failure_criteria(self, stress_state: Dict, material_props: Dict, 
                             criterion: str = 'Tsai-Hill') -> Dict:
        """
        Apply composite failure criteria to stress state.
        
        Parameters:
        -----------
        stress_state : Dict
            Stress components in material coordinates
        material_props : Dict
            Material strength properties
        criterion : str
            Failure criterion to apply
            
        Returns:
        --------
        Dict : Failure analysis results
        """
        # Extract stresses (assumed in material coordinates)
        sigma_1 = stress_state.get('sigma_1', 0)  # Longitudinal
        sigma_2 = stress_state.get('sigma_2', 0)  # Transverse
        tau_12 = stress_state.get('tau_12', 0)    # Shear
        
        # Extract strengths
        F_1t = material_props.get('F_1t_longitudinal_tensile_mpa', 1000)
        F_1c = material_props.get('F_1c_longitudinal_compressive_mpa', 600)
        F_2t = material_props.get('F_2t_transverse_tensile_mpa', 50)
        F_2c = material_props.get('F_2c_transverse_compressive_mpa', 100)
        F_12s = material_props.get('F_12s_inplane_shear_mpa', 70)
        
        failure_results = {'criterion': criterion}
        
        if criterion == 'Max Stress':
            # Maximum stress criterion
            checks = [
                abs(sigma_1) / (F_1t if sigma_1 >= 0 else F_1c),
                abs(sigma_2) / (F_2t if sigma_2 >= 0 else F_2c),
                abs(tau_12) / F_12s
            ]
            failure_index = max(checks)
            
        elif criterion == 'Tsai-Hill':
            # Tsai-Hill criterion
            term1 = (sigma_1 / (F_1t if sigma_1 >= 0 else F_1c))**2
            term2 = (sigma_2 / (F_2t if sigma_2 >= 0 else F_2c))**2
            term3 = (tau_12 / F_12s)**2
            term4 = -sigma_1 * sigma_2 / (F_1t * F_2t)
            
            failure_index = math.sqrt(term1 + term2 + term3 + term4)
            
        elif criterion == 'Max Strain':
            # Would need strain values and allowable strains
            failure_index = 0.5  # Placeholder
            
        else:
            failure_index = 1.0  # Conservative
        
        failure_results['failure_index'] = round(failure_index, 3)
        failure_results['safety_factor'] = round(1 / failure_index if failure_index > 0 else float('inf'), 2)
        failure_results['margin_of_safety'] = round(1/failure_index - 1 if failure_index > 0 else float('inf'), 2)
        
        return failure_results
        
    def calculate_fatigue_life(self, stress_amplitude: float, material_props: Dict, 
                             stress_ratio: float = 0.1) -> Dict:
        """
        Estimate fatigue life using simplified S-N curve approach.
        
        Parameters:
        -----------
        stress_amplitude : float
            Stress amplitude in MPa
        material_props : Dict
            Material properties
        stress_ratio : float
            Stress ratio (min stress / max stress)
            
        Returns:
        --------
        Dict : Fatigue life prediction
        """
        # Simplified fatigue analysis
        # Real implementation would use proper S-N curves for specific materials
        
        ultimate_strength = material_props.get('F_1t_longitudinal_tensile_mpa', 1000)
        
        # Simplified S-N relationship: N = A * (S/Sult)^(-m)
        # Where A and m are material constants
        A = 1e12  # Cycles (typical for composites)
        m = 10    # Slope parameter
        
        stress_fraction = stress_amplitude / ultimate_strength
        
        if stress_fraction <= 0:
            fatigue_life = float('inf')
        else:
            fatigue_life = A * (stress_fraction**(-m))
        
        fatigue_results = {
            'stress_amplitude_mpa': stress_amplitude,
            'stress_ratio': stress_ratio,
            'estimated_cycles_to_failure': int(min(fatigue_life, 1e8)),
            'fatigue_strength_fraction': round(stress_fraction, 3)
        }
        
        return fatigue_results

import numpy as np
from typing import Dict, List, Tuple, Optional
from data.material_database import FIBER_MATERIALS, RESIN_MATERIALS

class MaterialDatabase:
    """
    Material properties database and composite calculations for pressure vessel design.
    """
    
    def __init__(self):
        self.fiber_materials = FIBER_MATERIALS
        self.resin_materials = RESIN_MATERIALS
        
    def get_fiber_properties(self, fiber_type: str) -> Dict:
        """Get fiber material properties"""
        return self.fiber_materials.get(fiber_type, {})
        
    def get_resin_properties(self, resin_type: str) -> Dict:
        """Get resin material properties"""
        return self.resin_materials.get(resin_type, {})
        
    def calculate_composite_properties(self, fiber_type: str, resin_type: str, 
                                     fiber_volume_fraction: float, void_content: float = 0.02) -> Dict:
        """
        Calculate composite lamina properties using micromechanics.
        
        Parameters:
        -----------
        fiber_type : str
            Type of fiber material
        resin_type : str
            Type of resin material
        fiber_volume_fraction : float
            Fiber volume fraction (0 to 1)
        void_content : float
            Void content fraction (0 to 1)
            
        Returns:
        --------
        Dict : Composite material properties
        """
        fiber_props = self.get_fiber_properties(fiber_type)
        resin_props = self.get_resin_properties(resin_type)
        
        if not fiber_props or not resin_props:
            raise ValueError(f"Material properties not found for {fiber_type} or {resin_type}")
        
        # Input validation
        if not (0 <= fiber_volume_fraction <= 1):
            raise ValueError(f"Fiber volume fraction ({fiber_volume_fraction}) must be between 0 and 1.")
        if not (0 <= void_content < 1):
            raise ValueError(f"Void content ({void_content}) must be between 0 and 1.")
        
        # Volume fractions (accounting for voids)
        V_f = fiber_volume_fraction * (1 - void_content)
        V_m = (1 - fiber_volume_fraction) * (1 - void_content)
        V_v = void_content
        
        # Ensure V_f is valid for sqrt operations
        if V_f < 0:
            V_f = 0.01  # Minimum safe value
        
        # Extract properties
        E_f = fiber_props.get('tensile_modulus_gpa', 70.0)
        E_m = resin_props.get('tensile_modulus_gpa', 3.5)
        nu_f = fiber_props.get('poissons_ratio', 0.22)
        nu_m = resin_props.get('poissons_ratio', 0.35)
        G_m = resin_props.get('shear_modulus_gpa', 1.3)
        
        # Fiber tensile strength
        S_f = fiber_props.get('tensile_strength_mpa', 2000.0)
        S_m = resin_props.get('tensile_strength_mpa', 80.0)
        
        # Densities
        rho_f = fiber_props.get('density_g_cm3', 2.6)
        rho_m = resin_props.get('density_g_cm3', 1.2)
        
        # Rule of mixtures and micromechanics calculations
        
        # Longitudinal modulus (E11) - Rule of Mixtures
        E_11 = V_f * E_f + V_m * E_m
        
        # Transverse modulus (E22) - Modified rule of mixtures
        # Using Halpin-Tsai equations for better accuracy
        eta_E = (E_f/E_m - 1) / (E_f/E_m + 2)
        E_22 = E_m * (1 + 2*eta_E*V_f) / (1 - eta_E*V_f)
        
        # In-plane shear modulus (G12) - Halpin-Tsai
        G_f = E_f / (2 * (1 + nu_f))  # Estimate fiber shear modulus
        eta_G = (G_f/G_m - 1) / (G_f/G_m + 1)
        G_12 = G_m * (1 + eta_G*V_f) / (1 - eta_G*V_f)
        
        # Major Poisson's ratio (nu12) - Rule of mixtures
        nu_12 = V_f * nu_f + V_m * nu_m
        
        # Minor Poisson's ratio (nu21) - From symmetry condition
        nu_21 = nu_12 * E_22 / E_11
        
        # Strength properties
        
        # Longitudinal tensile strength - Rule of mixtures
        F_1t = V_f * S_f + V_m * S_m * 0.5  # Matrix contribution reduced
        
        # Longitudinal compressive strength (typically lower)
        F_1c = F_1t * 0.6  # Typical ratio for composites
        
        # Transverse tensile strength (matrix dominated)
        F_2t = S_m * (1 - np.sqrt(V_f)) * 1.2  # Empirical correction
        
        # Transverse compressive strength
        F_2c = F_2t * 2.0  # Typically higher than tensile
        
        # In-plane shear strength
        F_12s = S_m * 0.8 * (1 - V_f)  # Matrix-dominated property
        
        # Composite density
        rho_composite = V_f * rho_f + V_m * rho_m
        
        # Calculate anisotropy factor (ke)
        # Koussios definition: ke = E2(1+nu12) / E1(1+nu21)
        k_e = E_22 * (1 + nu_12) / (E_11 * (1 + nu_21))
        
        # Strength-based anisotropy factor
        k_s = F_2t / F_1t
        
        # Thermal expansion coefficients (if available)
        alpha_f = fiber_props.get('cte_longitudinal_1e6_k', 0.0)
        alpha_m = resin_props.get('cte_1e6_k', 50.0)
        
        # Longitudinal CTE
        alpha_11 = (V_f * alpha_f * E_f + V_m * alpha_m * E_m) / E_11
        
        # Transverse CTE (more complex, simplified here)
        alpha_22 = (1 + nu_f) * V_f * alpha_f + (1 + nu_m) * V_m * alpha_m - alpha_11 * nu_12
        
        # Ply thickness (estimated from fiber and resin)
        typical_ply_thickness = 0.125  # mm, typical for prepreg
        
        composite_properties = {
            # Elastic properties
            'E_11_longitudinal_gpa': round(E_11, 2),
            'E_22_transverse_gpa': round(E_22, 2),
            'G_12_shear_gpa': round(G_12, 2),
            'nu_12_major_poisson': round(nu_12, 3),
            'nu_21_minor_poisson': round(nu_21, 3),
            
            # Strength properties
            'F_1t_longitudinal_tensile_mpa': round(F_1t, 0),
            'F_1c_longitudinal_compressive_mpa': round(F_1c, 0),
            'F_2t_transverse_tensile_mpa': round(F_2t, 0),
            'F_2c_transverse_compressive_mpa': round(F_2c, 0),
            'F_12s_inplane_shear_mpa': round(F_12s, 0),
            
            # Physical properties
            'density_g_cm3': round(rho_composite, 2),
            'fiber_volume_fraction': round(V_f, 3),
            'matrix_volume_fraction': round(V_m, 3),
            'void_content': round(V_v, 3),
            'ply_thickness_mm': typical_ply_thickness,
            
            # Anisotropy factors
            'k_e_elastic_anisotropy': round(k_e, 3),
            'k_s_strength_anisotropy': round(k_s, 3),
            
            # Thermal properties
            'alpha_11_cte_longitudinal_1e6_k': round(alpha_11, 2),
            'alpha_22_cte_transverse_1e6_k': round(alpha_22, 2),
            
            # Material system
            'fiber_type': fiber_type,
            'resin_type': resin_type
        }
        
        return composite_properties
        
    def get_material_recommendations(self, application: str, operating_pressure: float, 
                                   operating_temperature: float) -> Dict:
        """
        Recommend materials based on application requirements.
        
        Parameters:
        -----------
        application : str
            Application type (e.g., 'CNG', 'LPG', 'rocket', 'industrial')
        operating_pressure : float
            Operating pressure in MPa
        operating_temperature : float
            Operating temperature in °C
            
        Returns:
        --------
        Dict : Material recommendations
        """
        recommendations = {
            'fiber_recommendations': [],
            'resin_recommendations': [],
            'rationale': []
        }
        
        # Fiber recommendations based on requirements
        if operating_pressure > 70:  # High pressure applications
            recommendations['fiber_recommendations'].extend(['Carbon Fiber T700', 'Carbon Fiber T800'])
            recommendations['rationale'].append("High pressure requires high-strength carbon fibers")
        elif operating_pressure > 30:  # Medium pressure
            recommendations['fiber_recommendations'].extend(['Carbon Fiber T700', 'S-Glass'])
            recommendations['rationale'].append("Medium pressure allows carbon or high-strength glass")
        else:  # Lower pressure
            recommendations['fiber_recommendations'].extend(['E-Glass', 'S-Glass'])
            recommendations['rationale'].append("Lower pressure applications can use glass fibers")
        
        # Temperature considerations
        if operating_temperature > 150:  # High temperature
            recommendations['resin_recommendations'].extend(['BMI', 'Epoxy High-Temp'])
            recommendations['rationale'].append("High temperature requires advanced resin systems")
        elif operating_temperature > 80:  # Medium temperature
            recommendations['resin_recommendations'].extend(['Epoxy Standard', 'Vinylester'])
            recommendations['rationale'].append("Medium temperature allows standard epoxy systems")
        else:  # Room temperature
            recommendations['resin_recommendations'].extend(['Epoxy Standard', 'Polyester'])
            recommendations['rationale'].append("Room temperature allows various resin systems")
        
        # Application-specific considerations
        if application.lower() == 'cng':
            recommendations['rationale'].append("CNG tanks require permeation-resistant liners")
        elif application.lower() == 'rocket':
            recommendations['rationale'].append("Rocket applications prioritize weight and strength")
        
        return recommendations
        
    def calculate_laminate_properties(self, ply_stacking: List[Tuple], composite_props: Dict) -> Dict:
        """
        Calculate laminate properties from ply stacking sequence.
        
        Parameters:
        -----------
        ply_stacking : List[Tuple]
            List of (angle, thickness) tuples for each ply
        composite_props : Dict
            Composite lamina properties
            
        Returns:
        --------
        Dict : Laminate-level properties
        """
        if not ply_stacking:
            return {}
        
        # Extract lamina properties
        E_11 = composite_props.get('E_11_longitudinal_gpa', 150.0)
        E_22 = composite_props.get('E_22_transverse_gpa', 10.0)
        G_12 = composite_props.get('G_12_shear_gpa', 5.0)
        nu_12 = composite_props.get('nu_12_major_poisson', 0.3)
        
        total_thickness = sum(thickness for angle, thickness in ply_stacking)
        
        # Simplified laminate analysis (Classical Lamination Theory would be more accurate)
        
        # Calculate effective moduli using simple averaging
        angles = [angle for angle, thickness in ply_stacking]
        thicknesses = [thickness for angle, thickness in ply_stacking]
        
        # Weight by thickness
        thickness_weights = [t/total_thickness for t in thicknesses]
        
        # Effective longitudinal modulus (simplified)
        E_x_eff = 0
        E_y_eff = 0
        G_xy_eff = 0
        
        for i, (angle, thickness) in enumerate(ply_stacking):
            weight = thickness_weights[i]
            
            # Transform properties to global coordinates (simplified)
            c = np.cos(np.radians(angle))
            s = np.sin(np.radians(angle))
            
            # Simplified transformation
            E_x_eff += weight * (E_11 * c**4 + E_22 * s**4 + 2*G_12 * s**2 * c**2)
            E_y_eff += weight * (E_11 * s**4 + E_22 * c**4 + 2*G_12 * s**2 * c**2)
            G_xy_eff += weight * G_12  # Simplified
        
        laminate_props = {
            'total_thickness_mm': total_thickness,
            'number_of_plies': len(ply_stacking),
            'effective_ex_gpa': round(E_x_eff, 2),
            'effective_ey_gpa': round(E_y_eff, 2),
            'effective_gxy_gpa': round(G_xy_eff, 2),
            'ply_angles': angles,
            'ply_thicknesses': thicknesses
        }
        
        return laminate_props

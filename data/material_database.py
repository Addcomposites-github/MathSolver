"""
Material properties database for composite pressure vessel design.
Data based on typical values from literature and manufacturer specifications.
"""

# Fiber material properties
FIBER_MATERIALS = {
    "E-Glass": {
        "description": "Standard E-glass fiber, most common and cost-effective",
        "tensile_strength_mpa": 3400,
        "tensile_modulus_gpa": 73,
        "density_g_cm3": 2.58,
        "poissons_ratio": 0.22,
        "strain_to_failure_percent": 4.7,
        "cte_longitudinal_1e6_k": 5.0,
        "typical_applications": "General purpose, low-cost applications",
        "relative_cost": 1.0
    },
    
    "S-Glass": {
        "description": "High-strength S-glass fiber for demanding applications",
        "tensile_strength_mpa": 4580,
        "tensile_modulus_gpa": 86,
        "density_g_cm3": 2.49,
        "poissons_ratio": 0.22,
        "strain_to_failure_percent": 5.3,
        "cte_longitudinal_1e6_k": 2.9,
        "typical_applications": "High-performance pressure vessels, aerospace",
        "relative_cost": 2.5
    },
    
    "Carbon Fiber T700": {
        "description": "Standard modulus carbon fiber, aerospace grade",
        "tensile_strength_mpa": 4900,
        "tensile_modulus_gpa": 230,
        "density_g_cm3": 1.80,
        "poissons_ratio": 0.20,
        "strain_to_failure_percent": 2.1,
        "cte_longitudinal_1e6_k": -0.5,
        "typical_applications": "Aerospace, high-performance automotive, sporting goods",
        "relative_cost": 8.0
    },
    
    "Carbon Fiber T800": {
        "description": "Intermediate modulus carbon fiber, high strength",
        "tensile_strength_mpa": 5490,
        "tensile_modulus_gpa": 294,
        "density_g_cm3": 1.80,
        "poissons_ratio": 0.20,
        "strain_to_failure_percent": 1.9,
        "cte_longitudinal_1e6_k": -0.5,
        "typical_applications": "Advanced aerospace structures, high-end pressure vessels",
        "relative_cost": 12.0
    },
    
    "Carbon Fiber T1000": {
        "description": "High modulus carbon fiber, maximum performance",
        "tensile_strength_mpa": 6370,
        "tensile_modulus_gpa": 294,
        "density_g_cm3": 1.80,
        "poissons_ratio": 0.20,
        "strain_to_failure_percent": 2.2,
        "cte_longitudinal_1e6_k": -0.5,
        "typical_applications": "Ultra-high performance aerospace, F1 racing",
        "relative_cost": 20.0
    },
    
    "Aramid Kevlar 49": {
        "description": "High-strength aramid fiber, excellent impact resistance",
        "tensile_strength_mpa": 3600,
        "tensile_modulus_gpa": 112,
        "density_g_cm3": 1.44,
        "poissons_ratio": 0.36,
        "strain_to_failure_percent": 3.2,
        "cte_longitudinal_1e6_k": -4.0,
        "typical_applications": "Ballistic protection, impact-resistant structures",
        "relative_cost": 6.0
    },
    
    "Basalt Fiber": {
        "description": "Natural basalt fiber, eco-friendly alternative to glass",
        "tensile_strength_mpa": 4840,
        "tensile_modulus_gpa": 89,
        "density_g_cm3": 2.65,
        "poissons_ratio": 0.26,
        "strain_to_failure_percent": 5.4,
        "cte_longitudinal_1e6_k": 8.0,
        "typical_applications": "Eco-friendly applications, chemical resistance",
        "relative_cost": 2.0
    }
}

# Resin/matrix material properties
RESIN_MATERIALS = {
    "Epoxy Standard": {
        "description": "Standard epoxy resin system for general applications",
        "tensile_strength_mpa": 80,
        "tensile_modulus_gpa": 3.5,
        "shear_modulus_gpa": 1.3,
        "density_g_cm3": 1.20,
        "poissons_ratio": 0.35,
        "glass_transition_temp_c": 120,
        "cte_1e6_k": 50,
        "cure_temperature_c": 120,
        "pot_life_hours_25c": 4,
        "typical_applications": "General composites, room temperature service",
        "relative_cost": 1.0
    },
    
    "Epoxy High-Temp": {
        "description": "High-temperature epoxy for elevated service temperatures",
        "tensile_strength_mpa": 90,
        "tensile_modulus_gpa": 4.2,
        "shear_modulus_gpa": 1.5,
        "density_g_cm3": 1.25,
        "poissons_ratio": 0.33,
        "glass_transition_temp_c": 180,
        "cte_1e6_k": 45,
        "cure_temperature_c": 177,
        "pot_life_hours_25c": 2,
        "typical_applications": "Aerospace, automotive underhood, high-temp service",
        "relative_cost": 2.5
    },
    
    "Vinylester": {
        "description": "Vinylester resin with excellent chemical resistance",
        "tensile_strength_mpa": 85,
        "tensile_modulus_gpa": 3.8,
        "shear_modulus_gpa": 1.4,
        "density_g_cm3": 1.15,
        "poissons_ratio": 0.36,
        "glass_transition_temp_c": 110,
        "cte_1e6_k": 55,
        "cure_temperature_c": 80,
        "pot_life_hours_25c": 6,
        "typical_applications": "Chemical tanks, marine applications, corrosion resistance",
        "relative_cost": 1.8
    },
    
    "Polyester": {
        "description": "Unsaturated polyester resin, cost-effective general purpose",
        "tensile_strength_mpa": 55,
        "tensile_modulus_gpa": 2.8,
        "shear_modulus_gpa": 1.0,
        "density_g_cm3": 1.10,
        "poissons_ratio": 0.38,
        "glass_transition_temp_c": 85,
        "cte_1e6_k": 60,
        "cure_temperature_c": 25,
        "pot_life_hours_25c": 8,
        "typical_applications": "Low-cost applications, construction, marine",
        "relative_cost": 0.6
    },
    
    "BMI (Bismaleimide)": {
        "description": "High-performance thermosetting resin for extreme conditions",
        "tensile_strength_mpa": 110,
        "tensile_modulus_gpa": 4.8,
        "shear_modulus_gpa": 1.8,
        "density_g_cm3": 1.30,
        "poissons_ratio": 0.32,
        "glass_transition_temp_c": 250,
        "cte_1e6_k": 35,
        "cure_temperature_c": 200,
        "pot_life_hours_25c": 1,
        "typical_applications": "Aerospace, jet engines, extreme temperature applications",
        "relative_cost": 8.0
    },
    
    "PEEK": {
        "description": "Polyetheretherketone thermoplastic, ultra-high performance",
        "tensile_strength_mpa": 100,
        "tensile_modulus_gpa": 4.0,
        "shear_modulus_gpa": 1.5,
        "density_g_cm3": 1.32,
        "poissons_ratio": 0.30,
        "glass_transition_temp_c": 143,
        "melting_point_c": 334,
        "cte_1e6_k": 47,
        "processing_temp_c": 380,
        "typical_applications": "Aerospace, medical implants, oil & gas downhole",
        "relative_cost": 15.0
    }
}

# Typical composite layup configurations
STANDARD_LAYUPS = {
    "Pressure Vessel Optimized": {
        "description": "Optimized layup for internal pressure loading",
        "sequence": [
            (90, 0.125),   # Hoop layer
            (55, 0.125),   # Helical +55°
            (-55, 0.125),  # Helical -55°
            (90, 0.125),   # Hoop layer
            (0, 0.125),    # Axial layer (if needed)
            (90, 0.125),   # Hoop layer
            (-55, 0.125),  # Helical -55°
            (55, 0.125),   # Helical +55°
            (90, 0.125)    # Hoop layer
        ],
        "total_thickness_mm": 1.125,
        "hoop_percentage": 44.4,
        "helical_percentage": 44.4,
        "axial_percentage": 11.1
    },
    
    "Quasi-Isotropic": {
        "description": "Balanced properties in all directions",
        "sequence": [
            (0, 0.125),
            (45, 0.125),
            (90, 0.125),
            (-45, 0.125),
            (-45, 0.125),
            (90, 0.125),
            (45, 0.125),
            (0, 0.125)
        ],
        "total_thickness_mm": 1.0,
        "hoop_percentage": 25.0,
        "helical_percentage": 50.0,
        "axial_percentage": 25.0
    },
    
    "High Hoop": {
        "description": "Maximum hoop strength for pressure vessels",
        "sequence": [
            (90, 0.125),
            (90, 0.125),
            (55, 0.125),
            (-55, 0.125),
            (90, 0.125),
            (90, 0.125),
            (-55, 0.125),
            (55, 0.125),
            (90, 0.125),
            (90, 0.125)
        ],
        "total_thickness_mm": 1.25,
        "hoop_percentage": 60.0,
        "helical_percentage": 40.0,
        "axial_percentage": 0.0
    }
}

# Material combination recommendations
MATERIAL_COMBINATIONS = {
    "High Performance CNG": {
        "fiber": "Carbon Fiber T700",
        "resin": "Epoxy Standard",
        "layup": "Pressure Vessel Optimized",
        "typical_vf": 0.60,
        "cost_index": 8.0,
        "performance_index": 9.5,
        "applications": ["CNG tanks", "High-pressure hydrogen storage"]
    },
    
    "Cost Effective Industrial": {
        "fiber": "E-Glass",
        "resin": "Vinylester",
        "layup": "High Hoop",
        "typical_vf": 0.55,
        "cost_index": 1.8,
        "performance_index": 6.0,
        "applications": ["Industrial tanks", "Water treatment vessels"]
    },
    
    "Aerospace Premium": {
        "fiber": "Carbon Fiber T800",
        "resin": "BMI",
        "layup": "Quasi-Isotropic",
        "typical_vf": 0.65,
        "cost_index": 20.0,
        "performance_index": 10.0,
        "applications": ["Rocket motor cases", "Spacecraft pressure vessels"]
    },
    
    "Marine Chemical": {
        "fiber": "S-Glass",
        "resin": "Vinylester",
        "layup": "Pressure Vessel Optimized",
        "typical_vf": 0.58,
        "cost_index": 4.5,
        "performance_index": 8.0,
        "applications": ["Chemical storage", "Marine pressure vessels"]
    }
}

# Operating environment limits for material selection
ENVIRONMENTAL_LIMITS = {
    "E-Glass + Epoxy Standard": {
        "max_temperature_c": 120,
        "min_temperature_c": -40,
        "max_pressure_mpa": 50,
        "chemical_resistance": "Limited",
        "uv_resistance": "Poor"
    },
    
    "Carbon + Epoxy High-Temp": {
        "max_temperature_c": 180,
        "min_temperature_c": -60,
        "max_pressure_mpa": 150,
        "chemical_resistance": "Good",
        "uv_resistance": "Poor"
    },
    
    "S-Glass + Vinylester": {
        "max_temperature_c": 110,
        "min_temperature_c": -30,
        "max_pressure_mpa": 80,
        "chemical_resistance": "Excellent",
        "uv_resistance": "Fair"
    },
    
    "Carbon + BMI": {
        "max_temperature_c": 250,
        "min_temperature_c": -80,
        "max_pressure_mpa": 200,
        "chemical_resistance": "Excellent",
        "uv_resistance": "Poor"
    }
}

def get_material_property(material_type: str, material_name: str, property_name: str):
    """
    Retrieve a specific property for a material.
    
    Parameters:
    -----------
    material_type : str
        Either 'fiber' or 'resin'
    material_name : str
        Name of the material
    property_name : str
        Name of the property to retrieve
        
    Returns:
    --------
    Property value or None if not found
    """
    if material_type.lower() == 'fiber':
        return FIBER_MATERIALS.get(material_name, {}).get(property_name)
    elif material_type.lower() == 'resin':
        return RESIN_MATERIALS.get(material_name, {}).get(property_name)
    else:
        return None

def get_recommended_combinations(max_cost_index: float = 10.0, 
                               min_performance_index: float = 6.0):
    """
    Get material combinations that meet cost and performance criteria.
    
    Parameters:
    -----------
    max_cost_index : float
        Maximum acceptable cost index
    min_performance_index : float
        Minimum required performance index
        
    Returns:
    --------
    Dict of qualifying material combinations
    """
    qualified_combinations = {}
    
    for name, combo in MATERIAL_COMBINATIONS.items():
        if (combo['cost_index'] <= max_cost_index and 
            combo['performance_index'] >= min_performance_index):
            qualified_combinations[name] = combo
            
    return qualified_combinations

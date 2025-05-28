"""
Unified Trajectory Configuration Management
Provides comprehensive settings and configuration control for the unified trajectory system
"""

import json
import streamlit as st
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
from pathlib import Path

@dataclass
class TrajectoryConfig:
    """Configuration settings for unified trajectory system"""
    
    # Default pattern settings
    default_pattern_type: str = 'geodesic'
    default_physics_model: str = 'clairaut'
    default_continuity_level: int = 1
    default_coverage_mode: str = 'optimized_coverage'
    
    # Performance settings
    performance_mode: str = 'balanced'  # 'fast', 'balanced', 'accurate'
    max_trajectory_points: int = 10000
    numerical_tolerance: float = 1e-6
    integration_step_size: float = 0.001
    
    # Quality control settings
    min_quality_threshold: float = 0.85
    enable_quality_validation: bool = True
    enable_continuity_analysis: bool = True
    enable_manufacturing_validation: bool = True
    
    # Visualization settings
    default_decimation_factor: int = 10
    default_surface_segments: int = 30
    enable_enhanced_visualization: bool = True
    default_view_mode: str = 'full'
    
    # Manufacturing settings
    enable_machine_kinematics: bool = True
    default_feed_rate: float = 100.0  # mm/min
    default_tension: float = 50.0  # N
    enable_collision_detection: bool = True
    
    # Advanced settings
    enable_experimental_features: bool = False
    debug_mode: bool = False
    log_level: str = 'INFO'
    save_intermediate_results: bool = False

class TrajectoryConfigManager:
    """Manages configuration loading, saving, and application"""
    
    def __init__(self):
        self.config_file = Path("trajectory_config.json")
        self.current_config = TrajectoryConfig()
    
    def load_config(self) -> TrajectoryConfig:
        """Load configuration from file or session state"""
        
        # Try to load from session state first
        if 'trajectory_config' in st.session_state:
            try:
                config_dict = st.session_state.trajectory_config
                return TrajectoryConfig(**config_dict)
            except Exception as e:
                st.warning(f"Could not load config from session: {e}")
        
        # Try to load from file
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config_dict = json.load(f)
                config = TrajectoryConfig(**config_dict)
                self.current_config = config
                return config
            except Exception as e:
                st.warning(f"Could not load config from file: {e}")
        
        # Return default config
        return TrajectoryConfig()
    
    def save_config(self, config: TrajectoryConfig):
        """Save configuration to file and session state"""
        
        # Save to session state
        st.session_state.trajectory_config = asdict(config)
        
        # Save to file
        try:
            with open(self.config_file, 'w') as f:
                json.dump(asdict(config), f, indent=2)
        except Exception as e:
            st.warning(f"Could not save config to file: {e}")
        
        self.current_config = config
    
    def get_current_config(self) -> TrajectoryConfig:
        """Get current configuration"""
        return self.current_config
    
    def reset_to_defaults(self) -> TrajectoryConfig:
        """Reset configuration to defaults"""
        default_config = TrajectoryConfig()
        self.save_config(default_config)
        return default_config
    
    def apply_config_to_planner(self, planner, config: TrajectoryConfig):
        """Apply configuration settings to a unified trajectory planner"""
        
        try:
            # Apply performance settings
            if hasattr(planner, 'set_performance_mode'):
                planner.set_performance_mode(config.performance_mode)
            
            if hasattr(planner, 'set_numerical_tolerance'):
                planner.set_numerical_tolerance(config.numerical_tolerance)
            
            if hasattr(planner, 'set_integration_step'):
                planner.set_integration_step(config.integration_step_size)
            
            # Apply quality control settings
            if hasattr(planner, 'set_quality_threshold'):
                planner.set_quality_threshold(config.min_quality_threshold)
            
            if hasattr(planner, 'enable_quality_validation'):
                planner.enable_quality_validation(config.enable_quality_validation)
            
            # Apply debug settings
            if hasattr(planner, 'set_debug_mode'):
                planner.set_debug_mode(config.debug_mode)
            
        except Exception as e:
            st.warning(f"Could not apply all config settings to planner: {e}")
    
    def get_performance_profile(self, mode: str) -> Dict[str, Any]:
        """Get performance profile settings for different modes"""
        
        profiles = {
            'fast': {
                'max_trajectory_points': 5000,
                'numerical_tolerance': 1e-4,
                'integration_step_size': 0.005,
                'default_decimation_factor': 20,
                'default_surface_segments': 20,
                'enable_quality_validation': False,
                'enable_continuity_analysis': False
            },
            'balanced': {
                'max_trajectory_points': 10000,
                'numerical_tolerance': 1e-6,
                'integration_step_size': 0.001,
                'default_decimation_factor': 10,
                'default_surface_segments': 30,
                'enable_quality_validation': True,
                'enable_continuity_analysis': True
            },
            'accurate': {
                'max_trajectory_points': 20000,
                'numerical_tolerance': 1e-8,
                'integration_step_size': 0.0005,
                'default_decimation_factor': 5,
                'default_surface_segments': 50,
                'enable_quality_validation': True,
                'enable_continuity_analysis': True,
                'enable_manufacturing_validation': True
            }
        }
        
        return profiles.get(mode, profiles['balanced'])
    
    def update_config_from_profile(self, config: TrajectoryConfig, profile_name: str) -> TrajectoryConfig:
        """Update configuration with performance profile settings"""
        
        profile = self.get_performance_profile(profile_name)
        
        # Update config with profile settings
        for key, value in profile.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        config.performance_mode = profile_name
        return config

def create_configuration_ui():
    """Create comprehensive configuration UI for trajectory settings"""
    
    st.markdown("### ⚙️ Unified Trajectory System Configuration")
    st.info("Configure advanced settings for optimal trajectory generation performance and quality")
    
    config_manager = TrajectoryConfigManager()
    current_config = config_manager.load_config()
    
    # Create tabs for different configuration categories
    tabs = st.tabs([
        "🚀 Performance", 
        "🎯 Quality Control", 
        "📊 Visualization", 
        "🏭 Manufacturing", 
        "🔧 Advanced"
    ])
    
    with tabs[0]:  # Performance tab
        st.markdown("#### Performance Settings")
        
        performance_mode = st.selectbox(
            "Performance Mode",
            options=['fast', 'balanced', 'accurate'],
            index=['fast', 'balanced', 'accurate'].index(current_config.performance_mode),
            help="Fast: Quick calculations, lower quality. Balanced: Good quality/speed. Accurate: High quality, slower."
        )
        
        if performance_mode != current_config.performance_mode:
            current_config = config_manager.update_config_from_profile(current_config, performance_mode)
            st.success(f"Updated to {performance_mode} performance profile!")
        
        col1, col2 = st.columns(2)
        with col1:
            max_points = st.number_input(
                "Max Trajectory Points",
                min_value=1000,
                max_value=50000,
                value=current_config.max_trajectory_points,
                step=1000,
                help="Maximum number of points in generated trajectories"
            )
            
            numerical_tolerance = st.number_input(
                "Numerical Tolerance",
                min_value=1e-10,
                max_value=1e-3,
                value=current_config.numerical_tolerance,
                format="%.2e",
                help="Precision for numerical calculations"
            )
        
        with col2:
            integration_step = st.number_input(
                "Integration Step Size",
                min_value=0.0001,
                max_value=0.01,
                value=current_config.integration_step_size,
                format="%.4f",
                help="Step size for trajectory integration"
            )
    
    with tabs[1]:  # Quality Control tab
        st.markdown("#### Quality Control Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            min_quality = st.slider(
                "Minimum Quality Threshold",
                min_value=0.5,
                max_value=1.0,
                value=current_config.min_quality_threshold,
                step=0.05,
                help="Minimum acceptable quality score for trajectories"
            )
            
            continuity_level = st.selectbox(
                "Default Continuity Level",
                options=[0, 1, 2],
                index=current_config.default_continuity_level,
                help="0: C0 (position), 1: C1 (velocity), 2: C2 (acceleration)"
            )
        
        with col2:
            enable_quality_validation = st.checkbox(
                "Enable Quality Validation",
                value=current_config.enable_quality_validation,
                help="Perform comprehensive quality checks during generation"
            )
            
            enable_continuity_analysis = st.checkbox(
                "Enable Continuity Analysis",
                value=current_config.enable_continuity_analysis,
                help="Analyze trajectory smoothness and continuity"
            )
            
            enable_manufacturing_validation = st.checkbox(
                "Enable Manufacturing Validation",
                value=current_config.enable_manufacturing_validation,
                help="Validate trajectories for manufacturing feasibility"
            )
    
    with tabs[2]:  # Visualization tab
        st.markdown("#### Visualization Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            decimation_factor = st.number_input(
                "Default Decimation Factor",
                min_value=1,
                max_value=50,
                value=current_config.default_decimation_factor,
                help="Show every Nth point for performance (higher = faster)"
            )
            
            surface_segments = st.number_input(
                "Surface Segments",
                min_value=10,
                max_value=100,
                value=current_config.default_surface_segments,
                help="Number of segments for mandrel surface rendering"
            )
        
        with col2:
            view_mode = st.selectbox(
                "Default View Mode",
                options=['full', 'half_y_positive', 'half_x_positive'],
                index=['full', 'half_y_positive', 'half_x_positive'].index(current_config.default_view_mode),
                help="Default 3D visualization view mode"
            )
            
            enhanced_viz = st.checkbox(
                "Enhanced Visualization",
                value=current_config.enable_enhanced_visualization,
                help="Enable advanced visualization features"
            )
    
    with tabs[3]:  # Manufacturing tab
        st.markdown("#### Manufacturing Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            enable_machine_kinematics = st.checkbox(
                "Machine Kinematics",
                value=current_config.enable_machine_kinematics,
                help="Include machine kinematic constraints"
            )
            
            feed_rate = st.number_input(
                "Default Feed Rate (mm/min)",
                min_value=1.0,
                max_value=1000.0,
                value=current_config.default_feed_rate,
                help="Default filament feed rate"
            )
        
        with col2:
            tension = st.number_input(
                "Default Tension (N)",
                min_value=1.0,
                max_value=500.0,
                value=current_config.default_tension,
                help="Default filament tension"
            )
            
            collision_detection = st.checkbox(
                "Collision Detection",
                value=current_config.enable_collision_detection,
                help="Enable collision detection during planning"
            )
    
    with tabs[4]:  # Advanced tab
        st.markdown("#### Advanced Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            experimental_features = st.checkbox(
                "Experimental Features",
                value=current_config.enable_experimental_features,
                help="Enable experimental and beta features"
            )
            
            debug_mode = st.checkbox(
                "Debug Mode",
                value=current_config.debug_mode,
                help="Enable detailed debugging information"
            )
        
        with col2:
            log_level = st.selectbox(
                "Log Level",
                options=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                index=['DEBUG', 'INFO', 'WARNING', 'ERROR'].index(current_config.log_level),
                help="Logging detail level"
            )
            
            save_intermediate = st.checkbox(
                "Save Intermediate Results",
                value=current_config.save_intermediate_results,
                help="Save intermediate calculation results for debugging"
            )
    
    # Update configuration with UI values
    updated_config = TrajectoryConfig(
        default_pattern_type=current_config.default_pattern_type,
        default_physics_model=current_config.default_physics_model,
        default_continuity_level=continuity_level,
        default_coverage_mode=current_config.default_coverage_mode,
        performance_mode=performance_mode,
        max_trajectory_points=max_points,
        numerical_tolerance=numerical_tolerance,
        integration_step_size=integration_step,
        min_quality_threshold=min_quality,
        enable_quality_validation=enable_quality_validation,
        enable_continuity_analysis=enable_continuity_analysis,
        enable_manufacturing_validation=enable_manufacturing_validation,
        default_decimation_factor=decimation_factor,
        default_surface_segments=surface_segments,
        enable_enhanced_visualization=enhanced_viz,
        default_view_mode=view_mode,
        enable_machine_kinematics=enable_machine_kinematics,
        default_feed_rate=feed_rate,
        default_tension=tension,
        enable_collision_detection=collision_detection,
        enable_experimental_features=experimental_features,
        debug_mode=debug_mode,
        log_level=log_level,
        save_intermediate_results=save_intermediate
    )
    
    # Save and action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Save Configuration", type="primary"):
            config_manager.save_config(updated_config)
            st.success("Configuration saved successfully!")
            st.rerun()
    
    with col2:
        if st.button("🔄 Reset to Defaults"):
            config_manager.reset_to_defaults()
            st.success("Configuration reset to defaults!")
            st.rerun()
    
    with col3:
        if st.button("📋 Show Current Config"):
            st.json(asdict(updated_config))
    
    return updated_config, config_manager

def get_trajectory_config() -> TrajectoryConfig:
    """Get current trajectory configuration (convenience function)"""
    config_manager = TrajectoryConfigManager()
    return config_manager.load_config()

def apply_config_to_unified_planner(planner, config: Optional[TrajectoryConfig] = None):
    """Apply configuration to unified trajectory planner"""
    if config is None:
        config = get_trajectory_config()
    
    config_manager = TrajectoryConfigManager()
    config_manager.apply_config_to_planner(planner, config)
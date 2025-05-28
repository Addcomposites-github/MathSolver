"""
Basic trajectory module for backward compatibility
Redirects to unified trajectory system
"""

from .missing_module_stubs import (
    TrajectoryPlanner,
    WindingPatternCalculator,
    RobustTurnaroundCalculator,
    PathContinuityManager,
    NonGeodesicKinematicsCalculator
)

# Re-export all classes for compatibility
__all__ = [
    'TrajectoryPlanner',
    'WindingPatternCalculator', 
    'RobustTurnaroundCalculator',
    'PathContinuityManager',
    'NonGeodesicKinematicsCalculator'
]
"""
Unified Trajectory Performance & Caching System
Provides intelligent caching, performance monitoring, and optimization for trajectory generation
"""

import time
import hashlib
import json
import pickle
from collections import defaultdict
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import asdict
import streamlit as st
import numpy as np
from pathlib import Path

from .unified_trajectory_core import TrajectoryResult
from .unified_trajectory_config import TrajectoryConfig

class TrajectoryPerformanceMonitor:
    """Monitors and tracks trajectory generation performance metrics"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.session_stats = {}
        self.start_time = None
        
    def start_generation(self, pattern_type: str, params: Dict[str, Any]):
        """Start timing a trajectory generation"""
        self.start_time = time.time()
        self.current_pattern = pattern_type
        self.current_params = params
        
    def end_generation(self, result: Optional[TrajectoryResult] = None):
        """End timing and record metrics"""
        if self.start_time is None:
            return
            
        generation_time = time.time() - self.start_time
        
        # Record basic metrics
        self.metrics[self.current_pattern].append({
            'generation_time': generation_time,
            'points_generated': len(result.points) if result and result.points else 0,
            'success': result is not None and result.points is not None,
            'timestamp': time.time()
        })
        
        # Update session stats
        if self.current_pattern not in self.session_stats:
            self.session_stats[self.current_pattern] = {
                'total_generations': 0,
                'total_time': 0.0,
                'total_points': 0,
                'successes': 0
            }
        
        stats = self.session_stats[self.current_pattern]
        stats['total_generations'] += 1
        stats['total_time'] += generation_time
        stats['total_points'] += len(result.points) if result and result.points else 0
        stats['successes'] += 1 if result is not None else 0
        
        self.start_time = None
        
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            'session_summary': {},
            'pattern_performance': {},
            'recommendations': []
        }
        
        # Session summary
        total_generations = sum(stats['total_generations'] for stats in self.session_stats.values())
        total_time = sum(stats['total_time'] for stats in self.session_stats.values())
        total_points = sum(stats['total_points'] for stats in self.session_stats.values())
        
        report['session_summary'] = {
            'total_generations': total_generations,
            'total_time': total_time,
            'total_points': total_points,
            'average_time_per_generation': total_time / max(total_generations, 1),
            'average_points_per_generation': total_points / max(total_generations, 1)
        }
        
        # Pattern-specific performance
        for pattern_type, stats in self.session_stats.items():
            if stats['total_generations'] > 0:
                report['pattern_performance'][pattern_type] = {
                    'average_time': stats['total_time'] / stats['total_generations'],
                    'average_points': stats['total_points'] / stats['total_generations'],
                    'success_rate': stats['successes'] / stats['total_generations'],
                    'total_generations': stats['total_generations']
                }
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(report)
        
        return report
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        # Check for slow patterns
        for pattern_type, perf in report['pattern_performance'].items():
            if perf['average_time'] > 5.0:  # > 5 seconds
                recommendations.append(f"Consider using 'fast' mode for {pattern_type} patterns to improve speed")
            
            if perf['average_points'] > 15000:  # Very high point density
                recommendations.append(f"High point density detected for {pattern_type}. Consider increasing decimation factor for visualization")
            
            if perf['success_rate'] < 0.9:  # Low success rate
                recommendations.append(f"Low success rate for {pattern_type}. Check parameter ranges and constraints")
        
        # Overall recommendations
        if report['session_summary']['average_time_per_generation'] > 3.0:
            recommendations.append("Consider enabling result caching to improve repeated calculations")
        
        if not recommendations:
            recommendations.append("Performance is optimal! No specific recommendations at this time.")
        
        return recommendations

class CachedTrajectoryPlanner:
    """Intelligent caching wrapper for unified trajectory planner"""
    
    def __init__(self, unified_planner, cache_size_limit: int = 100):
        self.planner = unified_planner
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_size_limit = cache_size_limit
        self.access_times = {}  # For LRU cache management
        
    def generate_trajectory(self, **params) -> Optional[TrajectoryResult]:
        """Generate trajectory with intelligent caching"""
        
        # Create cache key
        cache_key = self._create_cache_key(params)
        
        # Check cache first
        if cache_key in self.cache:
            self.cache_hits += 1
            self.access_times[cache_key] = time.time()
            
            # Add cache hit indicator to result
            cached_result = self.cache[cache_key]
            if cached_result and hasattr(cached_result, 'metadata'):
                cached_result.metadata['cache_hit'] = True
                cached_result.metadata['cache_timestamp'] = self.access_times[cache_key]
            
            return cached_result
        
        # Cache miss - generate new trajectory
        self.cache_misses += 1
        result = self.planner.generate_trajectory(**params)
        
        # Store in cache with size management
        if result is not None:
            self._add_to_cache(cache_key, result)
            
            # Add cache miss indicator
            if hasattr(result, 'metadata'):
                result.metadata['cache_hit'] = False
                result.metadata['generation_timestamp'] = time.time()
        
        return result
    
    def _create_cache_key(self, params: Dict[str, Any]) -> str:
        """Create deterministic cache key from parameters"""
        
        # Sort parameters for consistent hashing
        sorted_params = {}
        for key, value in sorted(params.items()):
            if isinstance(value, (float, np.floating)):
                # Round floats to avoid precision issues
                sorted_params[key] = round(float(value), 6)
            elif isinstance(value, (int, str, bool)):
                sorted_params[key] = value
            elif value is None:
                sorted_params[key] = None
            else:
                # Convert complex objects to string representation
                sorted_params[key] = str(value)
        
        # Create hash
        param_string = json.dumps(sorted_params, sort_keys=True)
        return hashlib.md5(param_string.encode()).hexdigest()
    
    def _add_to_cache(self, cache_key: str, result: TrajectoryResult):
        """Add result to cache with LRU eviction"""
        
        # Check size limit
        if len(self.cache) >= self.cache_size_limit:
            self._evict_lru_entry()
        
        # Add to cache
        self.cache[cache_key] = result
        self.access_times[cache_key] = time.time()
    
    def _evict_lru_entry(self):
        """Remove least recently used cache entry"""
        if not self.access_times:
            return
            
        # Find LRU entry
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        
        # Remove from cache
        del self.cache[lru_key]
        del self.access_times[lru_key]
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / max(total_requests, 1) * 100
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate_percent': hit_rate,
            'cache_size': len(self.cache),
            'cache_limit': self.cache_size_limit,
            'total_requests': total_requests
        }
    
    def clear_cache(self):
        """Clear all cached results"""
        self.cache.clear()
        self.access_times.clear()
        self.cache_hits = 0
        self.cache_misses = 0

class PerformanceOptimizer:
    """Automatically optimizes trajectory generation based on performance data"""
    
    def __init__(self, monitor: TrajectoryPerformanceMonitor):
        self.monitor = monitor
        self.optimization_history = []
        
    def suggest_config_optimization(self, current_config: TrajectoryConfig) -> Tuple[TrajectoryConfig, List[str]]:
        """Suggest configuration optimizations based on performance data"""
        
        optimized_config = TrajectoryConfig(**asdict(current_config))
        suggestions = []
        
        report = self.monitor.get_performance_report()
        avg_time = report['session_summary']['average_time_per_generation']
        avg_points = report['session_summary']['average_points_per_generation']
        
        # Optimize based on performance patterns
        if avg_time > 5.0:  # Slow generation
            if current_config.performance_mode != 'fast':
                optimized_config.performance_mode = 'fast'
                optimized_config.max_trajectory_points = min(optimized_config.max_trajectory_points, 5000)
                optimized_config.numerical_tolerance = 1e-4
                suggestions.append("Switched to fast mode for better performance")
            
            if current_config.enable_quality_validation:
                optimized_config.enable_quality_validation = False
                suggestions.append("Disabled quality validation to improve speed")
                
        elif avg_time < 1.0 and current_config.performance_mode == 'fast':  # Fast enough for better quality
            optimized_config.performance_mode = 'balanced'
            optimized_config.enable_quality_validation = True
            suggestions.append("Upgraded to balanced mode for better quality")
        
        # Optimize visualization settings
        if avg_points > 10000:
            optimized_config.default_decimation_factor = max(optimized_config.default_decimation_factor, 15)
            suggestions.append("Increased decimation factor for better visualization performance")
        
        if not suggestions:
            suggestions.append("Current configuration appears optimal for your usage patterns")
        
        return optimized_config, suggestions
    
    def auto_tune_parameters(self, pattern_type: str, base_params: Dict[str, Any]) -> Dict[str, Any]:
        """Auto-tune parameters based on historical performance"""
        
        tuned_params = base_params.copy()
        
        # Get historical data for this pattern type
        if pattern_type in self.monitor.session_stats:
            stats = self.monitor.session_stats[pattern_type]
            avg_time = stats['total_time'] / max(stats['total_generations'], 1)
            
            # Adjust parameters based on performance
            if avg_time > 3.0:  # Slow performance
                # Reduce complexity
                if 'num_layers_desired' in tuned_params:
                    tuned_params['num_layers_desired'] = min(tuned_params['num_layers_desired'], 8)
                if 'continuity_level' in tuned_params:
                    tuned_params['continuity_level'] = min(tuned_params['continuity_level'], 1)
            
            elif avg_time < 1.0:  # Fast performance, can increase quality
                if 'continuity_level' in tuned_params:
                    tuned_params['continuity_level'] = min(tuned_params['continuity_level'] + 1, 2)
        
        return tuned_params

def create_performance_dashboard():
    """Create comprehensive performance monitoring dashboard"""
    
    st.markdown("### 📊 Performance Monitoring Dashboard")
    
    # Initialize performance monitoring in session state
    if 'trajectory_monitor' not in st.session_state:
        st.session_state.trajectory_monitor = TrajectoryPerformanceMonitor()
    
    if 'trajectory_cache' not in st.session_state:
        st.session_state.trajectory_cache_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'hit_rate_percent': 0.0,
            'cache_size': 0,
            'cache_limit': 100
        }
    
    monitor = st.session_state.trajectory_monitor
    
    # Get current performance report
    report = monitor.get_performance_report()
    
    # Performance summary metrics
    st.markdown("#### 🎯 Current Session Performance")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Generations",
            report['session_summary']['total_generations']
        )
    
    with col2:
        avg_time = report['session_summary']['average_time_per_generation']
        st.metric(
            "Avg Generation Time",
            f"{avg_time:.2f}s",
            delta=f"{'🟢 Fast' if avg_time < 2.0 else '🟡 Moderate' if avg_time < 5.0 else '🔴 Slow'}"
        )
    
    with col3:
        avg_points = report['session_summary']['average_points_per_generation']
        st.metric(
            "Avg Points Generated",
            f"{avg_points:,.0f}",
            delta=f"{'🎯 Optimal' if 1000 <= avg_points <= 10000 else '⚠️ Check Settings'}"
        )
    
    with col4:
        cache_stats = st.session_state.trajectory_cache_stats
        st.metric(
            "Cache Hit Rate",
            f"{cache_stats['hit_rate_percent']:.1f}%",
            delta=f"{'🚀 Excellent' if cache_stats['hit_rate_percent'] > 50 else '📈 Building'}"
        )
    
    # Pattern-specific performance
    if report['pattern_performance']:
        st.markdown("#### 🔍 Pattern-Specific Performance")
        
        pattern_data = []
        for pattern_type, perf in report['pattern_performance'].items():
            pattern_data.append({
                "Pattern Type": pattern_type,
                "Avg Time (s)": f"{perf['average_time']:.2f}",
                "Avg Points": f"{perf['average_points']:,.0f}",
                "Success Rate": f"{perf['success_rate']:.1%}",
                "Generations": perf['total_generations']
            })
        
        if pattern_data:
            st.dataframe(pattern_data, use_container_width=True, hide_index=True)
    
    # Performance recommendations
    if report['recommendations']:
        st.markdown("#### 💡 Performance Recommendations")
        for i, recommendation in enumerate(report['recommendations']):
            if "optimal" in recommendation.lower():
                st.success(f"✅ {recommendation}")
            elif "consider" in recommendation.lower():
                st.info(f"💡 {recommendation}")
            else:
                st.warning(f"⚠️ {recommendation}")
    
    # Cache management
    with st.expander("🗄️ Cache Management", expanded=False):
        cache_stats = st.session_state.trajectory_cache_stats
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Cache Statistics:**")
            st.write(f"• Cache Size: {cache_stats['cache_size']}/{cache_stats['cache_limit']}")
            st.write(f"• Total Requests: {cache_stats['cache_hits'] + cache_stats['cache_misses']}")
            st.write(f"• Cache Hits: {cache_stats['cache_hits']}")
            st.write(f"• Cache Misses: {cache_stats['cache_misses']}")
        
        with col2:
            if st.button("🗑️ Clear Cache", help="Clear all cached trajectory results"):
                st.session_state.trajectory_cache_stats = {
                    'cache_hits': 0,
                    'cache_misses': 0,
                    'hit_rate_percent': 0.0,
                    'cache_size': 0,
                    'cache_limit': 100
                }
                st.success("Cache cleared successfully!")
                st.rerun()
            
            cache_enabled = st.checkbox(
                "Enable Intelligent Caching",
                value=True,
                help="Cache trajectory results for faster repeated calculations"
            )
            
            if cache_enabled:
                cache_limit = st.number_input(
                    "Cache Size Limit",
                    min_value=10,
                    max_value=500,
                    value=cache_stats['cache_limit'],
                    help="Maximum number of cached trajectory results"
                )
                st.session_state.trajectory_cache_stats['cache_limit'] = cache_limit
    
    return monitor

def enable_performance_optimization():
    """Enable performance optimization features globally"""
    if 'performance_optimization_enabled' not in st.session_state:
        st.session_state.performance_optimization_enabled = True
    
    if 'trajectory_monitor' not in st.session_state:
        st.session_state.trajectory_monitor = TrajectoryPerformanceMonitor()
    
    return st.session_state.performance_optimization_enabled
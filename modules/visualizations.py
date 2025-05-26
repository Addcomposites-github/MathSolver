import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from typing import Dict, List, Optional
from modules.geometry import VesselGeometry

class VesselVisualizer:
    """
    Visualization tools for composite pressure vessel design and analysis.
    """
    
    def __init__(self):
        # Set matplotlib style for engineering plots
        plt.style.use('default')
        self.colors = {
            'inner_wall': '#2E86C1',
            'outer_wall': '#1B4F72',
            'centerline': '#E74C3C',
            'trajectory': '#E67E22',
            'grid': '#BDC3C7',
            'background': '#FFFFFF'
        }
        
    def plot_vessel_profile(self, vessel: VesselGeometry, show_dimensions: bool = True) -> plt.Figure:
        """
        Plot the 2D meridian profile of the pressure vessel.
        
        Parameters:
        -----------
        vessel : VesselGeometry
            Vessel geometry object with generated profile
        show_dimensions : bool
            Whether to show dimension annotations
            
        Returns:
        --------
        plt.Figure : Matplotlib figure object
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        if vessel.profile_points is None:
            ax.text(0.5, 0.5, 'No profile data available.\nGenerate geometry first.', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=14)
            return fig
        
        # Extract profile data
        r_inner = vessel.profile_points['r_inner']
        r_outer = vessel.profile_points['r_outer']
        z = vessel.profile_points['z']
        
        # Plot inner and outer walls
        ax.plot(z, r_inner, color=self.colors['inner_wall'], linewidth=2, label='Inner wall')
        ax.plot(z, [-r for r in r_inner], color=self.colors['inner_wall'], linewidth=2)
        ax.plot(z, r_outer, color=self.colors['outer_wall'], linewidth=2, label='Outer wall')
        ax.plot(z, [-r for r in r_outer], color=self.colors['outer_wall'], linewidth=2)
        
        # Plot centerline
        z_range = [min(z), max(z)]
        ax.plot(z_range, [0, 0], '--', color=self.colors['centerline'], 
               linewidth=1, alpha=0.7, label='Centerline')
        
        # Fill the wall thickness
        ax.fill_between(z, r_inner, r_outer, alpha=0.3, color=self.colors['outer_wall'], label='Wall thickness')
        ax.fill_between(z, [-r for r in r_inner], [-r for r in r_outer], 
                       alpha=0.3, color=self.colors['outer_wall'])
        
        # Add dimension annotations if requested
        if show_dimensions:
            self._add_dimension_annotations(ax, vessel, z, r_inner, r_outer)
        
        # Formatting
        ax.set_xlabel('Axial Position Z (mm)', fontsize=12)
        ax.set_ylabel('Radial Position R (mm)', fontsize=12)
        ax.set_title(f'Pressure Vessel Profile - {vessel.dome_type} Domes', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, color=self.colors['grid'])
        ax.legend(loc='upper right')
        ax.set_aspect('equal')
        
        # Add vessel specifications text
        specs_text = self._generate_specs_text(vessel)
        ax.text(0.02, 0.98, specs_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=10, 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig
        
    def _add_dimension_annotations(self, ax, vessel: VesselGeometry, z, 
                                 r_inner, r_outer):
        """Add dimension annotations to the vessel profile plot"""
        
        # Inner diameter annotation
        max_r = max(r_inner)
        # Handle both numpy arrays and lists
        if hasattr(r_inner, 'tolist'):
            r_inner_list = r_inner.tolist()
            z_list = z.tolist()
        else:
            r_inner_list = list(r_inner)
            z_list = list(z)
        
        # Find index of max radius more safely
        max_r_idx = np.argmax(r_inner)
        z_at_max_r = z_list[max_r_idx]
        
        # Diameter line
        ax.annotate('', xy=(z_at_max_r, max_r), xytext=(z_at_max_r, -max_r),
                   arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
        ax.text(z_at_max_r + max_r * 0.1, 0, f'ID = {vessel.inner_diameter:.1f} mm', 
               rotation=90, ha='left', va='center', color='red', fontweight='bold')
        
        # Overall length annotation
        z_min, z_max = min(z), max(z)
        r_for_length = max(r_outer) * 1.2
        
        ax.annotate('', xy=(z_min, r_for_length), xytext=(z_max, r_for_length),
                   arrowprops=dict(arrowstyle='<->', color='green', lw=1.5))
        ax.text((z_min + z_max) / 2, r_for_length + max_r * 0.1, 
               f'Overall Length = {z_max - z_min:.1f} mm', 
               ha='center', va='bottom', color='green', fontweight='bold')
        
    def _generate_specs_text(self, vessel: VesselGeometry) -> str:
        """Generate specifications text for the plot"""
        specs = [
            f"Dome Type: {vessel.dome_type}",
            f"Inner Diameter: {vessel.inner_diameter:.1f} mm",
            f"Wall Thickness: {vessel.wall_thickness:.1f} mm",
            f"Cylindrical Length: {vessel.cylindrical_length:.1f} mm"
        ]
        
        if vessel.dome_type == "Isotensoid":
            specs.extend([
                f"q-factor: {vessel.q_factor:.2f}",
                f"r-factor: {vessel.r_factor:.2f}",
                f"s-factor: {vessel.s_factor:.2f}"
            ])
        elif vessel.dome_type == "Elliptical":
            specs.append(f"Aspect Ratio: {vessel.elliptical_aspect_ratio:.2f}")
        
        return '\n'.join(specs)
        
    def plot_winding_trajectory(self, vessel: VesselGeometry, trajectory_data: Dict) -> plt.Figure:
        """
        Plot filament winding trajectory on the vessel surface.
        
        Parameters:
        -----------
        vessel : VesselGeometry
            Vessel geometry object
        trajectory_data : Dict
            Trajectory calculation results
            
        Returns:
        --------
        plt.Figure : Matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        if not trajectory_data or 'path_points' not in trajectory_data:
            ax1.text(0.5, 0.5, 'No trajectory data available.\nCalculate trajectory first.', 
                    transform=ax1.transAxes, ha='center', va='center', fontsize=14)
            ax2.text(0.5, 0.5, 'No trajectory data available.\nCalculate trajectory first.', 
                    transform=ax2.transAxes, ha='center', va='center', fontsize=14)
            return fig
        
        path_points = trajectory_data['path_points']
        
        # Left plot: Meridian view (R-Z plane)
        self._plot_meridian_trajectory(ax1, vessel, path_points, trajectory_data)
        
        # Right plot: Cylindrical unwrapped view (θ-Z plane)  
        self._plot_unwrapped_trajectory(ax2, vessel, path_points, trajectory_data)
        
        plt.tight_layout()
        return fig
        
    def _plot_meridian_trajectory(self, ax, vessel: VesselGeometry, path_points: List[Dict], 
                                trajectory_data: Dict):
        """Plot trajectory in meridian view (R-Z plane)"""
        
        # First plot vessel outline
        if vessel.profile_points:
            r_inner = vessel.profile_points['r_inner']
            r_outer = vessel.profile_points['r_outer']
            z = vessel.profile_points['z']
            
            # Plot vessel outline
            ax.plot(z, r_inner, 'k-', linewidth=1, alpha=0.5, label='Vessel outline')
            ax.plot(z, [-r for r in r_inner], 'k-', linewidth=1, alpha=0.5)
        
        # Plot trajectory points
        if path_points:
            z_traj = [p['z'] for p in path_points if 'z' in p]
            r_traj = [p['r'] for p in path_points if 'r' in p]
            
            # Color by circuit number if available
            if any('circuit' in p for p in path_points):
                circuits = [p.get('circuit', 0) for p in path_points]
                scatter = ax.scatter(z_traj, r_traj, c=circuits, cmap='viridis', 
                                   s=20, alpha=0.7, label='Winding path')
                plt.colorbar(scatter, ax=ax, label='Circuit Number')
            else:
                ax.plot(z_traj, r_traj, 'o-', color=self.colors['trajectory'], 
                       markersize=3, linewidth=1, alpha=0.7, label='Winding path')
        
        ax.set_xlabel('Axial Position Z (mm)')
        ax.set_ylabel('Radial Position R (mm)')
        ax.set_title(f'Meridian View - {trajectory_data.get("pattern_type", "Unknown")} Pattern')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect('equal')
        
    def _plot_unwrapped_trajectory(self, ax, vessel: VesselGeometry, path_points: List[Dict], 
                                 trajectory_data: Dict):
        """Plot trajectory in unwrapped cylindrical view (θ-Z plane)"""
        
        if not path_points:
            return
            
        # Extract coordinates
        theta_traj = [p.get('theta', 0) for p in path_points if 'theta' in p]
        z_traj = [p['z'] for p in path_points if 'z' in p]
        
        # Convert theta to degrees for better readability
        theta_deg = [math.degrees(t) for t in theta_traj]
        
        # Color by circuit if available
        if any('circuit' in p for p in path_points):
            circuits = [p.get('circuit', 0) for p in path_points]
            scatter = ax.scatter(theta_deg, z_traj, c=circuits, cmap='viridis', 
                               s=20, alpha=0.7, label='Winding path')
            plt.colorbar(scatter, ax=ax, label='Circuit Number')
        else:
            ax.plot(theta_deg, z_traj, 'o-', color=self.colors['trajectory'], 
                   markersize=3, linewidth=1, alpha=0.7, label='Winding path')
        
        # Add cylinder boundaries
        if vessel.cylindrical_length > 0:
            ax.axhline(y=vessel.cylindrical_length/2, color='k', linestyle='--', alpha=0.5)
            ax.axhline(y=-vessel.cylindrical_length/2, color='k', linestyle='--', alpha=0.5)
            ax.text(max(theta_deg) * 0.95, vessel.cylindrical_length/2, 'Cylinder End', 
                   ha='right', va='bottom')
        
        ax.set_xlabel('Angular Position θ (degrees)')
        ax.set_ylabel('Axial Position Z (mm)')
        ax.set_title('Unwrapped Cylindrical View')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add trajectory information
        info_text = self._generate_trajectory_info(trajectory_data)
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
    def _generate_trajectory_info(self, trajectory_data: Dict) -> str:
        """Generate trajectory information text"""
        info = [
            f"Pattern: {trajectory_data.get('pattern_type', 'Unknown')}",
            f"Circuits: {trajectory_data.get('total_circuits', 'N/A')}",
            f"Winding Angle: {trajectory_data.get('winding_angle', 'N/A')}°"
        ]
        
        if 'total_fiber_length' in trajectory_data:
            info.append(f"Fiber Length: {trajectory_data['total_fiber_length']:.1f} m")
        
        if 'winding_time' in trajectory_data:
            info.append(f"Est. Time: {trajectory_data['winding_time']:.1f} min")
        
        return '\n'.join(info)
        
    def plot_stress_distribution(self, stress_results: Dict, vessel: VesselGeometry) -> plt.Figure:
        """
        Plot stress distribution in the vessel.
        
        Parameters:
        -----------
        stress_results : Dict
            Stress analysis results
        vessel : VesselGeometry
            Vessel geometry
            
        Returns:
        --------
        plt.Figure : Matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Extract stresses
        sigma_hoop = stress_results.get('hoop_stress_cyl', 0)
        sigma_axial = stress_results.get('axial_stress_cyl', 0)
        sigma_dome_max = stress_results.get('dome_stress_max', 0)
        
        # Left plot: Stress components bar chart
        stress_types = ['Hoop\n(Cylinder)', 'Axial\n(Cylinder)', 'Max\n(Dome)']
        stress_values = [sigma_hoop, sigma_axial, sigma_dome_max]
        colors = ['#3498DB', '#E74C3C', '#F39C12']
        
        bars = ax1.bar(stress_types, stress_values, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Stress (MPa)', fontsize=12)
        ax1.set_title('Stress Components', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, stress_values):
            if value != 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(stress_values)*0.01,
                        f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Right plot: Safety factor visualization
        if 'safety_factor_hoop' in stress_results or 'safety_factor_axial' in stress_results:
            sf_hoop = stress_results.get('safety_factor_hoop', 0)
            sf_axial = stress_results.get('safety_factor_axial', 0)
            sf_min = stress_results.get('safety_factor_min', min(sf_hoop, sf_axial) if sf_hoop and sf_axial else 0)
            
            sf_types = ['Hoop', 'Axial', 'Minimum']
            sf_values = [sf_hoop, sf_axial, sf_min]
            
            # Color code: green for safe (>2), yellow for marginal (1-2), red for unsafe (<1)
            sf_colors = []
            for sf in sf_values:
                if sf >= 2.0:
                    sf_colors.append('#27AE60')  # Green
                elif sf >= 1.0:
                    sf_colors.append('#F1C40F')  # Yellow
                else:
                    sf_colors.append('#E74C3C')  # Red
            
            bars2 = ax2.bar(sf_types, sf_values, color=sf_colors, alpha=0.7, edgecolor='black')
            ax2.set_ylabel('Safety Factor', fontsize=12)
            ax2.set_title('Safety Factors', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Minimum (SF=1)')
            ax2.axhline(y=2.0, color='green', linestyle='--', alpha=0.7, label='Target (SF=2)')
            
            # Add value labels
            for bar, value in zip(bars2, sf_values):
                if value > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sf_values)*0.01,
                            f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
            
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'Safety factor data\nnot available', 
                    transform=ax2.transAxes, ha='center', va='center', fontsize=14)
        
        plt.tight_layout()
        return fig
        
    def plot_material_properties_comparison(self, materials_data: List[Dict]) -> plt.Figure:
        """
        Plot comparison of material properties.
        
        Parameters:
        -----------
        materials_data : List[Dict]
            List of material property dictionaries
            
        Returns:
        --------
        plt.Figure : Matplotlib figure object
        """
        if not materials_data:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.text(0.5, 0.5, 'No material data available for comparison', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=14)
            return fig
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Extract material names and properties
        material_names = [mat.get('fiber_type', 'Unknown') + ' + ' + mat.get('resin_type', 'Unknown') 
                         for mat in materials_data]
        
        # Plot 1: Modulus comparison
        e11_values = [mat.get('E_11_longitudinal_gpa', 0) for mat in materials_data]
        e22_values = [mat.get('E_22_transverse_gpa', 0) for mat in materials_data]
        
        x = np.arange(len(material_names))
        width = 0.35
        
        ax1.bar(x - width/2, e11_values, width, label='E11 (Longitudinal)', alpha=0.7)
        ax1.bar(x + width/2, e22_values, width, label='E22 (Transverse)', alpha=0.7)
        ax1.set_ylabel('Modulus (GPa)')
        ax1.set_title('Elastic Modulus Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(material_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Strength comparison
        f1t_values = [mat.get('F_1t_longitudinal_tensile_mpa', 0) for mat in materials_data]
        f2t_values = [mat.get('F_2t_transverse_tensile_mpa', 0) for mat in materials_data]
        
        ax2.bar(x - width/2, f1t_values, width, label='F1t (Longitudinal)', alpha=0.7)
        ax2.bar(x + width/2, f2t_values, width, label='F2t (Transverse)', alpha=0.7)
        ax2.set_ylabel('Strength (MPa)')
        ax2.set_title('Tensile Strength Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(material_names, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Density comparison
        density_values = [mat.get('density_g_cm3', 0) for mat in materials_data]
        
        ax3.bar(material_names, density_values, alpha=0.7, color='orange')
        ax3.set_ylabel('Density (g/cm³)')
        ax3.set_title('Density Comparison')
        ax3.set_xticklabels(material_names, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Specific strength (strength/density)
        specific_strength = [f1t/density if density > 0 else 0 
                           for f1t, density in zip(f1t_values, density_values)]
        
        ax4.bar(material_names, specific_strength, alpha=0.7, color='green')
        ax4.set_ylabel('Specific Strength (MPa·cm³/g)')
        ax4.set_title('Specific Strength Comparison')
        ax4.set_xticklabels(material_names, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_3d_trajectory(self, trajectory_data: dict, vessel_profile_data: Optional[dict] = None, 
                          title: str = "COPV Fiber Trajectory") -> plt.Figure:
        """
        Plots the generated 3D fiber trajectory and optionally the vessel profile.

        Parameters:
        -----------
        trajectory_data : dict
            The dictionary returned by generate_geodesic_trajectory, expected to contain
            'x_points_m', 'y_points_m', 'z_points_m'.
        vessel_profile_data : Optional[dict]
            Optional: Dictionary containing the vessel's meridional profile points,
            e.g., {'r_m': array_of_radii, 'z_m': array_of_axial_coords}.
            This will be plotted by revolving it around the Z-axis.
        title : str
            The title for the plot.
            
        Returns:
        --------
        plt.Figure : Matplotlib figure object
        """
        if not trajectory_data or \
           'x_points_m' not in trajectory_data or \
           'y_points_m' not in trajectory_data or \
           'z_points_m' not in trajectory_data:
            print("Error: Trajectory data is missing required 'x_points_m', 'y_points_m', or 'z_points_m'.")
            return None

        x_path = trajectory_data['x_points_m']
        y_path = trajectory_data['y_points_m']
        z_path = trajectory_data['z_points_m']

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the fiber path
        ax.plot(x_path, y_path, z_path, label='Fiber Path', color=self.colors['trajectory'], 
                marker='o', markersize=1.5, linestyle='-', linewidth=2)

        # Optionally, plot the vessel profile if provided
        if vessel_profile_data and 'r_m' in vessel_profile_data and 'z_m' in vessel_profile_data:
            profile_r = vessel_profile_data['r_m']
            profile_z = vessel_profile_data['z_m']
            
            # Create a mesh for the revolved surface
            phi_surf = np.linspace(0, 2 * np.pi, 30)  # Azimuthal angle for surface
            Z_surf, PHI_surf = np.meshgrid(profile_z, phi_surf)
            R_surf, _ = np.meshgrid(profile_r, phi_surf) # R needs to match Z's dimension for meshgrid
            
            X_surf = R_surf * np.cos(PHI_surf)
            Y_surf = R_surf * np.sin(PHI_surf)
            
            # Plot the vessel surface
            ax.plot_surface(X_surf, Y_surf, Z_surf, alpha=0.2, color='lightgray', 
                          rstride=5, cstride=5)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (axial) (m)")
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Set limits to ensure aspect ratio is somewhat representative
        max_range = np.array([x_path.max()-x_path.min(), y_path.max()-y_path.min(), 
                             z_path.max()-z_path.min()]).max() / 2.0
        mid_x = (x_path.max()+x_path.min()) * 0.5
        mid_y = (y_path.max()+y_path.min()) * 0.5
        mid_z = (z_path.max()+z_path.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.legend()
        plt.tight_layout()
        return fig

# Import math for calculations
import math

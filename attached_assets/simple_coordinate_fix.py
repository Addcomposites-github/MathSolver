# Simple fix: Replace the _add_mandrel_surface method in your existing fixed_advanced_3d_visualizer.py

def _add_mandrel_surface(self, fig, vessel_geometry, quality_settings):
    """Add mandrel surface with proper error handling - FIXED VERSION"""
    try:
        # Get vessel profile
        profile = vessel_geometry.get_profile_points()
        if not profile or 'r_inner_mm' not in profile:
            st.error("No vessel profile data available")
            return False
        
        # Convert to meters
        z_profile_m = np.array(profile['z_mm']) / 1000.0
        r_profile_m = np.array(profile['r_inner_mm']) / 1000.0
        
        st.write(f"Original vessel profile: Z from {min(z_profile_m):.3f}m to {max(z_profile_m):.3f}m")
        st.write(f"Original vessel radius: {min(r_profile_m):.3f}m to {max(r_profile_m):.3f}m")
        
        # **CRITICAL FIX: DON'T CENTER THE VESSEL**
        # The trajectory data has already been aligned to vessel coordinates in app.py
        # Centering the vessel again will cause misalignment
        
        # REMOVE THIS CENTERING CODE:
        # z_center = (np.min(z_profile_m) + np.max(z_profile_m)) / 2
        # z_profile_m = z_profile_m - z_center
        
        # Keep original vessel coordinates
        st.write(f"Using original coordinates (no centering applied)")
        
        # Create surface mesh using original coordinates
        resolution = quality_settings.get('mandrel_resolution', 60)
        segments = quality_settings.get('surface_segments', 32)
        
        theta = np.linspace(0, 2*np.pi, segments)
        
        Z_mesh, Theta_mesh = np.meshgrid(z_profile_m, theta)
        R_mesh = np.tile(r_profile_m, (segments, 1))
        X_mesh = R_mesh * np.cos(Theta_mesh)
        Y_mesh = R_mesh * np.sin(Theta_mesh)
        
        fig.add_trace(go.Surface(
            x=X_mesh, y=Y_mesh, z=Z_mesh,
            colorscale='Greys',
            opacity=0.3,
            showscale=False,
            name='Mandrel Surface',
            hovertemplate='Mandrel Surface<br>R: %{customdata:.3f}m<extra></extra>',
            customdata=R_mesh
        ))
        
        return True
        
    except Exception as e:
        st.error(f"Mandrel surface error: {e}")
        return False
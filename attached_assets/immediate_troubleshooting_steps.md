# 🔍 Immediate Trajectory Troubleshooting Steps

## Step 1: Quick Integration (Add This to Your App)

Add this to your sidebar in `app.py`:

```python
# Add to your sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 Troubleshooting")

if st.sidebar.button("🚀 Quick Diagnostic"):
    from modules.trajectory_troubleshooting_guide import quick_trajectory_diagnostic
    quick_trajectory_diagnostic()

if st.sidebar.button("🔍 Full Diagnostic"):
    from modules.trajectory_troubleshooting_guide import run_trajectory_troubleshooting
    run_trajectory_troubleshooting()
```

## Step 2: Manual Quick Checks (Do This Now)

### 2.1 Check Your Trajectory Generation Call
In your trajectory planning page, add this debug right after calling `generate_all_layer_trajectories()`:

```python
# Add this debug code after trajectory generation
if 'all_layer_trajectories' in st.session_state:
    trajectories = st.session_state.all_layer_trajectories
    st.write(f"🔍 **Debug**: Generated {len(trajectories)} trajectories")
    
    for i, traj in enumerate(trajectories):
        st.write(f"  Trajectory {i+1}:")
        st.write(f"    Layer ID: {traj.get('layer_id', 'Unknown')}")
        st.write(f"    Layer Type: {traj.get('layer_type', 'Unknown')}")
        st.write(f"    Data Keys: {list(traj.get('trajectory_data', {}).keys())}")
        
        # Check if points exist
        traj_data = traj.get('trajectory_data', {})
        if 'points' in traj_data:
            points = traj_data['points']
            st.write(f"    Points Count: {len(points)}")
            if points:
                point = points[0]
                if hasattr(point, 'rho'):
                    st.write(f"    Sample Point: rho={point.rho:.6f}, z={point.z:.6f}")
                else:
                    st.write(f"    Point Type: {type(point)}")
        else:
            st.write(f"    ❌ NO POINTS FOUND")
```

### 2.2 Check Layer Stack Application
Add this to your layer stack page:

```python
# Check if layers are applied to mandrel
if st.button("🔍 Check Layer Application Status"):
    manager = st.session_state.layer_stack_manager
    summary = manager.get_layer_stack_summary()
    
    st.write(f"Total layers: {summary['total_layers']}")
    st.write(f"Applied to mandrel: {summary['layers_applied_to_mandrel']}")
    
    if summary['layers_applied_to_mandrel'] == 0:
        st.error("❌ NO LAYERS APPLIED TO MANDREL!")
        st.info("💡 You must apply layers to mandrel before trajectory planning")
    else:
        st.success("✅ Layers properly applied")
```

### 2.3 Test UnifiedTrajectoryPlanner Directly
Add this test to your trajectory planning page:

```python
# Test unified planner directly
if st.button("🧪 Test Unified Planner Directly"):
    try:
        from modules.unified_trajectory_planner import UnifiedTrajectoryPlanner
        
        planner = UnifiedTrajectoryPlanner(
            vessel_geometry=st.session_state.vessel_geometry,
            roving_width_m=0.003,  # 3mm
            payout_length_m=0.5,   # 500mm
            default_friction_coeff=0.1
        )
        
        result = planner.generate_trajectory(
            pattern_type='helical',
            coverage_mode='single_pass',
            physics_model='clairaut',
            continuity_level=1,
            num_layers_desired=1,
            target_params={'winding_angle_deg': 45.0},
            options={'num_points': 50}
        )
        
        if result and result.points:
            st.success(f"✅ Direct test successful: {len(result.points)} points")
            sample = result.points[0]
            st.write(f"Sample: rho={sample.rho:.6f}, z={sample.z:.6f}, phi={sample.phi:.6f}")
        else:
            st.error("❌ Direct test failed - no points generated")
            
    except Exception as e:
        st.error(f"❌ Direct test error: {e}")
```

## Step 3: Common Issues & Solutions

### Issue 1: "No Points Generated"
**Symptoms:** Trajectory data exists but contains no points
**Likely Causes:**
- UnifiedTrajectoryPlanner parameters are invalid
- Physics model incompatible with layer type
- Vessel geometry has issues

**Quick Fix:**
```python
# Try with minimal parameters
result = planner.generate_trajectory(
    pattern_type='geodesic',  # Try geodesic first
    coverage_mode='single_pass',
    physics_model='clairaut',
    continuity_level=0,  # No continuity requirements
    num_layers_desired=1,
    target_params={'winding_angle_deg': 30.0},  # Conservative angle
    options={'num_points': 20}  # Small number first
)
```

### Issue 2: "Wrong Coordinate Values"
**Symptoms:** Points generated but coordinates are wrong (too small, too large, NaN)
**Likely Causes:**
- Vessel geometry coordinate system issues
- Unit conversion problems (mm vs m)
- Physics calculation errors

**Quick Fix:**
```python
# Check vessel geometry units
vessel = st.session_state.vessel_geometry
profile = vessel.get_profile_points()
st.write("Vessel Z range (mm):", min(profile['z_mm']), "to", max(profile['z_mm']))
st.write("Vessel R range (mm):", min(profile['r_inner_mm']), "to", max(profile['r_inner_mm']))

# Check if these ranges are reasonable (should be 100s of mm typically)
```

### Issue 3: "Layer Stack Integration Failure"
**Symptoms:** Individual trajectory generation works, but multi-layer fails
**Likely Causes:**
- Layers not applied to mandrel
- Orchestrator data flow issues
- Mandrel evolution problems

**Quick Fix:**
```python
# Test with single layer first
manager = st.session_state.layer_stack_manager
if manager.layer_stack:
    # Apply first layer to mandrel if not already applied
    if len(manager.mandrel.layers_applied) == 0:
        success = manager.apply_layer_to_mandrel(0)
        st.write(f"Applied layer to mandrel: {success}")
    
    # Test trajectory for just this layer
    # [Use orchestrator._generate_single_layer_trajectory() test]
```

## Step 4: Regression Testing

Since you mentioned it worked before, try this:

### 4.1 Compare with Previous Working Version
- What changed since it last worked?
- Did you update vessel geometry parameters?
- Did you change layer definitions?
- Are you using different physics models?

### 4.2 Test with Known Good Parameters
Try these **conservative parameters** that should always work:

```python
# Known good test case
vessel: 200mm diameter, 300mm length, isotensoid domes
layer: helical, 45°, clairaut physics
roving: 3mm width, 0.125mm thickness
```

## Step 5: Enable Full Debug Mode

Add this to the top of your trajectory generation:

```python
# Enable full debug mode
import logging
logging.basicConfig(level=logging.DEBUG)

# Add debug prints to see where it fails
st.write("🔍 Starting trajectory generation...")
st.write(f"Vessel: {st.session_state.vessel_geometry.inner_diameter}mm diameter")
st.write(f"Layers: {len(st.session_state.layer_stack_manager.layer_stack)}")
```

## Step 6: If Still Stuck

**Message me with:**
1. Output from the Quick Diagnostic
2. Your vessel geometry parameters
3. Your layer definitions
4. Any error messages you see
5. What specifically is "wrong" with the generated trajectory

The systematic approach above should identify where in your pipeline the issue is occurring! 🎯
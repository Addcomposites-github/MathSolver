# Composite Pressure Vessel Design Tool

## Overview

This is a Streamlit-based engineering application for designing composite pressure vessels (COPVs). The tool provides comprehensive functionality for vessel geometry generation, material property analysis, filament winding trajectory planning, and performance calculations. The application follows a modular architecture with separate modules for different engineering disciplines.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application
- **Structure**: Multi-page application with sidebar navigation
- **Pages**: Vessel Geometry, Material Properties, Trajectory Planning, Performance Analysis, Export Results
- **Visualization**: Matplotlib-based plotting with custom engineering visualization tools
- **State Management**: Streamlit session state for maintaining data across page navigation

### Backend Architecture
- **Language**: Python 3.11
- **Structure**: Modular design with separate modules for different engineering calculations
- **Core Modules**:
  - `geometry.py`: Vessel geometry calculations and profile generation
  - `materials.py`: Material database and composite property calculations
  - `trajectories.py`: Filament winding trajectory planning
  - `calculations.py`: Stress analysis and performance calculations
  - `visualizations.py`: Engineering plotting and visualization tools

### Data Storage Solutions
- **Material Database**: Static Python dictionaries in `data/material_database.py`
- **Session Storage**: Streamlit session state for temporary data
- **No Persistent Database**: Currently uses in-memory storage only

## Key Components

### 1. Vessel Geometry Module (`modules/geometry.py`)
- **Purpose**: Generate 2D vessel profiles and calculate geometric properties
- **Features**: 
  - Multiple dome types (Isotensoid, Geodesic, Elliptical, Hemispherical)
  - Koussios qrs-parameterization for dome design
  - Profile point generation for visualization and analysis

### 2. Material Database (`modules/materials.py` + `data/material_database.py`)
- **Purpose**: Manage fiber and resin material properties
- **Features**:
  - Comprehensive material property database
  - Micromechanics calculations for composite properties
  - Support for E-Glass, S-Glass, Carbon Fiber variants
  - Resin property management

### 3. Trajectory Planning (`modules/trajectories.py`)
- **Purpose**: Calculate filament winding patterns and toolpaths
- **Features**:
  - Multiple winding patterns (Helical, Hoop, Polar, Transitional)
  - Geodesic and non-geodesic trajectory calculations
  - Winding angle optimization

### 4. Engineering Calculations (`modules/calculations.py`)
- **Purpose**: Perform stress analysis and failure predictions
- **Features**:
  - Classical pressure vessel stress calculations
  - Multiple failure criteria (Max Stress, Max Strain, Tsai-Hill, Tsai-Wu)
  - Performance metrics calculation

### 5. Visualization Tools (`modules/visualizations.py`)
- **Purpose**: Generate engineering plots and visualizations
- **Features**:
  - 2D vessel profile plotting
  - Trajectory visualization
  - Stress distribution plots
  - Engineering-style formatting and annotations

## Data Flow

1. **Input Phase**: User defines vessel requirements through Streamlit interface
2. **Geometry Generation**: VesselGeometry class generates vessel profile based on parameters
3. **Material Selection**: MaterialDatabase provides composite properties based on fiber/resin selection
4. **Trajectory Planning**: TrajectoryPlanner calculates winding patterns for the vessel geometry
5. **Analysis**: VesselCalculations performs stress analysis using geometry and material data
6. **Visualization**: VesselVisualizer generates plots and charts for results
7. **Export**: Results can be exported (functionality to be implemented)

## External Dependencies

### Core Dependencies
- **Streamlit**: Web application framework for the user interface
- **NumPy**: Numerical computing for engineering calculations
- **Matplotlib**: Plotting and visualization
- **Pandas**: Data manipulation and analysis (minimal usage currently)

### System Dependencies (via Nix)
- **Python 3.11**: Core runtime environment
- **Graphics Libraries**: Cairo, FreeType, Ghostscript for matplotlib rendering
- **GUI Libraries**: GTK3 for potential GUI components
- **Media Processing**: FFmpeg for potential video export features

## Deployment Strategy

### Development Environment
- **Platform**: Replit with Nix package management
- **Configuration**: Python 3.11 module with extensive graphics support
- **Port**: Application runs on port 5000

### Production Deployment
- **Target**: Autoscale deployment on Replit infrastructure
- **Command**: `streamlit run app.py --server.port 5000`
- **Configuration**: Headless server mode with light theme

### Key Architectural Decisions

1. **Modular Design**: Separated engineering disciplines into distinct modules for maintainability and testing
   - **Rationale**: Complex engineering calculations require clear separation of concerns
   - **Benefit**: Easier testing, debugging, and future enhancements

2. **Streamlit Framework**: Chosen for rapid prototyping and engineering visualization
   - **Rationale**: Allows quick development of interactive engineering tools
   - **Trade-off**: Less customization than full web frameworks, but much faster development

3. **In-Memory Data Storage**: Currently uses session state and static data
   - **Rationale**: Simplifies deployment and reduces infrastructure requirements
   - **Limitation**: No persistence between sessions, may need database integration later

4. **Matplotlib Visualization**: Selected for engineering-quality plots
   - **Rationale**: Industry standard for technical plotting with extensive customization
   - **Benefit**: Publication-quality graphics suitable for engineering reports

5. **Static Material Database**: Material properties stored as Python dictionaries
   - **Rationale**: Simplifies access patterns and reduces database complexity
   - **Trade-off**: Limited flexibility for user-defined materials, but sufficient for current scope
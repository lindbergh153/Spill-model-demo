# Spill-model-demo
A demo of 3D deepwater oil spill model
# DWOSM: Deep Water Oil Spill Model

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive numerical framework for simulating deep water oil spill transport and fate, integrating near-field plume dynamics with far-field advection-diffusion processes.

## Features

### Near-Field Model (NFM)
- Lagrangian integral plume model for buoyant multiphase jets
- Peng-Robinson equation of state for gas/liquid equilibrium
- SINTEF and Wang et al. droplet size distribution models
- Particle tracking with dissolution and separation

### Far-Field Model (FFM)
- Single Parcel Model (SPM) for transport simulation
- 4D spatiotemporal interpolation using RegularGridInterpolator
- Oil weathering: evaporation, emulsification, dispersion
- Multiprocessing for ensemble simulations
- Shoreline interaction detection

## Model Architecture

```
DWOSM
├── Near-Field Model (NFM)
│   ├── Plume dynamics (plume_model.py)
│   ├── DSD model (DSDmodel.py)
│   └── Particle tracking (particle_API.py)
│
└── Far-Field Model (FFM)
    ├── Transport (SPM.py, SPM_functions.py)
    ├── Weathering (weathering_functions.py)
    └── Ensemble simulation (FFM.py)
```


### Dependencies
- numpy, scipy, pandas, matplotlib
- netCDF4, shapely, geopandas
- numba (for JIT compilation)
- oil_library (NOAA ADIOS database)

## Project Structure

```
DWOSM/
├── dwosm/
│   ├── DWOSM_API.py      # Main API
│   ├── NFM.py            # Near-field interface
│   ├── FFM.py            # Far-field interface
│   ├── plume_model.py    # Lagrangian plume
│   ├── SPM.py            # Single Parcel Model
│   ├── DSDmodel.py       # Droplet size distribution
│   ├── particle.py       # Fluid particle classes
│   ├── seawater.py       # Thermodynamic properties
│   └── weathering_functions.py
└── README.md
```

## Validation

Model validated against:
- Deepwater Horizon field observations
- SINTEF Tower Basin experiments
- DeepSpill field experiment (2000)

## References

- Johansen, Ø. (2003). Development and verification of deep-water blowout models. Marine Pollution Bulletin.
- Socolofsky, S.A., et al. (2011). Intercomparison of oil spill prediction models. Marine Pollution Bulletin.
- Gill, A.E. (1982). Atmosphere-Ocean Dynamics. Academic Press.

## Author

Zhaoyang Yang, PhD  
Concordia University, Montreal  
zhaoyang.yang@concordia.ca

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments
- Marine Pollution Research Initiative (MPRI)

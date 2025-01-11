# Laser Diode Simulator

A comprehensive Python-based simulator for semiconductor laser diodes, implementing rate equations to analyze various operating conditions and dynamic responses.

## Features

- DC Characteristics Analysis
  - Carrier and photon density calculations
  - Power-Current (P-I) curves
  - Temperature dependence modeling

- Dynamic Response Analysis
  - Step response simulation
  - Ramp current response
  - Small-signal modulation
  - Frequency response curves

- Advanced Analysis
  - Phase modulation effects
  - Chirp analysis
  - Super-Gaussian pulse response
  - Multiple operating point analysis

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/laser-simulator.git
cd laser-simulator
```

2. Install required packages:
```bash
pip install numpy scipy matplotlib tqdm
```

## Usage

### Basic Operation
```python
from laser_model import LaserModel, SimulationConfig, PhysicsConstants

# Initialize model
config = SimulationConfig()
physics = PhysicsConstants()
laser = LaserModel(config, physics)

# Run simulations
laser.perform_dc_analysis()
```

### Running Complete Analysis
```python
python main.py
```

## Project Structure

- `main.py`: Entry point and simulation orchestration
- `laser_model.py`: Core laser simulation model
- `/plots`: Generated analysis plots

## Model Parameters

Key parameters can be adjusted in the `SimulationConfig` and `PhysicsConstants` classes:

- Cavity length: 250 μm
- Active region width: 2 μm
- Active region thickness: 0.2 μm
- Confinement factor: 0.3
- Wavelength: 1.3 μm

## Generated Plots

The simulation produces several plots:
- DC characteristics
- Transient responses
- Frequency response curves
- Phase modulation analysis
- Temperature dependence
- Power-Current characteristics

## License

This project is licensed under the MIT License

## Acknowledgments

Based on semiconductor laser rate equations and physical models from:
- "Physics of Semiconductor Lasers" by G.P. Agrawal
- "Semiconductor Optoelectronics" by P.S. Zory

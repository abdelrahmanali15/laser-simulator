# Project Overview
This project analyzes and simulates laser diode behavior using rate equations. It covers:
- DC performance calculations
- Transient analysis (step and ramp currents)
- Small-signal frequency response

## How It Works
- part1.py uses rate equations to compute carrier and photon densities under varying current conditions.
  - DC analysis finds steady-state densities.
  - Transient analysis studies laser behavior under step and ramp current changes.
- part2.py focuses on small-signal frequency response by solving the rate equations for different frequencies.

## Usage Instructions
1. Run part1.py to generate DC and transient plots.
   - DC analysis: executes solve_ivp for a range of currents to build characteristic curves.
   - Transient analysis: solves rate equations over time steps or ramped currents.
2. Run part2.py to analyze small-signal behavior.
   - Sweeps over a range of modulation frequencies to measure response amplitude.

## Installation
1. Clone or download this repository.
2. Install Python dependencies: NumPy, SciPy, Matplotlib.

## Contributing
Open issues and pull requests are welcome.

## License
Pending.

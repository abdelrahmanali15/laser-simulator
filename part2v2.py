"""
This module simulates the small signal frequency response of a laser model using rate equations.
Classes:
  SimulationConfig: Configuration settings for the simulation.
  PhysicsConstants: Physical constants and parameters for the laser model.
  LaserModel: Laser model that includes methods for calculating steady state, single frequency response, and plotting.
Functions:
  main: Main function to run the simulation and plot results.
Methods in LaserModel:
  __init__(self, config, physics): Initializes the laser model with given configuration and physical constants.
  rate_equations_dc(self, t, y, I_dc): Defines the rate equations for DC bias current.
  rate_equations_ac(self, t, y, freq, I_dc, I_ac_amplitude): Defines the rate equations for AC modulation.
  calculate_steady_state(self, I_dc): Calculates the steady state carrier and photon densities for a given DC current.
  calculate_single_frequency(self, params): Calculates the response for a single frequency.
  analyze_current(self, I_dc=None, I_ac=None): Analyzes the frequency response for given DC and AC currents.
  plot_specific_frequency(self, freq, I_dc=None, I_ac=None): Analyzes and plots the response at a specific frequency.
"""
import numpy as np
from scipy.integrate import solve_ivp
import concurrent.futures
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from scipy.interpolate import interp1d
from tqdm import tqdm


class SimulationConfig:
    def __init__(self):
        # Time settings
        self.T_STEADY = 10e-9  # Time to reach steady state [s]
        self.T_SIM = 40e-9  # Total simulation time [s]
        self.STEADY_POINTS = 750  # Points for steady-state calculation

        # Frequency analysis settings
        self.FREQ_MIN = 1e8  # Minimum frequency [Hz]
        self.FREQ_MAX = 1e10  # Maximum frequency [Hz]
        self.FREQ_POINTS = 150  # Number of frequency points

        # Current settings
        self.I_DC = 50e-3  # Default DC bias current [A]
        self.I_AC = 1e-3  # Default AC modulation amplitude [A]

        # Solver settings
        self.N_THREADS = 8
        self.MIN_POINTS_PER_PERIOD = 20
        self.MIN_TOTAL_POINTS = 0
        self.SOLVER_RTOL = 1e-9
        self.SOLVER_ATOL = 1e-12

        # Initial conditions
        self.N0 = 2e18  # Initial carrier density [cm^-3]
        self.S0 = 3.5e15  # Initial photon density [cm^-3]

        # Analysis configurations
        self.DC_CURRENTS = [10e-3, 30e-3, 50e-3, 80e-3]  # DC currents [A]
        self.AC_CURRENTS = [0.1e-3, 0.5e-3, 0.8e-3, 1e-3]  # AC currents [A]

        # interpolation settings
        self.INTERP_POINTS = 1000  # Number of points for interpolation


class PhysicsConstants:
    def __init__(self):
        self.c_0 = 3e10  # Speed of light [cm/s]
        self.n_g = 4  # Group index
        self.v_g = self.c_0 / self.n_g  # Group velocity [cm/s]
        self.L = 250e-4  # Length [cm]
        self.w = 2e-4  # Width [cm]
        self.d = 0.2e-4  # Thickness [cm]
        self.Gamma = 0.3  # Confinement factor
        self.a_gain = 2.5e-16  # Gain cross-section [cm^2]
        self.N_tr = 1e18  # Transparency carrier density [cm^-3]
        self.A_nr = 1e8  # Non-radiative recombination [s^-1]
        self.B = 1e-10  # Radiative recombination [cm^3/s]
        self.C = 3e-29  # Auger recombination [cm^6/s]
        self.beta_sp = 1e-4  # Spontaneous emission factor
        self.tau_p = 1.6e-12  # Photon lifetime [s]
        self.q = 1.602e-19  # Electron charge [C]
        self.Vact = self.L * self.w * self.d  # Active region volume [cm^3]


class LaserModel:
    """
    A class to model the behavior of a laser using rate equations.
    Attributes:
      config (object): Configuration parameters for the laser model.
      physics (object): Physical constants and parameters for the laser model.
    Methods:
      __init__(config, physics):
        Initializes the LaserModel with configuration and physical parameters.
      rate_equations_dc(t, y, I_dc):
        Computes the rate equations for the carrier and photon densities under DC conditions.
      rate_equations_ac(t, y, freq, I_dc, I_ac_amplitude):
        Computes the rate equations for the carrier and photon densities under AC conditions.
      calculate_steady_state(I_dc):
        Calculates the steady-state carrier and photon densities for a given DC current.
      calculate_single_frequency(params):
        Calculates the laser response for a single frequency.
      analyze_current(I_dc=None, I_ac=None):
        Analyzes the frequency response of the laser for given DC and AC currents.
      plot_specific_frequency(freq, I_dc=None, I_ac=None):
        Analyzes and plots the laser response at a specific frequency.
    """

    def __init__(self, config, physics):
        self.config = config
        self.physics = physics

    def rate_equations_dc(self, t, y, I_dc):
        N, S = y
        Rtot = (self.physics.A_nr * N + self.physics.B * N ** 2 +
                self.physics.C * N ** 3)
        G = (self.physics.Gamma * self.physics.v_g * self.physics.a_gain *
             max(N - self.physics.N_tr, 0))

        dNdt = ((I_dc / (self.physics.q * self.physics.Vact)) - Rtot - G * S)
        dSdt = (G * S - (S / self.physics.tau_p) +
                self.physics.beta_sp * self.physics.B * N ** 2)
        return [dNdt, dSdt]

    def rate_equations_ac(self, t, y, freq, I_dc, I_ac_amplitude):
        N, S = y
        I = I_dc + I_ac_amplitude * np.sin(2 * np.pi * freq * t)
        return self.rate_equations_dc(t, y, I)

    def calculate_steady_state(self, I_dc):
        """Calculate DC steady state for given current"""
        t_steady = np.linspace(0, self.config.T_STEADY,
                               self.config.STEADY_POINTS)

        sol_dc = solve_ivp(
            lambda t, y: self.rate_equations_dc(t, y, I_dc),
            [0, self.config.T_STEADY],
            [self.config.N0, self.config.S0],
            t_eval=t_steady,
            method='Radau',
            rtol=self.config.SOLVER_RTOL,
            atol=self.config.SOLVER_ATOL
        )

        return sol_dc.y[0][-1], sol_dc.y[1][-1]  # N_dc, S_dc

    def calculate_single_frequency(self, params):
        """Calculate response for a single frequency"""
        freq, N_dc, S_dc, S_tot, I_dc, I_ac_amplitude = params

        # Calculate dynamic T_sim to accommodate multiple cycles
        cycles_required = 20
        T_sim = cycles_required / freq
        T_sim = T_sim * 2  # Add margin for transients

        # Adjust number of points based on frequency
        num_points = max(self.config.MIN_TOTAL_POINTS,
                         int(freq * T_sim * self.config.MIN_POINTS_PER_PERIOD))

        # Create time points
        t_sim = np.linspace(0, T_sim, num_points)

        # print(f"Processing frequency: {freq:.2e} Hz, T_sim: {T_sim:.2e} s")

        sol_ac = solve_ivp(
            lambda t, y: self.rate_equations_ac(
                t, y, freq, I_dc, I_ac_amplitude),
            [0, T_sim],
            [N_dc, S_dc],
            t_eval=t_sim,
            method='Radau',
            rtol=self.config.SOLVER_RTOL,
            atol=self.config.SOLVER_ATOL
        )

        # Analyze last few cycles
        cycles_to_analyze = 5
        points_per_cycle = num_points / (cycles_required * 2)
        last_n_points = int(points_per_cycle * cycles_to_analyze)

        S_last_portion = sol_ac.y[1][-last_n_points:]
        S_ac = np.max(S_last_portion)
        S_amplitude = abs(S_ac - S_dc)
        response = S_amplitude / (S_tot - S_dc)

        return freq, response, sol_ac

    def analyze_current(self, I_dc=None, I_ac=None):
        """Analyze frequency response for given currents"""
        I_dc = I_dc if I_dc is not None else self.config.I_DC
        I_ac = I_ac if I_ac is not None else self.config.I_AC

        # Calculate DC steady state
        N_dc, S_dc = self.calculate_steady_state(I_dc)
        _, S_tot = self.calculate_steady_state(I_dc + I_ac)

        # Frequency analysis parameters
        f_ac = np.logspace(
            np.log10(self.config.FREQ_MIN),
            np.log10(self.config.FREQ_MAX),
            self.config.FREQ_POINTS
        )

        params_list = [(freq, N_dc, S_dc, S_tot, I_dc, I_ac) for freq in f_ac]

        results_dict = {}

        # Calculate frequency response with progress bar
        with ProcessPoolExecutor(max_workers=self.config.N_THREADS) as executor:
            futures = [executor.submit(
                self.calculate_single_frequency, params) for params in params_list]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Frequency Sweep"):
                try:
                    freq, response, _ = future.result()
                    results_dict[freq] = response
                except Exception as e:
                    print(f"Error processing frequency: {str(e)}")

        frequencies = np.array(sorted(results_dict.keys()))
        response_amp = np.array([results_dict[f] for f in frequencies])

        return frequencies, response_amp

    def plot_specific_frequency(self, freq, I_dc=None, I_ac=None):
        """Analyze and plot response at a specific frequency"""
        I_dc = I_dc if I_dc is not None else self.config.I_DC
        I_ac = I_ac if I_ac is not None else self.config.I_AC

        # Calculate DC steady state
        N_dc, S_dc = self.calculate_steady_state(I_dc)
        _, S_tot = self.calculate_steady_state(I_dc + I_ac)

        # Calculate response at specific frequency
        _, _, sol_ac = self.calculate_single_frequency(
            (freq, N_dc, S_dc, S_tot, I_dc, I_ac)
        )

        # Create figure with two y-axes
        fig, ax1 = plt.subplots(figsize=(12, 8))
        ax2 = ax1.twinx()

        # Plot carrier density
        ln1 = ax1.plot(sol_ac.t * 1e9, sol_ac.y[0] - N_dc, 'b-',
                       label="Carrier Density", linewidth=2)
        ax1.set_xlabel("Time (ns)")
        ax1.set_ylabel("ΔN (cm⁻³)", color='b')
        ax1.tick_params(axis='y', labelcolor='b')

        # Plot photon density
        ln2 = ax2.plot(sol_ac.t * 1e9, sol_ac.y[1] - S_dc, 'r-',
                       label="Photon Density", linewidth=2)
        ax2.set_ylabel("ΔS (cm⁻³)", color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        # Add legend
        lns = ln1 + ln2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc='upper right')

        plt.title(f"Laser Response at {freq / 1e9:.2f} GHz")
        plt.tight_layout()


def main():
    """
    Main function to initialize the simulation configuration and physics constants,
    calculate the DC steady state, and analyze the frequency response for different
    DC and AC currents. It also plots the transient response for specific frequencies.
    Steps:
    1. Initialize configuration and physics constants.
    2. Calculate and print the DC steady state carrier and photon densities.
    3. Plot the small signal frequency response for multiple DC currents with interpolation.
    4. Plot the small signal frequency response for multiple AC currents with interpolation.
    5. Plot the transient response for specific frequencies.
    Plots:
    - Small Signal Frequency Response vs. DC Current
    - Small Signal Frequency Response vs. AC Current
    - Transient response for specific frequencies
    Saves plots as:
    - 'plots/AC_multiple_large_signal.png'
    - 'plots/AC_multiple_small_signal.png'
    """
    # Initialize configuration and physics
    config = SimulationConfig()
    physics = PhysicsConstants()
    laser = LaserModel(config, physics)

    # Calculate DC steady state
    print("Calculating DC steady state...")
    N_dc, S_dc = laser.calculate_steady_state(config.I_DC)
    print(f"Carrier density = {N_dc:.2e} cm⁻³")
    print(f"Photon density = {S_dc:.2e} cm⁻³")

    # Plot frequency response for different DC currents with interpolation
    print("\nAnalyzing multiple DC currents...")
    plt.figure(figsize=(12, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, I_dc_value in enumerate(config.DC_CURRENTS):
        print(f"\nAnalyzing DC current: {I_dc_value * 1000:.1f} mA")
        freq_result, response_result = laser.analyze_current(I_dc=I_dc_value)

        # Create interpolation function
        response_db = 20 * np.log10(response_result)
        interp_func = interp1d(freq_result, response_db, kind='cubic')

        # Create smoother frequency points for plotting
        freq_smooth = np.logspace(
            np.log10(freq_result[0]),
            np.log10(freq_result[-1]),
            config.INTERP_POINTS
        )
        response_smooth = interp_func(freq_smooth)

        plt.semilogx(freq_smooth, response_smooth,
                     color=colors[i],
                     linewidth=2,
                     label=f'I_DC = {I_dc_value * 1000:.1f} mA')

    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.grid(True, which="major", ls="-", alpha=0.5)
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Response (dB)', fontsize=12)
    plt.title('Small Signal Frequency Response vs. DC Current',
              fontsize=14, pad=20)
    plt.legend(title='Bias Current', title_fontsize=12, fontsize=10,
               loc='lower left', bbox_to_anchor=(0.02, 0.02))
    plt.tight_layout()
    plt.savefig('plots/AC_multiple_large_signal.png',
                dpi=300, bbox_inches='tight')

    # Plot frequency response for different AC currents with interpolation
    print("\nAnalyzing multiple AC currents...")
    plt.figure(figsize=(12, 8))

    for i, I_ac_value in enumerate(config.AC_CURRENTS):
        print(f"\nAnalyzing AC current: {I_ac_value * 1000:.1f} mA")
        freq_result, response_result = laser.analyze_current(I_ac=I_ac_value)

        # Create interpolation function
        response_db = 20 * np.log10(response_result)
        interp_func = interp1d(freq_result, response_db, kind='cubic')

        # Create smoother frequency points for plotting
        freq_smooth = np.logspace(
            np.log10(freq_result[0]),
            np.log10(freq_result[-1]),
            config.INTERP_POINTS
        )
        response_smooth = interp_func(freq_smooth)

        plt.semilogx(freq_smooth, response_smooth,
                     color=colors[i],
                     linewidth=2,
                     label=f'I_AC = {I_ac_value * 1000:.1f} mA')

    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.grid(True, which="major", ls="-", alpha=0.5)
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Response (dB)', fontsize=12)
    plt.title('Small Signal Frequency Response vs. AC Current',
              fontsize=14, pad=20)
    plt.legend(title='AC Current', title_fontsize=12, fontsize=10,
               loc='lower left', bbox_to_anchor=(0.02, 0.02))
    plt.tight_layout()
    plt.savefig('plots/AC_multiple_small_signal.png',
                dpi=300, bbox_inches='tight')

    # Plot specific frequencies transient response
    test_frequencies = [1.3e8, 1e9, 5e9, 11e9]
    print("\nAnalyzing specific frequencies...")
    for freq in test_frequencies:
        print(f"\nAnalyzing frequency: {freq / 1e9:.1f} GHz")
        laser.plot_specific_frequency(freq)
    plt.show()


if __name__ == "__main__":
    import os
    os.makedirs('plots', exist_ok=True)
    main()

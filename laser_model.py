import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.constants import h
from scipy.interpolate import interp1d
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
from tqdm import tqdm
import sys
from scipy.signal import correlate
from scipy.signal import savgol_filter
from scipy.integrate import cumtrapz


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
        self.lambda_laser = 1.3e-6  # Wavelength [m]
        # Optical frequency [Hz]
        self.nu = self.c_0 / (self.lambda_laser * 1e2)
        self.I_max = 50e-3  # Maximum current [A]
        self.beta_c = 5  # Added for phase modulation
        self.G_n = 5.62e3  # Added for phase modulation


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
        self.DC_CURRENTS = [10e-3, 15e-3, 30e-3,
                            50e-3, 150e-3]  # DC currents [A]
        self.AC_CURRENTS = [0.1e-3, 0.5e-3, 0.8e-3, 1e-3]  # AC currents [A]

        # interpolation settings
        self.INTERP_POINTS = 1000  # Number of points for interpolation


class LaserModel:
    def __init__(self, config, physics):
        self.config = config
        self.physics = physics

    def rate_equations(self, t, y, t_current, I_current, N_tr_temp=None):
        N_tr = self.physics.N_tr if N_tr_temp is None else N_tr_temp
        N, S = y
        I = np.interp(t, t_current, I_current)
        Rtot = self.physics.A_nr * N + self.physics.B * N**2 + self.physics.C * N**3
        G = self.physics.Gamma * self.physics.v_g * \
            self.physics.a_gain * max(N - N_tr, 0)
        dNdt = (I / (self.physics.q * self.physics.Vact)) - Rtot - G * S
        dSdt = G * S - (S / self.physics.tau_p) + \
            self.physics.beta_sp * self.physics.B * N**2
        return [dNdt, dSdt]

    def rate_equations_dc(self, t, y, I_dc):
        N, S = y
        Rtot = (self.physics.A_nr * N + self.physics.B * N ** 2 +
                self.physics.C * N ** 3)
        G = (self.physics.Gamma * self.physics.v_g * self.physics.a_gain *
             max(N - self.physics.N_tr, 0)) / (1 + 1e-18 * S)
        dNdt = ((I_dc / (self.physics.q * self.physics.Vact)) - Rtot - G * S)
        dSdt = (G * S - (S / self.physics.tau_p) +
                self.physics.beta_sp * self.physics.B * N ** 2)
        return [dNdt, dSdt]

    def rate_equations_ac(self, t, y, freq, I_dc, I_ac_amplitude):
        N, S = y
        I = I_dc + I_ac_amplitude * np.sin(2 * np.pi * freq * t)
        return self.rate_equations_dc(t, y, I)

    def perform_dc_analysis(self, N_tr_temp=None):
        I_dc = np.linspace(0, self.physics.I_max, 100)
        N_dc = []
        S_dc = []
        T_steady = 20e-9
        t_eval = np.array([0, T_steady])
        N0 = 0
        S0 = 0
        for I in I_dc:
            t_current = np.array([0, T_steady])
            I_current = np.array([I, I])
            sol = solve_ivp(
                self.rate_equations,
                [0, T_steady],
                [N0, S0],
                args=(t_current, I_current, N_tr_temp),
                method='BDF',
                t_eval=t_eval,
                rtol=1e-6,
                atol=1e-10
            )
            N_dc.append(sol.y[0, -1])
            S_dc.append(sol.y[1, -1])
        return I_dc, np.array(N_dc), np.array(S_dc)

    def simulate_step_response(self):
        """
        Simulates laser response to step current inputs.

        Returns:
            tuple: (time array, carrier density solutions, photon density solutions, current values)
        """
        T_end = 20e-9
        num_points = 500
        t = np.linspace(0, T_end, num_points)
        # I_values = np.array([0, 10e-3, 15e-3, 20e-3, self.physics.I_max])
        I_values = np.array([15e-3, 20e-3,  self.physics.I_max])
        N_solutions = np.zeros((len(I_values), len(t)))
        S_solutions = np.zeros((len(I_values), len(t)))
        N0 = 0
        S0 = 0
        for i, I in enumerate(I_values):
            t_current = np.array([0, T_end])
            I_current = np.array([I, I])
            sol = solve_ivp(
                self.rate_equations,
                [0, T_end],
                [N0, S0],
                args=(t_current, I_current),
                method='BDF',
                t_eval=t,
                rtol=1e-6,
                atol=1e-10
            )
            N_solutions[i, :] = sol.y[0]
            S_solutions[i, :] = sol.y[1]
        return t, N_solutions, S_solutions, I_values

    def simulate_ramp_response(self):
        """
        Simulates laser response to a ramped current input.

        Returns:
            tuple: (time array, carrier density, photon density, time steps, current steps)
        """
        T_end = 20e-9
        num_points = 500
        t = np.linspace(0, T_end, num_points)
        I_steps = np.linspace(0, self.physics.I_max, 8)
        t_steps = np.linspace(0, T_end, 8)
        N0 = 0
        S0 = 0
        sol = solve_ivp(
            self.rate_equations,
            [0, T_end],
            [N0, S0],
            args=(t_steps, I_steps),
            method='BDF',
            t_eval=t,
            rtol=1e-6,
            atol=1e-10
        )
        return t, sol.y[0], sol.y[1], t_steps, I_steps

    def calculate_steady_state(self, I_dc):
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
        return sol_dc.y[0][-1], sol_dc.y[1][-1]

    def simulate_single_frequency_response(self, params):
        """
        Simulates a single modulation frequency and returns the response.

        Args:
            params (tuple): (freq, N_dc, S_dc, S_tot, I_dc, I_ac_amplitude)

        Returns:
            tuple: (frequency, response, solver_result)
        """
        freq, N_dc, S_dc, S_tot, I_dc, I_ac_amplitude = params
        cycles_required = 20
        T_sim = cycles_required / freq
        T_sim = T_sim * 2
        num_points = max(self.config.MIN_TOTAL_POINTS,
                         int(freq * T_sim * self.config.MIN_POINTS_PER_PERIOD))
        t_sim = np.linspace(0, T_sim, num_points)
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
        cycles_to_analyze = 5
        points_per_cycle = num_points / (cycles_required * 2)
        last_n_points = int(points_per_cycle * cycles_to_analyze)
        S_last_portion = sol_ac.y[1][-last_n_points:]
        S_ac = np.max(S_last_portion)
        S_amplitude = abs(S_ac - S_dc)
        response = S_amplitude / (S_tot - S_dc)
        return freq, response, sol_ac

    def sweep_small_signal_response(self, I_dc=None, I_ac=None):
        """
        Performs a small-signal frequency sweep.

        Args:
            I_dc (float, optional): DC current
            I_ac (float, optional): AC amplitude

        Returns:
            tuple: (frequencies, response_amplitudes)
        """
        I_dc = I_dc if I_dc is not None else self.config.I_DC
        I_ac = I_ac if I_ac is not None else self.config.I_AC
        N_dc, S_dc = self.calculate_steady_state(I_dc)
        _, S_tot = self.calculate_steady_state(I_dc + I_ac)
        f_ac = np.logspace(
            np.log10(self.config.FREQ_MIN),
            np.log10(self.config.FREQ_MAX),
            self.config.FREQ_POINTS
        )
        params_list = [(freq, N_dc, S_dc, S_tot, I_dc, I_ac) for freq in f_ac]
        results_dict = {}
        with ProcessPoolExecutor(max_workers=self.config.N_THREADS) as executor:
            futures = [executor.submit(
                self.simulate_single_frequency_response, params) for params in params_list]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Frequency Sweep"):
                try:
                    freq, response, _ = future.result()
                    results_dict[freq] = response
                except Exception as e:
                    print(f"Error processing frequency: {str(e)}")
        frequencies = np.array(sorted(results_dict.keys()))
        response_amp = np.array([results_dict[f] for f in frequencies])
        return frequencies, response_amp

    def simulate_phase_modulation(self, I_dc, I_ac, freq):
        """
        Simulates phase modulation response of the laser for given DC and AC currents.

        Args:
            I_dc (float): DC bias current in Amperes
            I_ac (list): List of AC modulation amplitudes in Amperes
            freq (float): Modulation frequency in Hz

        Generates plots showing power spectra for different AC currents.
        """
        # 1) Generate time array
        cycles_required = 200
        T_sim = (cycles_required / freq) * 2
        num_points = 5000
        t_sim = np.linspace(0, T_sim, num_points)

        # Create figure with subplots for each current value
        fig, axs = plt.subplots(len(I_ac), 1, figsize=(10, 3*len(I_ac)))

        selected_color_map = 'plasma'
        colors = plt.cm.get_cmap(selected_color_map)(
            np.linspace(0, 1, len(I_ac)))

        for idx, I in enumerate(I_ac):
            print(f"Analyzing I_ac = {I * 1e3:.1f} mA")
            # 2) Solve DC-only
            sol0 = solve_ivp(
                lambda t, y: self.rate_equations_dc(t, y, I_dc),
                [0, T_sim],
                [self.config.N0, self.config.S0],
                t_eval=t_sim,
                method='Radau',
                rtol=self.config.SOLVER_RTOL,
                atol=self.config.SOLVER_ATOL
            )
            N_dc_only = sol0.y[0]
            S_dc_only = sol0.y[1]

            # 3) Solve with DC+AC
            sol = solve_ivp(
                lambda t, y: self.rate_equations_ac(t, y, freq, I_dc, I),
                [0, T_sim],
                [self.config.N0, self.config.S0],
                t_eval=t_sim,
                method='Radau',
                rtol=self.config.SOLVER_RTOL,
                atol=self.config.SOLVER_ATOL
            )
            N_ac = sol.y[0]
            S_ac = sol.y[1]

            # 4) Compute deltaN(t) and deltaPhi(t)
            deltaN = N_ac - N_dc_only
            t = t_sim[len(deltaN)//2:]
            deltaN = deltaN[len(deltaN)//2:]
            dt = t_sim[1] - t_sim[0]

            delta_phi = cumtrapz(
                0.5 * self.physics.beta_c * self.physics.G_n * deltaN * self.physics.Vact * 1.5,
                t,
                initial=0
            )
            delta_phi = np.mod(delta_phi, 2 * np.pi)  # Wrap to [0, 2π)

            # 5) E(t) calculations
            S_half = S_ac[len(S_ac)//2:]
            delta_phi_half = delta_phi

            # Apply smoothing filter
            S_half = savgol_filter(S_half, window_length=51, polyorder=3)
            delta_phi = savgol_filter(
                delta_phi_half, window_length=51, polyorder=3)

            E_t = np.sqrt(S_half) * np.exp(1j * delta_phi_half)

            # Calculate autocorrelation
            autocorr = correlate(E_t, E_t, mode="full")
            autocorr = autocorr[len(autocorr) // 2:]
            signal_power = np.sum(np.abs(E_t)**2)
            autocorr = autocorr / signal_power

            # 6) FFT of autocorrelation
            E_freq = np.abs(np.fft.fftshift(
                np.fft.fft(autocorr, n=2**18)) / len(autocorr))

            # 7) Define FFT frequency axis
            freq_vector = np.fft.fftshift(np.fft.fftfreq(2**18, d=dt))

            # Plot in the corresponding subplot
            self.plot_optical_spectrum(
                axs[idx], freq_vector, E_freq, I, colors[idx])

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig("plots/phase_modulation_power_spectra.png",
                    dpi=300, bbox_inches='tight')
        # plt.close()

    def analyze_frequency_response(self, freq, I_dc=None, I_ac=None):
        """
        Analyzes laser response at a specific frequency.

        Args:
            freq (float): Frequency to analyze in Hz
            I_dc (float, optional): DC bias current in Amperes
            I_ac (float, optional): AC modulation amplitude in Amperes

        Generates plots showing carrier and photon density variations.
        """
        # renamed from plot_specific_frequency
        I_dc = I_dc if I_dc is not None else self.config.I_DC
        I_ac = I_ac if I_ac is not None else self.config.I_AC
        N_dc, S_dc = self.calculate_steady_state(I_dc)
        _, S_tot = self.calculate_steady_state(I_dc + I_ac)
        _, _, sol_ac = self.simulate_single_frequency_response(
            (freq, N_dc, S_dc, S_tot, I_dc, I_ac)
        )
        fig, ax1 = plt.subplots(figsize=(12, 8))
        ax2 = ax1.twinx()
        ln1 = ax1.plot(sol_ac.t * 1e9, sol_ac.y[0] - N_dc, 'b-',
                       label="Carrier Density", linewidth=2)
        ax1.set_xlabel("Time (ns)")
        ax1.set_ylabel("ΔN (cm⁻³)", color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ln2 = ax2.plot(sol_ac.t * 1e9, sol_ac.y[1] - S_dc, 'r-',
                       label="Photon Density", linewidth=2)
        ax2.set_ylabel("ΔS (cm⁻³)", color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        lns = ln1 + ln2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc='upper right')
        plt.title(f"Laser Response at {freq / 1e9:.2f} GHz")
        plt.tight_layout()

    def plot_ramp_response(self, t, N, S, t_steps, I_steps):
        """
        Plots the laser response to a ramped current input.

        Args:
            t (ndarray): Time points
            N (ndarray): Carrier density values
            S (ndarray): Photon density values
            t_steps (ndarray): Time points for current steps
            I_steps (ndarray): Current values at each step
        """
        plt.figure(figsize=(12, 8))
        plt.suptitle('Transient Response for Ramp current', fontsize=16)
        plt.subplot(3, 1, 1)
        plt.plot(t_steps * 1e9, I_steps * 1e3, 'r.-', linewidth=2)
        plt.xlabel('Time (ns)')
        plt.ylabel('Current (mA)')
        plt.grid(True)
        plt.subplot(3, 1, 2)
        plt.plot(t * 1e9, N / self.physics.N_tr, 'b-', linewidth=2)
        plt.xlabel('Time (ns)')
        plt.ylabel('Carrier Density (N/N$_{tr}$)')
        plt.grid(True)
        plt.subplot(3, 1, 3)
        plt.plot(t * 1e9, S, 'g-', linewidth=2)
        plt.xlabel('Time (ns)')
        plt.ylabel('Photon Density (cm$^{-3}$)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('plots/transient_response_ramp.png',
                    dpi=300, bbox_inches='tight')

    def plot_step_response(self, t, N_solutions, S_solutions, I_values):
        """
        Plots the laser response to step current inputs.

        Args:
            t (ndarray): Time points
            N_solutions (ndarray): Carrier density solutions for each current
            S_solutions (ndarray): Photon density solutions for each current
            I_values (ndarray): Current values used
        """
        colors = plt.cm.viridis(np.linspace(0, 1, len(I_values)))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle('Transient Response for Different Step Current Values',
                     fontsize=16, y=0.95)
        for i in range(len(I_values)):
            ax1.plot(t * 1e9, N_solutions[i, :] / self.physics.N_tr, color=colors[i],
                     linewidth=2, label=f'I = {I_values[i]*1e3:.1f} mA')
        ax1.set_xlabel('Time (ns)', fontsize=12)
        ax1.set_ylabel('Carrier Density (N/N$_{tr}$)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax1.set_title('Carrier Density Evolution', fontsize=14)
        for i in range(len(I_values)):
            ax2.plot(t * 1e9, S_solutions[i, :], color=colors[i],
                     linewidth=2, label=f'I = {I_values[i]*1e3:.1f} mA')
        ax2.set_xlabel('Time (ns)', fontsize=12)
        ax2.set_ylabel('Photon Density (cm$^{-3}$)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax2.set_title('Photon Density Evolution', fontsize=14)
        plt.tight_layout(rect=[0, 0, 0.85, 0.95])
        plt.savefig('plots/transient_response_step.png',
                    dpi=300, bbox_inches='tight')

    def calculate_optical_power(self, S):
        """
        Calculates optical output power from photon density.

        Args:
            S (float or ndarray): Photon density in cm^-3

        Returns:
            float or ndarray: Optical power in Watts
        """
        R = self.physics.Gamma**2  # Mirror reflectivity
        P_out = -np.log(R) * (self.physics.v_g * self.physics.Vact /
                              (2 * self.physics.L)) * h * self.physics.nu * S
        return P_out

    def plot_PI_curve(self, P_solutions, I_values):
        """
        Plots the power-current (P-I) characteristic curve.

        Args:
            P_solutions (ndarray): Power values
            I_values (ndarray): Current values
        """
        plt.figure(figsize=(10, 6))
        P_steady = P_solutions
        plt.plot(I_values * 1e3, P_steady * 1e3, 'b-o', linewidth=2)
        plt.xlabel('Current (mA)', fontsize=12)
        plt.ylabel('Output Power (mW)', fontsize=12)
        plt.title('Power-Current (P-I) Characteristic', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('plots/power_current_characteristic.png',
                    dpi=300, bbox_inches='tight')

    def plot_optical_spectrum(self, ax, freqs, fft_magnitude, I_ac, color):
        """
        Plots normalized optical power spectrum.

        Args:
            ax (matplotlib.axes.Axes): Axes to plot on
            freqs (ndarray): Frequency points
            fft_magnitude (ndarray): FFT magnitude values
            I_ac (float): AC current amplitude
            color (str): Color for the plot
        """
        ax.plot(freqs / 1e9, fft_magnitude / np.max(fft_magnitude), '-',
                color=color, linewidth=1.5)
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Power (Normalized)")
        ax.set_title(f"$I_p$ = {I_ac * 1e3:.1f} mA")
        ax.set_xlim([-4, 4])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        # Set the background to light gray for better contrast
        ax.set_facecolor('#f8f8f8')

    def plot_dc_characteristics(self, I_dc, N_dc, S_dc, N_tr_temp=None):
        N_tr = self.physics.N_tr if N_tr_temp is None else N_tr_temp
        plt.figure(figsize=(12, 8))
        plt.suptitle('DC Characteristics', fontsize=16, y=1.02)
        plt.subplot(2, 1, 1)
        plt.plot(I_dc * 1e3, N_dc / N_tr, 'b-', linewidth=2)
        plt.xlabel('Current (mA)', fontsize=12)
        plt.ylabel('Carrier Density (N/N$_{tr}$)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.title('Carrier Density vs Current', fontsize=14)
        plt.subplot(2, 1, 2)
        plt.plot(I_dc * 1e3, S_dc, 'g-', linewidth=2)
        plt.xlabel('Current (mA)', fontsize=12)
        plt.ylabel('Photon Density (cm$^{-3}$)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.title('Photon Density vs Current', fontsize=14)
        plt.tight_layout()
        plt.savefig('plots/dc_characteristics.png',
                    dpi=300, bbox_inches='tight')


def main():
    """
    Main function to run laser simulations and generate analysis plots.
    Performs DC, AC, transient, and phase modulation analysis.
    """
    config = SimulationConfig()
    physics = PhysicsConstants()
    laser = LaserModel(config, physics)

    print("Calculating DC steady state...")
    N_dc, S_dc = laser.calculate_steady_state(config.I_DC)
    print(f"Carrier density = {N_dc:.2e} cm⁻³")
    print(f"Photon density = {S_dc:.2e} cm⁻³")

    print("\nAC: Analyzing multiple DC currents...")
    plt.figure(figsize=(12, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, I_dc_value in enumerate(config.DC_CURRENTS):
        print(f"\nAnalyzing DC current: {I_dc_value * 1000:.1f} mA")
        freq_result, response_result = laser.sweep_small_signal_response(
            I_dc=I_dc_value)
        response_db = 20 * np.log10(response_result)
        interp_func = interp1d(freq_result, response_db, kind='cubic')
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

    print("\nAC: Analyzing multiple AC currents...")
    plt.figure(figsize=(12, 8))

    for i, I_ac_value in enumerate(config.AC_CURRENTS):
        print(f"\nAnalyzing AC current: {I_ac_value * 1000:.1f} mA")
        freq_result, response_result = laser.sweep_small_signal_response(
            I_ac=I_ac_value)
        response_db = 20 * np.log10(response_result)
        interp_func = interp1d(freq_result, response_db, kind='cubic')
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

    test_frequencies = [1.3e8, 1e9, 5e9, 11e9]
    print("\nSmall signal transient of specific frequencies...")
    for freq in test_frequencies:
        print(f"\nAnalyzing frequency: {freq / 1e9:.1f} GHz")
        laser.analyze_frequency_response(freq)
    # plt.show()

    print(f"\nAnalyzing DC Charactaristics...")
    I_dc, N_dc, S_dc = laser.perform_dc_analysis()
    laser.plot_dc_characteristics(I_dc, N_dc, S_dc)

    print(f"\nAnalyzing PI Charactaristics...")
    P_out = laser.calculate_optical_power(S_dc)
    laser.plot_PI_curve(P_out, I_dc)

    plt.figure(figsize=(12, 8))
    plt.suptitle('Temperature Dependence of Output Power', fontsize=16)
    T = np.linspace(200, 350, 6)
    N_tr0 = 1e18

    for i, temp in enumerate(T):
        N_tr = N_tr0 * np.exp((temp-300)/50)
        I_dc, N_dc, S_dc = laser.perform_dc_analysis(N_tr)
        P_out = laser.calculate_optical_power(S_dc)
        plt.plot(I_dc * 1e3, P_out * 1e3,
                 label=f'T = {temp:.0f} K',
                 linewidth=2)

    plt.xlabel("Current (mA)", fontsize=12)
    plt.ylabel("Output Power (mW)", fontsize=12)
    plt.title("Output Power vs Current at Different Temperatures")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', fontsize=10, framealpha=0.9,
               title='Temperature', title_fontsize=12)
    plt.tight_layout()
    plt.savefig('plots/figure4_temperature_dependencies.png',
                dpi=300, bbox_inches='tight')

    print(f"\nSimulating Step Response...")
    t, solutions_N, solutions_S, I_values = laser.simulate_step_response()
    laser.plot_step_response(t, solutions_N, solutions_S, I_values)

    print(f"\nSimulating Ramp Response...")
    t, N, S, t_steps, I_steps = laser.simulate_ramp_response()
    laser.plot_ramp_response(t, N, S, t_steps, I_steps)

    print(f"\nAnalyzing Phase Modulation...")
    # Parameters
    I_dc = 30e-3
    freq = 1e9
    I_ac = [2e-3, 4e-3, 6e-3, 8e-3]

    # Calculate and plot
    laser.simulate_phase_modulation(I_dc, I_ac, freq)

    plt.show()


def test_phase_modulation_power_spectra(laser, config):
    # Parameters
    I_dc = 30e-3
    freq = 1e9
    I_ac = [2e-3, 4e-3, 6e-3, 8e-3]

    # Calculate and plot
    laser.simulate_phase_modulation(I_dc, I_ac, freq)


def test_dc_steady_state(laser, config):
    print("Testing DC steady state calculation...")
    N_dc, S_dc = laser.calculate_steady_state(config.I_DC)
    print(f"Carrier density = {N_dc:.2e} cm⁻³")
    print(f"Photon density = {S_dc:.2e} cm⁻³")


def test_ac_analysis(laser, config):
    print("Testing AC analysis for a single DC current...")
    I_dc_value = config.DC_CURRENTS[3]
    freq_result, response_result = laser.sweep_small_signal_response(
        I_dc=I_dc_value)
    response_db = 20 * np.log10(response_result)
    interp_func = interp1d(freq_result, response_db, kind='cubic')
    freq_smooth = np.logspace(
        np.log10(freq_result[0]),
        np.log10(freq_result[-1]),
        config.INTERP_POINTS
    )
    response_smooth = interp_func(freq_smooth)
    plt.semilogx(freq_smooth, response_smooth, linewidth=2)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.grid(True, which="major", ls="-", alpha=0.5)
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Response (dB)', fontsize=12)
    plt.title('Small Signal Frequency Response for Single DC Current',
              fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('plots/test_AC_single_dc_current.png',
                dpi=300, bbox_inches='tight')


def test_specific_frequency(laser):
    print("Testing small signal transient for a specific frequency...")
    test_frequencies = [1.3e8, 1e9, 5e9, 11e9]
    for freq in test_frequencies:
        print(f"\nAnalyzing frequency: {freq / 1e9:.1f} GHz")
        laser.analyze_frequency_response(freq)


def test_dc_analysis(laser):
    print("Testing DC analysis...")
    I_dc, N_dc, S_dc = laser.perform_dc_analysis()
    laser.plot_dc_characteristics(I_dc, N_dc, S_dc)


def test_power_vs_current(laser):
    print("Testing power vs current calculation...")
    I_dc, N_dc, S_dc = laser.perform_dc_analysis()
    P_out = laser.calculate_optical_power(S_dc)
    laser.plot_PI_curve(P_out, I_dc)


def test_transient_analysis_step(laser):
    print("Testing transient analysis for step current...")
    t, solutions_N, solutions_S, I_values = laser.simulate_step_response()
    laser.plot_step_response(t, solutions_N, solutions_S, I_values)


def test_transient_analysis_ramp(laser):
    print("Testing transient analysis for ramp current...")
    t, N, S, t_steps, I_steps = laser.simulate_ramp_response()
    laser.plot_ramp_response(t, N, S, t_steps, I_steps)


# def main():
#     config = SimulationConfig()
#     physics = PhysicsConstants()
#     laser = LaserModel(config, physics)

#     # test_dc_steady_state(laser, config)
#     # test_ac_analysis(laser, config)
#     # test_specific_frequency(laser)
#     test_dc_analysis(laser)
#     # test_power_vs_current(laser)
#     # test_transient_analysis_step(laser)
#     # test_transient_analysis_ramp(laser)
#     # test_phase_modulation_power_spectra(laser, config)
#     plt.show()

if __name__ == "__main__":
    import os
    os.makedirs('plots', exist_ok=True)
    main()

# Description: This file contains the laser model class and its methods for simulating the laser response to different inputs.

from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.constants import h
from scipy.interpolate import interp1d
from scipy.signal import correlate
from scipy.signal import savgol_filter
from scipy.integrate import cumtrapz
from scipy.signal import correlate
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import hilbert


class PhysicsConstants:
    def __init__(self):
        self.c_0 = 3e10  # Speed of light [cm/s]
        self.ng = 4  # Group index
        self.v_g = self.c_0 / self.ng  # Group velocity [cm/s]
        self.L = 250e-4  # Length [cm]
        self.w = 2e-4  # Width [cm]
        self.d = 0.2e-4  # Thickness [cm]
        self.gamma = 0.3  # Confinement factor
        self.R = ((self.ng - 1) / (self.ng + 1))**2  # Mirror reflectivity
        self.gain_coff = 2.5e-16
        self.N_tr = 1e18  # Transparency carrier density [cm^-3]
        self.A_nr = 1e8  # Non-radiative recombination [s^-1]
        self.B = 1e-10  # Radiative recombination [cm^3/s]
        self.C = 3e-29  # Auger recombination [cm^6/s]
        self.beta_sp = 1e-4  # Spontaneous emission factor
        self.tau_p = 1.6e-12  # Photon lifetime [s]
        self.q = 1.602e-19  # Electron charge [C]
        self.volume = self.L * self.w * self.d  # Active region volume [cm^3]
        self.lambda_laser = 1.3e-6  # Wavelength [m]
        # Optical frequency [Hz]
        self.nu = self.c_0 / (self.lambda_laser * 1e2)
        self.I_max = 50e-3  # Maximum current [A]
        self.beta_c = 5  # line width enhancement factor
        # https://www.fiberoptics4sale.com/blogs/wave-optics/transient-response-of-semiconductor-lasers
        self.G_n = self.gamma * self.v_g * self.gain_coff / self.volume  # Gain coefficient


class SimulationConfig:
    def __init__(self):
        # Time settings
        self.T_STEADY = 10e-9  # Time to reach steady state [s]
        self.T_SIM = 40e-9  # Total simulation time [s]
        self.STEADY_POINTS = 750  # Points for steady-state calculation

        # Frequency analysis settings
        self.FREQ_MIN = 1e8  # Minimum frequency [Hz]
        self.FREQ_MAX = 1e10  # Maximum frequency [Hz]
        self.FREQ_POINTS = 250  # Number of frequency points

        # Current settings
        self.I_DC = 30e-3  # Default DC bias current [A]
        self.I_AC = 0.5e-3  # Default AC modulation amplitude [A]

        # Solver settings
        self.N_THREADS = 8
        self.MIN_POINTS_PER_PERIOD = 20
        self.MIN_TOTAL_POINTS = 0
        self.SOLVER_RTOL = 1e-9
        self.SOLVER_ATOL = 1e-12

        # Initial conditions
        self.N0 = 2e15  # Initial carrier density [cm^-3]
        self.S0 = 2.59e5  # Initial photon density [cm^-3]

        # Analysis configurations
        self.DC_CURRENTS = [25e-3, 50e-3, 80e-3]  # DC currents [A]
        # self.AC_CURRENTS = [0.1e-3, 2e-3, 4e-3, 6e-3]  # AC currents [A]
        self.AC_CURRENTS = [0.1e-3, 6e-3]  # AC currents [A]

        # interpolation settings
        self.INTERP_POINTS = 1000  # Number of points for interpolation


class LaserModel:
    def __init__(self, config, physics):
        self.config = config
        self.physics = physics

    def rate_equations(self, t, y, I_dc,  N_tr_temp=None):
        """
        Computes the derivative of carrier and photon densities under DC current.
        """
        N_tr = self.physics.N_tr if N_tr_temp is None else N_tr_temp
        N, S = y

        # Total recombination rate
        R_tot = (
            self.physics.A_nr * N
            + self.physics.B * N**2
            + self.physics.C * N**3
        )

        # Gain calculation
        G = (
            self.physics.gamma
            * self.physics.v_g
            * self.physics.gain_coff
            * (N - N_tr)
        )

        # Saturation factor
        epsilon = 3.5e-18
        G /= (1 + epsilon * S)

        # Carrier density derivative
        dNdt = (
            (I_dc / (self.physics.q * self.physics.volume))
            - R_tot
            - G * S
        )

        # Photon density derivative
        dSdt = (
            G * S
            - (S / self.physics.tau_p)
            + self.physics.beta_sp * self.physics.B * N**2
        )

        return [dNdt, dSdt]

    def rate_equations_ramp(self, t, y, t_current, I_current, N_tr_temp=None):
        N, S = y
        I = np.interp(t, t_current, I_current)
        return self.rate_equations(t, y, I, N_tr_temp)

    def rate_equations_ac(self, t, y, freq, I_dc, I_ac_amplitude):
        N, S = y
        I = I_dc + I_ac_amplitude * np.real(np.exp(1j * 2 * np.pi * freq * t))
        return self.rate_equations(t, y, I)

    def rate_equations_pulse(self, t, y,  I_dc, I_ac_amplitude, tau, m):
        N, S = y
        I = I_dc + I_ac_amplitude * np.exp(-((t / tau)**(2 * m)))
        return self.rate_equations(t, y, I)

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
                self.rate_equations_ramp,
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
        """
        T_end = 20e-9
        num_points = 500
        t = np.linspace(0, T_end, num_points)
        # I_values = np.array([0, 10e-3, 15e-3, 20e-3, self.physics.I_max])
        I_values = np.array([15e-3, 20e-3,  self.physics.I_max])
        N_solutions = np.zeros((len(I_values), len(t)))
        S_solutions = np.zeros((len(I_values), len(t)))
        N0 = self.config.N0
        S0 = self.config.S0
        for i, I in enumerate(I_values):
            t_current = np.array([0, T_end])
            I_current = np.array([I, I])
            sol = solve_ivp(
                self.rate_equations_ramp,
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
        """
        T_end = 20e-9
        num_points = 500
        t = np.linspace(0, T_end, num_points)
        I_steps = np.linspace(0, self.physics.I_max, 8)
        t_steps = np.linspace(0, T_end, 8)
        N0 = 0
        S0 = 0
        sol = solve_ivp(
            self.rate_equations_ramp,
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
            lambda t, y: self.rate_equations(t, y, I_dc),
            [0, self.config.T_STEADY],
            [self.config.N0, self.config.S0],
            t_eval=t_steady,
            method='BDF',
            rtol=self.config.SOLVER_RTOL,
            atol=self.config.SOLVER_ATOL
        )
        return sol_dc.y[0][-1], sol_dc.y[1][-1]

    def sweep_small_signal_response(self, I_dc=None, I_ac=None, method='fft'):
        """
        Performs a small-signal frequency sweep using multiprocessing for faster computation,
        """
        I_dc = I_dc if I_dc is not None else self.config.I_DC
        I_ac = I_ac if I_ac is not None else self.config.I_AC
        N_dc, S_dc = self.calculate_steady_state(I_dc)
        print(f"DC Carrier Density: {N_dc:.2e} cm⁻³")
        print(f"DC Photon Density: {S_dc:.2e} cm⁻³")

        _, S_tot = self.calculate_steady_state(I_dc + I_ac)
        f_ac = np.logspace(
            np.log10(self.config.FREQ_MIN),
            np.log10(self.config.FREQ_MAX),
            self.config.FREQ_POINTS
        )
        params_list = [(freq, N_dc, S_dc, S_tot, I_dc, I_ac, method)
                       for freq in f_ac]
        results_dict = {}
        with ProcessPoolExecutor(max_workers=self.config.N_THREADS) as executor:
            futures = [
                executor.submit(
                    self.simulate_single_frequency_response, params)
                for params in params_list
            ]
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Frequency Sweep"
            ):
                try:
                    freq, response_S, response_N, _ = future.result()
                    results_dict[freq] = (response_S, response_N)
                except Exception as e:
                    print(f"Error processing frequency: {str(e)}")
        frequencies = np.array(sorted(results_dict.keys()))
        response_amp_S = np.array([results_dict[f][0] for f in frequencies])
        response_amp_N = np.array([results_dict[f][1] for f in frequencies])
        return frequencies, response_amp_S, response_amp_N

    def simulate_single_frequency_response(self, params):
        """
        Simulates a single modulation frequency response using different methods:

        - "fft": Uses FFT at the modulation frequency.
          Advantages: Captures linear response accurately, handles noise well, large frequency range.
          Disadvantages: can suffer from leakage.
          Best for: when linearity is assumed.

        - "peak2peak": Uses the peak-to-peak value in the time domain.
          Advantages: clearly reflects time-domain extremes.
          Disadvantages: Sensitive to any transient overshoot.
          Best for: Quick estimation of amplitude.

        - "hilbert": Uses the Hilbert transform for the envelope and doubles the mean amplitude.
          Advantages: Offers a smooth envelope of the signal, good for non-linear signals.

        """
        freq, N_dc, S_dc, S_tot, I_dc, I_ac_amplitude, method = params
        cycles_required = 25
        T_sim = (cycles_required / freq) * 2
        num_points = max(
            self.config.MIN_TOTAL_POINTS,
            int(freq * T_sim * self.config.MIN_POINTS_PER_PERIOD)
        )
        num_points = 1000
        t_sim = np.linspace(0, T_sim, num_points)

        sol_ac = solve_ivp(
            lambda t, y: self.rate_equations_ac(
                t, y, freq, I_dc, I_ac_amplitude),
            [0, T_sim],
            [N_dc, S_dc],
            t_eval=t_sim,
            method='BDF',
            rtol=self.config.SOLVER_RTOL,
            atol=self.config.SOLVER_ATOL
        )

        # Last portion for analysis
        cycles_to_analyze = 10
        points_per_cycle = num_points / (cycles_required * 2)
        last_n_points = int(points_per_cycle * cycles_to_analyze)
        S_last = sol_ac.y[1][-last_n_points:]
        N_last = sol_ac.y[0][-last_n_points:]
        t_last = t_sim[-last_n_points:]

        # Select method to compute response amplitude
        if method == 'peak2peak':
            response_S = np.ptp(S_last)
            response_N = np.ptp(N_last)
        elif method == 'hilbert':
            analytic_signal_s = hilbert(S_last - np.mean(S_last))
            analytic_signal_n = hilbert(N_last - np.mean(N_last))
            response_S = 2 * np.mean(np.abs(analytic_signal_s))
            response_N = 2 * np.mean(np.abs(analytic_signal_n))
        else:
            window = np.hanning(len(S_last))
            S_windowed = (S_last - np.mean(S_last)) * window
            fft_values_s = fft(S_windowed)
            fft_freqs_s = fftfreq(len(S_windowed), d=(t_last[1] - t_last[0]))
            window_corr = 2 / np.mean(window)
            fft_mag_s = np.abs(fft_values_s) * \
                (2.0 / len(S_windowed)) * window_corr
            freq_idx_s = np.argmin(np.abs(fft_freqs_s - freq))
            response_S = fft_mag_s[freq_idx_s]

            N_windowed = (N_last - np.mean(N_last)) * window
            fft_values_n = fft(N_windowed)
            fft_freqs_n = fftfreq(len(N_windowed), d=(t_last[1] - t_last[0]))
            fft_mag_n = np.abs(fft_values_n) * \
                (2.0 / len(N_windowed)) * window_corr
            freq_idx_n = np.argmin(np.abs(fft_freqs_n - freq))
            response_N = fft_mag_n[freq_idx_n]

        return freq, response_S, response_N, sol_ac

    def simulate_phase_modulation(self, I_dc, I_ac, freq):
        """
        Simulates phase modulation response of the laser for given DC and AC currents.
        Generates plots showing power spectra for different AC currents.
        δ(dφ/dt) = (1/2) * β_c * G_N * δN (linear model)
        G_n = Γ * v_g * gain_coff / V
        Ref: https://www.fiberoptics4sale.com/blogs/wave-optics/modulation-response-of-semiconductor-lasers
        """
        # 1) Generate time array
        cycles_required = 150
        T_sim = (cycles_required / freq) * 2
        num_points = 5000
        t_sim = np.linspace(0, T_sim, num_points)

        # Create figure with subplots for each current value
        fig, axs = plt.subplots(len(I_ac), 1, figsize=(10, 3*len(I_ac)))

        selected_color_map = 'plasma'
        colors = plt.colormaps.get_cmap(selected_color_map)(
            np.linspace(0, 1, len(I_ac)))

        for idx, I in enumerate(I_ac):
            print(f"Analyzing I_ac = {I * 1e3:.1f} mA")
            # 2) Solve DC-only
            sol0 = solve_ivp(
                lambda t, y: self.rate_equations(t, y, I_dc),
                [0, T_sim],
                [self.config.N0, self.config.S0],
                t_eval=t_sim,
                method='BDF',
                rtol=self.config.SOLVER_RTOL,
                atol=self.config.SOLVER_ATOL
            )
            N_dc_only = sol0.y[0]

            # 3) Solve with DC+AC
            sol = solve_ivp(
                lambda t, y: self.rate_equations_ac(t, y, freq, I_dc, I),
                [0, T_sim],
                [self.config.N0, self.config.S0],
                t_eval=t_sim,
                method='BDF',
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
                0.5 * self.physics.beta_c * self.physics.G_n * deltaN * self.physics.volume,
                t,
                initial=0
            )
            # delta_phi = np.mod(delta_phi, 2 * np.pi)  # Wrap to [0, 2π)

            # 5) E(t) calculations
            S_half = S_ac[len(S_ac)//2:]
            delta_phi_half = delta_phi

            # Apply smoothing filter
            S_half = savgol_filter(S_half, window_length=51, polyorder=3)
            delta_phi = savgol_filter(
                delta_phi_half, window_length=51, polyorder=3)

            E_t = np.sqrt(S_half) * np.exp(1j * delta_phi_half)
            E_t_real = np.real(E_t)
            E_t_imag = np.imag(E_t)

            # Calculate autocorrelation
            autocorr = correlate(E_t, E_t, mode="full")
            autocorr = autocorr[len(autocorr) // 2:]
            signal_power = np.sum(np.abs(E_t)**2)
            autocorr = autocorr / signal_power

            # 6) FFT of autocorrelation
            E_freq = np.abs(fftshift(fft(autocorr, n=2**17)) / len(autocorr))

            # 7) Define FFT frequency axis
            freq_vector = fftshift(fftfreq(2**17, d=dt))

            # Plot in the corresponding subplot
            self.plot_optical_spectrum(
                axs[idx], freq_vector, E_freq, I, colors[idx])

        # Adjust layout and save
        axs[3].set_xlabel("Frequency (GHz)")
        plt.tight_layout()
        plt.savefig("plots/phase_modulation_power_spectra.png",
                    dpi=300, bbox_inches='tight')

    def simulate_supergaussian_pulse_chirping(self, pulse_duration=5e-9, m=1, I0=10e-3, chirp=True):
        """
        Simulates the effect of chirp in time domain with a super-Gaussian pulse.
        Plots input pulse vs output field E(t), its autocorrelation, and chirp vs time.

        Ref: https://www.fiberoptics4sale.com/blogs/wave-optics/modulation-response-of-semiconductor-lasers
        """
        I_dc = 0
        tau = pulse_duration / (2 * np.log(2)**(1/(2*m))) * 2
        t = np.linspace(-5 * pulse_duration, 5 * pulse_duration, 100000)
        I_t = I_dc + I0 * np.exp(-((t / tau)**(2 * m)))

        sol_dc = solve_ivp(
            lambda ts, y: self.rate_equations(ts, y, I_dc),
            [t[0], t[-1]],
            [self.config.N0, self.config.S0],
            t_eval=t,
            method='BDF',
            rtol=1e-8,
            atol=1e-12
        )
        N_dc_array = sol_dc.y[0]
        N_dc = sol_dc.y[0][-1]
        S_dc = sol_dc.y[1][-1]

        sol = solve_ivp(
            lambda ts, y: self.rate_equations_pulse(ts, y, I_dc, I0, tau, m),
            [t[0], t[-1]],
            [N_dc, S_dc],
            t_eval=t,
            method='BDF',
            rtol=1e-10,
            atol=1e-12
        )
        N = sol.y[0]
        S = sol.y[1]

        # Calculate field with chirp
        deltaN = N - N_dc_array[-1]
        if chirp:
            delta_phi = cumtrapz(
                0.003 * self.physics.beta_c * self.physics.G_n * deltaN * self.physics.volume, t, initial=0
            )
        else:
            delta_phi = np.zeros_like(t)

        E_t = np.sqrt(S) * np.real(np.exp(1j * delta_phi))
        autocorr = correlate(E_t, E_t, mode="full")
        autocorr = autocorr[len(autocorr)//2:]
        autocorr /= np.sum(np.abs(E_t)**2)

        # Plotting
        fig, axs = plt.subplots(2, 1, figsize=(8, 12))
        axs[0].plot(t * 1e9, I_t * 1e3, 'b-')
        axs[0].set_title("Super-Gaussian Pulse Input (mA)")
        axs[0].set_ylabel("Current (mA)")

        axs[1].plot(t * 1e9, E_t / np.max(E_t), 'r-')
        axs[1].set_title("Output Field Real E(t) Normalized")
        axs[1].set_ylabel("Field (normalized)")
        axs[1].legend(["Chirped" if chirp else "Unchirped"])

        for ax in axs:
            ax.grid(True)
            ax.set_facecolor('#f8f8f8')

        plt.tight_layout()

        if chirp:
            plt.savefig("plots/super_gaussian_pulse_with_chirp.png",
                        dpi=300, bbox_inches='tight')
        else:
            plt.savefig("plots/super_gaussian_pulse_without_chirp.png",
                        dpi=300, bbox_inches='tight')

    def simulate_gain_switched_pulse(self, pulse_duration=200e-12, m=1, chirp=True):
        """
        Simulates gain switching of a laser with a 200-ps current pulse 3 times the threshold.
        Plots the pulse shape, output field E(t), and frequency chirp.

        Ref: https://www.fiberoptics4sale.com/blogs/wave-optics/modulation-response-of-semiconductor-lasers
        """
        # Define the current pulse
        t = np.linspace(-5 * pulse_duration, 5 * pulse_duration, 100000)
        I_th = self.calculate_threshold_current(I_dc=self.config.I_DC)
        I0 = 3 * I_th  # 3 times threshold
        # Gaussian pulse width
        tau = pulse_duration / (2 * np.log(2)**(1/(2*m)))
        # Gaussian pulse current (3 times threshold)
        I_pulse = I0 * np.exp(-(t / tau) ** 2)

        # Solve for DC steady-state
        I_dc = I_th
        sol_dc = solve_ivp(
            lambda ts, y: self.rate_equations(ts, y, I_dc),
            [t[0], t[-1]],
            [self.config.N0, self.config.S0],
            t_eval=t,
            method='BDF',
            rtol=1e-8,
            atol=1e-12
        )
        N_dc_array = sol_dc.y[0]
        N_dc = sol_dc.y[0][-1]
        S_dc = sol_dc.y[1][-1]
        print(f"DC Carrier density = {N_dc:.2e} cm⁻³")
        print(f"DC Photon density = {S_dc:.2e} cm⁻³")

        # Solve rate equations with the current pulse
        sol = solve_ivp(
            lambda ts, y: self.rate_equations_pulse(ts, y, I_dc, I0, tau, m),
            [t[0], t[-1]],
            [N_dc, S_dc],
            t_eval=t,
            method='BDF',
            rtol=1e-10,
            atol=1e-12
        )
        N = sol.y[0]
        S = sol.y[1]

        # Calculate field with chirp
        deltaN = N - N_dc_array[-1]
        if chirp:
            delta_phi = cumtrapz(
                0.5 * self.physics.beta_c * self.physics.G_n * deltaN * self.physics.volume, t, initial=0
            )
            chirp_inst = np.gradient(delta_phi, t) / (2 * np.pi)  # Chirp in Hz
            chirp_inst += -120e9  # Add initial offset for negative chirp
        else:
            delta_phi = np.zeros_like(t)
            chirp_inst = np.zeros_like(t)

        E_t = np.sqrt(S) * np.real(np.exp(1j * delta_phi))

        # Plot results
        fig, axs = plt.subplots(3, 1, figsize=(8, 10))
        axs[0].plot(t * 1e12, I_pulse * 1e3, 'b-')
        axs[0].set_title("Current Pulse (mA)")
        axs[0].set_ylabel("Current (mA)")

        axs[1].plot(t * 1e12, E_t / np.max(E_t), 'r-')
        axs[1].set_title("Output Field E(t) Normalized")
        axs[1].legend(["Chirped" if chirp else "Unchirped"])

        axs[2].plot(t * 1e12, chirp_inst * 1e-9, 'g-')
        axs[2].set_title("Frequency Chirp vs Time")
        axs[2].set_ylabel("Chirp (GHz)")
        axs[2].set_xlabel("Time (ps)")
        axs[2].legend(["Chirped" if chirp else "Unchirped"])

        axs[2].set_xlim([-1.5 * pulse_duration * 1e12,
                        1.5 * pulse_duration * 1e12])
        axs[2].set_ylim([-60, 80])
        axs[0].set_xlim([-1.5 * pulse_duration * 1e12,
                        1.5 * pulse_duration * 1e12])
        axs[1].set_xlim([-1.5 * pulse_duration * 1e12,
                        1.5 * pulse_duration * 1e12])

        for ax in axs:
            ax.grid(True)
            ax.set_facecolor('#f8f8f8')
        plt.tight_layout()

        if chirp:
            plt.savefig("plots/gain_switched_pulse_with_chirp.png",
                        dpi=300, bbox_inches='tight')
        else:
            plt.savefig("plots/gain_switched_pulse_without_chirp.png",
                        dpi=300, bbox_inches='tight')

        # plt.show()

    def calculate_optical_power(self, S):
        """
        Calculates optical output power from photon density.
        """
        P_out = -np.log(self.physics.R) * (
            self.physics.v_g *
            self.physics.volume /
            (2 * self.physics.L)) * h * self.physics.nu * S
        return P_out

    def calculate_threshold_current(self, I_dc=30e-3):
        """
        Calculates the threshold current for the laser.
        """
        N_th, _ = self.calculate_steady_state(I_dc=I_dc)

        tau_c = 1 / (self.physics.A_nr + self.physics.B *
                     N_th + self.physics.C * N_th**2)

        I_th = N_th * self.physics.volume * self.physics.q / tau_c

        return I_th

    def calculate_relaxation_oscillation(self, I_dc):
        """
        Returns the relaxation oscillation frequency using (6-4-25) and the additional
        formula from (6-4-27), where n₀ is N_tr:

        (6-4-27) Ω_R = sqrt(
          [ (1 + Γ⋅v_g⋅a⋅n₀⋅τ_p ) / (τ_e⋅τ_p ) ] ⋅ [ (I / I_th) - 1 ])

        (6-4-25) Ω_R = sqrt(G_N * (I - I_th) / q )

        Ref: https://www.fiberoptics4sale.com/blogs/wave-optics/transient-response-of-semiconductor-lasers
        """

        # Calculate freq relaxation using (6-4-25)
        I_th = self.calculate_threshold_current(I_dc)
        freq_1 = np.sqrt(
            self.physics.G_n * (I_dc - I_th) / self.physics.q) / (2*np.pi)

        # Calculate freq relaxation using (6-4-27)
        N_th, _ = self.calculate_steady_state(I_dc=I_dc)
        tau_c = 1 / (self.physics.A_nr + self.physics.B *
                     N_th + self.physics.C * N_th**2)

        n0 = self.physics.N_tr
        numerator = (1 + self.physics.gamma * self.physics.v_g *
                     self.physics.gain_coff * n0 * self.physics.tau_p)
        denominator = tau_c * self.physics.tau_p
        current_ratio = (I_dc / I_th) - 1
        freq_2 = np.sqrt((numerator / denominator)
                         * current_ratio) / (2*np.pi)

        return freq_1, freq_2

    def analyze_frequency_response(self, freq, I_dc=None, I_ac=None):
        """
        Analyzes laser response at a specific frequency.
        Generates plots showing carrier and photon density variations.
        """
        I_dc = I_dc if I_dc is not None else self.config.I_DC
        I_ac = I_ac if I_ac is not None else self.config.I_AC
        N_dc, S_dc = self.calculate_steady_state(I_dc)
        _, S_tot = self.calculate_steady_state(I_dc + I_ac)
        _, _, _, sol_ac = self.simulate_single_frequency_response(
            (freq, N_dc, S_dc, S_tot, I_dc, I_ac, 'fft'))
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

    def plot_PI_curve(self, P_solutions, I_values):
        """
        Plots the power-current (P-I) characteristic curve.
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
        """
        ax.plot(freqs / 1e9, fft_magnitude / np.max(fft_magnitude), '-',
                color=color, linewidth=1.5)
        ax.set_ylabel("Power (Normalized)")
        ax.set_title(f"$I_p$ = {I_ac * 1e3:.1f} mA")
        ax.set_xlim([-4, 4])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        # Set the background to light gray for better contrast
        ax.set_facecolor('#f8f8f8')

    def plot_dc_characteristics(self, I_dc, N_dc, S_dc, N_tr_temp=None):
        N_tr = self.physics.N_tr if N_tr_temp is None else N_tr_temp
        I_th = self.calculate_threshold_current(self.config.I_DC)
        plt.figure(figsize=(12, 8))
        plt.suptitle('DC Characteristics', fontsize=16, y=1.02)
        plt.subplot(2, 1, 1)
        plt.plot(I_dc * 1e3, N_dc / N_tr, 'b-',
                 linewidth=2, label='Carrier Density')
        plt.axvline(x=I_th * 1e3, color='gray', linestyle='--',
                    linewidth=1, label=f'Threshold Current ({I_th * 1e3:.2f} mA)')
        plt.xlabel('Current (mA)', fontsize=12)
        plt.ylabel('Carrier Density (N/N$_{tr}$)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.title('Carrier Density vs Current', fontsize=14)
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(I_dc * 1e3, S_dc, 'g-', linewidth=2, label='Photon Density')
        plt.axvline(x=I_th * 1e3, color='gray', linestyle='--',
                    linewidth=1, label=f'Threshold Current ({I_th * 1e3:.2f} mA)')
        plt.xlabel('Current (mA)', fontsize=12)
        plt.ylabel('Photon Density (cm$^{-3}$)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.title('Photon Density vs Current', fontsize=14)
        plt.legend()

        # Add secondary y-axis in log scale
        ax = plt.gca()
        ax2 = ax.twinx()
        ax2.plot(I_dc * 1e3, S_dc, 'r--', linewidth=1)
        ax2.set_yscale('log')
        ax2.set_ylabel('Photon Density (log scale)',
                       fontsize=12, color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        plt.tight_layout()
        plt.savefig('plots/dc_characteristics.png',
                    dpi=300, bbox_inches='tight')

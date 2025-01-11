# Description: Test file for the laser_model.py module.

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from laser_model import LaserModel, SimulationConfig, PhysicsConstants
import cProfile
import pstats
import io
from scipy.signal import hilbert
from scipy.fft import fft, fftfreq

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


class LaserModel2(LaserModel):
    def __init__(self, config, physics):
        super().__init__(config, physics)

    def simulate_single_frequency_response(self, params):
        """
        Simulates a single modulation frequency and calculates the response using
        Peak-to-Peak, Hilbert Transform, and FFT with proper scaling.

        Args:
            params (tuple): (freq, N_dc, S_dc, S_tot, I_dc, I_ac_amplitude)

        Returns:
            tuple: (frequency, responses_dict, solver_result)
        """
        freq, N_dc, S_dc, S_tot, I_dc, I_ac_amplitude = params
        cycles_required = 25
        T_sim = cycles_required / freq
        T_sim = T_sim * 2  # Increase simulation time for better accuracy

        # Ensure enough points per cycle for accurate FFT
        points_per_cycle = 20  # Minimum points per cycle
        num_points = max(
            self.config.MIN_TOTAL_POINTS,
            int(freq * T_sim * points_per_cycle)
        )

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

        # Extract steady-state portion (last few cycles)
        cycles_to_analyze = 10  # Increase number of cycles for better FFT accuracy
        points_per_cycle = num_points / (cycles_required * 2)
        last_n_points = int(points_per_cycle * cycles_to_analyze)
        S_last_portion = sol_ac.y[1][-last_n_points:]
        t_last_portion = t_sim[-last_n_points:]

        # Apply window function to reduce spectral leakage
        window = np.hanning(len(S_last_portion))
        S_windowed = (S_last_portion - np.mean(S_last_portion)) * window

        # Method 1: Peak-to-Peak (using original signal)
        peak_to_peak_response = np.ptp(S_last_portion)

        # Method 2: Hilbert Transform (using original signal)
        analytic_signal = hilbert(S_last_portion - np.mean(S_last_portion))
        # Scale by 2 for amplitude
        hilbert_response = 2 * np.mean(np.abs(analytic_signal))

        # Method 3: FFT with proper scaling
        fft_values = fft(S_windowed)
        fft_frequencies = fftfreq(len(S_windowed), d=(
            t_last_portion[1] - t_last_portion[0]))

        # Compute magnitude spectrum with proper scaling
        # Factor of 2 accounts for negative frequencies
        # Window correction factor compensates for the energy loss due to windowing
        # Scale factor for Hanning window
        window_correction = 2 / np.mean(window)
        fft_magnitudes = np.abs(fft_values) * \
            (2.0 / len(S_windowed)) * window_correction

        # Find the magnitude at the fundamental frequency
        freq_resolution = fft_frequencies[1] - fft_frequencies[0]
        freq_idx = np.argmin(np.abs(fft_frequencies - freq))
        fft_response = fft_magnitudes[freq_idx]

        # Store results in a dictionary
        responses = {
            "Peak-to-Peak": peak_to_peak_response,
            "Hilbert": hilbert_response,
            "FFT": fft_response
        }

        return freq, responses, sol_ac

    def sweep_and_compare(self, I_dc=None, I_ac=None):
        """
        Performs a frequency sweep and compares response amplitude measurement methods.

        Args:
            I_dc (float, optional): DC current
            I_ac (float, optional): AC amplitude

        Returns:
            None: Plots the results directly.
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
        results_dict = {"Peak-to-Peak": [], "Hilbert": [], "FFT": []}

        with ProcessPoolExecutor(max_workers=self.config.N_THREADS) as executor:
            futures = [executor.submit(
                self.simulate_single_frequency_response, params) for params in params_list]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Frequency Sweep"):
                try:
                    freq, responses, _ = future.result()
                    for method, response in responses.items():
                        results_dict[method].append((freq, response))
                except Exception as e:
                    print(f"Error processing frequency: {str(e)}")

        # Sort results by frequency
        for method in results_dict:
            results_dict[method] = sorted(
                results_dict[method], key=lambda x: x[0])

        # Extract frequencies and responses for plotting
        frequencies = [x[0] for x in results_dict["Peak-to-Peak"]]
        responses = {method: [x[1] for x in results]
                     for method, results in results_dict.items()}

        # Plot the results
        plt.figure(figsize=(12, 8))
        for method, response in responses.items():
            plt.semilogx(frequencies, 20 * np.log10(response),
                         label=f'{method}', linewidth=2)

        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.grid(True, which="major", ls="-", alpha=0.5)
        plt.xlabel('Frequency (Hz)', fontsize=12)
        plt.ylabel('Response (dB)', fontsize=12)
        plt.title('Comparison of Amplitude Measurement Methods',
                  fontsize=14, pad=20)
        plt.legend(title='Method', title_fontsize=12,
                   fontsize=10, loc='lower left')
        plt.tight_layout()
        plt.savefig('plots/AC_amplitude_comparison.png',
                    dpi=300, bbox_inches='tight')
        plt.show()


def test_phase_modulation_power_spectra(laser, config):
    # Parameters
    I_dc = 30e-3
    freq = 1e9
    I_ac = [2e-3, 4e-3, 6e-3, 8e-3]

    # Calculate and plot
    laser.simulate_phase_modulation(I_dc, I_ac, freq)


def tst_ac_analysis2():
    print("Testing AC analysis for a single DC current...")
    I_dc_value = 30e-3
    I_ac = 0.2 * I_dc_value
    config = SimulationConfig()
    physics = PhysicsConstants()
    laser = LaserModel2(config, physics)
    laser.sweep_and_compare(I_dc=I_dc_value, I_ac=I_ac)


def test_dc_steady_state(laser, config):
    print("Testing DC steady state calculation...")
    N_dc, S_dc = laser.calculate_steady_state(config.I_DC)
    print(f"Carrier density = {N_dc:.2e} cm⁻³")
    print(f"Photon density = {S_dc:.2e} cm⁻³")


def test_ac_analysis(laser, config):
    print("Testing AC analysis for a single DC current...")
    I_dc_value = 30e-3
    I_ac = 0.1 * I_dc_value
    freq_result, response_amp_S, response_amp_N = laser.sweep_small_signal_response(
        I_dc=I_dc_value, I_ac=I_ac)
    # response_db = 20 * np.log10(response_amp_S / response_amp_S[10])
    # response_db = response_result / response_result[10]
    # interp_func = interp1d(freq_result, response_db, kind='cubic')
    fig_dc, (ax_s_dc, ax_n_dc) = plt.subplots(2, 1, figsize=(12, 8))
    response_db_s = 20 * np.log10(response_amp_S / response_amp_S[1])
    response_db_n = 20 * np.log10(response_amp_N / response_amp_N[1])
    interp_func_s = interp1d(freq_result, response_db_s, kind='cubic')
    interp_func_n = interp1d(freq_result, response_db_n, kind='cubic')
    freq_smooth = np.logspace(np.log10(freq_result[0]), np.log10(
        freq_result[-1]), config.INTERP_POINTS)
    resp_s_smooth = interp_func_s(freq_smooth)
    resp_n_smooth = interp_func_n(freq_smooth)
    ax_s_dc.semilogx(freq_smooth, resp_s_smooth,
                     linewidth=2, label=f'I_DC = {I_dc_value * 1000:.1f} mA')
    ax_n_dc.semilogx(freq_smooth, resp_n_smooth,
                     linewidth=2, label=f'I_DC = {I_dc_value * 1000:.1f} mA')

    ax_s_dc.grid(True, which="both", ls="-", alpha=0.2)
    ax_s_dc.grid(True, which="major", ls="-", alpha=0.5)
    ax_s_dc.set_xlabel('Frequency (Hz)', fontsize=12)
    ax_s_dc.set_ylabel('Photon Response (dB)', fontsize=12)
    ax_s_dc.set_title(' ')
    ax_s_dc.legend(title='Bias Current', title_fontsize=12,
                   fontsize=10, loc='lower left', bbox_to_anchor=(0.02, 0.02))

    ax_n_dc.grid(True, which="both", ls="-", alpha=0.2)
    ax_n_dc.grid(True, which="major", ls="-", alpha=0.5)
    ax_n_dc.set_xlabel('Frequency (Hz)', fontsize=12)
    ax_n_dc.set_ylabel('Carrier Response (dB)', fontsize=12)
    ax_n_dc.set_title('Small Signal Frequency Response vs. DC Current (N)')
    ax_n_dc.legend(title='Bias Current', title_fontsize=12,
                   fontsize=10, loc='lower left', bbox_to_anchor=(0.02, 0.02))
    fig_dc.tight_layout()


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


def test_supergaussian_pulse_chirping(laser, config):
    """
    Test function for super-Gaussian pulse chirping simulation.
    """
    print("Testing super-Gaussian pulse chirping with chirp = True")
    laser.simulate_supergaussian_pulse_chirping(chirp=True)
    print("Testing super-Gaussian pulse chirping with chirp = False")
    laser.simulate_supergaussian_pulse_chirping(chirp=False)


def main():
    config = SimulationConfig()
    physics = PhysicsConstants()
    laser = LaserModel(config, physics)
    # test_supergaussian_pulse_chirping(laser, config)
    # test_dc_steady_state(laser, config)
    # test_ac_analysis(laser, config)
    tst_ac_analysis2()
    # test_specific_frequency(laser)
    # test_dc_analysis(laser)
    # test_power_vs_current(laser)
    # test_transient_analysis_step(laser)
    # test_transient_analysis_ramp(laser)
    # test_phase_modulation_power_spectra(laser, config)
    plt.show()


if __name__ == "__main__":
    import os
    os.makedirs('plots', exist_ok=True)
    main()

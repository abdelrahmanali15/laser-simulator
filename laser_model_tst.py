import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from laser_model import LaserModel, SimulationConfig, PhysicsConstants
import cProfile
import pstats
import io


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
    I_dc_value = 30e-3
    I_ac = 0.1 * I_dc_value
    freq_result, response_result = laser.sweep_small_signal_response(
        I_dc=I_dc_value, I_ac=I_ac)
    response_db = 20 * np.log10(response_result / response_result[10])
    # response_db = response_result / response_result[10]
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
    # test_specific_frequency(laser)
    # test_dc_analysis(laser)
    # test_power_vs_current(laser)
    # test_transient_analysis_step(laser)
    # test_transient_analysis_ramp(laser)
    test_phase_modulation_power_spectra(laser, config)
    plt.show()


if __name__ == "__main__":
    import os
    os.makedirs('plots', exist_ok=True)
    main()

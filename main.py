# Description: Main script to run laser simulations and generate analysis plots.

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from laser_model import LaserModel, SimulationConfig, PhysicsConstants


def run_dc_analysis(laser):
    print("Calculating DC steady state...")
    N_dc, S_dc = laser.calculate_steady_state(laser.config.I_DC)
    print(f"Carrier density = {N_dc:.2e} cm^-3")
    print(f"Photon density = {S_dc:.2e} cm^-3")


def run_ac_analysis(laser):
    print("\nAC: Analyzing multiple DC currents...")
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    fig_dc, (ax_s_dc, ax_n_dc) = plt.subplots(2, 1, figsize=(12, 8))

    method = 'fft'  # 'peak2peak', 'hilbert', or 'fft'
    for i, I_dc_value in enumerate(laser.config.DC_CURRENTS):
        print(f"\nAnalyzing DC current: {I_dc_value * 1000:.1f} mA")
        freq_result, response_amp_S, response_amp_N = laser.sweep_small_signal_response(
            I_dc=I_dc_value, method=method)
        response_db_s = 20 * np.log10(response_amp_S / response_amp_S[1])
        response_db_n = 20 * np.log10(response_amp_N / response_amp_N[1])
        interp_func_s = interp1d(freq_result, response_db_s, kind='cubic')
        interp_func_n = interp1d(freq_result, response_db_n, kind='cubic')
        freq_smooth = np.logspace(np.log10(freq_result[0]), np.log10(
            freq_result[-1]), laser.config.INTERP_POINTS)
        resp_s_smooth = interp_func_s(freq_smooth)
        resp_n_smooth = interp_func_n(freq_smooth)
        ax_s_dc.semilogx(freq_smooth, resp_s_smooth, color=colors[i],
                         linewidth=2, label=f'I_DC = {I_dc_value * 1000:.1f} mA')
        ax_n_dc.semilogx(freq_smooth, resp_n_smooth, color=colors[i],
                         linewidth=2, label=f'I_DC = {I_dc_value * 1000:.1f} mA')

    ax_s_dc.grid(True, which="both", ls="-", alpha=0.2)
    ax_s_dc.grid(True, which="major", ls="-", alpha=0.5)
    ax_s_dc.set_xlabel('Frequency (Hz)', fontsize=12)
    ax_s_dc.set_ylabel('Photon Response (dB)', fontsize=12)
    ax_s_dc.set_title('Small Signal Frequency Response vs. DC Current (S)')
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
    plt.savefig('plots/AC_multiple_large_signal.png',
                dpi=300, bbox_inches='tight')

    print("\nAC: Analyzing multiple AC currents...")
    fig_ac, (ax_s_ac, ax_n_ac) = plt.subplots(2, 1, figsize=(12, 8))

    for i, I_ac_value in enumerate(laser.config.AC_CURRENTS):
        print(f"\nAnalyzing AC current: {I_ac_value * 1000:.1f} mA")
        freq_result, response_amp_S, response_amp_N = laser.sweep_small_signal_response(
            I_ac=I_ac_value, method=method)
        response_db_s = 20 * np.log10(response_amp_S / response_amp_S[1])
        response_db_n = 20 * np.log10(response_amp_N / response_amp_N[1])
        interp_func_s = interp1d(freq_result, response_db_s, kind='cubic')
        interp_func_n = interp1d(freq_result, response_db_n, kind='cubic')
        freq_smooth = np.logspace(np.log10(freq_result[0]), np.log10(
            freq_result[-1]), laser.config.INTERP_POINTS)
        resp_s_smooth = interp_func_s(freq_smooth)
        resp_n_smooth = interp_func_n(freq_smooth)
        ax_s_ac.semilogx(freq_smooth, resp_s_smooth, color=colors[i],
                         linewidth=2, label=f'I_AC = {I_ac_value * 1000:.1f} mA')
        ax_n_ac.semilogx(freq_smooth, resp_n_smooth, color=colors[i],
                         linewidth=2, label=f'I_AC = {I_ac_value * 1000:.1f} mA')

    ax_s_ac.grid(True, which="both", ls="-", alpha=0.2)
    ax_s_ac.grid(True, which="major", ls="-", alpha=0.5)
    ax_s_ac.set_xlabel('Frequency (Hz)', fontsize=12)
    ax_s_ac.set_ylabel('Photon Response (dB)', fontsize=12)
    ax_s_ac.set_title('Small Signal Frequency Response vs. AC Current (S)')
    ax_s_ac.legend(title='AC Current', title_fontsize=12,
                   fontsize=10, loc='lower left', bbox_to_anchor=(0.02, 0.02))

    ax_n_ac.grid(True, which="both", ls="-", alpha=0.2)
    ax_n_ac.grid(True, which="major", ls="-", alpha=0.5)
    ax_n_ac.set_xlabel('Frequency (Hz)', fontsize=12)
    ax_n_ac.set_ylabel('Carrier Response (dB)', fontsize=12)
    ax_n_ac.set_title('Small Signal Frequency Response vs. AC Current (N)')
    ax_n_ac.legend(title='AC Current', title_fontsize=12,
                   fontsize=10, loc='lower left', bbox_to_anchor=(0.02, 0.02))

    fig_ac.tight_layout()
    plt.savefig('plots/AC_multiple_small_signal.png',
                dpi=300, bbox_inches='tight')


def run_specific_frequency_analysis(laser):
    test_frequencies = [1.3e8, 1e9, 5e9, 11e9]
    print("\nSmall signal transient of specific frequencies...")
    for freq in test_frequencies:
        print(f"\nAnalyzing frequency: {freq / 1e9:.1f} GHz")
        laser.analyze_frequency_response(freq)


def run_dc_characteristics_analysis(laser):
    print(f"\nAnalyzing DC Characteristics...")
    I_dc, N_dc, S_dc = laser.perform_dc_analysis()
    laser.plot_dc_characteristics(I_dc, N_dc, S_dc)


def run_pi_analysis(laser):
    print(f"\nAnalyzing PI Characteristics...")
    I_dc, N_dc, S_dc = laser.perform_dc_analysis()
    P_out = laser.calculate_optical_power(S_dc)
    laser.plot_PI_curve(P_out, I_dc)


def run_temperature_analysis(laser):
    print("\nAnalyzing Temperature Dependence...")
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


def run_step_and_ramp(laser):
    print(f"\nSimulating Step Response...")
    t, solutions_N, solutions_S, I_values = laser.simulate_step_response()
    laser.plot_step_response(t, solutions_N, solutions_S, I_values)
    print(f"\nSimulating Ramp Response...")
    t, N, S, t_steps, I_steps = laser.simulate_ramp_response()
    laser.plot_ramp_response(t, N, S, t_steps, I_steps)


def run_phase_and_pulse(laser):
    print("\nAnalyzing Phase Modulation...")
    # Parameters
    I_dc = 30e-3
    freq = 1e9
    I_ac = [2e-3, 4e-3, 6e-3, 8e-3]

    # Calculate and plot
    laser.simulate_phase_modulation(I_dc, I_ac, freq)

    print("Testing super-Gaussian pulse chirping with chirp = True")
    laser.simulate_supergaussian_pulse_chirping(chirp=True)
    print("Testing super-Gaussian pulse chirping with chirp = False")
    laser.simulate_supergaussian_pulse_chirping(chirp=False)


def main():
    """
    Main function to run laser simulations and generate analysis plots.
    Performs DC, AC, transient, and phase modulation analysis.
    """
    config = SimulationConfig()
    physics = PhysicsConstants()
    laser = LaserModel(config, physics)

    run_dc_analysis(laser)
    run_ac_analysis(laser)
    run_specific_frequency_analysis(laser)
    run_dc_characteristics_analysis(laser)
    run_pi_analysis(laser)
    run_temperature_analysis(laser)
    run_step_and_ramp(laser)
    run_phase_and_pulse(laser)

    plt.show()


if __name__ == "__main__":
    import os
    os.makedirs('plots', exist_ok=True)
    main()

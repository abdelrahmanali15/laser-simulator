import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.constants import h

# Constants and Parameters (keeping the same as before)
c_0 = 3e10  # Speed of light [cm/s]
n_g = 4  # Group index
v_g = c_0 / n_g  # Group velocity [cm/s]
L = 250e-4  # 250 µm in cm
w = 2e-4  # 2 µm in cm
d = 0.2e-4  # 0.2 µm in cm
Vact = L * w * d  # Active region volume [cm^3]
Gamma = 0.3  # Confinement factor
a_gain = 2.5e-16  # Gain cross-section [cm^2]
N_tr0 = 1e18  # Transparency carrier density [cm^-3]
A_nr = 1e8  # Non-radiative recombination [s^-1]
B = 1e-10  # Radiative recombination coefficient [cm^3/s]
C = 3e-29  # Auger recombination [cm^6/s]
beta_sp = 1e-4  # Spontaneous emission factor
tau_p = 1.6e-12  # Photon lifetime [s]
q = 1.602e-19  # Electron charge [C]
lambda_laser = 1.3e-6  # Wavelength [m]
nu = c_0 / (lambda_laser * 1e2)  # Optical frequency [Hz]
I_max = 50e-3  # Maximum current [A]


def rate_equations(t, y, t_current, I_current, N_tr_temp=None):
    """
    Rate equations with current interpolation
    Args:
        t: time point
        y: state vector [N, S]
        t_current: time array for current interpolation
        I_current: current array for interpolation
    """
    N_tr = N_tr0 if N_tr_temp is None else N_tr_temp
    N, S = y
    # Interpolate current at time t
    I = np.interp(t, t_current, I_current)

    # Total carrier recombination
    Rtot = A_nr * N + B * N**2 + C * N**3
    # Net gain
    G = Gamma * v_g * a_gain * max(N - N_tr, 0)
    # Carrier density rate equation
    dNdt = (I / (q * Vact)) - Rtot - G * S
    # Photon density rate equation
    dSdt = G * S - (S / tau_p) + beta_sp * B * N**2

    return [dNdt, dSdt]

# DC Analysis


def perform_dc_analysis(N_tr_temp=None):
    # Create current points for DC analysis
    I_dc = np.linspace(0, I_max, 100)
    N_dc = []
    S_dc = []
    T_steady = 20e-9  # Time to reach steady state
    t_eval = np.array([0, T_steady])

    # Initial conditions
    N0 = 0
    S0 = 0

    for I in I_dc:
        # Create constant current array for this current value
        t_current = np.array([0, T_steady])
        I_current = np.array([I, I])

        # Solve ODE
        sol = solve_ivp(
            rate_equations,
            [0, T_steady],
            [N0, S0],
            args=(t_current, I_current, N_tr_temp),
            method='BDF',
            t_eval=t_eval,
            rtol=1e-6,
            atol=1e-10
        )

        # Store final values (steady state)
        N_dc.append(sol.y[0, -1])
        S_dc.append(sol.y[1, -1])

    return I_dc, np.array(N_dc), np.array(S_dc)

# Transient Analysis


def perform_transient_analysis_step():
    # Time points
    T_end = 20e-9
    num_points = 500
    t = np.linspace(0, T_end, num_points)

    # Current values (8 points from 0 to I_max)
    # I_values = np.linspace(0, I_max, 6)
    I_values = np.array([0, 10e-3, 15e-3, 20e-3, I_max])

    # Initialize 2D arrays to store solutions
    # Each row corresponds to a current value
    # Each column corresponds to a time point
    N_solutions = np.zeros((len(I_values), len(t)))
    S_solutions = np.zeros((len(I_values), len(t)))

    # Initial conditions
    N0 = 0
    S0 = 0

    # Solve for each current value
    for i, I in enumerate(I_values):
        # Create constant current array
        t_current = np.array([0, T_end])
        I_current = np.array([I, I])

        # Solve ODE
        sol = solve_ivp(
            rate_equations,
            [0, T_end],
            [N0, S0],
            args=(t_current, I_current),
            method='BDF',
            t_eval=t,
            rtol=1e-6,
            atol=1e-10
        )

        # Store solutions in the corresponding row
        N_solutions[i, :] = sol.y[0]
        S_solutions[i, :] = sol.y[1]

    return t, N_solutions, S_solutions, I_values

# Transient Analysis


def perform_transient_analysis_ramp():
    # Create time points for transient analysis
    T_end = 20e-9
    num_points = 500
    t = np.linspace(0, T_end, num_points)

    # Create stepped current profile (8 points)
    I_steps = np.linspace(0, I_max, 8)
    t_steps = np.linspace(0, T_end, 8)

    # Initial conditions
    N0 = 0
    S0 = 0

    # Solve ODE with stepped current
    sol = solve_ivp(
        rate_equations,
        [0, T_end],
        [N0, S0],
        args=(t_steps, I_steps),
        method='BDF',
        t_eval=t,
        rtol=1e-6,
        atol=1e-10
    )

    return t, sol.y[0], sol.y[1], t_steps, I_steps


def plot_transient_response_ramp(t, N, S, t_steps, I_steps):
    plt.figure(figsize=(12, 8))
    plt.suptitle('Transient Response for Ramp current', fontsize=16)

    # Current steps
    plt.subplot(3, 1, 1)
    plt.plot(t_steps * 1e9, I_steps * 1e3, 'r.-', linewidth=2)
    plt.xlabel('Time (ns)')
    plt.ylabel('Current (mA)')
    plt.grid(True)

    # Carrier density
    plt.subplot(3, 1, 2)
    plt.plot(t * 1e9, N / N_tr0, 'b-', linewidth=2)
    plt.xlabel('Time (ns)')
    plt.ylabel('Carrier Density (N/N$_{tr}$)')
    plt.grid(True)

    # Photon density
    plt.subplot(3, 1, 3)
    # plt.semilogy(t * 1e9, S, 'g-', linewidth=2)
    plt.plot(t * 1e9, S, 'g-', linewidth=2)
    plt.xlabel('Time (ns)')
    plt.ylabel('Photon Density (cm$^{-3}$)')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('plots/transient_response_ramp.png',
                dpi=300, bbox_inches='tight')


def plot_dc_characteristics(I_dc, N_dc, S_dc, N_tr_temp=None):
    N_tr = N_tr0 if N_tr_temp is None else N_tr_temp

    plt.figure(figsize=(12, 8))
    plt.suptitle('DC Characteristics', fontsize=16, y=1.02)

    # Carrier density vs current
    plt.subplot(2, 1, 1)
    plt.plot(I_dc * 1e3, N_dc / N_tr, 'b-', linewidth=2)
    plt.xlabel('Current (mA)', fontsize=12)
    plt.ylabel('Carrier Density (N/N$_{tr}$)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.title('Carrier Density vs Current', fontsize=14)

    # Photon density vs current
    plt.subplot(2, 1, 2)
    # plt.semilogy(I_dc * 1e3, S_dc, 'g-', linewidth=2)
    plt.plot(I_dc * 1e3, S_dc, 'g-', linewidth=2)
    plt.xlabel('Current (mA)', fontsize=12)
    plt.ylabel('Photon Density (cm$^{-3}$)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.title('Photon Density vs Current', fontsize=14)

    plt.tight_layout()
    plt.savefig('plots/dc_characteristics.png', dpi=300, bbox_inches='tight')


def plot_transient_response_step(t, N_solutions, S_solutions, I_values):
    # Create color maps for the plots
    colors = plt.cm.viridis(np.linspace(0, 1, len(I_values)))

    # Create figures
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Transient Response for Different Step Current Values',
                 fontsize=16, y=0.95)

    # Plot carrier density for each current value
    for i in range(len(I_values)):
        ax1.plot(t * 1e9, N_solutions[i, :] / N_tr0, color=colors[i],
                 linewidth=2, label=f'I = {I_values[i]*1e3:.1f} mA')

    ax1.set_xlabel('Time (ns)', fontsize=12)
    ax1.set_ylabel('Carrier Density (N/N$_{tr}$)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax1.set_title('Carrier Density Evolution', fontsize=14)

    # Plot photon density for each current value
    for i in range(len(I_values)):
        ax2.plot(t * 1e9, S_solutions[i, :], color=colors[i],
                 linewidth=2, label=f'I = {I_values[i]*1e3:.1f} mA')

    ax2.set_xlabel('Time (ns)', fontsize=12)
    ax2.set_ylabel('Photon Density (cm$^{-3}$)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax2.set_title('Photon Density Evolution', fontsize=14)

    # Adjust layout to prevent label overlap
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    plt.savefig('plots/transient_response_step.png',
                dpi=300, bbox_inches='tight')


def calculate_power(S):
    """Calculate output power from photon density"""
    R = Gamma**2  # Mirror reflectivity
    P_out = -np.log(R) * (v_g * Vact / (2 * L)) * h * nu * S
    return P_out


def plot_power_vs_current(P_solutions, I_values):
    # Create a new figure for P-I characteristic
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


# Main execution
if __name__ == "__main__":
    # Perform DC analysis,  I_dc, N_dc, S_dc are arrays of different steady state solutions across current
    I_dc, N_dc, S_dc = perform_dc_analysis()
    plot_dc_characteristics(I_dc, N_dc, S_dc)

    P_out = calculate_power(S_dc)
    plot_power_vs_current(P_out, I_dc)

    plt.figure(figsize=(12, 8))
    plt.suptitle('Temperature Dependence of Output Power', fontsize=16)
    # Temperature-dependent analysis
    T = np.linspace(200, 350, 6)
    N_tr0 = 1e18

    for i, temp in enumerate(T):
        N_tr = N_tr0 * np.exp((temp-300)/50)

        I_dc, N_dc, S_dc = perform_dc_analysis(N_tr)

        # Calculate output power
        P_out = calculate_power(S_dc)

        # Plot with custom colormap
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

    # Perform transient analysis
    t, solutions_N, solutions_S, I_values = perform_transient_analysis_step()
    plot_transient_response_step(t, solutions_N, solutions_S, I_values)

    # Perform transient analysis
    t, N, S, t_steps, I_steps = perform_transient_analysis_ramp()
    plot_transient_response_ramp(t, N, S, t_steps, I_steps)

    plt.show()

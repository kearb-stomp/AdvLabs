import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pandas as pd

# --- Load and normalize data ---
file_path_gold = r'Data/Reflectivity/Gold Mirror baseline reflectivity.ASC'
file_path_si_rf = r'Data/Reflectivity/Silicone reflectivity spectrum.ASC'
header_lines = 25

# Load gold mirror reflectivity
data_gold = pd.read_csv(
    file_path_gold,
    skiprows=header_lines,
    delim_whitespace=True,
    header=None,
    decimal=',',
    encoding='latin1'
)
x_gold = np.array(data_gold[0])
y_gold = np.array(data_gold[1])

# Load silicon reflectivity
data_si_rf = pd.read_csv(
    file_path_si_rf,
    skiprows=header_lines,
    delim_whitespace=True,
    header=None,
    decimal=',',
    encoding='latin1'
)
x_si1_rf = np.array(data_si_rf[0])
y_si1_rf = np.array(data_si_rf[1])

# Normalize and convert wavenumber to frequency
c = 3 * 10**8
x_si2_rf = x_si1_rf * 100 * c  # Convert cm^-1 to Hz
x_data = x_si2_rf
y_data = y_si1_rf / y_gold

# Apply mask to remove noisy low-frequency and high-frequency edges
mask = (x_data > 2e13) & (x_data < 9e13)
x_data_fit = x_data[mask]
y_data_fit = y_data[mask]

# --- Define 2-phonon model ---
def R_2phonon_model(omega, eps_inf, omega_LO1, omega_TO1, gamma1, omega_LO2, omega_TO2, gamma2):
    kai_1 = (omega_LO1**2 - omega_TO1**2) / (omega_TO1**2 - omega**2 - 1j * omega * gamma1)
    kai_2 = (omega_LO2**2 - omega_TO2**2) / (omega_TO2**2 - omega**2 - 1j * omega * gamma2)
    eps = eps_inf + kai_1 + kai_2
    n = np.sqrt(eps)
    reflectivity = np.abs((1 - n) / (1 + n))**2
    return reflectivity

# --- Error function ---
def fit_and_score(params):
    eps_inf, LO1, TO1, g1, LO2, TO2, g2 = params
    try:
        y_pred = R_2phonon_model(x_data_fit, eps_inf, LO1, TO1, g1, LO2, TO2, g2)
        mse = np.mean((y_pred - y_data_fit)**2)
        return mse, params
    except:
        return np.inf, params

if __name__ == '__main__':
    import multiprocessing as mp

    # --- Expanded Parameter Grid (~250,000 iterations) ---
    eps_inf_vals = np.linspace(0.01, 0.2, 10)  # 10
    omega_LO1_vals = np.linspace(1.0e13, 1.4e13, 10)  # 10
    omega_TO1_vals = np.linspace(1.0e13, 1.3e13, 10)  # 10
    gamma1_vals = np.linspace(1e12, 6e12, 5)  # 5
    omega_LO2_vals = np.linspace(2.5e13, 3.5e13, 10)  # 10
    omega_TO2_vals = np.linspace(2.0e13, 3.2e13, 10)  # 10
    gamma2_vals = np.linspace(1e12, 6e12, 5)  # 5

    param_grid = list(product(
        eps_inf_vals,
        omega_LO1_vals, omega_TO1_vals, gamma1_vals,
        omega_LO2_vals, omega_TO2_vals, gamma2_vals
    ))

    print(f"Running {len(param_grid)} fits on {cpu_count()} cores...")

    best_mse = np.inf
    best_params = None
    best_fit = None

    with mp.Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(fit_and_score, param_grid), total=len(param_grid)))

    best_result = min(results, key=lambda x: x[0])
    best_mse, best_params = best_result

    # --- Plot best fit ---
    x_fit = np.linspace(min(x_data_fit), max(x_data_fit), 2000)
    y_fit = R_2phonon_model(x_fit, *best_params)

    plt.figure(figsize=(10, 6))
    plt.plot(x_data_fit, y_data_fit, label='Reflection Data', color='steelblue')
    plt.plot(x_fit, y_fit, '--', label='Best 2-Phonon Fit', color='orange')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Reflectivity")
    plt.title("2-Phonon Parallel Grid Search Fit")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\nBest Fit Parameters:")
    labels = ['epsilon_inf', 'omega_LO1', 'omega_TO1', 'gamma1', 'omega_LO2', 'omega_TO2', 'gamma2']
    for name, val in zip(labels, best_params):
        print(f"{name:>12} = {val:.3e}")
    print(f"Best MSE: {best_mse:.4e}")
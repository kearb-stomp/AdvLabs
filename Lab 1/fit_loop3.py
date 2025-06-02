import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pandas as pd
from scipy.optimize import curve_fit

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

# --- Define 3-phonon model ---
def R_3phonon_model(omega, eps_inf, LO1, TO1, g1, LO2, TO2, g2, LO3, TO3, g3):
    kai_1 = (LO1**2 - TO1**2) / (TO1**2 - omega**2 - 1j * omega * g1)
    kai_2 = (LO2**2 - TO2**2) / (TO2**2 - omega**2 - 1j * omega * g2)
    kai_3 = (LO3**2 - TO3**2) / (TO3**2 - omega**2 - 1j * omega * g3)
    eps = eps_inf + kai_1 + kai_2 + kai_3
    n = np.sqrt(eps)
    reflectivity = np.abs((1 - n) / (1 + n))**2
    return reflectivity

# --- Fit with curve_fit ---
def R_model_flat(w, eps_inf, LO1, TO1, g1, LO2, TO2, g2, LO3, TO3, g3):
    return R_3phonon_model(w, eps_inf, LO1, TO1, g1, LO2, TO2, g2, LO3, TO3, g3)

# Updated initial guess and tighter bounds for 3rd oscillator
initial_guess = [0.1, 3.3e13, 3.0e13, 2e12, 2.6e13, 2.3e13, 2e12, 2.2e13, 2.0e13, 2e12]
bounds_lower = [0.001, 1e13, 1e13, 1e11, 1e13, 1e13, 1e11, 1.8e13, 1.7e13, 1e11]
bounds_upper = [1.0,   5e13, 5e13, 1e13, 5e13, 5e13, 1e13, 2.4e13, 2.3e13, 1e13]

# Fit
popt, _ = curve_fit(
    R_model_flat, x_data_fit, y_data_fit,
    p0=initial_guess,
    bounds=(bounds_lower, bounds_upper),
    maxfev=10000
)

# --- Plot refined 3-phonon fit ---
x_fit = np.linspace(min(x_data_fit), max(x_data_fit), 2000)
y_fit_refined = R_3phonon_model(x_fit, *popt)

plt.figure(figsize=(10, 6))
plt.plot(x_data_fit, y_data_fit, label='Reflection Data', color='steelblue')
plt.plot(x_fit, y_fit_refined, '--', label='Refined 3-Phonon Fit', color='orange')
plt.xlabel("Frequency (Hz)")
plt.ylabel("Reflectivity")
plt.title("3-Phonon Fit with curve_fit")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("\nRefined 3-Phonon Fit Parameters:")
labels = ['epsilon_inf', 'omega_LO1', 'omega_TO1', 'gamma1', 'omega_LO2', 'omega_TO2', 'gamma2', 'omega_LO3', 'omega_TO3', 'gamma3']
for name, val in zip(labels, popt):
    print(f"{name:>12} = {val:.3e}")
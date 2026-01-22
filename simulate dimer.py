import os
import numpy as np
import matplotlib.pyplot as plt
import time

# Set the font to Times New Roman globally
plt.rcParams['font.family'] = 'Times New Roman'

# Create output folders for figures and data
figs_dir = "figs"
data_dir = "data"
os.makedirs(figs_dir, exist_ok=True)  # Create the figures folder
os.makedirs(data_dir, exist_ok=True)  # Create the data folder

# Function to save figures with a unique name based on timestamp
def save_figure(fig, name_prefix, figs_dir):
    timestamp = time.strftime("%Y%m%d-%H%M%S")  # Create a timestamp for uniqueness
    file_name = f"{name_prefix}_{timestamp}.pdf"  # Name the file with the timestamp
    file_path = os.path.join(figs_dir, file_name)  # Save in the 'figs' folder
    fig.savefig(file_path, format='pdf')  # Save the figure in PDF format
    print(f"Figure saved as: {file_path}")  # Print the saved file's path

#Part I
# Constants for Track 1 (constant refractive index)
n_m = 1.0  # Refractive index of the background medium (vacuum)
r1 = 90e-9  # Radius of sphere 1 in meters
r2 = 65e-9  # Radius of sphere 2 in meters
wavelength = np.linspace(500, 1000, 400) * 1e-9  # Wavelength range (500 nm to 1000 nm)

# Mie Coefficient Calculation (Track 1: constant refractive index)
def mie_coefficient(radius, wavelength, n_m):
    k = 2 * np.pi / wavelength  # Wave number (k = 2π/λ)
    a1 = 6 * np.pi * 1j * (radius ** 3) * k / (3 * np.pi)  # Simplified Mie coefficient for electric dipole
    return a1

# Calculate Mie coefficients for both spheres
a1_sphere1 = mie_coefficient(r1, wavelength, n_m)
a1_sphere2 = mie_coefficient(r2, wavelength, n_m)

# Plot the absolute value of the polarizability for both spheres
fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(wavelength * 1e9, np.abs(a1_sphere1), label='Sphere 1 (Track 1)', color='blue', linestyle='-', linewidth=2)
ax.plot(wavelength * 1e9, np.abs(a1_sphere2), label='Sphere 2 (Track 1)', color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Wavelength (nm)', fontsize=14, fontweight='bold')
ax.set_ylabel('Polarizability |α(λ)|', fontsize=14, fontweight='bold')
ax.set_title('Track 1: Mie Coefficients for Dielectric Spheres', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(False)
# Save the figure with a unique name
save_figure(fig, "mie_coefficients_track1", figs_dir)
# Show the figure
plt.show()

#Track 2
import pandas as pd
from scipy.interpolate import interp1d

# Load the refractive index and extinction coefficient data for crystalline Si from CSV
data = pd.read_csv('data\Aspnes.csv')

# Extract the wavelength, n (refractive index), and k (extinction coefficient)
wavelength_data = data['Wavelength'].values * 1e-9  # Convert to meters
n_data = data['n'].values
k_data = data['k'].values

# Interpolate the n(λ) and k(λ) values onto the simulation wavelength grid
wavelength_simulation = np.linspace(500, 1000, 400) * 1e-9  # Wavelength grid (500 nm to 1000 nm)

# Interpolate refractive index and extinction coefficient
n_interp = interp1d(wavelength_data, n_data, kind='cubic', fill_value="extrapolate")
k_interp = interp1d(wavelength_data, k_data, kind='cubic', fill_value="extrapolate")

# Get the interpolated values for n(λ) and k(λ)
n_interp_values = n_interp(wavelength_simulation)
k_interp_values = k_interp(wavelength_simulation)

# Calculate the complex refractive index ˜n(λ) = n(λ) + i k(λ)
n_complex = n_interp_values + 1j * k_interp_values

# Calculate the dielectric function ε(λ) = ˜n(λ)^2
epsilon = n_complex**2

# Plot the Mie coefficients for Sphere 1 using the complex refractive index
def mie_coefficient_dispersion(radius, wavelength, n_complex):
    k = 2 * np.pi / wavelength  # Wave number (k = 2π/λ)
    a1 = 6 * np.pi * 1j * (radius ** 3) * k / (3 * np.pi)  # Simplified Mie coefficient for electric dipole
    return a1 * n_complex

# Calculate Mie coefficients for both spheres using the complex refractive index
a1_sphere1_disp = mie_coefficient_dispersion(r1, wavelength_simulation, n_complex)
a1_sphere2_disp = mie_coefficient_dispersion(r2, wavelength_simulation, n_complex)

# Plot the absolute value of the polarizability for both spheres (Track 2)
fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(wavelength_simulation * 1e9, np.abs(a1_sphere1_disp), label='Sphere 1 (Track 2)', color='blue', linestyle='-', linewidth=2)
ax.plot(wavelength_simulation * 1e9, np.abs(a1_sphere2_disp), label='Sphere 2 (Track 2)', color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Wavelength (nm)', fontsize=14, fontweight='bold')
ax.set_ylabel('Polarizability |α(λ)|', fontsize=14, fontweight='bold')
ax.set_title('Track 2: Mie Coefficients for Dielectric Spheres (Dispersion)', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(False)
# Save the figure with a unique name based on timestamp
save_figure(fig, "mie_coefficients_track2", figs_dir)
# Show the figure
plt.show()

#Part II: Coupled-Dipole Equations

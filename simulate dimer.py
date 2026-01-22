import os
import numpy as np
import matplotlib.pyplot as plt
import time

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

# Constants
n_m = 1.0  # Refractive index of the background medium (assumed vacuum)
r1 = 90e-9  # Radius of sphere 1 in meters
r2 = 65e-9  # Radius of sphere 2 in meters
wavelength = np.linspace(500, 1000, 400) * 1e-9  # Wavelength range from 500 nm to 1000 nm

# Mie Coefficient Calculation
def mie_coefficient(radius, wavelength, n_m):
    k = 2 * np.pi / wavelength  # Wave number (k = 2π/λ)
    a1 = 6 * np.pi * 1j * (radius ** 3) * k / (3 * np.pi)  # Simplified Mie coefficient for electric dipole
    return a1

# Calculate Mie coefficients for both spheres
a1_sphere1 = mie_coefficient(r1, wavelength, n_m)
a1_sphere2 = mie_coefficient(r2, wavelength, n_m)

# Plot the absolute value of the polarizability for both spheres
fig, ax = plt.subplots()
ax.plot(wavelength * 1e9, np.abs(a1_sphere1), label='Sphere 1')
ax.plot(wavelength * 1e9, np.abs(a1_sphere2), label='Sphere 2')
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Polarizability |α(λ)|')
ax.legend()
ax.set_title('Mie Coefficients for Dielectric Spheres')
ax.grid(False)

# Save the figure with a unique name
save_figure(fig, "mie_coefficients", figs_dir)

# Show the figure
plt.show()

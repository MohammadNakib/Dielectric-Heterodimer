import numpy as np
import matplotlib.pyplot as plt

# Constants
n_m = 1.0  # Refractive index of the background medium (assumed vacuum)
r1 = 90e-9  # Radius of sphere 1 in meters
r2 = 65e-9  # Radius of sphere 2 in meters
wavelength = np.linspace(500, 1000, 400) * 1e-9  # Wavelength range from 500 nm to 1000 nm

# Mie Coefficient Calculation
def mie_coefficient(radius, wavelength, n_m):
    k = 2 * np.pi / wavelength  # Wave number
    a1 = 6 * np.pi * 1j * (radius ** 3) * k / (3 * np.pi)  # Simplified Mie coefficient for electric dipole
    return a1

# Calculate Mie coefficients for both spheres
a1_sphere1 = mie_coefficient(r1, wavelength, n_m)
a1_sphere2 = mie_coefficient(r2, wavelength, n_m)

# Plot the absolute value of the polarizability for both spheres
plt.plot(wavelength * 1e9, np.abs(a1_sphere1), label='Sphere 1')
plt.plot(wavelength * 1e9, np.abs(a1_sphere2), label='Sphere 2')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Polarizability |α(λ)|')
plt.legend()
plt.title('Mie Coefficients for Dielectric Spheres')
plt.grid(True)
plt.show()

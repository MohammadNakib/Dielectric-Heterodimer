import os
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from scipy.interpolate import interp1d
from scipy.linalg import solve
from matplotlib import font_manager

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

# Constants
n_m1 = 3.47  # Refractive index of Sphere 1 (constant)
n_m2 = 2.41  # Refractive index of Sphere 2 (constant)
r1 = 90e-9  # Radius of sphere 1 in meters
r2 = 65e-9  # Radius of sphere 2 in meters
n_m = 1.0  # Background index
gaps = np.array([10, 20, 40, 60, 80]) * 1e-9
wavelength_simulation = np.linspace(500, 1000, 400) * 1e-9  # Wavelength grid (500 nm to 1000 nm)

# Mie Coefficient Calculation for constant refractive index
def mie_coefficient(radius, wavelength, n_m):
    k = 2 * np.pi / wavelength  # Wave number (k = 2π/λ)
    a1 = 6 * np.pi * 1j * (radius ** 3) * k / (3 * np.pi)  # Simplified Mie coefficient for electric dipole
    return a1

# Track 1: Baseline Calculation for constant refractive indices
a1_sphere1 = mie_coefficient(r1, wavelength_simulation, n_m1)
a1_sphere2 = mie_coefficient(r2, wavelength_simulation, n_m2)

# Plot Mie coefficients for both spheres (Track 1)
fig, ax = plt.subplots(figsize=(10, 10))  # Square-shaped figure
ax.plot(wavelength_simulation * 1e9, np.abs(a1_sphere1), label=f'Sphere 1 (n={n_m1})', color='blue', linestyle='-', linewidth=2)
ax.plot(wavelength_simulation * 1e9, np.abs(a1_sphere2), label=f'Sphere 2 (n={n_m2})', color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Wavelength (nm)', fontsize=14, fontweight='bold')
ax.set_ylabel('Polarizability |α(λ)| (C·m²/V)', fontsize=14, fontweight='bold')
ax.set_title('Track 1: Mie Coefficients for Dielectric Spheres (Constant Refractive Index)', fontsize=16, fontweight='bold')
ax.tick_params(axis='both', which='major', labelsize=12, width=2)  # Bold the ticks and make them thicker
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontweight('bold')  # Bold the tick numbers (labels)
ax.grid(False)

# Draw a rectangular border around the figure
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)

# Add legend with bold font properties
legend_font = font_manager.FontProperties(weight='bold')
ax.legend(fontsize=12, prop=legend_font)

# Save the figure with a unique name based on timestamp
save_figure(fig, "mie_coefficients_track1", figs_dir)

# Show the figure
plt.show()

# Track 2: Dispersion for Crystalline Si (c-Si) for Sphere 1 only
# Load the refractive index and extinction coefficient data for crystalline Si
data = pd.read_csv('Wang-25C.csv')
wavelength_data = data['Wavelength'].values * 1e-9  # Convert to meters
n_data = data['n'].values
k_data = data['k'].values

# Interpolate the n(λ) and k(λ) values onto the simulation wavelength grid
n_interp = interp1d(wavelength_data, n_data, kind='cubic', fill_value="extrapolate")
k_interp = interp1d(wavelength_data, k_data, kind='cubic', fill_value="extrapolate")
n_interp_values = n_interp(wavelength_simulation)
k_interp_values = k_interp(wavelength_simulation)

# Calculate the complex refractive index for Sphere 1 (only)
n_complex = n_interp_values + 1j * k_interp_values

# Mie Coefficient Calculation with Dispersion for Sphere 1
def mie_coefficient_dispersion(radius, wavelength, n_complex):
    k = 2 * np.pi / wavelength  # Wave number (k = 2π/λ)
    a1 = 6 * np.pi * 1j * (radius ** 3) * k / (3 * np.pi)  # Simplified Mie coefficient for electric dipole
    return a1 * n_complex

# Constants for spheres
a1_sphere1_disp = mie_coefficient_dispersion(r1, wavelength_simulation, n_complex)
a1_sphere2_const = mie_coefficient(r2, wavelength_simulation, 2.41)  # Use constant refractive index for Sphere 2

# Plot the absolute value of the polarizability for both spheres (Track 2)
fig, ax = plt.subplots(figsize=(10, 10))  # Square-shaped figure
ax.plot(wavelength_simulation * 1e9, np.abs(a1_sphere1_disp), label=f'Sphere 1 (c-Si)', color='blue', linestyle='-', linewidth=2)
ax.plot(wavelength_simulation * 1e9, np.abs(a1_sphere2_const), label=f'Sphere 2 (n=2.41)', color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Wavelength (nm)', fontsize=14, fontweight='bold')
ax.set_ylabel('Polarizability |α(λ)| (C·m²/V)', fontsize=14, fontweight='bold')
ax.set_title('Track 2: Mie Coefficients for Dielectric Spheres (Dispersion)', fontsize=16, fontweight='bold')
ax.tick_params(axis='both', which='major', labelsize=12, width=2)  # Bold the ticks and make them thicker
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontweight('bold')  # Bold the tick numbers (labels)
ax.grid(False)

# Draw a rectangular border around the figure
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)

# Add legend with bold font properties
legend_font = font_manager.FontProperties(weight='bold')
ax.legend(fontsize=12, prop=legend_font)

# Save the figure with a unique name based on timestamp
save_figure(fig, "mie_coefficients_track2", figs_dir)

# Show the figure
plt.show()


# Plot Track 1 and Track 2 together for comparison
fig, ax = plt.subplots(figsize=(10, 10))  # Square-shaped figure

# Plot Track 1: Mie coefficients for Sphere 1 (n=3.47) and Sphere 2 (n=2.41)
ax.plot(wavelength_simulation * 1e9, np.abs(a1_sphere1), label=f'Track 1: Sphere 1 (n={n_m1})', color='blue', linestyle='-', linewidth=2)
ax.plot(wavelength_simulation * 1e9, np.abs(a1_sphere2), label=f'Track 1 and 2: Sphere 2 (n={n_m2})', color='red', linestyle='--', linewidth=2)

# Plot Track 2: Mie coefficients for Sphere 1 with dispersion (c-Si) and Sphere 2 with constant refractive index
ax.plot(wavelength_simulation * 1e9, np.abs(a1_sphere1_disp), label=f'Track 2: Sphere 1 (c-Si)', color='green', linestyle='-', linewidth=2)


# Add labels and title for the combined plot
ax.set_xlabel('Wavelength (nm)', fontsize=14, fontweight='bold')
ax.set_ylabel('Polarizability |α(λ)| (C·m²/V)', fontsize=14, fontweight='bold')
ax.set_title('Comparison: Track 1 vs Track 2 (Mie Coefficients)', fontsize=16, fontweight='bold')

# Make ticks bolder
ax.tick_params(axis='both', which='major', labelsize=12, width=2)  # Bold the ticks and make them thicker
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontweight('bold')  # Bold the tick numbers (labels)

# Remove grid lines for clarity
ax.grid(False)

# Draw a rectangular border around the figure
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)

# Add legend with bold font properties
legend_font = font_manager.FontProperties(weight='bold')
ax.legend(fontsize=12, prop=legend_font)

# Save the figure with a unique name based on timestamp
save_figure(fig, "track1_vs_track2_comparison", figs_dir)

# Show the figure
plt.show()


# Coupled-Dipole Model
def dyadic_green_function(r, wavelength):
    R = np.linalg.norm(r)  # Distance between spheres
    R_hat = r / R  # Unit vector in the direction of r
    k = 2 * np.pi / wavelength  # Wave number (k = 2π/λ)
    
    # Green's function for each wavelength
    G = (np.exp(1j * k * R) / (4 * np.pi * R**3)) * (
        (k**2 / R**2) * (np.eye(3) - np.outer(R_hat, R_hat)) + (1 - 1j * k * R) * (3 * np.outer(R_hat, R_hat) - np.eye(3))
    )
    return G

def solve_coupled_dipoles(r1, r2, a1_sphere1, a1_sphere2, wavelengths):
    dipoles = []
    condition_numbers = []
    for wavelength in wavelengths:
        r = r2 - r1  # Relative position between the spheres
        G = dyadic_green_function(r, wavelength)  # Dyadic Green's function
        
        # Form the 6x6 system (3 components per sphere)
        A = np.zeros((6, 6), dtype=complex)
        
        # For each wavelength, update the matrix A with the corresponding a1 values
        a1 = mie_coefficient(r1, wavelength, n_m1)
        A[:3, :3] = np.eye(3) * a1  # Sphere 1's contribution
        a2 = mie_coefficient(r2, wavelength, n_m2)
        A[3:, 3:] = np.eye(3) * a2  # Sphere 2's contribution
        
        A[:3, 3:] = G  # Interaction between spheres
        A[3:, :3] = G.T  # Symmetric interaction
        
        # Define the incident field (assuming propagation along +z-axis)
        E_inc1 = np.exp(1j * 2 * np.pi / wavelength)  # Incident field at Sphere 1
        E_inc2 = np.exp(1j * 2 * np.pi / wavelength)  # Incident field at Sphere 2
        
        # Right-hand side vector of the coupled equations (incident field + interaction terms)
        rhs = np.zeros(6, dtype=complex)
        rhs[:3] = E_inc1 * np.ones(3)  # Sphere 1's field
        rhs[3:] = E_inc2 * np.ones(3)  # Sphere 2's field
        
        # Solve the 6x6 system for this wavelength
        try:
            dipoles_wavelength = solve(A, rhs)
            dipoles.append(dipoles_wavelength)
            # Calculate the condition number of the matrix A
            cond_number = np.linalg.cond(A)
            condition_numbers.append(cond_number)
        except np.linalg.LinAlgError:
            print(f"Warning: Ill-conditioned matrix at wavelength {wavelength * 1e9} nm")

    return np.array(dipoles), np.array(condition_numbers)

# Wavelength grid (500 nm to 1000 nm)
wavelength_simulation = np.linspace(500, 1000, 400) * 1e-9  # Convert to meters

# Calculate dipoles and condition numbers for each wavelength
dipoles, condition_numbers = solve_coupled_dipoles(np.array([-10e-9 / 2, 0, 0]), np.array([10e-9 / 2, 0, 0]), a1_sphere1, a1_sphere2, wavelength_simulation)

# Extract dipole moments for both spheres
dipole_moment_sphere1 = dipoles[:, :3]  # Dipole moments of Sphere 1 (3 components)
dipole_moment_sphere2 = dipoles[:, 3:]  # Dipole moments of Sphere 2 (3 components)

# Plot Dipole Moments for both spheres
fig, ax = plt.subplots(figsize=(10, 10))

# Initialize arrays for storing dipole moment magnitudes for each wavelength
dipole_magnitude_sphere1 = []
dipole_magnitude_sphere2 = []

# Compute magnitude for each wavelength and plot
for i in range(len(wavelength_simulation)):
    # Compute magnitude of the dipole moment for Sphere 1 and Sphere 2 (for each wavelength)
    magnitude_sphere1 = np.linalg.norm(dipole_moment_sphere1[i])  # Compute magnitude for Sphere 1
    magnitude_sphere2 = np.linalg.norm(dipole_moment_sphere2[i])  # Compute magnitude for Sphere 2
    
    # Store the magnitudes for each wavelength
    dipole_magnitude_sphere1.append(magnitude_sphere1)
    dipole_magnitude_sphere2.append(magnitude_sphere2)

# Convert the lists to numpy arrays for plotting
dipole_magnitude_sphere1 = np.array(dipole_magnitude_sphere1)
dipole_magnitude_sphere2 = np.array(dipole_magnitude_sphere2)

# Plot the magnitudes of the dipole moments for both spheres
ax.plot(wavelength_simulation * 1e9, dipole_magnitude_sphere1, label=f'Sphere 1', linestyle='-', linewidth=2)
ax.plot(wavelength_simulation * 1e9, dipole_magnitude_sphere2, label=f'Sphere 2', linestyle='--', linewidth=2)

ax.set_xlabel('Wavelength (nm)', fontsize=14, fontweight='bold')
ax.set_ylabel('Dipole Moment Magnitude |d(λ)| (C·m²/V)', fontsize=14, fontweight='bold')
ax.set_title('Dipole Moments for Dielectric Spheres', fontsize=16, fontweight='bold')
ax.tick_params(axis='both', which='major', labelsize=12, width=2)

# Remove grid lines as per your request
ax.grid(False)

# Draw a rectangular border around the figure
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)

# Add legend with bold font properties
legend_font = font_manager.FontProperties(weight='bold')
ax.legend(fontsize=12, prop=legend_font)

# Save the figure with a unique name based on timestamp
save_figure(fig, "dipole_moments", figs_dir)

# Save dipole moments and condition number data to CSV
dipole_data = pd.DataFrame({
    'Wavelength (nm)': wavelength_simulation * 1e9,
    'Dipole Moment Sphere 1 (C·m²/V)': dipole_magnitude_sphere1,
    'Dipole Moment Sphere 2 (C·m²/V)': dipole_magnitude_sphere2
})
dipole_data.to_csv(os.path.join(data_dir, "dipole_moments.csv"), index=False)

condition_number_data = pd.DataFrame({
    'Wavelength (nm)': wavelength_simulation * 1e9,
    'Condition Number': condition_numbers.flatten()
})
condition_number_data.to_csv(os.path.join(data_dir, "condition_numbers.csv"), index=False)

# Show the figures
plt.show()

# Calculate dipoles and condition numbers for each wavelength
dipoles, condition_numbers = solve_coupled_dipoles(np.array([-10e-9 / 2, 0, 0]), np.array([10e-9 / 2, 0, 0]), a1_sphere1, a1_sphere2, wavelength_simulation)

# Plot Condition Number vs Wavelength
fig, ax = plt.subplots(figsize=(10, 10))

# Plot the data with bold font
ax.plot(wavelength_simulation * 1e9, condition_numbers, color='purple', linestyle='-', linewidth=2)

# Bold the x and y axis labels, and the title
ax.set_xlabel('Wavelength (nm)', fontsize=14, fontweight='bold')
ax.set_ylabel('Condition Number', fontsize=14, fontweight='bold')
ax.set_title('Condition Number vs Wavelength', fontsize=16, fontweight='bold')

# Bold the ticks and make them thicker
ax.tick_params(axis='both', which='major', labelsize=12, width=2)
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontweight('bold')

# Add grid lines for clarity (if needed)
ax.grid(False)

# Add legend with bold font properties
legend_font = font_manager.FontProperties(weight='bold')
ax.legend(fontsize=12, prop=legend_font)

# Save the figure with a unique name based on timestamp
save_figure(fig, "condition_number", figs_dir)

# Save condition number data to CSV
condition_number_data = pd.DataFrame({
    'Wavelength (nm)': wavelength_simulation * 1e9,  # Convert to nm
    'Condition Number': condition_numbers.flatten()
})
condition_number_data.to_csv(os.path.join(data_dir, "condition_numbers.csv"), index=False)

# Show the figure
plt.show()

#Step 4:Spectrum to plot (normalized extinction proxy)
#Material Ingestion (Track 2: c-Si)
try:
    data = pd.read_csv(os.path.join(data_dir, 'Wang-25C.csv'))
    n_interp = interp1d(data['Wavelength']*1e-9, data['n'], kind='cubic', fill_value="extrapolate")
    k_interp = interp1d(data['Wavelength']*1e-9, data['k'], kind='cubic', fill_value="extrapolate")
    n_si = n_interp(wavelength_simulation) + 1j * k_interp(wavelength_simulation)
except FileNotFoundError:
    print("Warning: c-Si data not found. Falling back to Track 1 constant indices.")
    n_si = np.full_like(wavelength_simulation, 3.47 + 0j) 

#  Physics Functions :Converts Mie a1 to polarizability alpha
def get_alpha_from_mie(radius, wvl, n_sphere):
    k = 2 * np.pi * n_m / wvl
    # Simplified a1 for dielectric sphere
    m = n_sphere / n_m
    x = k * radius
    a1 = (2j/3) * (x**3) * (m**2 - 1) / (m**2 + 2) # Quasi-static approximation
    # Conversion formula from PDF [cite: 71, 73]
    alpha = a1 * (3 * np.pi * 1j) / (k**3)
    return alpha

# Dyadic Green function
def dyadic_G(r_vec, wvl):
    R = np.linalg.norm(r_vec)
    R_hat = r_vec / R
    k = 2 * np.pi * n_m / wvl
    exp_term = np.exp(1j * k * R) / (4 * np.pi * R**3)
    term1 = (k*R)**2 * (np.eye(3) - np.outer(R_hat, R_hat))
    term2 = (1 - 1j*k*R) * (3*np.outer(R_hat, R_hat) - np.eye(3))
    return exp_term * (term1 + term2)

#Coupled-Dipole Solver
def solve_dimer(wvl, gap, n1, n2, polarization='x'):
    d_dist = r1 + r2 + gap
    r01, r02 = np.array([-d_dist/2, 0, 0]), np.array([d_dist/2, 0, 0])
    k_vec = (2 * np.pi * n_m / wvl) * np.array([0, 0, 1])  
    alpha1 = get_alpha_from_mie(r1, wvl, n1)
    alpha2 = get_alpha_from_mie(r2, wvl, n2)
    G12 = dyadic_G(r01 - r02, wvl)
    
    # E_inc polarization 
    E0_vec = np.array([1, 0, 0]) if polarization == 'x' else np.array([0, 1, 0])
    E_inc1 = E0_vec * np.exp(1j * np.dot(k_vec, r01))
    E_inc2 = E0_vec * np.exp(1j * np.dot(k_vec, r02))
    
    # Matrix: [I, -alpha1*G; -alpha2*G, I] * [d1; d2] = [alpha1*E1; alpha2*E2]
    M = np.eye(6, dtype=complex)
    M[0:3, 3:6] = -alpha1 * G12
    M[3:6, 0:3] = -alpha2 * G12
    rhs = np.concatenate([alpha1 * E_inc1, alpha2 * E_inc2])
    
    sol = solve(M, rhs)
    d1, d2 = sol[0:3], sol[3:6]
    
    # S(lambda) proxy
    S = np.imag(np.dot(np.conj(E_inc1), d1) + np.dot(np.conj(E_inc2), d2))
    return S, np.linalg.cond(M)

# Parametric Sweep: Gap and Polarizibility
results = {'x': [], 'y': []}
for pol in ['x', 'y']:
    plt.figure(figsize=(8, 6))
    for g in gaps:
        spectrum = [solve_dimer(w, g, n_si[i], 2.41, pol)[0] for i, w in enumerate(wavelength_simulation)]
        results[pol].append(spectrum)
        plt.plot(wavelength_simulation*1e9, spectrum, label=f'g={g*1e9:.0f}nm')
    #Plot
    plt.title(f"Normalized Extinction Proxy: Polarization {pol.upper()}", fontsize=14, fontweight='bold')
    plt.xlabel("Wavelength (nm)", fontsize=12, fontweight='bold')
    plt.ylabel("S(λ) (Normalized Extinction Proxy)", fontsize=12, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=12, width=2)
    for label in (plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
        label.set_fontweight('bold')
    
    plt.legend(fontsize=14, prop={'weight': 'bold'})
    plt.grid(False) 
    plt.savefig(f"figs/sweep_pol_{pol}.pdf")
    plt.show()

import os
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import miepython
import scipy.linalg as linalg
from scipy.optimize import curve_fit

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

# Function to save data as CSV
def save_data(data, file_name, data_dir):
    data.to_csv(os.path.join(data_dir, file_name), index=False)  # Save data in CSV format
    print(f"Data saved as: {file_name}")  # Print the saved file's path

print('Track 1: Sphere 1 (n=3.47) and Sphere 2 (n=2.41)')

# Constants for Sphere 1 and Sphere 2
n_m1 = 3.47  # Refractive index of Sphere 1 (constant)
n_m2 = 2.41  # Refractive index of Sphere 2 (constant)
n1_imag = 0.0  # Imaginary part of refractive index for sphere 1 (lossless)
n2_imag = 0.0  # Imaginary part of refractive index for sphere 2 (lossless)
n_m = 1.0  # Background index
r1 = 90e-9  # Radius of sphere 1 in meters
r2 = 65e-9  # Radius of sphere 2 in meters
R = r1 + r2 + (20 * 1e-9)  # Total Distance between two particles; Gap = 20 nm
wavelength = np.linspace(500, 1000, 400) * 1e-9  # Wavelength grid (500 nm to 1000 nm)

# Part I: Mie Coefficient Calculation (a1)
def mie_coefficients(n_particle_real, n_particle_imag, radius, wavelength):
    k = 2 * np.pi / wavelength  # Wave number (m^-1)
    n_particle = n_particle_real + 1j * n_particle_imag  # Complex refractive index
    m = n_particle / n_m   # Relative refractive index (m = n_particle / n_medium)
    a1 = np.zeros_like(wavelength, dtype=complex)  # Initialize array for a1 coefficient

    # Loop over all wavelengths and calculate Mie coefficients
    for i in range(len(wavelength)):
        # Calculate Mie scattering efficiencies using miepython.efficiencies
        qext, qsca, qback, g = miepython.efficiencies(m, 2 * radius * 1e9, wavelength[i] * 1e9)  # Convert wavelength to nm for mie.efficiencies()
        a1[i] = qext  # We use the extinction efficiency qext as a1
    
    return a1

# Compute Mie Coefficients for Both Spheres
a1_sphere1 = mie_coefficients(n_m1, n1_imag, r1, wavelength)
a1_sphere2 = mie_coefficients(n_m2, n2_imag, r2, wavelength)

# Plot Mie coefficients for both spheres
plt.figure(figsize=(10, 8))  # Set the figure size
# Plot for Sphere 1
plt.plot(wavelength * 1e9, np.abs(a1_sphere1), label=r'$\mathbf{Sphere\ 1\ (n\ =\ 3.47)}$', color='blue', linewidth=2)
# Plot for Sphere 2
plt.plot(wavelength * 1e9, np.abs(a1_sphere2), label=r'$\mathbf{Sphere\ 2\ (n\ =\ 2.41)}$', color='red', linewidth=2)
# Customize plot
plt.title('Mie Coefficients for Sphere 1 and Sphere 2', fontsize=16, fontweight='bold')  # Title with bold font
plt.xlabel('Wavelength (nm)', fontsize=14, fontweight='bold')  # X-axis label with bold font
plt.ylabel('Mie Coefficient |a1(λ)|', fontsize=14, fontweight='bold')  # Y-axis label with bold font
plt.legend()
plt.tight_layout()

# Remove grid and ticks
plt.grid(False)  # No grid
plt.tick_params(axis='both', which='major', labelsize=14, width=2, direction='in', length=6, colors='black')  # Bold ticks

# Save the plot in PDF format
save_figure(plt, "Mie_Coefficients_Track_1", figs_dir)

# Save the data in CSV format
data_sphere1 = pd.DataFrame({
    'Wavelength (nm)': wavelength * 1e9,  # Wavelength in nm
    'Sphere 1 |a1(λ)|': np.abs(a1_sphere1)
})
data_sphere2 = pd.DataFrame({
    'Wavelength (nm)': wavelength * 1e9,  # Wavelength in nm
    'Sphere 2 |a1(λ)|': np.abs(a1_sphere2)
})
save_data(data_sphere1, 'sphere1_mie_coefficients.csv', data_dir)
save_data(data_sphere2, 'sphere2_mie_coefficients.csv', data_dir)

# Show plot
plt.show()

# Step 1: Mie-derived electric dipole polarizability calculation
def mie_derived_polarizability(a1, wavelength, n_m):
    """
    Calculate Mie-derived electric dipole polarizability for each sphere.
    """
    k = 2 * np.pi / wavelength  # Wave number
    polarizability = (6 * np.pi * 1j / k**3) * a1  # Using the formula for polarizability
    return polarizability

# Compute polarizabilities for both spheres
alpha_sphere1 = mie_derived_polarizability(a1_sphere1, wavelength, n_m)
alpha_sphere2 = mie_derived_polarizability(a1_sphere2, wavelength, n_m)

# Plot Mie-derived polarizability for both spheres
plt.figure(figsize=(10, 8))  # Set the figure size
# Plot for Sphere 1
plt.plot(wavelength * 1e9, np.abs(alpha_sphere1), label=r'$\mathbf{Sphere\ 1\ (n\ =\ 3.47)}$', color='blue', linewidth=2)
# Plot for Sphere 2
plt.plot(wavelength * 1e9, np.abs(alpha_sphere2), label=r'$\mathbf{Sphere\ 2\ (n\ =\ 2.41)}$', color='red', linewidth=2)
# Customize plot
plt.title('Mie-derived Electric Dipole Polarizability for Sphere 1 and Sphere 2', fontsize=16, fontweight='bold')  # Title with bold font
plt.xlabel('Wavelength (nm)', fontsize=14, fontweight='bold')  # X-axis label with bold font
plt.ylabel('Polarizability |α(λ)|', fontsize=14, fontweight='bold')  # Y-axis label with bold font
plt.legend()
plt.tight_layout()

# Remove grid and ticks
plt.grid(False)  # No grid
plt.tick_params(axis='both', which='major', labelsize=14, width=2, direction='in', length=6, colors='black')  # Bold ticks

# Save the plot in PDF format
save_figure(plt, "Electric_Dipole_Polarizability_Track_1", figs_dir)

# Save the data in CSV format for both spheres
data_alpha_sphere1 = pd.DataFrame({
    'Wavelength (nm)': wavelength * 1e9,  # Wavelength in nm
    'Sphere 1 |α1(λ)|': np.abs(alpha_sphere1)
})
data_alpha_sphere2 = pd.DataFrame({
    'Wavelength (nm)': wavelength * 1e9,  # Wavelength in nm
    'Sphere 2 |α2(λ)|': np.abs(alpha_sphere2)
})

# Save data to CSV
save_data(data_alpha_sphere1, 'sphere1_polarizability_Track_1.csv', data_dir)
save_data(data_alpha_sphere2, 'sphere2_polarizability_Track_1.csv', data_dir)

# Show plot
plt.show()

def dyadic_green_function(R, k):
    """
    Computes Green function G(R) for R = r_i - r_j.
    Formula: (e^ikR / 4piR^3) * [ (kR)^2(I-RR) + (1-ikR)(3RR-I) ]
    """
    dist = np.linalg.norm(R)
    if dist == 0: return np.zeros((3,3), dtype=complex)
    
    R_hat = R / dist
    RR = np.outer(R_hat, R_hat)
    I = np.eye(3)
    
    # Scalar Green function part
    scalar_G = np.exp(1j * k * dist) / (4 * np.pi * dist**3)
    
    # Far-field term (proportional to 1/R)
    term1 = (k * dist)**2 * (I - RR)
    
    # Near-field/Intermediate terms (proportional to 1/R^3 and 1/R^2)
    term2 = (1 - 1j * k * dist) * (3 * RR - I)
    
    return scalar_G * (term1 + term2)

# Function to solve coupled-dipoles for each wavelength
def solve_coupled_dipoles(alpha1, alpha2, E_inc1, E_inc2, R, k_values):
    """
    Solve the coupled-dipole equations for two particles for multiple wavelengths.
    alpha1, alpha2: Polarizabilities of sphere 1 and sphere 2 (arrays of size len(wavelength))
    E_inc1, E_inc2: Incident fields for spheres 1 and 2
    R: Distance between the two spheres
    k_values: Array of wave numbers (for each wavelength)
    """
    dipoles = np.zeros((len(k_values), 6), dtype=complex)  # Store dipoles for each wavelength

    # Loop over each wavelength and solve the system
    for i, k in enumerate(k_values):
        G_12 = dyadic_green_function(R, k)  # Interaction between spheres
        G_21 = G_12.T  # Green function is symmetric

        # Create the 6x6 matrix A for this specific wavelength
        A = np.zeros((6, 6), dtype=complex)

        # Set self-interactions for sphere 1 and sphere 2 using the polarizability for this wavelength
        A[:3, :3] = np.eye(3) * alpha1[i]  # Use alpha1[i] for the current wavelength
        A[3:, 3:] = np.eye(3) * alpha2[i]  # Use alpha2[i] for the current wavelength

        # Set interaction terms between spheres
        A[:3, 3:] = alpha1[i] * G_12  # Interaction between spheres for sphere 1
        A[3:, :3] = alpha2[i] * G_21  # Interaction between spheres for sphere 2

        # Right-hand side (incident fields)
        rhs = np.concatenate([alpha1[i] * E_inc1, alpha2[i] * E_inc2])  # Incident field scaled by polarizability

        # Solve for dipoles for this wavelength
        dipoles[i, :] = linalg.solve(A, rhs)

    return dipoles

# Compute the condition number of matrix A
def compute_condition_number(A):
    cond_number = np.linalg.cond(A)
    return cond_number

# Example usage (based on the parameters you've provided)
R = np.array([r1 + r2 + 20e-9, 0, 0])  # Distance vector between spheres (gap = 20 nm)
k_values = 2 * np.pi / wavelength  # Wave number for each wavelength
E_inc1 = np.ones(3)  # Incident field for sphere 1
E_inc2 = np.ones(3)  # Incident field for sphere 2

# Solve for dipoles for each wavelength
dipoles = solve_coupled_dipoles(alpha_sphere1, alpha_sphere2, E_inc1, E_inc2, R, k_values)

# Print the dipoles for the first few wavelengths
print("Dipoles for Sphere 1 and Sphere 2:", dipoles[:5, :])  # Display first 5 values

# Compute condition number for matrix A (Example: Identity matrix)
A_example = np.eye(6)
cond_number_example = compute_condition_number(A_example)
print("Condition Number:", cond_number_example)

def solve_dimer_system(alpha1_arr, alpha2_arr, r1_pos, r2_pos, wl_arr, pol_vec):
    """
    Solves (I - alpha*G) * d = alpha * E_inc
    Returns Extinction Proxy Spectrum.
    """
    num_pts = len(wl_arr)
    S_lambda = np.zeros(num_pts)
    R_vec = r1_pos - r2_pos # Vector from 2 to 1
    
    for i in range(num_pts):
        k = 2 * np.pi / wl_arr[i]
        a1 = alpha1_arr[i]
        a2 = alpha2_arr[i]
        
        # 1. Build Interaction Matrix M (6x6)
        G12 = dyadic_green_function(R_vec, k)
        G21 = G12.T
        
        M = np.eye(6, dtype=complex)
        M[0:3, 3:6] = -a1 * G12  # Interaction block 1-2
        M[3:6, 0:3] = -a2 * G21  # Interaction block 2-1
        
        # 2. Build RHS (Incident Field)
        E0 = 1.0
        E_inc = E0 * pol_vec
        b = np.concatenate([a1 * E_inc, a2 * E_inc])
        
        # 3. Solve Linear System
        d_vec = linalg.solve(M, b)
        d1, d2 = d_vec[0:3], d_vec[3:6]
        
        # 4. Extinction Proxy: Im(E_inc* . d)
        ext = np.imag(np.vdot(E_inc, d1) + np.vdot(E_inc, d2))
        S_lambda[i] = ext
        
    return S_lambda



# Normalized Extinction Proxy

gaps_nm = [10, 20, 40, 60, 80]
polarizations = {
    'A': {'vec': np.array([1, 0, 0]), 'name': r'$E \parallel \hat{x}$ (Longitudinal)'},
    'B': {'vec': np.array([0, 1, 0]), 'name': r'$E \parallel \hat{y}$ (Transverse)'}
}

simulation_data = {} # To store results for Fano fit

for pol_key, pol_info in polarizations.items():
    plt.figure(figsize=(10, 7))
    print(f"Processing Polarization {pol_key}...")
    
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(gaps_nm)))
    
    for idx, g_nm in enumerate(gaps_nm):
        # Define Geometry
        gap = g_nm * 1e-9
        center_dist = r1 + r2 + gap
        pos1 = np.array([-center_dist/2, 0, 0])
        pos2 = np.array([+center_dist/2, 0, 0])
        
        # Run Solver
        spectrum = solve_dimer_system(alpha_sphere1, alpha_sphere2, pos1, pos2, wavelength, pol_info['vec'])
        
        # Store
        simulation_data[(pol_key, g_nm)] = spectrum
        
        # Plot
        plt.plot(wavelength*1e9, spectrum, linewidth=2.5, color=colors[idx], label=f'Gap = {g_nm} nm')

    plt.title(f'Polarization : {pol_info["name"]}', fontsize=16, fontweight='bold')
    plt.xlabel('Wavelength (nm)', fontsize=14, fontweight='bold')
    plt.ylabel('Normalized Extinction Proxy (a.u.)', fontsize=14, fontweight='bold')
    plt.legend(title='Gap Size', frameon=True, edgecolor='black')
    plt.tick_params(direction='in', length=6, width=2)
    plt.tight_layout()
    save_figure(plt, f"Normalized_Extinction_Polarization_{pol_key}", figs_dir)  # Save the figure for polarization
    plt.show()


# PART IV: FANO FIT ANALYSIS (WITH ANNOTATION)
print('\n--- Performing Fano Fit Analysis ---')

# Select Data: Polarization A (Longitudinal), Gap 20nm
target_pol = 'A'
target_gap = 20
y_obs = simulation_data[(target_pol, target_gap)]
x_obs = wavelength * 1e9

# Fano Formula
def fano_func(lam, y0, A, lam0, Gamma, q):
    eps = (lam - lam0) / Gamma
    return y0 + A * ((q + eps)**2) / (1 + eps**2)

# Initial Guesses
peak_idx = np.argmax(y_obs)
p0 = [
    np.min(y_obs),              # y0 (offset)
    np.max(y_obs) - np.min(y_obs), # A (amplitude)
    x_obs[peak_idx],            # lam0 (center)
    50.0,                       # Gamma (width)
    1.0                         # q (asymmetry)
]

try:
    popt, pcov = curve_fit(fano_func, x_obs, y_obs, p0=p0, maxfev=10000)
    y_fit = fano_func(x_obs, *popt)
    residuals = y_obs - y_fit
    
    # Unpack parameters
    fit_y0, fit_A, fit_lam0, fit_Gamma, fit_q = popt
    print(f"Fano Fit Results:")
    print(f"  Resonance (λ0): {fit_lam0:.2f} nm")
    print(f"  Width (Γ):      {fit_Gamma:.2f} nm")
    print(f"  Asymmetry (q):  {fit_q:.4f}")

    # --- Plotting with Annotation ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    # Main Plot
    ax1.plot(x_obs, y_obs, 'ko', markersize=4, alpha=0.5, label='Simulation')
    ax1.plot(x_obs, y_fit, 'r-', linewidth=2.5, label='Fano Fit')
    ax1.set_ylabel('Normalized Extinction (a.u.)', fontsize=14, fontweight='bold')
    ax1.set_title(f'Fano Analysis: Gap {target_gap} nm, Pol {target_pol}', fontsize=16, fontweight='bold')
    ax1.legend(frameon=True, edgecolor='black', loc='upper right')
    ax1.tick_params(direction='in', length=6, width=2)
    
    # Add Text Box with Parameters
    text_str = (f"Fit Parameters:\n"
                f"$\\lambda_0 = {fit_lam0:.2f}$ nm\n"
                f"$\\Gamma = {fit_Gamma:.2f}$ nm\n"
                f"$q = {fit_q:.4f}$")
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, linewidth=1.5, edgecolor='black')
    ax1.text(0.03, 0.95, text_str, transform=ax1.transAxes, fontsize=12,
             verticalalignment='top', bbox=props, fontweight='bold')

    # Residuals Plot
    ax2.plot(x_obs, residuals, 'b-', linewidth=1.5)
    ax2.axhline(0, color='gray', linestyle='--', linewidth=2)
    ax2.set_ylabel('Residuals', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Wavelength (nm)', fontsize=14, fontweight='bold')
    ax2.tick_params(direction='in', length=6, width=2)
    
    plt.tight_layout()
    save_figure(plt, "Fano_Fit_Result", figs_dir)  # Save the figure for Fano fit
    plt.show()

except Exception as e:
    print(f"Fano fit failed: {e}")

print("\n--- Track 1 Complete. Check 'figs/' folder. ---")








# Track 2: Mie Coefficient Calculation for Crystalline Si Dispersion (for both spheres)
print('Track 2: Mie Coefficient Calculation for Crystalline Si Dispersion (for both spheres)')
print('Calculating Mie Coeffiecnt for Track 2...')
# Load the crystalline Si dispersion data (n, k from the CSV file)
# Assuming the CSV file has columns: Wavelength, n, k
# You can adjust the file path as needed
si_dispersion_data = pd.read_csv('data/Wang-25C.csv')  # CSV file containing Wavelength, n, k

# Extract the wavelength, n, and k values
wavelength_data = si_dispersion_data['Wavelength'].values  # Wavelength in nm
n_si_real = si_dispersion_data['n'].values  # Real part of the refractive index
k_si_imag = si_dispersion_data['k'].values  # Imaginary part of the refractive index

# Interpolate the data to match the simulation wavelength grid
n_si_real_interp = np.interp(wavelength * 1e9, wavelength_data, n_si_real)  # Interpolated n values
k_si_imag_interp = np.interp(wavelength * 1e9, wavelength_data, k_si_imag)  # Interpolated k values

# Now, we use the interpolated refractive index data for both spheres (crystalline Si for both)
n_m1_real = n_si_real_interp  # Real part for Sphere 1 (Si)
n_m1_imag = k_si_imag_interp  # Imaginary part for Sphere 1 (Si)

n_m2_real = n_si_real_interp  # Real part for Sphere 2 (Si)
n_m2_imag = k_si_imag_interp  # Imaginary part for Sphere 2 (Si)

# Track 2: Mie Coefficient Calculation for Crystalline Si Dispersion (for both spheres)

def mie_coefficients(n_particle_real, n_particle_imag, radius, wavelength):
    k = 2 * np.pi / wavelength  # Wave number (m^-1)
    n_particle = n_particle_real + 1j * n_particle_imag  # Complex refractive index
    m = n_particle / n_m  # Relative refractive index (m = n_particle / n_medium)

    a1 = np.zeros_like(wavelength, dtype=complex)  # Initialize array for a1 coefficient

    # Loop over all wavelengths and calculate Mie coefficients
    for i in range(len(wavelength)):
        # Ensure that the wavelength is passed as a scalar value (not an array)
        wavelength_scalar = wavelength[i]  # Extract scalar wavelength for each iteration
        # Calculate Mie scattering efficiencies using miepython.efficiencies
        qext, qsca, qback, g = miepython.efficiencies(m, 2 * radius * 1e9, wavelength_scalar * 1e9)  # Convert wavelength to nm for mie.efficiencies()
        
        # Check if qext is a scalar
        if np.ndim(qext) > 0:
            qext = qext[0]  # In case qext is an array, take the first element
        
        a1[i] = qext  # We use the extinction efficiency qext as a1
    
    return a1

# Now, using the updated function to calculate Mie coefficients for both spheres
a12_sphere1 = mie_coefficients(n_m1_real, n_m1_imag, r1, wavelength)
a12_sphere2 = mie_coefficients(n_m2_real, n_m2_imag, r2, wavelength)

# Plotting Mie coefficients for both spheres
plt.figure(figsize=(10, 8))  # Set the figure size

# Plot for Sphere 1 with crystalline Si dispersion
plt.plot(wavelength * 1e9, np.abs(a12_sphere1), label=r'$\mathbf{Sphere\ 1\ (c-Si\ Dispersion)}$', color='blue', linewidth=2)

# Plot for Sphere 2 with crystalline Si dispersion
plt.plot(wavelength * 1e9, np.abs(a12_sphere2), label=r'$\mathbf{Sphere\ 2\ (c-Si\ Dispersion)}$', color='red', linewidth=2)

# Customize plot
plt.title('Mie Coefficients for Sphere 1 and Sphere 2 (Si Dispersion)', fontsize=16, fontweight='bold')  # Title with bold font
plt.xlabel('Wavelength (nm)', fontsize=14, fontweight='bold')  # X-axis label with bold font
plt.ylabel('Mie Coefficient |a1(λ)|', fontsize=14, fontweight='bold')  # Y-axis label with bold font
plt.legend()

# Customize plot appearance
plt.tight_layout()

# Remove grid and ticks
plt.grid(False)  # No grid
plt.tick_params(axis='both', which='major', labelsize=14, width=2, direction='in', length=6, colors='black')  # Bold ticks

# Save the plot in PDF format
save_figure(plt, "mie_coefficients_plot_sphere1_and_2_crystalline_Si_dispersion", figs_dir)

# Show the plot
plt.show()
print('Mie-derived electric dipole polarizability for each sphere (Track 2)...')
# Track 2: Mie-derived Electric Dipole Polarizability Calculation (for both spheres)

def mie_derived_polarizability(a1, wavelength, n_m):
    """
    Calculate Mie-derived electric dipole polarizability for each sphere.
    """
    k = 2 * np.pi / wavelength  # Wave number (m^-1)
    polarizability = (6 * np.pi * 1j / k**3) * a1  # Using the formula for polarizability
    return polarizability

# Calculate polarizability for Sphere 1 and Sphere 2
alpha2_sphere1 = mie_derived_polarizability(a12_sphere1, wavelength, n_m)
alpha2_sphere2 = mie_derived_polarizability(a12_sphere2, wavelength, n_m)

# Print the first few polarizability values for verification
print("Polarizability for Sphere 1 (First 5 values):", alpha2_sphere1[:5])
print("Polarizability for Sphere 2 (First 5 values):", alpha2_sphere2[:5])

# Plotting Mie-derived polarizability for both spheres
plt.figure(figsize=(10, 8))  # Set the figure size

# Plot for Sphere 1 with crystalline Si dispersion
plt.plot(wavelength * 1e9, np.abs(alpha2_sphere1), label=r'$\mathbf{Sphere\ 1\ (c-Si\ Dispersion)}$', color='blue', linewidth=2)

# Plot for Sphere 2 with crystalline Si dispersion
plt.plot(wavelength * 1e9, np.abs(alpha2_sphere2), label=r'$\mathbf{Sphere\ 2\ (c-Si\ Dispersion)}$', color='red', linewidth=2)

# Customize plot
plt.title('Mie-derived Electric Dipole Polarizability for Sphere 1 and Sphere 2 (Si Dispersion)', fontsize=16, fontweight='bold')  # Title with bold font
plt.xlabel('Wavelength (nm)', fontsize=14, fontweight='bold')  # X-axis label with bold font
plt.ylabel('Polarizability |α(λ)|', fontsize=14, fontweight='bold')  # Y-axis label with bold font
plt.legend()

# Customize plot appearance
plt.tight_layout()

# Remove grid and ticks
plt.grid(False)  # No grid
plt.tick_params(axis='both', which='major', labelsize=14, width=2, direction='in', length=6, colors='black')  # Bold ticks

# Save the plot in PDF format
save_figure(plt, "polarizability_plot_sphere1_and_2_crystalline_Si_dispersion", figs_dir)

# Show the plot
plt.show()

print('Comaparison between Mie Coefficient between Track 1 and Track 2')

# Plotting comparison between Mie Coefficients from Track 1 and Track 2
plt.figure(figsize=(12, 10))  # Set the figure size

# Plot for Track 1: Sphere 1 with crystalline Si dispersion
plt.plot(wavelength * 1e9, np.abs(a1_sphere1), label=r'$\mathbf{Track\ 1: Sphere\ 1\ (n=3.47)}$', color='blue', linewidth=2)

# Plot for Track 1: Sphere 2 with crystalline Si dispersion
plt.plot(wavelength * 1e9, np.abs(a1_sphere2), label=r'$\mathbf{Track\ 1: Sphere\ 2\ (n=2.41)}$', color='red', linewidth=2)

# Plot for Track 2: Sphere 1 with crystalline Si dispersion (Track 2)
plt.plot(wavelength * 1e9, np.abs(a12_sphere1), label=r'$\mathbf{Track\ 2: Sphere\ 1\ (c-Si Dispersion)}$', color='green', linewidth=2)

# Plot for Track 2: Sphere 2 with crystalline Si dispersion (Track 2)
plt.plot(wavelength * 1e9, np.abs(a12_sphere2), label=r'$\mathbf{Track\ 2: Sphere\ 2\ (c-Si Dispersion)}$', color='orange', linewidth=2)

# Customize plot
plt.title('Comparison of Mie Coefficients between Track 1 and Track 2', fontsize=16, fontweight='bold')  # Title with bold font
plt.xlabel('Wavelength (nm)', fontsize=14, fontweight='bold')  # X-axis label with bold font
plt.ylabel('Mie Coefficient |a1(λ)|', fontsize=14, fontweight='bold')  # Y-axis label with bold font

# Add legend to differentiate Track 1 and Track 2
plt.legend(loc='upper right', fontsize=12, title_fontsize=14)

# Customize plot appearance
plt.tight_layout()

# Remove grid and ticks
plt.grid(False)  # No grid
plt.tick_params(axis='both', which='major', labelsize=14, width=2, direction='in', length=6, colors='black')  # Bold ticks

# Save the plot in PDF format
save_figure(plt, "mie_coefficients_comparison__track1_and_2", figs_dir)

# Show the plot
plt.show()
print("\n--- Project Complete. Check 'figs/' folder. ---")

---

# Dielectric Sphere Mie Coefficients and Coupled-Dipole Model

This code simulates the optical properties of dielectric spheres using Mie theory and a coupled-dipole model. The simulation includes calculations for Mie coefficients, dipole moments, and condition numbers for dielectric spheres with varying refractive indices and sizes. The code also compares the Mie coefficients for constant and dispersive refractive indices (e.g., crystalline silicon) and saves the results in CSV files.

### Requirements

To run this code, you will need to install the following Python libraries:

* `numpy`
* `matplotlib`
* `pandas`
* `scipy`

You can install these dependencies using `pip`:

```bash
pip install numpy matplotlib pandas scipy
```

### File Structure

* `figs`: Directory where the generated figures (plots) will be saved.
* `data`: Directory where the resulting data (e.g., dipole moments and condition numbers) will be saved as CSV files.
* `Wang-25C.csv`: A CSV file containing the refractive index (`n`) and extinction coefficient (`k`) data for crystalline silicon (c-Si). This file must be present in the working directory for dispersion calculations.

### Code Explanation

#### 1. **Mie Coefficient Calculation**

The code calculates the Mie coefficients for two dielectric spheres with radii `r1` and `r2`, and refractive indices `n_m1` and `n_m2`. It then plots the Mie coefficients as a function of wavelength for both spheres.

#### 2. **Track 1: Constant Refractive Index**

This track uses constant refractive indices for the spheres and calculates the Mie coefficients for both spheres. The results are plotted, showing the polarizability as a function of wavelength.

#### 3. **Track 2: Dispersion for Crystalline Silicon**

In this track, the refractive index and extinction coefficient of crystalline silicon are interpolated and used to calculate the Mie coefficients for Sphere 1 with dispersive refractive index values. The results are compared with the constant refractive index case for Sphere 2.

#### 4. **Coupled-Dipole Model**

The code implements a coupled-dipole model to solve for the dipole moments of two interacting dielectric spheres. The interaction is calculated using the Dyadic Green's function, and the dipole moments are computed and plotted for various wavelengths.

#### 5. **Parametric Sweep: Gap and Polarization**

The code performs a parametric sweep over the gap distance between the two spheres and polarization states (`x` and `y`). For each gap, the normalized extinction proxy `S(λ)` is calculated and plotted for both polarization directions.

### Functions

* **`mie_coefficient(radius, wavelength, n_m)`**: Calculates the Mie coefficient for a dielectric sphere with a given radius, wavelength, and refractive index.
* **`mie_coefficient_dispersion(radius, wavelength, n_complex)`**: Calculates the Mie coefficient for a sphere with a dispersive refractive index.
* **`dyadic_green_function(r, wavelength)`**: Computes the Dyadic Green's function for two spheres separated by a distance `r` at a given wavelength.
* **`solve_coupled_dipoles(r1, r2, a1_sphere1, a1_sphere2, wavelengths)`**: Solves the coupled-dipole system for two spheres using their Mie coefficients and computes the dipole moments.
* **`solve_dimer(wvl, gap, n1, n2, polarization)`**: Solves the coupled-dipole model for a pair of spheres with varying gap distance and polarization direction.
* **`save_figure(fig, name_prefix, figs_dir)`**: Saves the generated figure as a PDF with a unique timestamped filename.

### How to Use

1. **Run the code**: Simply execute the script in a Python environment with the necessary dependencies installed.
2. **Wavelength Grid**: The code simulates over a wavelength range of 500-1000 nm with 400 points.
3. **Refractive Index Data**: The code expects a CSV file named `Wang-25C.csv` containing refractive index and extinction coefficient data for crystalline silicon. If this file is not found, the code falls back to using a constant refractive index of 3.47 for Sphere 1.
4. **Outputs**: The following outputs will be generated:

   * Mie coefficient plots for both spheres (`Track 1` and `Track 2`).
   * Dipole moment plots for both spheres.
   * Condition number vs. wavelength plot.
   * Polarization sweep plots (for `x` and `y` polarizations).

### Example Outputs

* **Mie Coefficients Plot**: Shows the polarizability of the two spheres as a function of wavelength.
* **Dipole Moment Magnitude**: Plots the magnitude of the dipole moments for both spheres.
* **Condition Number Plot**: Displays the condition number of the system matrix for each wavelength.

### Saving Figures and Data

Figures will be saved in the `figs` folder, and data will be saved in the `data` folder as CSV files. The filenames are timestamped to avoid overwriting previous files.

### Notes

* **Wavelengths**: The wavelength range is set from 500 nm to 1000 nm. Modify the `wavelength_simulation` array to change the wavelength range or resolution.
* **Data File**: The `Wang-25C.csv` file must be present in the directory for the code to work properly. It contains the refractive index and extinction coefficient data for crystalline silicon.
* **Plot Customization**: The plots are customized with bold axis labels, grid lines are removed for clarity, and a rectangular border is drawn around the figures.

### References

* The formulas for calculating Mie coefficients are based on standard Mie theory for dielectric spheres.
* The Dyadic Green's function implementation follows a commonly used form for coupled-dipole models.

---


# Dielectric Heterodimer Simulation and Fano Fit Analysis

## Overview

This Python project simulates the optical response of a two-sphere dielectric heterodimer using an electric-dipole coupled-dipole model. The goal is to analyze asymmetric spectral features and fit them with a Fano lineshape, evaluating how the asymmetry changes with the gap and polarization. The simulation includes the effects of dispersion in materials like crystalline silicon (Si), with refractive index data interpolated onto the simulation wavelength grid.

## Key Features

- **Mie Coefficients and Polarizability Calculation**: The Mie scattering coefficients are computed for two spheres, followed by the calculation of their electric dipole polarizability.
- **Coupled-Dipole Model**: The dipole moments are calculated by solving the coupled-dipole equations, considering the interactions between the two spheres.
- **Dyadic Green's Function**: The dyadic Green's function is employed to model particle-particle interactions, solving a 6×6 system for each wavelength.
- **Fano Fit**: The asymmetric spectral features are fitted using the Fano lineshape, extracting resonance wavelength, width, and asymmetry.
- **Dispersion Model**: The refractive index data for crystalline silicon (Si) is included and interpolated onto the simulation wavelength grid for realistic material behavior.
  
## Tools

- **Python 3.10+**: The programming language for the simulation.
- **Required Libraries**:
  - `numpy`
  - `scipy`
  - `matplotlib`
  - `miepython`
  
## Project Structure

- `simulate_dimer.py`: Python script to run the full simulation, which includes Mie coefficients, polarizabilities, extinction spectra, and Fano fit analysis. All figures and data are saved.
- `requirements.txt`: A file containing the required Python libraries for the project.
- `report.tex`: LaTeX report template for the simulation results, with references and figures.
- `figs/`: Directory containing the output figures (in PDF format).
- `data/`: Directory containing data files (CSV format), including those for dispersive materials like crystalline Si.

## Requirements

1. **Set Up the Environment**:
   - Ensure that you have **Python 3.10+** installed on your system.
   - It's recommended to use a **virtual environment** or **conda** for managing the dependencies.
   - Install the required libraries by running the following command:

2. **Run the Simulation**:
   - Once the environment is set up and dependencies are installed, you can run the full simulation by executing the following command:

     ```bash
     python simulate_dimer.py
     ```

   - This will execute the following tasks:
     - **Mie Coefficients Calculation**: Computes the Mie scattering coefficients for both spheres.
     - **Polarizability Calculation**: Converts the Mie coefficients to electric dipole polarizability for each sphere.
     - **Extinction Proxy Spectra**: Generates extinction proxy spectra for different gap sizes and polarization configurations.
     - **Fano Fit**: Fits a selected asymmetric spectral feature using the Fano lineshape model, extracting the resonance wavelength, width, and asymmetry.

3. **View the Outputs**:
   After running the simulation, the results will be saved in the following directories:
   - **`figs/`**: Contains PDF figures showing Mie coefficients, polarizabilities, extinction spectra, and Fano fit results.
   - **`data/`**: Contains CSV data files for Mie coefficients, polarizabilities, and extinction spectra. These files are saved with clear timestamps for easy tracking.

4. **Verify the Results**:
   - The code will save all output data and figures automatically. You can inspect these to validate the results of the simulation.

## Interpreting the Output

Once the simulation has finished running, you will find the following outputs:

1. **Figures (Saved in `figs/` Directory)**:
   - **Mie Coefficients Plot**: A graph showing the Mie coefficients for both spheres over the specified wavelength range.
   - **Polarizability Plot**: A plot depicting the Mie-derived electric dipole polarizability for each sphere.
   - **Extinction Proxy Spectra**: Graphs displaying the extinction proxy spectra for different gap sizes and polarization directions.
   - **Fano Fit Results**: A figure showing the Fano fit overlaying the extinction proxy data, with a residual plot to assess the quality of the fit.

2. **Data Files (Saved in `data/` Directory)**:
   - **Mie Coefficients**: CSV files containing the wavelength-dependent Mie coefficients for each sphere.
   - **Polarizabilities**: CSV files containing the calculated electric dipole polarizabilities for each sphere.
   - **Extinction Proxy Spectra**: CSV files with the calculated extinction spectra for different gaps and polarization directions.

   You can use these data files for further analysis or to regenerate plots if needed.

## Report

The final deliverable includes a **LaTeX report** (`report.tex`) which should be compiled to a PDF (`report.pdf`). This report should contain the following sections:

1. **Model Summary**:
   - Include a brief explanation of the coupled-dipole model, the assumptions made, and the limits of the model's validity. You can reference the equations used to compute the Mie coefficients, polarizabilities, and the coupled-dipole system.

2. **Results**:
   - Present the trends observed for both polarizations (longitudinal and transverse). Include a figure that overlays the extinction spectra for all gap sizes for a given polarization.
   - Discuss the variation in extinction spectra with respect to the gap size and polarization direction.

3. **Fano Fit**:
   - Select a representative extinction spectrum that shows an asymmetric feature (dip or peak) and fit it using the Fano lineshape model. Report the fitted parameters (λ0, Γ, q).
   - Include a plot showing the data with the Fano fit overlaid, as well as a residual plot.

4. **Dispersion**:
   - Compare the results using a constant refractive index (e.g., n1 = 3.47, n2 = 2.41) with the results using **crystalline silicon (c-Si)** as a dispersive material. Discuss how the inclusion of dispersion changes the spectral features.

5. **Conclusion**:
   - Summarize the key findings, such as how the gap size and polarization direction affect the spectra, and any trends observed in the Fano fit parameters.
   - If you have explored any additional extensions (such as using other dispersive materials or adjusting the model), describe what was tested, what worked, and what could be done next.

6. **Research Memo (Self-Directed Extension)**:
   - Write a half-page research memo that discusses an extension you tried with minimal guidance. This could involve adding a magnetic dipole response, exploring dispersion effects, changing the geometry, or modifying the Fano model. Describe what worked, what didn’t, and what you would do next if you had more time.


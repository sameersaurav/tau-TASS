# Tutorial to perform kinetics from Temperature Accelerated Sliced Sampling Simulations

## Description
This tutorial outlines the workflow for analyzing the kinetics of barrier-crossing events from Temperature Accelerated Sliced Sampling (TASS) simulations.

## Environment Setup
Before running the analysis, ensure the necessary Python environment is set up.

- **Create a Conda Environment:**
  
  Run the following command to create a new conda environment named `pytorch`:
  ```bash
  conda create -n pytorch
  ```

- **Activate the Environment:**
  
  Activate the newly created environment:
  ```bash
  conda activate pytorch
  ```

- **Install PyTorch and dependencies:**
  
  Install PyTorch using the following command:
  ```bash
  conda install pytorch::pytorch
  ```

## Simulation and Analysis Protocol

- **Compute the free energy surface $F(s)$**
  - Generate the free energy surface (FES) $F(s)$ from your existing TASS simulation data.
  - **Output:** Save the resulting data as `free_energy.dat`.

- **Train an ANN to Represent $F(s)$**
  Train a neural network to learn the continuous free energy landscape from the discrete data points.
  - **Input:** Ensure `free_energy.dat` is in the working directory.
  - **Execution:**
  ```bash
  python NN.py
  ```
- **Verification:**
  - Check the training and validation loss plots. Both should converge close to zero.
  - Compare the predicted FES against the original FES to ensure accuracy.
- **Output:** The trained model is saved as `free_energy.pt`.

## Compute the Static Bias $V^b_0(s)$
Calculate the static bias potential required in the subsequent infrequent metadynamics simulations.
- **Requirements:**
  - The trained model: `free_energy.pt`.
  - Simulation Parameters: Molecular dynamics (MD) time step, Well-tempered metadynamics (WTMetaD) parameters (Gaussian height, width, bias factor), and the product basin definition.
- **Execution:**
  ```bash
  python MD.py
  ```
- **Bias Extraction:**
  - From the generated bias output, identify the potential values corresponding to 90% filling of the transition barrier.
  - Extract these values and save them into a file named `HILLS` (ensure it is formatted correctly for PLUMED input).

## Perform Infrequent Metadynamics (IMetaD)
- **Preparation:** Place the following files in your working directory:
  - `plumed.dat`
  - `HILLS` (containing the static bias $V^b_0(s)$ )
  - System topology and parameter files.
- **Execution:**
  - Start the IMetaD simulation applying the static bias $V^b_0(s)$ as the initial potential.
  - Run the simulation until a successful barrier-crossing event occurs.
- **Time Calculation:**
  - Record the simulation time ($t_{sim}$) required for the crossing.
  - Calculate the unbiased transition time ($t_{unbiased}$) by multiplying $t_{sim}$ by the acceleration factor ($\alpha$).

## Statistical Analysis
- **Data Collection:** Perform multiple independent simulations. Assign different initial velocities and compute  $t_{unbiased}$ for each run. 
- **Estimate Characteristic Time ($\tau$):** Run the cumulative distribution function script:
  ```bash
  python cdf.py
  ```
  This computes the empirical cumulative distribution function (ECDF) from the collected unbiased transition times and fits it to the theoretical Poisson distribution (TCDF) to estimate the characteristic timescale $\tau$.
- **p-Value calculation:** Run the Kolmogorov-Smirnov test script:
  ```bash
  python ks.py
  ```
  This calculates the p-value for the fit.

### Citation
If you use this workflow in your research, please cite the following paper:

S. Saurav, D. Das, R. Javed, and N. N. Nair, "Kinetics of barrier crossing events from Temperature Accelerated Sliced Sampling simulations", Phys. Chem. Chem. Phys., 2026 (Accepted Manuscript).

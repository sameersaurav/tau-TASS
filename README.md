# Kinetics of Barrier Crossing Events from Temperature Accelerated Sliced Sampling Simulations
## Description
This tutorial outlines the workflow for computing the rate constant of barrier-crossing events from Temperature Accelerated Sliced Sampling (TASS) simulations.

## Environment Setup
Before running the analysis, ensure the necessary Python environment is set up.

- **Create a Conda Environment:** Run the following command to create a new conda environment named `pytorch`:
  ```
  conda create -n pytorch
  ```
- **Activate the Environment:** Activate the newly created environment:
  ```
  conda activate pytorch
  ```
- **Install PyTorch and dependencies:** Install PyTorch using the following command:
  ```
  conda install pytorch::pytorch
  ```
## Artificial Neural Network (ANN) Representation of the Free Energy Surfaces (FES)

- **Compute the FES $F(s)$:**
  - Generate the $F(s)$ from your existing TASS simulation data.
  - **Output:** Save the resulting data as `free_energy.dat`.

- **Train an ANN to Represent $F(s)$:**

  Train a neural network to learn the continuous free energy landscape from the discrete data points.
  - **Input:** Ensure `free_energy.dat` is in the working directory.
  - **Execution:**
    ```
    python NN.py
    ```
- **Verification:**
  - Check the training and validation loss plots. Both should converge close to zero.
  - Compare the predicted FES against the original FES to ensure accuracy.
- **Output:** The trained model is saved as `free_energy.pt`.

## Compute the Static Bias $V^{\mathrm{b}}_0(s)$
Calculate the static bias potential required in the subsequent infrequent metadynamics simulations.
- **Requirements:**
  - The trained model: `free_energy.pt`.
  - Simulation Parameters: Molecular dynamics time step, Well-tempered metadynamics parameters (Gaussian height, width, bias factor), and the product basin definition.
- **Execution:**
  ```
  python MD.py
   ```
- **Bias Extraction:**
  - From the generated bias output, identify the potential values corresponding to 90% filling of the transition barrier.
  - Extract these values and save them into a file named `HILLS` (ensure it is formatted correctly for PLUMED input).

## Perform Infrequent Metadynamics (IMetaD)
- **Preparation:** Place the following files in your working directory:
  - `plumed.dat`
  - `HILLS` (containing the static bias $V^{\mathrm{b}}_0(s)$ )
  - System topology and parameter files.
- **Execution:**
  - Start the IMetaD simulation applying the static bias $V^{\mathrm{b}}_0(s)$ as the initial potential.
  - Simulate until a successful barrier-crossing event occurs.
- **Time Calculation:**
  - Record the simulation time ($t_{\rm{sim}}$) required for the crossing.
  - Calculate the unbiased transition time ($t_{\rm{unbiased}}$) by multiplying $t_{\rm{sim}}$ by the acceleration factor ($\alpha$).

## Statistical Analysis
- **Data Collection:** Perform multiple independent simulations. Assign different initial velocities and compute  $t_{\rm{unbiased}}$ for each run. 
- **Estimate Mean First Passage Time ($\tau$):** Run the cumulative distribution function script:
  ```
  python cdf.py
   ```
  This computes the empirical cumulative distribution function (ECDF) from the collected unbiased transition times and fits it to the theoretical Poisson distribution (TCDF) to estimate the $\tau$ .
- **p-Value calculation:** Run the Kolmogorov-Smirnov test script:
   ```
  python ks.py
   ```
  This calculates the p-value for the fit.

## Contributors:
- Prof. Nisanth N. Nair, IIT Kanpur
 - Sameer Saurav, IIT Kanpur
## Citation
Please cite the following references if using this protocol:

- S. Saurav, D. Das, R. Javed, and N. N. Nair, "Kinetics of barrier crossing events from Temperature Accelerated Sliced Sampling simulations", _Phys. Chem. Chem. Phys._, 2026.

  arXiv: https://doi.org/10.48550/arXiv.2509.05068.

## License 

MIT License

Copyright (c) 2026 Nisanth N. Nair

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.



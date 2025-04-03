# Tutorial to perform kinetics from Temperature Accelerated Sliced Sampling Simulations

## Introduction
This tutorial outlines the steps to analyze the kinetics of barrier-crossing events from Temperature Accelerated Sliced Sampling (TASS) simulations. For further details, please refer to the following sources:

1. Temperature Accelerated Sliced Sampling (TASS) ([Awasthi & Nair, _JCP_ 2017](https://doi.org/10.1063/1.4977704),  [Pal et al., _JCC_ 2021](https://doi.org/10.1063/1.4977704), [Tutorial](https://sites.google.com/view/the-nnn-group/tutorials/tass?authuser=0))
2. Artificial Neural Networks (ANNs) ([Schneider et al., _PRL_ 2017](https://doi.org/10.1103/PhysRevLett.119.150601))
3. Infrequent Metadynamics (IMetaD) ([Tiwary and Parrinello, _PRL_ 2013](https://doi.org/10.1103/PhysRevLett.111.230602), [Salvalaglio et al., _JCTC_ 2014](https://doi.org/10.1021/ct500040r))

## Steps
1. ### Compute the free energy surface _F(**s**)_ from TASS simulations
   - Generate the free energy surface  _F(**s**)_ using your TASS simulation data.
2. ### Train an ANN to represent _F(**s**)_
   - use the file `free_energy.dat` as input for the neural network.
   - Run the command:
     ```
     python NN.py

      ``` 
   - Output Verification:
     - Check the training and validation loss plots. Both should ideally be close to zero.
     - Verify the generated plot of the predicted free energy surface for correctness.
     - **Model Saving**: The trained model is saved as `free_energy.pt`.
3. ### Compute the bias V<sup>b</sup><sub>0</sub> (<strong>s</strong>)
   - **Parameters**: Specify the molecular dynamics (MD) time step and the well-tempered metadynamics (WTMetaD) parameters (Gaussian's height, width, and bias factor) along with the location of the transition state.
   - **Model file:** Ensure that `free_energy.pt` is in the same directory.
   - **Execution:** Run the command:
     ```
     python MD.py
     ``` 
   - **Bias Extraction:** From the resulting bias file, extract the bias value corresponding to 90% of the transition barrier filling. Save these values in the `HILL` file.
4. ### Perform MD with "Infrequent Metadynamics (IMetaD)".
   - **Preparation:** Place the files `plumed.dat` and `HILL` in the working directory along with the system’s topology and parameter files.
   - **Execution:** Start the IMetaD simulation using V<sup>b</sup><sub>0</sub> (<strong>s</strong>) as the initial bias. Simulate until a barrier-crossing event occurs.
   - **Time Calculation:** Record the simulation time and multiply it by the acceleration factor to obtain the corresponding unbiased simulation time.
5. ### Statistical Analysis 
   - **Multiple Simulations:** Run several simulations using the same initial structure but with different initial velocities.
   - **$\tau$ estimation**
     ````
     python cdf.py
     ````
     This script computes the empirical cumulative distribution function (ECDF) from the unbiased transition times and fits it to the theoretical cumulative distribution function (TCDF) to estimate the characteristic time $\tau$.
   - **P-Value Calculation:**
      ````
     python ks.py
     ````
     This will compute the p-value for the fit.
  
  

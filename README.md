# Tutorial to perform kinetics from Temperature Accelerated Sliced Sampling Simulations

## Introduction
This is a tutorial for performing Kinetics of Barrier Crossing Events from  Temperature Accelerated Sliced Sampling Simulations (TASS) simulations. 

For details, kindly see the following -

1. Temperature Accelerated Sliced Sampling (TASS) ([Awasthi & Nair, _JCP_ 2017](https://doi.org/10.1063/1.4977704),  [Pal et al., _JCC_ 2021](https://doi.org/10.1063/1.4977704), [Tutorial](https://sites.google.com/view/the-nnn-group/tutorials/tass?authuser=0))
2. Artificial Neural Networks (ANNs) ([Schneider et al., _PRL_ 2017](https://doi.org/10.1103/PhysRevLett.119.150601))
3. Inrefequent Metadynamics (IMetaD) ([Tiwary and Parrinello, _PRL_ 2013](https://doi.org/10.1103/PhysRevLett.111.230602), [Salvalaglio et al., _JCTC_ 2014](https://doi.org/10.1021/ct500040r))

## Steps
1. ### Compute free energy surface _F(**s**)_ from TASS simulations.
2. ### Train an ANN to represent F(**s**)
   - Give "free_energy.dat" as input to NN_training.py.
   - Run the command "python NN_training.py". 
   - It will plot "training loss" and "validation loss". Ensure that they are within reasonable limits (close to zero).
   - It will also generate a "predicted free energy" surface plot. Ensure that it is correct.
   - Code will save the model "free_energy_net.pt". 
3. ### Compute the bias V<sup>b</sup><sub>0</sub> (<strong>s</strong>)
   - Specify the molecular dynamics (MD) time step, well-tempered metadynamics (WTMetaD) parameters- Gaussian's height, width, and bias factor, and location of the transition state.
   - Keep "free_energy_net.pt" in the same directory.
   - Run the command "python Analytical MD.py". 
   - From the bias file keep only the bias at which 90 % of the transition barrier is filled. Save it as "HILL".
4. ### Perform MD with "IMetaD".
   - Keep "plumed.dat", "HILL" file with the system's topolpgy and paramater file.
   - Start IMetaD, taking  V<sup>b</sup><sub>0</sub> (<strong>s</strong>) as the initial bias, until barrier crossing.
   - Note down the "simulation time". Multiply with the "acceleration factor" to recover "unbiased simulation time".
5. #### Statistics 
   - Perform multiple simulations with the same initial structure but different velocities.
   - Run "python cdf.py". It will compute the _empirical_ cumulative distribution function (ECDF) from the computed unbiased transition times and fit to Theoretical CDF (TCDF) to estimate $\tau$.
   - Run "python ks.py". It will compute the "p-value".
  
  

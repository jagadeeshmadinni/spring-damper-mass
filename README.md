Exploratory study of parametrizing the spring-mass-damper problem for Gaussian Process Regression

This repo contains the matlab code, simscape and Simulink model, and the Jupyter notebook files to evaluate the interpolation performance of Gaussian Process Regression in system identification.

What is Gaussian Process Regression?

Gaussian Process Regression (GPR) is a stochastic non-parametric technique used in modeling complex, non-linear relationships between variables. The objective of a GPR is to provide a probability distribution over functions that fit a known set of training points. A Gaussian Process is defined by a Mean and a Covariance function.

Y(x)?GP(m(x),k(x, x?)), where m is the mean and k is the covariance function.

The experiment is split into three tasks:
1) Build a known system model – a simple mass-spring-damper system. This is an abstraction that allows us to translate the outcomes to any mechanical system that can be represented by a spring-mass-damper.
2) Evaluate the system response to a standard input signal over a wide parameter range – The input signal is chosen to be a unit step response. The spring rate of the model is chosen as the parameter over which the model is varied. Evaluating the response for all possible parameters generates a source of truth to measure the performance of the GPR.
3) Provide a portion of the system response as training data to the Gaussian Process Regressor, predict the response for the remaining “test data” and evaluate the results against standard metrics.

The first two steps are conducted on a desktop computer with MATLAB Simscape and the Parallel Computing Toolbox. The third task is completed on a Linux based Compute cluster with access to an NVIDIA A100 GPU. The conda environment can be recreated with the yml file above.


Procedure:

The MATLAB script spring_damper_system.m contains a state-space model of the spring-mass-damper system and generates the system response to a unit step input. This is now commented out in favor a scalable simscape model that is titled SimscapeSpringMass.slx. The intent is to be able to replace the mass-spring-damper with more complex models like a half-car or full-car model and still run the entire pipeline with minimal changes to the code. 

Once the script file is executed, the displacement values of the mass are written to csv files. The file fineSpringParameters.csv contains the entire sweep of the spring rate values. This would be from 0.1 N/m to 50.1 N/m at 0.1 N/m increments. The file fineDisplacements.csv captures the system response for a period of 60s at 0.1s intervals. This is the source of truth to validate the performance of the GPR. The write to csv lines for these two are currently commented out. To use them, uncomment the lines and replace the filenames in the arguments with full path filenames in your system.

The Jupyter notebook medSpringRateSweepGPR.ipynb takes the inputs from these two csv files, extracts train_x, train_y and test_x, test_y. The train_x tensor is a mesh grid of the spring-rate and timestep where the spring-rate is 0.1:5:50.1 N/m. Similarly the test_x tensor is a mesh-grid of the spring-rate and timestep but with spring rate as 0.1:1:50.1. The tensors train_y and test_y are, as expected, the corresponding displacements.  

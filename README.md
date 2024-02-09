**Exploratory study of parametrizing the spring-mass-damper problem for Gaussian Process Regression**

This repo contains the matlab code, simscape and Simulink model, and the Jupyter notebook files to evaluate the interpolation performance of Gaussian Process Regression in system identification.

**What is Gaussian Process Regression?**

Gaussian Process Regression (GPR) is a stochastic non-parametric technique used in modeling complex, non-linear relationships between variables. The objective of a GPR is to provide a probability distribution over functions that fit a known set of training points. A Gaussian Process is defined by a Mean and a Covariance function. A typical GPR algorithm tries to fit the covariance model and its parameters to achieve loss minimization.

_Y(x)~GP(m(x),k(x, x'))_, where m is the mean and k is the covariance function.

**Procedure:**

The experiment is split into three tasks:
1) Build a known system model – a simple mass-spring-damper system. This is an abstraction that allows us to translate the outcomes to any mechanical system that can be represented by a spring-mass-damper.
2) Evaluate the system response to a standard input signal over a wide parameter range – The input signal is chosen to be a unit step response. The spring rate of the model is chosen as the parameter over which the model is varied. Evaluating the response for all possible parameters generates a source of truth to measure the performance of the GPR.
3) Provide a portion of the system response as training data to the Gaussian Process Regressor, predict the response for the remaining “test data” and evaluate the results against standard metrics.

The first two steps are conducted on a desktop computer with MATLAB Simscape and the Parallel Computing Toolbox. The third task is completed on a Linux based Compute cluster with access to an NVIDIA A100 GPU. The conda environment can be recreated with the yml file above.


The MATLAB script _spring_damper_system.m_ contains a state-space model of the spring-mass-damper system and generates the system response to a unit step input. This is now commented out in favor a scalable simscape model that is titled _SimscapeSpringMass.slx_. The intent is to be able to replace the mass-spring-damper with more complex models like a half-car or full-car model and still run the entire pipeline with minimal changes to the code. 

Once the script file is executed, the displacement values of the mass are written to csv files. The file _fineSpringParameters.csv_ contains the entire sweep of the spring rate values. This would be from 0.1 N/m to 50.1 N/m at 0.1 N/m increments. The file _fineDisplacements.csv_ captures the system response for a period of 60s at 0.1s intervals. This is the source of truth to validate the performance of the GPR. The write to csv lines for these two are currently commented out. To use them, uncomment the lines and replace the filenames in the arguments with full path filenames in your system.

The Jupyter notebook _medSpringRateSweepGPR.ipynb_ takes the inputs from these two csv files, extracts train_x, train_y and test_x, test_y. The train_x tensor is a mesh grid of the spring-rate and timestep where the spring-rate is 0.1:5:50.1 N/m. Similarly the test_x tensor is a mesh-grid of the spring-rate and timestep but with spring rate as 0.1:1:50.1. The tensors train_y and test_y are, as expected, the corresponding displacements.

**Tools:**

It is essential to have the Parallel Computing Toolbox installed to be able to run the matlab script file as it uses both parfor and parsim functionality. If you don't have the toolbox license, it is possible to run the simulations in a regular for loop but it will take a signficant amount of time to finish. The GPR implementation utilizes the open source GPyTorch toolbox that leverages the capabilities of PyTorch. This enables us to use CUDA GPU computing and makes the training and evaluation faster. Please note that the training and evaluation of a GPR is computationally expensive and smaller GPUs can easily run into CUDA Out of Memory errors. In addition to the A100, this code was run successfully on an NVIDIA V100 and a P100 but not tested on any smaller versions. It is also worth mentioning that trying to run inference with a 0.1 N/m increment on the spring rate failed even on the A100 due to the memory limitation.

**Metrics:**

The GPyTorch toolbox offers a number of metrics to evaluate the GPR performance but I have used only two. The first is the commonly used Root Mean Square Error(RMSE) and the other the Quantile Coverage Error(QCE) with a 95% confidence interval. RMSE is widely used to understand errors in Euclidean distance and it fits the displacement prediction problem but for a GPR that produces a probability distribution, this is not ideal. The QCE is more apt metric and provides an intuitive understanding of how well the predictions are distributed from our known output values.

**Tuning the GPR:**

Like any machine learning algorithm, there are a number of ways to tune the GPR, most prominently by adjusting the prior belief about the input scale and the output scale, the type of noise distribution and the model of the covariance. There is extensive literature available out there but this exercise did not venture out into too much detail on that front. Knowing the model of the system beforehand, I adjusted the priors to correspond to a smoother input/output representation and noticed a faster convergence as well as reduced QCE. However, RMSE did not correlate to the decrease in QCE.


**Results:**

| Plot of the Run | Prior | RMSE | QCE(95%) Confidence Interval | Computational Cost(s) |
|--|-------|-----|-------------------------------|----------------------|
| ![image](https://github.com/jagadeeshmadinni/spring-damper-mass/assets/31152033/6e2f49f0-f9d3-4ad1-8be0-2f7445aebcbd)| Default - No lengthscale | 2.1970 | 0.1284 | 30.7936 |
|![image](https://github.com/jagadeeshmadinni/spring-damper-mass/assets/31152033/d847b98c-5ead-4a53-9f85-a73e645d31ec)| Gamma Distribution - Both lengthscale and outputscale at -0.05368 | 2.2929 | 0.0978 | 22.3018
|![image](https://github.com/jagadeeshmadinni/spring-damper-mass/assets/31152033/8ddfbd41-f328-4ca7-9def-ec81553c4926)| Gamma Distribution - Both lengthscale and outputscale at -0.9273 | 2.1938 | 0.0224 | 16.4506


**References:**
1. https://ekamperi.github.io/mathematics/2021/03/30/gaussian-process-regression.html
2. https://www.sciencedirect.com/science/article/pii/S0022249617302158#:~:text=More%20formally%2C%20a%20Gaussian%20process,0%20%2C%20%CF%83%20%CF%B5%202%20)%20.
3. https://katbailey.github.io/post/gaussian-processes-for-dummies/
4. https://docs.gpytorch.ai/en/v1.6.0/examples/02_Scalable_Exact_GPs/Simple_GP_Regression_CUDA.html

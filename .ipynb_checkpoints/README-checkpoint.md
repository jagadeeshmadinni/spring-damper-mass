# spring-damper-mass
Exploratory study of parametrizing the spring-damper-mass problem for Kriging simulation

-- The MATLAB script spring_damper_system.m has a standard spring-damper-mass model which is being evaluated for the sensitivity of displacement of the mass to spring rate.
-- The objective is to run a parameter sweep with varying levels of spring rate intervals, use Kriging( more broadly Gaussian Processes Regression) to interpolate the results for untested parameter settings
-- The simscape model SimscapeSpringMass.slx replaces the mathematical model in the previous matlab script as a means to scale up the system easily. The script now has parallel processing ability to do a wide parameter sweep utilizing multi-core systems.
-- The Jupyter notebook KrigingTrial.ipynb has a working model of the base GPR function from GPyTorch and the result is visualized towards the end.

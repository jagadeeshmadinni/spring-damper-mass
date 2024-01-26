clear vars;
% Mass-spring-damper parameters starting with the values from
% ctms.engin.umich.edu state space model
m = 1; % Mass = 1.0 kg
%k = 1; % Spring constant = 1.0 N/m
b = 0.2; % Damping constant = 0.2 Ns/m
F = 1; % Input Force = 1.0 N



B = [0;1/m];
C = [1 0];
D = 0;
t = 0:0.1:60;
Z = zeros(601,100);
<<<<<<< HEAD
=======
K = 1:1:100;
>>>>>>> 075eddf53775ca7c6ce37f3e26039c8c1a9d071b
for k = 1:1:100
    i = k;
    A = [0,1;-k/m,-b/m];
    sys = ss(A,B,C,D);
    Z(:,i)= step(sys,t);
end 

<<<<<<< HEAD
[X,Y] = meshgrid(t,1:1:100);
=======
[X,Y] = meshgrid(t,K);
>>>>>>> 075eddf53775ca7c6ce37f3e26039c8c1a9d071b
fig = figure();
surf(X,Y,Z')
ylabel("Spring Constant-K(N/m)")
xlabel("Time(s)")
zlabel("Displacement(m)")
<<<<<<< HEAD
%saveas(fig,"Spring Parameter Plot.png")
%save("C:\Users\jmadinn\Documents\Jagadeesh\springParameters.mat","X","Y","Z")
=======
saveas(fig,"/home/jmadinn/ARM_Lab/Spring_Parameter_Plot.png")
writematrix(K,'/home/jmadinn/ARM_Lab/springParameters.csv');
writematrix(t,'/home/jmadinn/ARM_Lab/timesteps.csv');
writematrix(Z,'/home/jmadinn/ARM_Lab/displacement.csv');

>>>>>>> 075eddf53775ca7c6ce37f3e26039c8c1a9d071b

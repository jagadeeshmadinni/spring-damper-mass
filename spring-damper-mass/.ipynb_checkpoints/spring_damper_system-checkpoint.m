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
Z = zeros(601,91);
for k = 1:0.1:10
    i = int32((k-1)/0.1+1);
    A = [0,1;-k/m,-b/m];
    sys = ss(A,B,C,D);
    Z(:,i)= step(sys,t);
end 

[X,Y] = meshgrid(1:0.1:10,t);
fig = figure();
surf(X,Y,Z)
xlabel("Spring Constant-K(N/m)")
ylabel("Time(s)")
zlabel("Displacement(m)")
saveas(fig,"/home/jmadinn/ARM_Lab/Spring_Parameter_Plot.png")
save("/home/jmadinn/ARM_Lab/springParameters.mat","X","Y","Z")

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


springMax = 50;
fineIncrement = 0.1;
mediumIncrement = 1;
coarseIncrement = 5;
fineK = 0.1:fineIncrement:springMax+0.1;
coarseK = 0.1:coarseIncrement:springMax+0.1;
mediumK = 0.1:mediumIncrement:springMax+0.1;
fineZ = zeros(length(K),length(t));

%Generate the source of truth by iterating over K with fineIncrement
%interval

for i = 1:length(fineK)
    A = [0,1;-fineK(i)/m,-b/m];
    sys = ss(A,B,C,D);
    fineZ(i,:)= step(sys,t);
end 

[X,Y] = meshgrid(t,fineK);

fig = figure();
surf(X,Y,fineZ)
ylabel("Spring Constant-K(N/m)")
xlabel("Time(s)")
zlabel("Displacement(m)")



%saveas(fig,"Spring Parameter Plot.png")
%save("C:\Users\jmadinn\Documents\Jagadeesh\springParameters.mat","X","Y","Z")

%saveas(fig,"/home/jmadinn/ARM_Lab/Spring_Parameter_Plot.png")
writematrix(fineK,'C:\Users\jmadinn\Documents\Jagadeesh\fineSpringParameters.csv');
writematrix(t,'C:\Users\jmadinn\Documents\Jagadeesh\timesteps.csv');
writematrix(fineZ,'C:\Users\jmadinn\Documents\Jagadeesh\fineDisplacements.csv');


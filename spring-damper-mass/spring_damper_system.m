% Mass-spring-damper parameters starting with the values from
% ctms.engin.umich.edu state space model
m = 1; % Mass = 1.0 kg
k = 1; % Spring constant = 1.0 N/m
b = 0.2; % Damping constant = 0.2 Ns/m
F = 1; % Input Force = 1.0 N


tspan = [0 5];
y_init = [0.1;0];

[t,y] = ode45(@springDamperResponse,tspan, y_init);

plot(t,y(:,1));
grid on;
xlabel("time");
ylabel("Displacement")
title("Displacement vs time");


function Y = springDamperResponse(y)
Y = [y(2);F/m-(b/m)*y(2)-(k/m)*y(1)];
end 

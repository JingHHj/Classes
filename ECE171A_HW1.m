  %% p2,a1
% numerical
dx = @(t,x)2*x
tspan = [0 1.5]
x0 = 1
[t,x] = ode45(dx,tspan,x0)
plot(t,x)
title('numerical solution of the ODE')
yticks(0:2:20)  % changing the y axis range for a better demostration

%analytical
x = exp(2*t)
t = 0:0.01:1.5
plot(t,x)
title('analytical solution of the ODE')


  %%p2,a2
% numerical
f = @(t,z)[z(2);-z(1)]
tspan = [0 10]
[t,z] = ode45(f,tspan,[1;1])
plot(t,z(:,1),'-o',t,z(:,2),'-o')
title('numerical solution of the ODE')
legend('z_1 = z','z_2 = dz/dt')

%analytical
z = cos(t)+sin(t)
t = 0:0.01:10
plot(t,z,'-o')
title('analytical solution of the ODE')

  %%p2,a3
% numerical
dx = @(t,x)-3*x+5
tspan = [0,2.5]
x0 = 2
[t,x] = ode45(dx,tspan,x0)
plot(t,x,'-o')
yticks(1:0.04:2.5)
xticks(0:0.2:2)
title('numerical solution of the ODE')

%analytical
x = (exp(-3*t)+5)/3
t = 0:0.01:2.5
plot(t,x,'-o')
title('analytical solution of the ODE')

  %%p4,b
% numerical
f = @(t,i)[i(2);100*i(1)-18*i(2)]
tspan = [0 5]
[t,i] = ode45(f,tspan,[1;-6])
plot(t,i(:,1),t,i(:,2))
title('numerical solution of the ODE')
legend('i_1 = i','i_2 = di/dt')

%analytical
i = ((3+181^0.5)*exp((-9+181^0.5)*t)+(181^0.5-3)*exp((-9-181^0.5)*t))/(2*181^0.5)
t = 0:0.01:5
plot(t,i)
legend('i_1 = i')
title('analytical solution of the ODE')





clear all
close all

%% Problem 1 
% Solve the normal equations

% n = m
A1 = [1 0 0
     0 1 0
     0 0 1];
B1 = [1;1;1];

Theta1 = (A1'*A1)\(A1'*B1);

% n < m
A2 = [1 0 0
     0 1 0];
B2 = [1;1];

% Theta2 = (A2'*A2)\(A2'*B2);

% n > m

A3 =[1 0 0
     0 1 0
     0 0 1
     1 1 1];

B3 = [1;1;1;1];

Theta3 = (A3'*A3)\(A3'*B3);


%% Problem 2
% Solve the optimization problem with gradient descend and 
% plot f(theta) and ||Gradf(theta)|| during iterations

tau = 0.1;

% n = m
theta_now = [0;0;0];
M = 50;
for i=1:M
    f_now          = 0.5*theta_now'*A1'*A1*theta_now-(A1'*B1)'*theta_now+0.5*B1'*B1;
    grad_now       = A1'*A1*theta_now-A1'*B1;
    f_values(i)    = f_now;
    grad_values(i) = norm(grad_now);
    theta_next     = theta_now-tau*grad_now;
    theta_now      = theta_next;
end

% Plot f_values and grad_values
figure;
plot(1:M, f_values, 'b-', 'LineWidth', 2, 'MarkerSize', 6);
hold on;
plot(1:M, grad_values, 'r-', 'LineWidth', 2, 'MarkerSize', 6);
hold off;

xlabel('Iteration');
ylabel('Value');
title('Convergence of f_{now} and ||grad_{now}||');
legend('f_{now}', '||grad_{now}||');
grid on;

% n < m
theta_now = [0;0;0];
M = 50;
for i=1:M
    f_now          = 0.5*theta_now'*(A2')*A2*theta_now-(A2'*B2)'*theta_now+0.5*(B2')*B2;
    grad_now       = (A2')*A2*theta_now-(A2')*B2;
    f_values(i)    = f_now;
    grad_values(i) = norm(grad_now);
    theta_next     = theta_now-tau*grad_now;
    theta_now      = theta_next;
end

% Plot f_values and grad_values
figure;
plot(1:M, f_values, 'b-', 'LineWidth', 2, 'MarkerSize', 6);
hold on;
plot(1:M, grad_values, 'r-', 'LineWidth', 2, 'MarkerSize', 6);
hold off;

xlabel('Iteration');
ylabel('Value');
title('Convergence of f_{now} and ||grad_{now}||');
legend('f_{now}', '||grad_{now}||');
grid on;

% n > m
theta_now = [0;0;0];
M = 50;
for i=1:M
    f_now          = 0.5*theta_now'*A3'*A3*theta_now-(A3'*B3)'*theta_now+0.5*B3'*B3;
    grad_now       = A3'*A3*theta_now-A3'*B3;
    f_values(i)    = f_now;
    grad_values(i) = norm(grad_now);
    theta_next     = theta_now-tau*grad_now;
    theta_now      = theta_next;
end

% Plot f_values and grad_values
figure;
plot(1:M, f_values, 'b-', 'LineWidth', 2, 'MarkerSize', 6);
hold on;
plot(1:M, grad_values, 'r-', 'LineWidth', 2, 'MarkerSize', 6);
hold off;

xlabel('Iteration');
ylabel('Value');
title('Convergence of f_{now} and ||grad_{now}||');
legend('f_{now}', '||grad_{now}||');
grid on;

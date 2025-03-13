clear all
close all

%% L2 norm

% Generate Data
x = (linspace(0,10,30))';
A = [ones(size(x,1),1), x]; % Construct matrix A
X = A;
theta = [1;2];
for i=1:size(x,1)
noise(i,1) = rand(1)*x(i)*3;
end
y = A*theta + noise;

% Gradient Descent Parameters
tau = 0.0001;
tol = 1e-4;
theta_now = theta;
grad_now = 10;
M = 50; % Number of iterations

% Store values
f_values = zeros(M,1);
grad_values = zeros(M,1);

% Gradient Descent Loop
for i = 1:M
    f_now = 0.5 * theta_now' * A' * A * theta_now - (A' * y)' * theta_now + 0.5 * y' * y;
    grad_now = A' * A * theta_now - A' * y;
    f_values(i) = f_now;
    grad_values(i) = norm(grad_now);
    theta_now = theta_now - tau * grad_now;
end

% Plot data and L2 regression line
figure;
hold on;
scatter(x, y, 50, 'b', 'o', 'filled'); % Plot noisy data points
plot(x, A * theta_now, 'r-', 'LineWidth', 2); % L1 regression line
xlabel('x');
ylabel('y');
title('L2 Norm Regression: Original Data vs. Fit');
legend('Noisy Data', 'L2 Regression Line', 'Location', 'Best');
grid on;
set(gca, 'FontSize', 12);
hold off;

% Plot f_values and grad_values
figure;
yyaxis left;
plot(1:M, f_values, 'b-', 'LineWidth', 2);
ylabel('f_{now}');
hold on;

yyaxis right;
plot(1:M, grad_values, 'r-', 'LineWidth', 2);
ylabel('||grad_{now}||');

xlabel('Iteration');
title('Convergence of f_{now} and ||grad_{now}||');
legend('f_{now}', '||grad_{now}||');
grid on;
hold off;

% Plot error L2
figure;
bar(x,abs(y-A*theta_now),'b')
xlabel('x');
ylabel('Error Magnitude');
title('Error magnitude for L2 norm');
grid on;


%% L1 norm - linprog

C = [0 0 ones(1,size(x,1))];
b = [y;-y];
A = [X -eye(size(x,1)); -X -eye(size(x,1))];

sol = linprog(C,A,b);
thetaL1 = sol(1:2,1);
errL1 = abs(y-X*thetaL1);


% Plot error L1
figure;
bar(x,errL1,'b')
xlabel('x');
ylabel('Error Magnitude');
title('Error magnitude for L1 norm');
grid on;




%% Linf norm - linprog

C = [0 0 1];
b = [y;-y];
A = [X -ones(size(x,1),1); -X -ones(size(x,1),1)];

sol = linprog(C,A,b);
thetaLinf = sol(1:2,1);
errLinf = abs(y-X*thetaLinf);

% Plot error Linf
figure;
bar(x,errLinf,'b')
xlabel('x');
ylabel('Error Magnitude');
title('Error magnitude for L_{inf} norm');
grid on;

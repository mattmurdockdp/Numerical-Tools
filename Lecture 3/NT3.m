%% Cross-Validation on Polynomial Regression

clear; close all; clc;

%% Synthetic quadratic data: y = θ₀ + θ₁x + θ₂x² + noise

N = 100;
x = linspace(-3, 3, N)';
theta_true = [2; -1; 0.5];
noise = randn(N, 1) * 0.5;
y = theta_true(1) + theta_true(2)*x + theta_true(3)*x.^2 + noise;

%% Plot data
fig = figure;
scatter(x, y, 30, 'filled');
title('Synthetic Quadratic Data');
xlabel('x'); ylabel('y');
grid on;
set(fig, 'Color', 'w');
exportgraphics(fig, 'NT3Fig4.png', 'BackgroundColor', 'white');

%% Cross-validation setup (5-fold)
K = 5;
cv = cvpartition(N, 'KFold', K);

% Model capacities to test
max_degree = 10;
MSE_test_all = zeros(K, max_degree+1);

%% Cross-validation loop
for d = 0:max_degree
    for k = 1:K
        idx_train = training(cv, k);
        idx_test = test(cv, k);

        x_train = x(idx_train);
        y_train = y(idx_train);
        x_test = x(idx_test);
        y_test = y(idx_test);

        % Design matrix for train/test
        Phi_train = zeros(length(x_train), d+1);
        Phi_test = zeros(length(x_test), d+1);
        for j = 0:d
            Phi_train(:,j+1) = x_train.^j;
            Phi_test(:,j+1) = x_test.^j;
        end

        % Fit model (least squares)
        theta = Phi_train \ y_train;

        % Predict and compute MSE on test set
        y_pred = Phi_test * theta;
        MSE_test_all(k, d+1) = mean((y_test - y_pred).^2);
    end
end

%% Compute mean and std of MSE
mu = mean(MSE_test_all);
sigma = std(MSE_test_all);

%% Plot μ ± 1.96σ
degrees = 0:max_degree;
fig = figure;
hold on;
errorbar(degrees, mu, 1.96 * sigma, 'o-', 'LineWidth', 2, 'DisplayName', 'Cross-validation');
xlabel('Model Capacity (Degree)');
ylabel('Test MSE');
title('Cross-Validation: Mean ± 1.96 Std of Test MSE');
grid on;
legend;
set(fig, 'Color', 'w');
exportgraphics(fig, 'NT3Fig5.png', 'BackgroundColor', 'white');


%% Ridge Regression with high capacity
lambda = 1; % Ridge regularization parameter
d_ridge = 10;

% Full data
Phi_ridge = zeros(N, d_ridge+1);
for j = 0:d_ridge
    Phi_ridge(:, j+1) = x.^j;
end

% Ridge solution
theta_ridge = (Phi_ridge' * Phi_ridge + lambda * eye(d_ridge+1)) \ (Phi_ridge' * y);

% Predict
x_fit = linspace(-3.2, 3.2, 200)';
Phi_fit = zeros(length(x_fit), d_ridge+1);
for j = 0:d_ridge
    Phi_fit(:, j+1) = x_fit.^j;
end
y_fit = Phi_fit * theta_ridge;

% Plot result
fig = figure;
scatter(x, y, 25, 'filled'); hold on;
plot(x_fit, y_fit, 'r-', 'LineWidth', 2);
title(sprintf('Ridge Regression (degree = %d, \\lambda = %.2f)', d_ridge, lambda));
xlabel('x'); ylabel('y');
legend('Data', 'Ridge fit');
grid on;
set(fig, 'Color', 'w');
exportgraphics(fig, 'NT3Fig6.png', 'BackgroundColor', 'white');


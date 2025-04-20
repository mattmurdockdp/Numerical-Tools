%% Cross-Validation on Polynomial Regression

clear; close all; clc;

%% Synthetic quadratic data: y = θ₀ + θ₁x + θ₂x² + noise

N = 100;
x = linspace(-3, 3, N)';
theta_true = [2; -1; 0.5];
noise = randn(N, 1) * 0.5;
y = theta_true(1) + theta_true(2)*x + theta_true(3)*x.^2 + noise;

rng(1); 
idx = randperm(N);
n_train = round(0.8 * N);
idx_train = idx(1:n_train);
idx_test = idx(n_train+1:end);

x_train = x(idx_train);
y_train = y(idx_train);
x_test = x(idx_test);
y_test = y(idx_test);

fig = figure;
scatter(x_train, y_train, 40, 'b', 'filled'); hold on;
scatter(x_test, y_test, 40, 'r', 'filled');
title('Synthetic Quadratic Data: Train vs Test');
xlabel('x'); ylabel('y');
legend('Train Data', 'Test Data');
grid on;
set(fig, 'Color', 'w');
exportgraphics(fig, 'NT3Fig4.png', 'BackgroundColor', 'white');

%% Solve with small capacity (Theta0 and Theta1) and plot MSE_test and MSE_train

Theta_small_init = randn(2,1) * 0.1;  % reinit every time
MSE_test_small = zeros(size(x_train,1),1);  
MSE_train_small = zeros(size(x_train,1),1);  

for i = 2:size(x_train,1)  % start from 2 to avoid singularities
    
    Theta_small = Theta_small_init;
    eps = 1e-4;
    diff = [1000;1000];
    Theta_old = Theta_small;

    % Prepare training data
    x_small_train = [ones(i,1), x_train(1:i)];
    y_small = y_train(1:i);
    
    % Fixed test data
    x_test_small = [ones(length(x_test),1), x_test];

    tau = 0.001;

    % Gradient descent loop
    while (norm(diff) > eps)
        gradient = x_small_train' * x_small_train * Theta_small - x_small_train' * y_small;
        Theta_small = Theta_old - tau * gradient;
        diff = Theta_small - Theta_old;
        Theta_old = Theta_small;
    end

    % Compute predictions
    y_est_train = x_small_train * Theta_small;
    y_est_test = x_test_small * Theta_small;

    % Compute MSEs
    MSE_train_small(i,1) = mean((y_small - y_est_train).^2);
    MSE_test_small(i,1) = mean((y_test - y_est_test).^2);

end

% Plot both MSE curves
fig = figure;
plot(2:size(x_train,1), MSE_train_small(2:end), 'b-', 'LineWidth', 2); hold on;
plot(2:size(x_train,1), MSE_test_small(2:end), 'r--', 'LineWidth', 2);
xlabel('Number of Training Points');
ylabel('Mean Squared Error');
legend('Train MSE','Test MSE');
title('Train vs Test MSE (Small Capacity Model)');
grid on;
set(fig, 'Color', 'w');
exportgraphics(fig, 'NT3Fig5.png', 'BackgroundColor', 'white');

%% Solve with larger capacity and plot MSE train vs capacity

max_degree = 2;
MSE_train_large = zeros(max_degree, 1);
N = length(x_train);

for d = 1:max_degree
    X_poly_train = ones(N, d + 1);  % [1 x x^2 ... x^d]
    for k = 1:d
        X_poly_train(:, k+1) = x_train.^k;
    end
    
    % Gradient descent for Theta of size (d+1)
    Theta_large = randn(d+1, 1) * 0.1;
    Theta_old = Theta_large;
    eps = 1e-4;
    diff = ones(d+1,1) * 1000;
    tau = 0.001;

    % Gradient descent loop
    while norm(diff) > eps
        gradient = X_poly_train' * X_poly_train * Theta_large - X_poly_train' * y_train;
        Theta_large = Theta_old - tau * gradient;
        diff = Theta_large - Theta_old;
        Theta_old = Theta_large;
    end

    % Estimate output
    y_est_train = X_poly_train * Theta_large;

    % Compute MSE for training data
    MSE_train_large(d) = mean((y_train - y_est_train).^2);
end

% Plot MSE vs Capacity
fig = figure;
plot(1:max_degree, MSE_train_large, 'bo-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Model Capacity (Polynomial Degree)');
ylabel('Mean Squared Error (Train)');
title('Training MSE vs Model Capacity');
grid on;
set(fig, 'Color', 'w');
exportgraphics(fig, 'NT3Fig6.png', 'BackgroundColor', 'white');

%% Cross-Validation on Polynomial Regression

clear; close all; clc;

% Synthetic quadratic data: y = θ₀ + θ₁x + θ₂x² + noise

N = 100;
x = linspace(-3, 3, N)';
theta_true = [2; -1; 0.5];
noise = randn(N, 1) * 0.5;
y = theta_true(1) + theta_true(2)*x + theta_true(3)*x.^2 + noise;

% Cross-validation setup (5-fold)
K = 5;
cv = cvpartition(N, 'KFold', K);

% Model capacities to test
max_degree = 10;
MSE_test_all = zeros(K, max_degree+1);

% Cross-validation loop
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

% Compute mean and std of MSE
mu = mean(MSE_test_all);
sigma = std(MSE_test_all);

% Plot μ ± 1.96σ
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
exportgraphics(fig, 'NT3Fig7.png', 'BackgroundColor', 'white');

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
exportgraphics(fig, 'NT3Fig8.png', 'BackgroundColor', 'white');



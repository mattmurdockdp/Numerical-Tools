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

% Ridge Regression using polynomial features of degree 3
max_degree = 3;
lambda = 0.1;  % Regularization parameter

% Construct polynomial features for training and test sets
X_ridge_train = ones(N, max_degree + 1);
X_ridge_test = ones(size(x_test, 1), max_degree + 1);
for k = 1:max_degree
    X_ridge_train(:, k+1) = x_train.^k;
    X_ridge_test(:, k+1) = x_test.^k;
end

%% Solve Ridge Regression: Theta = inv(X'*X + lambda*I) * X'*y
I = eye(max_degree + 1);
Theta_ridge = (X_ridge_train' * X_ridge_train + lambda * I) \ (X_ridge_train' * y_train);

% Predictions
y_est_train_ridge = X_ridge_train * Theta_ridge;
y_est_test_ridge = X_ridge_test * Theta_ridge;

% MSE calculations
MSE_train_ridge = mean((y_train - y_est_train_ridge).^2);
MSE_test_ridge = mean((y_test - y_est_test_ridge).^2);

% Plot Ridge Regression predictions
fig = figure;
hold on;
scatter(x_train, y_train, 40, 'b', 'filled');
scatter(x_test, y_test, 40, 'r', 'filled');

% Smooth prediction line
x_range = linspace(min(x_train), max(x_train), 100)';
X_range = ones(length(x_range), max_degree + 1);
for k = 1:max_degree
    X_range(:, k+1) = x_range.^k;
end
y_range = X_range * Theta_ridge;

plot(x_range, y_range, 'k-', 'LineWidth', 2);
xlabel('x');
ylabel('y');
legend('Train data', 'Test data', 'Ridge prediction');
title(['Ridge Regression (degree = ', num2str(max_degree), ', \lambda = ', num2str(lambda), ')']);
grid on;
hold off;
set(fig, 'Color', 'w');
exportgraphics(fig, 'NT3Fig7.png', 'BackgroundColor', 'white');

%% Cross validation

% Generate synthetic quadratic data
N = 100;
x = linspace(-3, 3, N)';
y = 1 + 2*x + 3*x.^2 + 0.5*randn(N, 1);  % true model + noise

% Number of cross-validation folds
K = 5;
indices = crossvalind('Kfold', N, K);
MSE_test_all = zeros(K, 1);

for k = 1:K
    test_idx = (indices == k);
    train_idx = ~test_idx;

    % Split data
    x_train = x(train_idx);
    y_train = y(train_idx);
    x_test = x(test_idx);
    y_test = y(test_idx);

    % Create polynomial features (degree = 2)
    X_train = [ones(length(x_train),1), x_train, x_train.^2];
    X_test = [ones(length(x_test),1), x_test, x_test.^2];

    % Least squares solution
    Theta = (X_train' * X_train) \ (X_train' * y_train);

    % Test prediction
    y_pred_test = X_test * Theta;

    % MSE on test
    MSE_test_all(k) = mean((y_test - y_pred_test).^2);
end

% Compute mean and standard deviation
mu = mean(MSE_test_all);
sigma = std(MSE_test_all);

% Plotting mu ± 1.96*sigma
fig = figure;
hold on;
bar(1, mu, 'FaceColor', [0.6 0.6 0.9]);
errorbar(1, mu, 1.96*sigma, 'k', 'LineWidth', 2, 'CapSize', 15);
xlim([0.5 1.5]);
ylabel('Test MSE');
title('\mu \pm 1.96\sigma (Cross-Validation)');
set(gca, 'XTick', []);
grid on;
hold off;
set(fig, 'Color', 'w');
exportgraphics(fig, 'NT3Fig8.png', 'BackgroundColor', 'white');



%% Initial data
clear all
close all

%% Data definition
N     = 15; 
x1    = rand(N,1)*10; 
x2    = rand(N,1)*10;
Theta = [1;1];
noise = 30 * randn(size(x2));
x     = [x1, x2];
h     = x*Theta+noise;
y     = zeros(N,1);
for i=1:N
    if h(i)<0
        y(i)=0;
    else
        y(i)=1;
    end
end
D = [x1, x2, y];

%% Separate the data between train (70% of points) and test (30% of points)

idx = randperm(N); 
train_size = round(0.7 * N); 
train_idx = idx(1:train_size);    
test_idx = idx(train_size+1:end); 

D_train = D(train_idx, :);
D_test = D(test_idx, :);

x1_train = D_train(:,1);
x2_train = D_train(:,2);
y_train = D_train(:,3);

x1_test = D_test(:,1);
x2_test = D_test(:,2);
y_test = D_test(:,3);


%% Solve logistic regression and find Theta1 and Theta2, decision line is x2 = -Theta1/Theta2 * X1

tau = 0.01;
iter = 500;
lambda = 0.1;
X_train = [x1_train, x2_train];
N_linear = size(X_train,1);
Cost = zeros(iter,1);
Grad_save = zeros(iter,1);

for k = 1:iter
    h = X_train * Theta;
    g = 1 ./ (1 + exp(-h));

    Gradient = zeros(size(Theta));

    for i = 1:N_linear
        xi = X_train(i,:)';
        Gradient = Gradient + (g(i) - y_train(i)) * xi;
    end

    Gradient = (-1 / N_linear) * Gradient;
    Grad_save(k) = norm(Gradient);
    Theta = Theta + tau * Gradient;

    sum_cost = 0;
    epsilon = 1e-8;

    for i = 1:N_linear
        sum_cost = sum_cost + (1 - y_train(i)) * (-log(1 - g(i) + epsilon)) ...
            + y_train(i) * (-log(g(i) + epsilon));
    end

    cost = (1 / N_linear) * sum_cost;

    reg_term = lambda * (Theta' * Theta);

    Cost(k,1) = cost + reg_term;

end

% fig = figure;
% plot(1:iter, Grad_save, 'b-', 'LineWidth', 2);
% xlabel('Iteration');
% ylabel('||\nabla J(\Theta)||');
% title('Gradient Norm Over Iterations');
% grid on;
% set(fig, 'Color', 'w');
% exportgraphics(fig, 'NT4Fig6.png', 'BackgroundColor', 'white');
% 
% fig = figure;
% plot(1:iter, Cost, 'r-', 'LineWidth', 2);
% xlabel('Iteration');
% ylabel('Cost J(\Theta)');
% title('Cost Function Over Iterations');
% grid on;
% set(fig, 'Color', 'w');
% exportgraphics(fig, 'NT4Fig7.png', 'BackgroundColor', 'white');
% 
% decision_line = -(Theta(1)/Theta(2)) * x1;
% 
% fig = figure;
% 
% % === TRAINING DATA ===
% % y = 1 → orange, filled
% idx_train_1 = y_train == 1;
% scatter(x1_train(idx_train_1), x2_train(idx_train_1), 60, [1 0.5 0], 'filled'); hold on;
% 
% % y = 0 → green, filled
% idx_train_0 = y_train == 0;
% scatter(x1_train(idx_train_0), x2_train(idx_train_0), 60, [0 0.7 0], 'filled');
% 
% % === TEST DATA ===
% % y = 1 → orange, empty
% idx_test_1 = y_test == 1;
% scatter(x1_test(idx_test_1), x2_test(idx_test_1), 60, [1 0.5 0]);
% 
% % y = 0 → green, empty
% idx_test_0 = y_test == 0;
% scatter(x1_test(idx_test_0), x2_test(idx_test_0), 60, [0 0.7 0]);
% 
% % === DECISION LINE ===
% plot(x1, decision_line, 'k--', 'LineWidth', 2);
% 
% xlabel('x1');
% ylabel('x2');
% title('Data Points Colored by Class with Decision Boundary');
% 
% legend({'Train y = 1', 'Train y = 0', 'Test y = 1', 'Test y = 0', 'Decision Boundary'}, 'Location', 'best');
% grid on;
% axis([0 10 0 10]);
% hold off;
% set(fig, 'Color', 'w');
% exportgraphics(fig, 'NT4Fig8.png', 'BackgroundColor', 'white');

%% Polynomial Logistic Regression

X_poly = [ones(size(x1_train)), ...
          x1_train.^2, ...
          x2_train.^2, ...
          x1_train .* x2_train];

Theta_poly = randn(size(X_poly, 2), 1);  
Cost_poly = zeros(iter, 1);
Grad_save_poly = zeros(iter, 1);
tau = 0.001;
iter = 500;
lambda = 0.3;

for k = 1:iter
    h = X_poly * Theta_poly;
    g = 1 ./ (1 + exp(-h));
    Gradient = zeros(size(Theta_poly));
    
    for i = 1:N_linear
        xi = X_poly(i,:)';
        Gradient = Gradient + (g(i) - y_train(i)) * xi;
    end

    Gradient = (-1 / N_linear) * Gradient;
    Grad_save_poly(k) = norm(Gradient);
    Theta_poly = Theta_poly + tau * Gradient;

    sum_cost = 0;
    for i = 1:N_linear
        sum_cost = sum_cost + (1 - y_train(i)) * (-log(1 - g(i) + epsilon)) ...
            + y_train(i) * (-log(g(i) + epsilon));
    end

    cost = (1 / N_linear) * sum_cost;
    reg_term = lambda * (Theta_poly' * Theta_poly);
    Cost_poly(k) = cost + reg_term;
end

% Plot polynomial training evolution
fig = figure;
plot(1:iter, Grad_save_poly, 'b-', 'LineWidth', 2);
xlabel('Iteration'); ylabel('||\nabla J(\Theta)||');
title('Poly Gradient Norm Over Iterations'); grid on;
set(fig, 'Color', 'w');
exportgraphics(fig, 'NT4Fig9.png', 'BackgroundColor', 'white');

fig = figure;
plot(1:iter, Cost_poly, 'r-', 'LineWidth', 2);
xlabel('Iteration'); ylabel('Cost J(\Theta)');
title('Poly Cost Function Over Iterations'); grid on;
set(fig, 'Color', 'w');
exportgraphics(fig, 'NT4Fig10.png', 'BackgroundColor', 'white');

[x1_grid, x2_grid] = meshgrid(linspace(0, 10, 300), linspace(0, 10, 300));
x1_sq = x1_grid.^2;
x2_sq = x2_grid.^2;
x1x2 = x1_grid .* x2_grid;

X_mesh = [ones(numel(x1_grid), 1), ...
          x1_sq(:), ...
          x2_sq(:), ...
          x1x2(:)];

h_mesh = X_mesh * Theta_poly;
g_mesh = reshape(1 ./ (1 + exp(-h_mesh)), size(x1_grid));

fig = figure;
contour(x1_grid, x2_grid, g_mesh, [0.5, 0.5], 'k--', 'LineWidth', 2); hold on;
scatter(x1_train(y_train==1), x2_train(y_train==1), 60, [1 0.5 0], 'filled');
scatter(x1_train(y_train==0), x2_train(y_train==0), 60, [0 0.7 0], 'filled');
scatter(x1_test(y_test==1), x2_test(y_test==1), 60, [1 0.5 0]);
scatter(x1_test(y_test==0), x2_test(y_test==0), 60, [0 0.7 0]);
xlabel('x1');
ylabel('x2');
title('Polynomial Decision Boundary');
legend({'Decision Boundary', 'Train y = 1', 'Train y = 0', 'Test y = 1', 'Test y = 0'}, 'Location', 'best');
axis([0 10 0 10]);
grid on;
set(fig, 'Color', 'w');
exportgraphics(fig, 'NT4Fig11.png', 'BackgroundColor', 'white');
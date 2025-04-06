%% Neural Networks

clear all
close all

%% Data definition

Npoints_sector = 100;
N_sectors = 3;

x1_sector1 = rand(Npoints_sector,1)*10+25;
x2_sector1 = rand(Npoints_sector,1)*10+25;

x1_sector2 = rand(Npoints_sector,1)*10+15;
x2_sector2 = rand(Npoints_sector,1)*10+10;

x1_sector3 = rand(Npoints_sector,1)*10+10;
x2_sector3 = rand(Npoints_sector,1)*10+30;

y_sector1 = ones(Npoints_sector,1);
y_sector2 = ones(Npoints_sector,1)*2;
y_sector3 = ones(Npoints_sector,1)*3;

x1 = [x1_sector1; x1_sector2; x1_sector3];
x2 = [x2_sector1; x2_sector2; x2_sector3];
y  = [y_sector1; y_sector2; y_sector3];

D = [x1, x2, y];

% % Colors for each sector
% % colors = lines(3); 
% % Points
% % figure;
% % hold on;
% % scatter(x1(y==1), x2(y==1), 50, colors(1,:), 'filled'); % Sector 1
% % scatter(x1(y==2), x2(y==2), 50, colors(2,:), 'filled'); % Sector 2
% % scatter(x1(y==3), x2(y==3), 50, colors(3,:), 'filled'); % Sector 3
% % hold off;
% % xlabel('x1');
% % ylabel('x2');
% % title('Distribución de puntos en los 3 sectores');
% % legend({'Sector 1', 'Sector 2', 'Sector 3'}, 'Location', 'best');
% % grid on;

%% Divide the data into train data (70% of D) and test data (30% of D)

Npoints_total = size(D, 1);

idx = randperm(Npoints_total); 

train_size = round(0.7 * Npoints_total); 

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


%% Solve logistic regression considering linear model x = [1, x1, x2]

N_train = size(x1_train,1);
N_dim = 3;

% Define matrix X
X_linear = [ones(N_train,1), x1_train, x2_train];

% Define matrix Y of 1's and 0's
Y = zeros(N_train,N_sectors);
for i=1:N_train
    Y(i,y_train(i))=1;
end

% Theta linear model x = [1,x1,x2]
Theta_linear = randn(N_dim,N_sectors) * 0.1; 

% Heavyside function 
noise = 30 * randn(N_train,N_sectors);
h_linear = X_linear*Theta_linear;

% Signoid function
g_linear = zeros(N_train,N_sectors);
for i=1:N_train
    for j=1:N_sectors
        g_linear(i,j) = 1/(1+exp(-h_linear(i,j)));
    end
end

% Gradient method
tau = 0.001;
iter = 200;
Lambda = 0.01;
Cost = zeros(iter,1);
Grad_save = zeros(iter, N_sectors); 

for k=1:iter
    sum_grad = zeros(N_dim, N_sectors); 
    
    % Compute gradient
    for i=1:N_train
        for j=1:N_sectors
            sum_grad(:,j) = sum_grad(:,j) + (g_linear(i,j) - Y(i,j)) * X_linear(i,:)';
        end
    end
    
    Gradient = (1/N_train) * sum_grad;
    
    % Store gradient norms
    for j=1:N_sectors
        Grad_save(k,j) = norm(Gradient(:,j));
    end
    
    % Update Theta
    Theta_linear = Theta_linear - tau * Gradient;
    
    % Update heavyside and sigmoid
    h_linear = X_linear * Theta_linear;
    g_linear = 1 ./ (1 + exp(-h_linear)); 

    % Compute cost
    sum_cost = 0;
    for i=1:N_train
        for j=1:N_sectors
            sum_cost = sum_cost + (1 - Y(i,j)) * (-log(1 - g_linear(i,j))) + Y(i,j) * (-log(g_linear(i,j)));
        end
    end
    
    reg_term = 0;
    for j=1:N_sectors
        reg_term = reg_term + Lambda * sum(Theta_linear(:,j).^2);
    end

    Cost(k,1) = (1/N_train) * sum_cost + reg_term;
end

% Cost function
figure;
plot(1:iter, Cost, 'LineWidth', 2);
xlabel('Iteration');
ylabel('Cost');
title('Cost evolution for linear model');
grid on;

% Gradient norm per class
figure;
hold on;
for j = 1:N_sectors
    plot(1:iter, Grad_save(:,j), 'LineWidth', 2, 'DisplayName', ['Class ', num2str(j)]);
end
xlabel('Iteration');
ylabel('Gradient norm');
title('Gradient norm evolution per class for linear model');
legend('show');
grid on;

% Define the range of x1 for plotting the decision boundaries
x1_range = linspace(min(x1)-5, max(x1)+5, 200);

% Colors for each class
colors = lines(N_sectors); 

% Plot all data points (filled)
figure; hold on;
scatter(x1(y==1), x2(y==1), 50, colors(1,:), 'filled');
scatter(x1(y==2), x2(y==2), 50, colors(2,:), 'filled');
scatter(x1(y==3), x2(y==3), 50, colors(3,:), 'filled');

% Overlay training points (empty)
scatter(x1_train, x2_train, 70, 'k', 'LineWidth', 1.5);

% Plot decision boundaries
for j = 1:N_sectors
    theta_j = Theta_linear(:,j);
    % Avoid division by zero
    if abs(theta_j(3)) > 1e-6
        x2_line = -(theta_j(1) + theta_j(2)*x1_range) / theta_j(3);
        plot(x1_range, x2_line, '--', 'Color', colors(j,:), 'LineWidth', 2, ...
            'DisplayName', ['Boundary for class ', num2str(j)]);
    end
end

xlabel('x1');
ylabel('x2');
title('Data points with decision boundaries and training set for linear model');
legend({'Sector 1', 'Sector 2', 'Sector 3', 'Training points'}, 'Location', 'best');
grid on;
axis tight;

%% Solve logistic regression considering quadratic model x = [1, x1, x2, x1^2, x1*x2, x2^2]

N_train = size(x1_train, 1);
N_dim = 6; 

% Define matrix X
X_quadratic = [ones(N_train,1), x1_train, x2_train, x1_train.^2, x1_train.*x2_train, x2_train.^2];

% Target matrix Y 
Y = zeros(N_train, N_sectors);
for i = 1:N_train
    Y(i, y_train(i)) = 1;
end

% Theta for quadratic model
Theta_quadratic = randn(N_dim, N_sectors) * 0.1;

% Gradient descent parameters
tau = 0.00001;
iter = 400;
Lambda = 0.01;
Cost_quad = zeros(iter,1);
Grad_save_quad = zeros(iter, N_sectors);

% Initial h and g
h_quad = X_quadratic * Theta_quadratic;
g_quad = 1 ./ (1 + exp(-h_quad));

for k = 1:iter
    sum_grad = zeros(N_dim, N_sectors);

    % Compute gradient
    for i = 1:N_train
        for j = 1:N_sectors
            sum_grad(:,j) = sum_grad(:,j) + (g_quad(i,j) - Y(i,j)) * X_quadratic(i,:)';
        end
    end

    Gradient = (1/N_train) * sum_grad;

    % Store gradient norms
    for j = 1:N_sectors
        Grad_save_quad(k,j) = norm(Gradient(:,j));
    end

    % Update Theta
    Theta_quadratic = Theta_quadratic - tau * Gradient;

    % Recalculate h and g
    h_quad = X_quadratic * Theta_quadratic;
    g_quad = 1 ./ (1 + exp(-h_quad));

    % Compute cost
    sum_cost = 0;
    for i = 1:N_train
        for j = 1:N_sectors
            sum_cost = sum_cost + (1 - Y(i,j)) * (-log(1 - g_quad(i,j))) + Y(i,j)  * (-log(g_quad(i,j)));
        end
    end
    
    reg_term = 0;
    for j=1:N_sectors
        reg_term = reg_term + Lambda * sum(Theta_quadratic(:,j).^2);
    end

    Cost_quad(k,1) = (1/N_train) * sum_cost + reg_term;
end

% Plot cost for quadratic model
figure;
plot(1:iter, Cost_quad, 'LineWidth', 2);
xlabel('Iteration');
ylabel('Cost');
title('Cost evolution for quadratic model');
grid on;

% Plot gradient norms for each sector
figure;
hold on;
for j = 1:N_sectors
    plot(1:iter, Grad_save_quad(:,j), 'LineWidth', 2);
end
xlabel('Iteration');
ylabel('Gradient Norm');
title('Gradient norm evolution for quadratic model');
legend({'Sector 1', 'Sector 2', 'Sector 3'}, 'Location', 'best');
grid on;
hold off;




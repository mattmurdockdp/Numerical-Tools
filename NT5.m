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

% Colors for each sector
colors = lines(3); 

% Points
figure;
hold on;
scatter(x1(y==1), x2(y==1), 50, colors(1,:), 'filled'); % Sector 1
scatter(x1(y==2), x2(y==2), 50, colors(2,:), 'filled'); % Sector 2
scatter(x1(y==3), x2(y==3), 50, colors(3,:), 'filled'); % Sector 3
hold off;
xlabel('x1');
ylabel('x2');
title('Distribución de puntos en los 3 sectores');
legend({'Sector 1', 'Sector 2', 'Sector 3'}, 'Location', 'best');
grid on;

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

% Heavyside function with noise
noise = 30 * randn(N_train,1);
h_linear = X_linear*Theta_linear+noise;

% Signoid function
g_linear = zeros(N_train,N_sectors);
for i=1:N_train
    for j=1:N_sectors
        g_linear(i,j) = 1/(1+exp(-h_linear(i,j)));
    end
end

% Gradient method
tau = 0.0001;
iter = 200;
Lambda = 0;
Cost = zeros(iter,1);
Grad_save = zeros(iter, N_sectors); % Save gradient norms for each sector

for k=1:iter
    sum_grad = zeros(N_dim, N_sectors); % Gradient sum initialization
    
    % Compute gradient using a single loop
    for i=1:N_train
        for j=1:N_sectors
            sum_grad(:,j) = sum_grad(:,j) + (g_linear(i,j) - Y(i,j)) * X_linear(i,:)';
        end
    end
    
    % Compute the gradient
    Gradient = (1/N_train) * sum_grad;
    
    % Store gradient norms
    for j=1:N_sectors
        Grad_save(k,j) = norm(Gradient(:,j));
    end
    
    % Update Theta
    Theta_linear = Theta_linear - tau * Gradient;
    
    % Recompute hypothesis
    h_linear = X_linear * Theta_linear;
    g_linear = 1 ./ (1 + exp(-h_linear)); % Apply sigmoid function to all elements

    % Compute cost
    sum_cost = 0;
    for i=1:N_train
        for j=1:N_sectors
            sum_cost = sum_cost + ...
                (1 - Y(i,j)) * (-log(1 - g_linear(i,j))) + ...
                 Y(i,j) * (-log(g_linear(i,j))) + ...
                 Lambda * (Theta_linear(:,j)' * Theta_linear(:,j));
        end
    end
    Cost(k,1) = (1/N_train) * sum_cost;
end




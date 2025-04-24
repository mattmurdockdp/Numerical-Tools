%% NT Final Project
% Matheus Victor Do Prado Amaral

close all
clear 
clc

%% Data Preprocessing

opts = detectImportOptions('aircraft_data_csv.csv', 'VariableNamingRule', 'preserve');
opts = setvartype(opts, 'Aircraft', 'string');  % Treat aircraft names as strings

data = readtable('aircraft_data_csv.csv', opts);

% Extract numerical data only (all columns except 'Aircraft')
X = table2array(data(:, 2:end));

% Normalize the data (zero mean, unit std)
X_norm = (X - mean(X)) ./ std(X);

% Extract MTOW and Cruise Speed
mtow_raw = X(:, 1);             % MTOW (column 1)
cruise_raw = X(:, 8);           % Cruise Speed (column 8)

mtow_norm = X_norm(:, 1);       % Normalized MTOW
cruise_norm = X_norm(:, 8);     % Normalized Cruise Speed

% Index for aircraft (1 to number of aircraft)
n = height(data);
idx = 1:n;

% Create figure
figure;

% Subplot 1: MTOW (raw)
subplot(2,2,1);
scatter(1:height(data), X(:,1), 'filled');
title('MTOW (Raw)');
xlabel('Aircraft Index');
ylabel('MTOW (kg)');

% Subplot 2: MTOW (normalized)
subplot(2,2,2);
scatter(1:height(data), X_norm(:,1), 'filled');
title('MTOW (Norm)');
xlabel('Aircraft Index');
ylabel('MTOW (z-score)');

% Subplot 3: Cruise Speed (raw)
subplot(2,2,3);
scatter(1:height(data), X(:,8), 'filled');
title('Cruise Speed (Raw)');
xlabel('Aircraft Index');
ylabel('Speed (km/h)');

% Subplot 4: Cruise Speed (normalized)
subplot(2,2,4);
scatter(1:height(data), X_norm(:,8), 'filled');
title('Cruise Speed (Norm)');
xlabel('Aircraft Index');
ylabel('Speed (z-score)');

set(gcf, 'Color', 'w');
exportgraphics(gcf, 'mtow_cruise_comparison.png', 'BackgroundColor', 'white', 'ContentType', 'image');

%% SVD Method

% Compute SVD
[U, S, V] = svd(X_norm, 'econ');

% Choose number of components to keep (e.g., 2D projection)
k = 2;

% Project data onto the first k principal directions
Z = U(:, 1:k) * S(1:k, 1:k);  % Z = X_norm * V(:, 1:k);

% Plot the 2D representation
figure;
scatter(Z(:,1), Z(:,2), 50, 'filled');
title('2D Representation of Aircraft Data via SVD');
xlabel('x_1');
ylabel('x_2');
grid on;
hold on;

% Add numeric labels to each point
for i = 1:size(Z, 1)
    text(Z(i,1) + 0.2, Z(i,2), num2str(i), 'FontSize', 8);
end

% Save figure with white background
set(gcf, 'Color', 'w');
exportgraphics(gcf, 'svd_2d_projection.png', 'BackgroundColor', 'white', 'ContentType', 'image');



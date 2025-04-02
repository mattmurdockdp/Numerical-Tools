%% Initial data
clear all
close all

% Definim els punts
N = 15; 
x1 = sort(10 * rand(1, N))'; 
x2 = 10 * rand(1, N)';
X = zeros(N,3);
X(:,1) = 1;
X(:,2) = x1;
X(:,3) = x2;

% Rectes per definir y
y1 = @(m) -2*m + 5;
y2 = @(m)  2*m - 5;

% Calculem y
y = zeros(N,1);
for i=1:N
    if x2(i)<=y1(x1(i))
        y(i)=1;
    elseif x2(i)<=y2(x1(i))
        y(i)=2;
    else
        y(i)=3;
    end
end

% Definim matriu Y de 1's i 0's
Y = zeros(N,3);
for i=1:N
    Y(i,y(i))=1;
end

noise = 30 * randn(size(x1));

% Definim Theta pels 3 subsets considerant model lineal x = [1,x1,x2]
Theta = randn(3,3) * 0.1; 

% Calculem la h per cada subset
h = X*Theta+noise;

% Apliquem la signoid function
g = zeros(N,3);
for i=1:N
    for j=1:3
        g(i,j) = 1/(1+exp(-h(i,j)));
    end
end

% % Plot de los puntos y sus clases
% figure; hold on; grid on;
% scatter(x1(y == 1), x2(y == 1), 50, 'r', 'filled');
% scatter(x1(y == 2), x2(y == 2), 50, 'g', 'filled');
% scatter(x1(y == 3), x2(y == 3), 50, 'b', 'filled');
% % Trazamos las líneas de decisión
% fplot(y1, [0, 10], 'r--', 'LineWidth', 2);
% fplot(y2, [0, 10], 'g--', 'LineWidth', 2);
% 
% xlabel('x_1');
% ylabel('x_2');
% title('Puntos Agrupados por Sectores con Líneas de Decisión');
% legend('Sector 1', 'Sector 2', 'Sector 3', 'Location', 'Best');
% hold off;

%% Solve logistic regression
tau = 0.0001;
iter = 200;
Theta_old = Theta;
Lambda = 0;
h_old = h;
g_old = g;
Cost = zeros(iter,1);
Grad_save1 = zeros(iter,1);
Grad_save2 = zeros(iter,1);
Grad_save3 = zeros(iter,1);

for k=1:iter
    sum_grad1 = zeros(3,1);
    sum_grad2 = zeros(3,1);
    sum_grad3 = zeros(3,1);
    for i=1:N
        sum_grad1 = sum_grad1 + (g_old(i,1)-Y(i,1))*X(i,:)';
        sum_grad2 = sum_grad2 + (g_old(i,2)-Y(i,2))*X(i,:)';
        sum_grad3 = sum_grad3 + (g_old(i,3)-Y(i,3))*X(i,:)';
    end

    % Gradient
    Gradient1 = (1/N)*sum_grad1;
    Gradient2 = (1/N)*sum_grad2;
    Gradient3 = (1/N)*sum_grad3;

    Grad_save1(k) = norm(Gradient1);
    Grad_save2(k) = norm(Gradient2);
    Grad_save3(k) = norm(Gradient3);

    % Update values
    Theta_old(1,:) = Theta_old(1,:) + tau * Gradient1';
    Theta_old(2,:) = Theta_old(2,:) + tau * Gradient2';
    Theta_old(3,:) = Theta_old(3,:) + tau * Gradient3';
    h_old = X*Theta_old;
    for i=1:N
        for j=1:3
            g_old(i,j) = 1/(1+exp(-h_old(i,j)));
        end
    end

    % Cost function
    sum_cost = 0;
    for i=1:N
        for j=1:3
            sum_cost = sum_cost + (1-Y(i,j))*(-log(1-g_old(i,j)))+Y(i,j)*(-log(g_old(i,j)))+Lambda*Theta_old(:,j)'*Theta_old(:,j);
        end
    end
    Cost(k,1)=(1/N)*sum_cost;

end

figure;
plot(1:iter,Grad_save1,'r')
hold on
plot(1:iter,Grad_save2,'g')
plot(1:iter,Grad_save3,'b')
legend('Gradient Sector 1', 'Gradient Sector 2', 'Gradient Sector 3', 'Location', 'Best')
title('Gradient per cada iteració')
xlabel('Iteració');
ylabel('Gradient');

figure;
plot(1:iter,Cost)
title('Cost')

% Decision lines
DL1 = (-Theta_old(1,1)-Theta_old(1,2)*X(:,2))/Theta_old(1,3);
DL2 = (-Theta_old(2,1)-Theta_old(2,2)*X(:,2))/Theta_old(2,3);
DL3 = (-Theta_old(3,1)-Theta_old(3,2)*X(:,2))/Theta_old(3,3);




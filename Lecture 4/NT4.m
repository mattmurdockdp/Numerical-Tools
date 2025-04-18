%% Initial data
clear all
close all

N = 15; % Number of points

x1 = sort(10 * rand(1, N)); % Generate random x1 values and sort them
x2 = 10 * rand(1, N);

Theta1old = 1;
Theta2old = 1;

noise = 30 * randn(size(x2));

htheta = Theta1old*x1 + Theta2old*x2+noise;
y = zeros(N,1);

for i=1:N
    if htheta(i)<0
        y(i)=0;
    else
        y(i)=1;
    end
end


%% a) Solve logistic regression and find Theta1 and Theta2, decision line is x2 = -Theta1/Theta2 * X1

tau = 0.01;
iter = 100;
X = [x1',x2'];
Theta_old = [Theta1old;Theta2old];
Cost = zeros(iter,1);
Grad_save = zeros(iter,1);

for k=1:iter

    htheta = X*Theta_old;
    g_theta = zeros(N,1);
    sum = zeros(1,2);

    for i=1:N
        g_theta(i)=1/(1+exp(-htheta(i))); % Signoid Function
    end

    for i=1:N
    sum = sum + (g_theta(i)-y(i)).*X(i,:);
    end

    Gradient = (-1/N)*sum;
    Grad_save(k)=norm(Gradient);
    Theta_old = Theta_old+tau*Gradient';

    % Cost
    sum_cost = 0;
    for i=1:N
    sum_cost = sum_cost + (1-y(i))*(-log(1-g_theta(i)))+y(i)*(-log(g_theta(i)));
    end
    Cost(k,1)=(1/N)*sum_cost;
   
end

figure;
plot(1:iter,Grad_save)
title('Gradient')

figure;
plot(1:iter,Cost)
title('Cost')

% Decision Line
Decision = -Theta_old(1)/Theta_old(2)*x1;

figure;
scatter(x1,x2)
hold on
plot(x2,Decision)
hold off
title('Data points')


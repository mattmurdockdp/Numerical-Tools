%% 1. Solve the normal equations

% n = m
A1 = [1 0 0
     0 1 0
     0 0 1]; 
B1 = [1;1;1];

Theta1 = inv(A1'*A1)*A1'*B1
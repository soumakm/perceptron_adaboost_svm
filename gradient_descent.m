% This function performs gradient descent for single sample
% input: 
%       y: training samples already augmented with 1
%       a: weight vector
%     eta: learning rate
% output:
%        s: 0 if all tarining samples are classified correctly
%           1 if weight vectors are adjusted


function a = gradient_descent(y,b, eta)
a = b;
k = 1;
n= size(y,1);
for i=k:n
    if a*y(i,:)' > 0
        continue;
    else
       a = a + eta*y(i, :); 
       k = 1;
    end
end    
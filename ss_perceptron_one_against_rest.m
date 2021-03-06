% This functions calculates the Perceptron criterion for one against rest
% using gradient descent procedure using single sample update
% input: x: a matrix of nxd dimentions, rows represent number of training
%           samples, d represents dimension of feature vector.
%           First column of x is class number
%        c: class number for which it will generate weight vector
%        a: initial weight vector for training
%      eta: learning rate
% output:
%        a: weight vector trained by the Perceptron criterion

function a = ss_perceptron_one_against_rest(x, c, a, eta)

% number of samples
n = size(x,1);

% first add 1 to feature to make augmented vector
I  = ones(n, 1);

% augmented matrix add 1, strip the class information
y = [I x(:,2:end)];

% -negate ys which do not belong to class denoted by c
for i = 1:n
   if (x(i) ~=  c)
      y(i,:) =  y(i,:)*(-1);
   end    
end   

% call gradient descent
a = gradient_descent(y, a, eta);

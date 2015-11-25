% This functions calculates the Perceptron criterion
% using gradient descent procedure
% input: x, y: a matrix of nxd dimentions, rows represent number of training
%           samples, d represents dimension of feature vector.
%           First column of x is class number
%        a: initial weight vector for training
%      eta: learning rate
% output:
%        a: weight vector trained by the Perceptron criterion

function a = batch_perceptron_one_against_other(x, y, a, eta, b)

% number of samples
n = size(x,1);
m = size(y,1);

% first add 1 to feature to make augmented vector
Ix  = ones(n, 1);
Iy  = ones(m, 1);

% augmented matrix add 1, strip the class information
x = [Ix x(:,2:end)];
y = [Iy y(:,2:end)];

% -negate y
y = y*(-1);

% add them to a single matrix
x = [x;y];
% call gradient descent
a = gradient_descent_batch(x, a, eta, b);













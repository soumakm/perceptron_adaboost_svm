% Multi class single sample perceptron test on handwritten digit data set 
% This file should read files one for training, and one for test
% It should arrange the file data so that 1st column is the class 
% rest of the columns are feature vector
% it should also order the data according to class number

%modify next two lines based on data sets
% number of class
close all;
clc;
c = 3;
% learning rate
eta = 0.6;

% read data, 1st column is the class
x = dlmread('handwritten_0_2_train.txt');
y = dlmread('handwritten_0_2_test.txt');

% number of training samples
n = size(x,1);

%dimesion of feature vector
d = size(x,2) - 1;

% initialize weight vector with all ones
a0 = ones(1, d+1);

%trained weight vector 
a = zeros(c,d+1);

for i=1:c
    a(i,:) = ss_perceptron_one_against_rest(x, i-1, a0, eta);
end    

%test data
k = size(y,1);
%scalar to hold number of correct classification
h = 0;
%count to hold ambigous
l=0;
% first add 1 to feature to make augmented vector
I  = ones(k, 1);

% augmented matrix add 1, 
y = [y(:, 1) I y(:,2:end)];
fprintf('Sample No.  Actual Class  Classified Class  Corrrect?\n');
%loop through each test sample
for i=1:k
    l = 0;
    % loop through weaight vectors for each class 
    for j=1:c
        if a(j,:)*y(i,2:end)' > 0
             l = l + 1;
             class = j-1; %clas 0 to 2
        end    
    end
    if (y(i) == class && l == 1) % if they are correct
        h = h+1;
        fprintf('%d\t\t\t\t %d\t\t\t\t %d\t\t\t\t yes\n', i, y(i), class);
    elseif (y(i) ~= class && l == 1)
        fprintf('%d\t\t\t\t %d\t\t\t\t %d\t\t\t\t no\n', i, y(i), class);
    else    
        fprintf('%d\t\t\t\t %d\t\t\t\t ambiguous\t\t no\n', i, y(i));
    end   
    
end
p = h/k*100;
h
fprintf('The performance of two class  classifier on wine data set is %.2f\n',p);
   

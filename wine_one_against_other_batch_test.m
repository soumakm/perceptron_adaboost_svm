% Multi class single sample perceptron test on wine data set 
% using one against other
% This file should read files one for training, and one for test
% It should arrange the file data so that 1st column is the class 
% rest of the columns are feature vector
% it should also order the data according to class number

%modify next two lines based on data sets
% number of class
close all;
clear all;
clc;

%margin
b = 5;
%number of classes
c = 3;
% learning rate
eta = 0.6;

% read data, 1st column is the class
ux = dlmread('wine_uci_train.txt');
uy = dlmread('wine_uci_test.txt');

%normalize
x = [ux(1:end,1),normalize(ux(:,2:end))];
y = [uy(1:end,1),normalize(uy(:,2:end))];

% number of training samples
n = size(x,1);

%dimesion of feature vector
d = size(x,2) - 1;

% initialize weight vector with all ones
a0 = ones(1, d+1);

%trained weight vector 
a = zeros(c,d+1);

%separate matrix for each class
b1=1;
b2=1;
b3=1;

for i=1:n
    if (x(i) == 1)
        x1(b1,:) = x(i,:);
        b1 = b1+1;
    end  
    
    if (x(i) == 2)
        x2(b2,:) = x(i,:);
        b2 = b2+1;
    end 
    
    if (x(i) == 3)
        x3(b3,:) = x(i,:);
        b3 = b3+1;
    end 
end    

    a12 = batch_perceptron_one_against_other(x1, x2, a0, eta, b);
    a13 = batch_perceptron_one_against_other(x1, x3, a0, eta, b);
    a23 = batch_perceptron_one_against_other(x2, x3, a0, eta, b);

%test data
k = size(y,1);
%scalar to hold number of correct classification
h = 0;

% first add 1 to feature to make augmented vector
I  = ones(k, 1);

%class vector for each classifier
class = zeros(1,3);

% augmented matrix add 1, 
y = [y(:, 1) I y(:,2:end)];
fprintf('Sample No.  Actual Class  Classified Class  Corrrect?\n');

%loop through each test sample
for i=1:k
    
    % loop through weaight vectors for each class 
    
        if a12*y(i,2:end)' > b
             class(1,1) = 1;
        else
             class(1,1) = 2;
        end 
        
        if a13*y(i,2:end)' > b
             class(1,2) = 1;
        else
             class(1,2) = 3;
        end 
        
        if a23*y(i,2:end)' > b
             class(1,3) = 2;
        else
             class(1,3) = 3;
        end 
        
       [M, F] = mode(class); 
    if(F == 1)   
        fprintf('%d\t\t\t\t %d\t\t\t\t ambiguous\t\t no\n', i, y(i));
    elseif (y(i) == M) % if they are correct
        h = h+1;
        fprintf('%d\t\t\t\t %d\t\t\t\t %d\t\t\t\t yes\n', i, y(i), M);
    elseif (y(i) ~= M)
        fprintf('%d\t\t\t\t %d\t\t\t\t %d\t\t\t\t no\n', i, y(i), M);
    end   
    
end
p = h/k*100;
fprintf('The performance of two class  classifier on wine data set is %.2f\n',p);
   

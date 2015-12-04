% Two class batch perceptron test on handwritten digit data set 
% using one against other batch update rule
% It then uses Adaboost to create stronger classifier.
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
b = -0.1;
%number of classes
c = 3;
% learning rate
eta = 0.01;
%number of adaboost iterations
kmax = 20;

%number of samples
s=1500;

% read data, 1st column is the class
x = dlmread('handwritten_0_2_train.txt');
y = dlmread('handwritten_0_2_test.txt');

x(:,1) = x(:,1) + 1;
y(:,1) = y(:,1) + 1;

% number of training samples
n = size(x,1);

%dimesion of feature vector
d = size(x,2) - 1;

% initialize weight vector with all ones
a0 = ones(1, d+1);

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

% first add 1 to feature to make augmented vector
I  = ones(k, 1);

%class predicted by classifier
class = 0;

% augmented matrix add 1, 
y = [y(:, 1) I y(:,2:end)];

%adaboost for class 1-2 classifier
% call adaboost function
[alpha, hk] = adaboost(x, s, 1,2, eta, b, kmax);
%g_ada will be computed by sum(alpha*hk(x))
g_ada = 0;

% Testing Adaboost for class1-2 classifier

%{
j = 0; h = 0; h1 = 0;
%loop through each test sample
for i=1:k
    g_ada = 0;
    %test only class 1 and class 2 samples
    if(y(i) == 1 || y(i) == 2)
        j = j + 1;
        for l=1:kmax
            if hk(l,:)*y(i,2:end)' > b
                 g_ada = g_ada + alpha(l);
            else
                 g_ada = g_ada - alpha(l);
            end 
        end
        if g_ada > 0
            class = 1;
        else
            class = 2;
        end    
        
        if(y(i) == class)   %correct
            h = h+1;
        end  
        %without adaboost
        if hk(1,:)*y(i,2:end)' > b
             class = 1;
        else
             class = 2;
        end 
        if(y(i) == class)   %correct
            h1 = h1+1;
        end    
    end
end
p = h/j*100;
p1 = h1/j*100;
fprintf('The performance of class 0-1  classifier without AdaBoost on handwritten digit data set is %.2f\n',p1);
fprintf('The performance of class 0-1  classifier using AdaBoost on handwritten digit data set is %.2f\n',p);

%adaboost for class 1-3 classifier
% call adaboost function
[alpha, hk] = adaboost(x, s, 1,3, eta, b, kmax);
%g_ada will be computed by sum(alpha*hk(x))
g_ada = 0;

% Testing Adaboost for class1-2 classifier

j = 0; h = 0; h1 = 0;
%loop through each test sample
for i=1:k
    g_ada = 0;
    %test only class 1 and class 3 samples
    if(y(i) == 1 || y(i) == 3)
        j = j + 1;
        for l=1:kmax
            if hk(l,:)*y(i,2:end)' > b
                 g_ada = g_ada + alpha(l);
            else
                 g_ada = g_ada - alpha(l);
            end 
        end
        if g_ada > 0
            class = 1;
        else
            class = 3;
        end    
        if(y(i) == class)   %correct
            h = h+1;
        end   
        
        %without adaboost
        if hk(1,:)*y(i,2:end)' > b
             class = 1;
        else
             class = 3;
        end 
        if(y(i) == class)   %correct
            h1 = h1+1;
        end    
    end
end
p = h/j*100;
p1 = h1/j*100;
fprintf('The performance of class 0-2  classifier without AdaBoost on handwritten digit data set is %.2f\n',p1);
fprintf('The performance of class 0-2  classifier using AdaBoost on handwritten digit data set is %.2f\n',p);
%}

%adaboost for class 2-3 classifier
% call adaboost function
[alpha, hk] = adaboost(x, s, 2, 3, eta, b, kmax);
%g_ada will be computed by sum(alpha*hk(x))
g_ada = 0;

% Testing Adaboost for class2-3 classifier

j = 0; h = 0; h1 = 0;
%loop through each test sample
for i=1:k
    g_ada = 0;
    %test only class 2 and class 3 samples
    if(y(i) == 2 || y(i) == 3)
        j = j + 1;
        for l=1:kmax
            if hk(l,:)*y(i,2:end)' > b
                 g_ada = g_ada + alpha(l);
            else
                 g_ada = g_ada - alpha(l);
            end 
        end
        if g_ada > 0
            class = 2;
        else
            class = 3;
        end    
        if(y(i) == class)   %correct
            h = h+1;
        end   
        
        %without adaboost
        if hk(1,:)*y(i,2:end)' > b
             class = 2;
        else
             class = 3;
        end 
        if(y(i) == class)   %correct
            h1 = h1+1;
        end    
    end
end
p = h/j*100;
p1 = h1/j*100;
fprintf('The performance of class 1-2  classifier without AdaBoost on handwritten digit data set is %.2f\n',p1);
fprintf('The performance of class 1-2  classifier using AdaBoost on handwritten digit data set is %.2f\n',p);
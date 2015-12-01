% Two class batch perceptron test on wine data set 
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
b = 5;
%number of classes
c = 3;
% learning rate
eta = 1;
%number of adaboost iterations
kmax = 10;

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

fprintf('Computing perfromance for Class 1-2 classifier\n');
fprintf('Sample No.  Actual Class  Classified Class  Corrrect?\n');
j = 0;
h = 0;
%loop through each test sample
for i=1:k
    %test only class 1 and class 2 samples
    if(y(i) == 1 || y(i) == 2)
        j = j + 1;
        if a12*y(i,2:end)' > b
             class = 1;
        else
             class = 2;
        end 
   
        if(y(i) == class)   %correct
            h = h+1;
      %      fprintf('%d\t\t\t\t %d\t\t\t\t %d\t\t\t\t yes\n', i, y(i), class);
        else
      %      fprintf('%d\t\t\t\t %d\t\t\t\t %d\t\t\t\t no\n', i, y(i), class);
        end   
    end
end
p = h/j*100;
fprintf('The performance of class 1-2  classifier on wine data set is %.2f\n',p);

%classifier for 1-3
fprintf('Computing perfromance for Class 1-3 classifier\n');
fprintf('Sample No.  Actual Class  Classified Class  Corrrect?\n');
j = 0;
h = 0;
%loop through each test sample
for i=1:k
    %test only class 1 and class 3 samples
    if(y(i) == 1 || y(i) == 3)
        j = j + 1;
        if a13*y(i,2:end)' > b
             class = 1;
        else
             class = 3;
        end 
   
        if(y(i) == class)   %correct
            h = h+1;
         %   fprintf('%d\t\t\t\t %d\t\t\t\t %d\t\t\t\t yes\n', i, y(i), class);
        else
         %   fprintf('%d\t\t\t\t %d\t\t\t\t %d\t\t\t\t no\n', i, y(i), class);
        end   
    end
end
p = h/j*100;
fprintf('The performance of class 1-3  classifier on wine data set is %.2f\n',p);

%classifier for 2-3
fprintf('Computing perfromance for Class 2-3 classifier\n');
fprintf('Sample No.  Actual Class  Classified Class  Corrrect?\n');
j = 0;
h = 0;
%loop through each test sample
for i=1:k
    %test only class 1 and class 3 samples
    if(y(i) == 2 || y(i) == 3)
        j = j + 1;
        if a23*y(i,2:end)' > b
             class = 2;
        else
             class = 3;
        end 
        if(y(i) == class)   %correct
            h = h+1;
         %   fprintf('%d\t\t\t\t %d\t\t\t\t %d\t\t\t\t yes\n', i, y(i), class);
        else
         %   fprintf('%d\t\t\t\t %d\t\t\t\t %d\t\t\t\t no\n', i, y(i), class);
        end   
    end
end
p = h/j*100;
fprintf('The performance of class 2-3  classifier on wine data set is %.2f\n',p);

%adaboost for class 1-2 classifier

% initialize weight vector with 1/n
a0 = ones(1, d+1)/n;

% first add 1 to feature to make augmented vector
Ix  = ones(n, 1);

%class predicted by classifier
class = 0;

% augmented matrix add 1, 
z = [x(:, 1) Ix x(:,2:end)];

alpha = 0;
%loop for AdaBoost
for q=1:kmax
    a12 = batch_perceptron_one_against_other(x1, x2, a0, eta, b);

    j = 0;
    e = 0;
    %loop through each test sample
    for i=1:n
        %test only class 1 and class 2 samples
        if(z(i) == 1 || z(i) == 2)
            j = j + 1;
            if a12*y(i,2:end)' > b
                 class = 1;
            else
                 class = 2;
            end 
            if(z(i) ~= class)   %incorrect
                e = e+1;
            end   
        end
    end
    fprintf('The error rate of class 1-2  classifier number %d on wine data set is %.2f\n',q,e/j);
    alpha = 0.5*ln((1-e)/e);
    
    %The for loop is just for weight update
    for i=1:n
        %test only class 1 and class 2 samples
        if(z(i) == 1 || z(i) == 2)
            if a12*y(i,2:end)' > b
                 class = 1;
            else
                 class = 2;
            end 
            if(z(i) ~= class)   %incorrect
               a0 = a0*exp(alpha);
            else %correct
               a0 = a0*exp(alpha); 
            end   
        end
    end
    
end

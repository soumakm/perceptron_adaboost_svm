% Two class batch perceptron test on wine data set 
% using one against other single sample update rule
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

    a12 = ss_perceptron_one_against_other(x1, x2, a0, eta);
    a13 = ss_perceptron_one_against_other(x1, x3, a0, eta);
    a23 = ss_perceptron_one_against_other(x2, x3, a0, eta);

%test data
k = size(y,1);

h = 0;

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
        if a12*y(i,2:end)' > 0
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
        if a13*y(i,2:end)' > 0
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
        if a23*y(i,2:end)' > 0
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
  

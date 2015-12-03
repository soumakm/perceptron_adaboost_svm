% Two class single sample perceptron test on wine data set 
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
eta = 0.8;

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

for i=1:c
    a(i,:) = ss_perceptron_one_against_rest(x, i, a0, eta);
end    

%test data
k = size(y,1);
%scalar to hold number of correct classification
h1 = 0; h2 = 0; h3 = 0;

% first add 1 to feature to make augmented vector
I  = ones(k, 1);

% augmented matrix add 1, 
y = [y(:, 1) I y(:,2:end)];

%loop through each test sample
for i=1:k
    
    % loop through weaight vectors for each class 
    % for class 1
    if a(1,:)*y(i,2:end)' > 0
         class = 1;
    else
         class = -1;
    end    
    
    if ((y(i) == 1 && class == 1) || ((y(i) == 2) || y(i) == 3) && class == -1) % if they are correct
        h1 = h1+1;
    end   
    %for class 2
    if a(2,:)*y(i,2:end)' > 0
         class = 2;
    else
         class = -1;
    end    
    
    if ((y(i) == 2 && class == 2) || ((y(i) == 1) || y(i) == 3) && class == -1) % if they are correct
        h2 = h2+1;
    end 
    %for class 3
    if a(3,:)*y(i,2:end)' > 0
         class = 3;
    else
         class = -1;
    end    
    
    if ((y(i) == 3 && class == 3) || ((y(i) == 1) || y(i) == 2) && class == -1) % if they are correct
        h3 = h3+1;
    end 
end
p1 = h1/k*100;
p2 = h2/k*100;
p3 = h3/k*100;

fprintf('The performance of two-class classifier for class 1 against the rest using single sample on wine data set is %.2f\n',p1);
fprintf('The performance of two-class classifier for class 2 against the rest using single sample on wine data set is %.2f\n',p2);
fprintf('The performance of two-class classifier for class 3 against the rest using single sample on wine data set is %.2f\n',p3);
   

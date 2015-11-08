% Two class single sample perceptron test on wine data set 
% This file should read files one for training, and one for test
% It should arrange the file data so that 1st column is the class 
% rest of the columns are feature vector
% it should also order the data according to class number

%modify next two lines based on data sets
% number of class
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

%weight vector 
a = zeros(c,d+1);

for i=1:c
    a(i,:) = single_sample_perceptron(x, i, eta);
end    

%test data
k = size(y,1);
%scalar to hold number of correct classification
h = 0;
% first add 1 to feature to make augmented vector
I  = ones(k, 1);

% augmented matrix add 1, strip the class information
y = [y(:, 1) I y(:,2:end)];
fprintf('Sample No.  Actual Class  Classified Class  Corrrect?\n');
%loop through each test sample
for i=1:k
    % loop through weaight vectors for each class 
    for j=1:c
        if a(j,:)*y(i,2:end)' > 0
            if (y(i) == j) % if they are correct
                 h = h+1;
                 fprintf('%d\t\t\t\t %d\t\t\t\t %d\t\t\t\t yes\n', i, y(i), j);
            else
                fprintf('%d\t\t\t\t %d\t\t\t\t %d\t\t\t\t no\n', i, y(i), j);
            end    
            break;
        end    
    end
end    
p = h/k*100;
fprintf('The performance of two class  classifier on wine data set is %.2f\n',p);
   

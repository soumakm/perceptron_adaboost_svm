% This function will boost a weak perceptron clasifier 
% and return a vector of weak classifier hk and the parameter alpha


function [alpha, hk] = adaboost(x, s, c1, c2, eta, b, kmax)

%size of training set
n=size(x,1);
d=size(x,2)-1;
%weight vector , column
W = ones(n,1)/n;

% initialize weight vector with one, this is for perceptron
a0 = ones(1, d+1);

% first add 1 to feature to make augmented vector
Ix  = ones(n, 1);

% add weight, add 1, 
z = [x(:, 1) W Ix x(:,2:end)];



%class predicted by classifier
class = 0;

alpha = zeros(1,kmax);
hk = zeros(kmax,d+1);
%loop for AdaBoost
for q=1:kmax
    
    zs = sortrows(z,2);
    zz = zs(1:s,:);
    
    b1=1;
    b2=1;

    for i=1:s
        if (zz(i) == c1)
            z1(b1,:) = zz(i,:);
            b1 = b1+1;
        end  

        if (zz(i) == c2)
            z2(b2,:) = zz(i,:);
            b2 = b2+1;
        end 
    end    
    z11 = [z1(:,1) z1(:,4:end)];
    z22 = [z2(:,1) z2(:,4:end)];
    
    hk(q,:) = batch_perceptron_one_against_other(z11, z22, a0, eta, b);

    j = 0;
    e = 0;
    %loop through each test sample
    for i=1:n
        %test only class 1 and class 2 samples
        if(z(i) == c1 || z(i) == c2)
            j = j + 1;
            if hk(q,:)*z(i,3:end)' > b
                 class = c1;
            else
                 class = c2;
            end 
            if(z(i) ~= class)   %incorrect
                e = e+1;
            end   
        end
    end
    fprintf('The error rate of class 1-2  AdaBoost classifier number %d on wine data set is %.2f\n',q,e/j);
    alpha(q) = 0.5*log((1-e)/e);
    
    %The for loop is just for weight update
    for i=1:n
        %test only class 1 and class 2 samples
        if(z(i) == c1 || z(i) == c2)
            if hk(q, :)*z(i,3:end)' > b
                 class = c1;
            else
                 class = c2;
            end 
            if(z(i) ~= class)   %incorrect
               z(i,2) = z(i,2)*exp(alpha(q));
            else %correct
               z(i,2) = z(i,2)*exp(-alpha(q)); 
            end   
        end
    end
    
    %normalize weights
    z(:,2) = z(:,2)/sum(z(:,2));
    
end
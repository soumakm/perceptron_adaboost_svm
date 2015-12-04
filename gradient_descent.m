% This function performs gradient descent for single sample update
% input: 
%       y: training samples already augmented with 1
%       b: initial weight vector
%     eta: learning rate
% output:
%        a: trained weight vector
%          

function a = gradient_descent(y,b, eta)
a = b;
n= size(y,1);
miss = 1;
while miss == 1
    miss =0;
    for i=1:n
        if a*y(i,:)' <= 0
           a = a + eta*y(i, :); 
           miss =1;
        end
    end    
end    
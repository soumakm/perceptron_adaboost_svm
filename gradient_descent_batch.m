% This function performs gradient descent using batch relaxation
% input: 
%       y: training samples already augmented with 1
%       b: initial weight vector
%     eta: learning rate
% output:
%        a: trained weight vector
%          

function a = gradient_descent_batch(y,b, eta, m)
a = b;
n= size(y,1);
col = size(y,2);
miss = 1;
l = 1;
while miss == 1 && l < 5000
    x = zeros(1,col);
    
    miss =0;
    for i=1:n
        if a*y(i,:)' <= m
           x = x + (m - a*y(i,:)')*y(i,:)/(norm(y(i,:))^2);
           miss =1;
        end
    end 
    l = l + 1;
    a = a + eta*x; 
end    
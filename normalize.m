% This function will normalize the input matrix x
% input :
%       x = input matrix
%       y = normalized output


function y = normalize(x)

rows = size(x,1);
cols = size(x,2);
x1 = zeros(rows, cols);
y = zeros(rows, cols);
m = mean(x);
for i=1:rows
    x1(i,:) = x(i,:) - m;
end
v = var(x1);
for i=1:rows
    for j=1:cols
        y(i,j) = x1(i,j)/sqrt(v(j));  
    end
end
function [ m ] = samplecounter( traindata,nrow,nc)
%number of sample counter for each class
m = zeros(1,nc);
for n=1:nc
    for i=1:nrow
        if traindata(i,1)== n
            m(n) = m(n) + 1;
        end
    end
end

end


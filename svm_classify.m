%SVM train and classifier function
function [err] = svm_classify(xdata,group,uy)
    svmStruct = svmtrain(xdata,group);
    err = 0;
    for i=1:length(uy)
        if uy(i,1)~= svmclassify(svmStruct,uy(i,2:end))
            err = err + 1;
        end
    end
end
clear all;
clc;
% SVM classifier 
% read data, 1st column is the class
ux = dlmread('wine_uci_train.txt');
uy= dlmread('wine_uci_test.txt');
nrow = size(ux,1);
nc = max(ux(:,1));
nrow_ts = size(uy,1);
m = samplecounter(ux,nrow,nc);
ts = samplecounter(uy,nrow_ts,nc);

%% class 1 vs 2
xdata12 = ux(1:(m(1)+m(2)),2:end);
group12 = ux(1:(m(1)+ m(2)),1);
uy12 = uy(1:ts(1)+ts(2),:);

svmStruct = svmtrain(xdata12,group12);

err =  svm_classify(xdata12,group12,uy12);
sprintf('performance class 1 vs 2: %.2f', (1- err/length(uy))*100)
%% class 1 vs 3
xdata13 =[ux(1:m(1),2:end);ux(m(1)+m(2)+1:sum(m),2:end)];
group13 = [ux(1:m(1),1); ux(m(1)+m(2)+1:sum(m),1)];
uy13 = [uy(1:ts(1),:);uy(ts(1)+ts(2)+1:sum(ts),:)];

svmStruct2 = svmtrain(xdata13,group13);

err =  svm_classify(xdata13,group13,uy13);
sprintf('performance class 1 vs 3: %.2f', (1- err/length(uy))*100)

%% class 2 vs 3
xdata23 =ux(m(1)+1:sum(m),2:end);
group23 =ux(m(1)+1:sum(m),1);
uy23 = uy(ts(1)+1:sum(ts),:);

svmStruct3 = svmtrain(xdata23,group23);
err =  svm_classify(xdata23,group23,uy23);
sprintf('performance class 2 vs 3: %.2f', (1- err/length(uy))*100)
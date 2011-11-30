function [OTrain OTest] = pca(Train,Test)
[coeff OTrain] = princomp(Train);
%truncate:
dim = 50;
OTrain = OTrain(:,1:dim);
OTest = Test*coeff(:,1:dim);
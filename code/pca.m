function [OTrain OTest] = pca(Train,Test,opt)
[coeff OTrain] = princomp(Train);
%truncate:
dim = opt.dim;
OTrain = OTrain(:,1:dim);
OTest = Test*coeff(:,1:dim);
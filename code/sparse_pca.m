function [OTrain OTest] = sparse_pca(Train,Test)
dim = 50;
[U S V] = svds(Train,dim);
OTrain = U*S;
OTest = Test*V;
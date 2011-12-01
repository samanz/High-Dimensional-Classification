function [OTrain ODev OTest] = sparse_pca(Train,Dev,Test)
dim = 50;
[U S V] = svds(Train,dim);
OTrain = U*S;
ODev = Dev*V;
OTest = Test*V;
function [OTrain ODev OTest] = sparse_pca(Train,Dev,Test,opt,M)
Vd = M.V(:,1:opt.dim);
OTrain = Train*Vd;
ODev = Dev*Vd;
OTest = Test*Vd;
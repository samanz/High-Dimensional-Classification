function [OTrain ODev OTest] = rand_proj_read(Train,Dev,Test,opt,M)
    k = opt.dim; % Dimentions to project to
  
    OTrain = (sqrt(3)/sqrt(k))*Train*squeeze(M(:,1:k,opt.fold));
    ODev = (sqrt(3)/sqrt(k))*Dev*squeeze(M(:,1:k,opt.fold));
    OTest = (sqrt(3)/sqrt(k))*Test*squeeze(M(:,1:k,opt.fold));
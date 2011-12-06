function [OTrain ODev OTest] = rand_proj_read(Train,Dev,Test,opt,M)
    k = opt.dim; % Dimentions to project to
  
    OTrain = (sqrt(3)/sqrt(k))*Train*M(:,1:k);
    ODev = (sqrt(3)/sqrt(k))*Dev*M(:,1:k);
    OTest = (sqrt(3)/sqrt(k))*Test*M(:,1:k);
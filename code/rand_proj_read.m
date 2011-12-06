function [OTrain ODev OTest] = rand_proj_read(Train,Dev,Test,opt,M)
    k = opt.dim; % Dimensions to project to
    if(k == 1)
        OTrain = (sqrt(3)/sqrt(k))*Train*squeeze(M(opt.fold,:,1:k))';
        ODev = (sqrt(3)/sqrt(k))*Dev*squeeze(M(opt.fold,:,1:k))';
        OTest = (sqrt(3)/sqrt(k))*Test*squeeze(M(opt.fold,:,1:k))';
    else
        OTrain = (sqrt(3)/sqrt(k))*Train*squeeze(M(opt.fold,:,1:k));
        ODev = (sqrt(3)/sqrt(k))*Dev*squeeze(M(opt.fold,:,1:k));
        OTest = (sqrt(3)/sqrt(k))*Test*squeeze(M(opt.fold,:,1:k));
    end

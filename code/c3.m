clear all; close all;

paths = getLocalPaths();
addpath(paths.matlab);
addpath(paths.glmnet);
outdatadir = paths.outdatadir;

[all_lab all] = load_uci(paths.data);
all_lab = all_lab + 1;
%partition into train and test
r = randperm(length(all_lab));
all = all(r,:);
all_lab = all_lab(r);
xTrain = all(1:4000,:);
yTrain = all_lab(1:4000,:);
xTest = all(4001:end,:);
yTest = all_lab(4001:end,:);

numfolds = 10;
noreduce = @deal;
c = cvpartition(all_lab,'k',numfolds);
classifiers =        struct('svm',struct('function',@libsvmWrapper,'reduce',noreduce,'options', '-t 0'), ...
                     'ridge',struct('function',@glmnetWrapper,'reduce',noreduce,'options', struct('family','binomial','alpha',0,'type','')),...
                     'lasso',struct('function',@glmnetWrapper,'reduce',noreduce,'options', struct('family','binomial','alpha',1,'type','')),...
                     'elnet',struct('function',@glmnetWrapper,'reduce',noreduce,'options', struct('family','binomial','alpha',.5,'type','')),...
                     'naivebayes_nosmooth',  struct('function',@naiveBayesWrapper,'reduce',noreduce,'options', 'diagLinear'),...
                     'quad_analysis',struct('function',@matlabclassifierWrapper,'reduce',noreduce,'options', 'diagLinear'));



Techniques = {'naivebayes_nosmooth','svm','ridge','lasso'};
nT = length(Techniques);
rate = zeros(1,nT);
train_set_size = 100;
for Ti = 1:nT
    technique = Techniques{Ti};
    disp(['using technique: ' technique]);
    T = getfield(classifiers,technique);
    classifier = T.function;
    options = T.options;
    reduce = T.reduce;
    out = zeros(1,numfolds);
    for i = 1:numfolds
        out = subsample_and_reduce_and_classify(train_set_size,classifier,reduce,xTrain,yTrain,xTest,yTest,options);
    end

    savefile = [outdatadir technique '.' train_set_size '.mat'];
    save(savefile,'out');
end

nT = length(Techniques);
data = zeros(nT,numfolds);
for Ti = 1:nT
    technique = Techniques{Ti};
    infile = [outdatadir technique '.mat'];
    S = load(infile);
    data(Ti,:) = S.out;
end
boxplot(data');
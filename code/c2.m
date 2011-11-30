clear all; close all;
addpath('/Users/belanger/bin/libsvm/matlab');
addpath('./glmnet_matlab');
outdatadir = './data/';

[all_lab all] = load_uci('../../homework/hw4/matlab/spam/spambase/spambase.data');
all_lab = all_lab + 1;
if(1 ==0)
    r = randperm(length(all_lab));
    r = r(1:100);
    all_lab = all_lab(r);
    all = all(r,:);
end
numfolds = 10;
noreduce = @deal;
c = cvpartition(all_lab,'k',numfolds);
classifiers =        struct('svm',struct('function',@libsvmWrapper,'reduce',noreduce,'options', '-t 0'), ...
                     'ridge',struct('function',@glmnetWrapper,,'reduce',noreduce,'options', struct('family','binomial','alpha',0,'type','')),...
                     'lasso',struct('function',@glmnetWrapper,,'reduce',noreduce,'options', struct('family','binomial','alpha',1,'type','')),...
                     'elnet',struct('function',@glmnetWrapper,,'reduce',noreduce,'options', struct('family','binomial','alpha',.5,'type','')),...
                     'naivebayes_nosmooth',  struct('function',@naiveBayesWrapper,'reduce',noreduce,'options', 'diagLinear'),...
                     'quad_analysis',struct('function',@matlabclassifierWrapper,'reduce',noreduce,'options', 'diagLinear'));



Techniques = {'naivebayes_nosmooth','svm','ridge','lasso'};
nT = length(Techniques);
rate = zeros(1,nT);
for Ti = 1:nT
    technique = Techniques{Ti};
    disp(['using technique: ' technique]);
    T = getfield(classifiers,technique);
    classifier = T.function;
    options = T.options;
    fun = @(xTrain,yTrain,xTest,yTest)classifier(xTrain,yTrain,xTest,yTest,options);
    out= crossval(fun,all,all_lab,'partition',c);
    savefile = [outdatadir technique '.mat'];
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
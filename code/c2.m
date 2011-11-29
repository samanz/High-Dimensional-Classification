clear all; close all;
addpath('/Users/belanger/bin/libsvm/matlab');
addpath('./glmnet_matlab');

[all_lab all] = load_uci('../../homework/hw4/matlab/spam/spambase/spambase.data');
all_lab = all_lab + 1;
all_lab = full(all_lab);

c = cvpartition(all_lab,'k',2);
classifiers =        struct('svm',struct('function',@libsvmWrapper,'options', '-t 0 -c %f'), ...
                     'ridge',struct('function',@glmnetWrapper,'options', struct('family','binomial','alpha',0,'type','')),...
                     'lasso',struct('function',@glmnetWrapper,'options', struct('family','binomial','alpha',1,'type','')),...
                     'naivebayes',struct('function',@glmnetWrapper,'options', struct('family','gaussian','alpha',1,'type','naive')));



nT = 2;
Techniques = {'ridge','lasso'};
rate = zeros(1,nT);
for Ti = 1:nT
    disp(['using technique ' technique]);
    technique = Techniques{Ti};
    disp(['using technique ' technique]);

    T = getfield(classifiers,technique);
    classifier = T.function;
    options = T.options;
    fun = @(xTrain,yTrain,xTest,yTest)classifier(xTrain,yTrain,xTest,yTest,options);
    out = crossval(fun,all,all_lab,'partition',c)
    
    rate(nT) = sum(out)/length(out);
end

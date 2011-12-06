clear All; close All;
runExperiments = 1;
%figure out how to automaticAlly select which lambda value to use in glmnet
paths = getLocalPaths();
addpath(paths.matlab);
addpath(paths.glmnet);
outdatadir = paths.outdatadir;

%[All_lab All] = load_uci(paths.data);
datafile = paths.data;
load(datafile);
All_lab = labels;
if(exist('expanded'))
    All = expanded;
end;
if(exist('data'))
    All = data;
end;
All_lab = All_lab + 1;
%partition into train and test
r = randperm(length(All_lab));
trainsize = 3300;
devsize = 200;
%z = find(sum(All,1) ~= 0);
%All = All(r,z);  
All_lab = All_lab(r);
xTrain = All(1:trainsize,:);
yTrain = All_lab(1:trainsize,:);
xDev = All((trainsize + 1):(trainsize + devsize),:);
yDev = All_lab((trainsize + 1):(trainsize + devsize),:);
xTest = All((trainsize + devsize + 1):end,:);
yTest = All_lab((trainsize + devsize + 1):end,:);

numfolds = 10;
noreduce = @(x,y,z,w)deal(x,y,z);
load_pca_data = 0;
load_pca_data = 0;
load_rand_data = 0;
if(load_pca_data)
    PCA_Matrix = load(paths.PCA);  
end
if(load_rand_data)
    RAND_Matrix = load(paths.RAND);
    disp('successfully read rand matrix');
end

PCA = @(train,dev,test,options)sparse_pca(train,dev,test,options,PCA_Matrix.M);
RP = @(train,dev,test,options)rand_proj_read(train,dev,test,options,RAND_Matrix.R);  

nonnative_indices = size(xTrain,2)/2:size(xTrain,2);

classifiers =        struct('svm',struct('function',@libsvmWrapper,'reduce',noreduce,'options', '-t 0'), ...
                     'ridge',struct('function',@glmnetWrapper,'reduce',noreduce,'options', struct('family','binomial','alpha',0,'type','')),...
                     'lasso',struct('function',@glmnetWrapper,'reduce',noreduce,'options', struct('family','binomial','alpha',1,'type','')),...
                     'lassoSupport',struct('function',@glmnetWrapperFindSupport,'reduce',noreduce,'options', struct('family','binomial','alpha',1,'type','','nonnative_indices',nonnative_indices)),...
                     'elnet',struct('function',@glmnetWrapper,'reduce',noreduce,'options', struct('family','binomial','alpha',.5,'type','')),...
                     'naivebayes_nosmooth',  struct('function',@naiveBayesWrapper,'reduce',noreduce,'options', struct('smooth',0,'dim',50)),...
                     'naivebayes_nosmooth_pca',  struct('function',@naiveBayesWrapper,'reduce',PCA,'options', struct('smooth',0,'dim',50)),...
                     'naivebayes_smooth',  struct('function',@naiveBayesWrapper,'reduce',noreduce,'options', struct('smooth',1,'dim',50)),...
                     'naivebayes_smooth_pca',  struct('function',@naiveBayesWrapper,'reduce',PCA,'options', struct('smooth',1,'dim',50)),...    
                     'lasso_pca',struct('function',@glmnetWrapper,'reduce',PCA,'options', struct('family','binomial','alpha',1,'type','')),...
                     'quad_analysis',struct('function',@matlabclassifierWrapper,'reduce',noreduce,'options', 'diagLinear'),...
                     'naivebayes_smooth_rand',  struct('function',@naiveBayesWrapper,'reduce',RP,'options', struct('smooth',1,'dim',50)));

%Techniques = {'svm','ridge','naivebayes_nosmooth_pca','lasso'};%,'lasso_pca'};

Techniques = {'lassoSupport'};

nT = length(Techniques);
rate = zeros(1,nT);
%train_set_sizes = [trainsize];
train_set_sizes = floor(linspace(1000,trainsize,10));
if(runExperiments == 1)
for train_set_size = train_set_sizes
for Ti = 1:nT
    technique = Techniques{Ti};
    disp(['using technique: ' technique ' with train set size ' int2str(train_set_size)]);
    T = getfield(classifiers,technique);
    classifier = T.function;
    options = T.options;
    reduce = T.reduce;
    out = {}; % = zeros(1,numfolds);
    times = zeros(1,numfolds);
    for i = 1:numfolds
        tic 
        options.fold = i;
        if(strcmp(evaluator_style,'subsample'))
            out(i) = subsample_and_reduce_and_classify(train_set_size,classifier,reduce,xTrain,yTrain,xDev,yDev,xTest,yTest,options);
        elseif(strcmp(evaluator_style,'crossval'))
            out(i) = crossval_and_classify(train_set_size,classifier,reduce,xTrain,yTrain,xDev,yDev,xTest,yTest,options); 
        end
            times(i) = toc;
        disp(['finished iteration ' i]);
    end

    savefile = [outdatadir technique '.' int2str(train_set_size) '.mat'];
    save(savefile,'out');
    savefile = [outdatadir technique '.' int2str(train_set_size) '.times.mat'];
    save(savefile,'times');
end
end
end
nT = length(Techniques);
Accdata = zeros(nT,length(train_set_sizes),numfolds);
Timedata = zeros(nT,length(train_set_sizes),numfolds);

for ii = 1:length(train_set_sizes);
    train_set_size = train_set_sizes(ii);
    for Ti = 1:nT
        technique = Techniques{Ti};
        infile = [outdatadir technique '.' int2str(train_set_size) '.mat'];
        S = load(infile);
        Accdata(Ti,ii,:) = S.out;
        infile = [outdatadir technique '.' int2str(train_set_size) '.times.mat'];
        S = load(infile);
        Timedata(Ti,ii,:) = S.times;
    end
end
colors = {'red','blue','green','cyan','black','yellow','purple','cyan'};
hold on;
for Ti = 1:nT
    %accuracy
    hold on; figure(1);
    E = std(Accdata(Ti,:,:),0,3);
    M = mean(Accdata(Ti,:,:),3);
    errorbar(train_set_sizes,M,E,'color',colors{Ti});
    hold off;
    %time 
    hold on; figure(2);
    E = std(Timedata(Ti,:,:),0,3);
    M = mean(Timedata(Ti,:,:),3);
    errorbar(train_set_sizes,M,E,'color',colors{Ti});
end
hold off;

clear All; close All;
runExperiments = 1 ;
%figure out how to automaticAlly select which lambda value to use in glmnet
paths = getLocalPaths();
addpath(paths.matlab);
addpath(paths.glmnet);
outdatadir = paths.outdatadir;

%[All_lab All] = load_uci(paths.data);
datafile = paths.data;
load(datafile);
All_lab = labels;
All = data;
All_lab = All_lab + 1;
%partition into train and test
r = randperm(length(All_lab));
trainsize = 3300;
devsize = 200;
z = find(sum(All,1) ~= 0);
All = All(r,z);
All_lab = All_lab(r);
xTrain = All(1:trainsize,:);
yTrain = All_lab(1:trainsize,:);
xDev = All((trainsize + 1):(trainsize + devsize),:);
yDev = All_lab((trainsize + 1):(trainsize + devsize),:);
xTest = All((trainsize + devsize + 1):end,:);
yTest = All_lab((trainsize + devsize + 1):end,:);

numfolds = 1;
noreduce = @deal;
load_pca_data = 0;
load_rand_data = 1;
if(load_pca_data)
    PCA_Matrix = load(paths.PCA);  
end
if(load_rand_data)
    RAND_Matrix = load(paths.RAND);
    disp('successfully read rand matrix');
end

PCA = @(train,dev,test,options)sparse_pca(train,dev,test,options,PCA_Matrix.M);
RP = @(train,dev,test,options)rand_proj_read(train,dev,test,options,RAND_Matrix.R);  
nonnative_indices = [];


classifiers =        struct('svm',struct('function',@libsvmWrapper,'reduce',noreduce,'options', '-t 0'), ...
                     'ridge',struct('function',@glmnetWrapper,'reduce',noreduce,'options', struct('family','binomial','alpha',0,'type','')),...
                     'lasso',struct('function',@glmnetWrapper,'reduce',noreduce,'options', struct('family','binomial','alpha',1,'type','')),...
                     'elnet',struct('function',@glmnetWrapper,'reduce',noreduce,'options', struct('family','binomial','alpha',.5,'type','')),...
                     'naivebayes_nosmooth',  struct('function',@naiveBayesWrapper,'reduce',noreduce,'options', struct('smooth',0,'dim',50)),...
                     'naivebayes_nosmooth_pca',  struct('function',@naiveBayesWrapper,'reduce',PCA,'options', struct('smooth',0,'dim',50)),...
                     'naivebayes_smooth',  struct('function',@naiveBayesWrapper,'reduce',noreduce,'options', struct('smooth',1,'dim',50)),...
                     'naivebayes_smooth_pca',  struct('function',@naiveBayesWrapper,'reduce',PCA,'options', struct('smooth',1,'dim',50)),...    
                     'lasso_pca',struct('function',@glmnetWrapper,'reduce',PCA,'options', struct('family','binomial','alpha',1,'type','')),...
                     'quad_analysis',struct('function',@matlabclassifierWrapper,'reduce',noreduce,'options', 'diagLinear'),...
                     'naivebayes_smooth_rand',  struct('function',@naiveBayesWrapper,'reduce',RP,'options',struct('smooth',1,'dim',50)));

Techniques = {'naivebayes_smooth_rand'};%,'lasso_pca'};

nT = length(Techniques);
rate = zeros(1,nT);

train_set_sizes = [trainsize];
dims = 1:10;%round(logspace(log(10),log(size(xTrain,2)),10));
numdims = length(dims);
if(runExperiments == 1)
for train_set_size = train_set_sizes
for Ti = 1:nT
    technique = Techniques{Ti};
    T = getfield(classifiers,technique);
    classifier = T.function;
    options = T.options;
    reduce = T.reduce;
    out = zeros(numdims,numfolds);
    times = zeros(numdims,numfolds);
    for d=1:length(dims);
        options.dim = dims(d);
            disp(['using technique: ' technique ' with train set size ' int2str(train_set_size) ' and dim ' int2str(d)]);
    for i = 1:numfolds
        options.fold = i;
        tic 
        out(d,i) = subsample_and_reduce_and_classify(train_set_size,classifier,reduce,xTrain,yTrain,xDev,yDev,xTest,yTest,options);
        times(d,i) = toc;
        disp(['finished iteration ' int2str(i)]);
    end
    end
    savefile = [outdatadir technique '.' int2str(train_set_size) '.mat'];
    save(savefile,'out');
    savefile = [outdatadir technique '.' int2str(train_set_size) '.refine.times.mat'];
    save(savefile,'times');
end
end
end
data = zeros(numdims,numfolds);
for ii = 1:length(train_set_sizes);
    train_set_size = train_set_sizes(ii);
    for Ti = 1:Ti
        technique = Techniques{Ti};
        infile = [outdatadir technique '.' int2str(train_set_size) '.mat'];
        S = load(infile);
        data(:,:) = S.out;
    end    
end
figure(1);
colors = {'red','blue','green','cyan','black','yellow','purple','cyan'};
hold on;
    E = std(data(:,:),0,2);
    M = mean(data(:,:),2);
    errorbar(dims,M,E,'color','red')
hold off;

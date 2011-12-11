function out = glmnetWrapperFindSupport(xTrain,yTrain,xDev,yDev,xTest,yTest,opt)
native_indices = setdiff(1:size(xTrain,2),opt.nonnative_indices);
xTrainNative = full(xTrain(:,native_indices));
xDevNative = full(xDev(:,native_indices));
xTestNative = full(xTest(:,native_indices));
xTrain = full(xTrain);
xDev = full(xDev);
xTest = full(xTest);
%the family variable specifies which type of response variables we're using
%family = 'gaussian'; % refers to quantitiative y.  
%family = 'binomial'; % refers to 0-1 y
%family = 'multinomial'; % refers to multiclass y
family = opt.family;
%the options object has most of the specifications for the model we're
%estimating

options = glmnetSet;
options.alpha = opt.alpha; %this is the mixing parameter for L1 vs. L2 in the 
%elastic net. 1 refers to Lasso
options.nlambda = 100; %number of lambda values to try
options.standardize = 'true'; % boolean for whether you standardize input data
if(strcmp(opt.type,'naive'))
    options.type = 'naive'; %uncomment this if you want a naive bayes assumption
    %when using 'gaussian' classification family
end

fitNative = glmnet(xTrainNative,yTrain,family,options);
fit = glmnet(xTrain,yTrain,family,options);


%%%options below are for glmnetPredict
%type = 'response'; %gives dot product values
type = 'link'; %gives probability of 1 for binomial models and fitted values 
%for gaussian ones (same as 'link')
%type = 'class'; %returns the classification result for binomial models
%type = 'nonzero'; %returns list of indices for nonzero coeffs for a given
%value of s. need to add a fourth input to glmnetPredict called s. 
%s = []; % give s as a fourth input to glmnetPredict and only have it do
%prediction at lambda values along vector s

%here p(i,j) refers to the prediction for the ith instance at lambda index j
pD = glmnetPredict(fit,'class',xDev); % make predictions
pT = glmnetPredict(fit,'class',xTest); % make predictions
pDNative = glmnetPredict(fitNative,'class',xDevNative); % make predictions
pTNative = glmnetPredict(fitNative,'class',xTestNative); % make predictions
nl = length(fit.lambda);
acc = zeros(1,nl);
for ii = 1:nl
     acc(ii) = mean( yDev == pD(:,ii));
end
[m im] = max(acc);

accNative = zeros(1,nl);
for ii = 1:nl
     accNative(ii) = mean( yDev == pDNative(:,ii));
end
[m imNative] = max(accNative);


v = (fit.beta(:,im) > 0);
vNative = (fitNative.beta(:,imNative) > 0);

out.Nativemean = mean( yTest == pTNative(:,imNative));
out.mean = mean( yTest == pT(:,im));
out.Precision = sum(v(1:length(vNative))&vNative)/sum(v);
out.Recall = sum(v(1:length(vNative))&vNative)/sum(vNative);

out.selected = [];

betaGZ = (fit.beta > 0);
betaGZ = (fitNative.beta > 0);

for k=linspace(10, min(max(sum(betaGZ)),max(sum(betaGZ)))-1, 6)
    ik = round(k);
    results = {};
    kIndex = find(sum(betaGZ) > ik,1);
    kNativeIndex = find(sum(betaGZNative) > ik,1);
    
    v = (fit.beta(:,kIndex) > 0);
    vNative = (fitNative.beta(:,kNativeIndex) > 0);

    results.Nativemean = mean( yTest == pTNative(:,kNativeIndex));
    results.mean = mean( yTest == pT(:,kIndex));
    results.Precision = sum(v(1:length(vNative))&vNative)/sum(v);
    results.Recall = sum(v(1:length(vNative))&vNative)/sum(vNative);
    results.k = ik;
    results.featuresSelected = find( betaGZ(:,kIndex) == 1 );
    results.featuresSelectedNative = find( betaGZNative(:,kNativeIndex) == 1 );

    out.selected = [out.selected results];
end;
k=10;
precision = 1;
while (precision==1)
    ik = round(k);

    betaGZ = (fit.beta > 0);
    betaGZNative = (fitNative.beta > 0);
    kIndex = find(sum(betaGZ) > ik,1);
    kNativeIndex = find(sum(betaGZNative) > ik,1);
    
    v = (fit.beta(:,kIndex) > 0);
    vNative = (fitNative.beta(:,kNativeIndex) > 0);

    precision = sum(v(1:length(vNative))&vNative)/sum(v);
    k
    precision
    k=k+1;
end;
featuresSelected = find( betaGZ(:,kIndex) == 1 );
featuresSelectedNative = find( betaGZNative(:,kNativeIndex) == 1 );
out.firstUnselected = setdiff(featuresSelectedNative,featuresSelected);
out.firstRedundant = setdiff(featuresSelected,featuresSelectedNative);
out.redundantAt = k;
end
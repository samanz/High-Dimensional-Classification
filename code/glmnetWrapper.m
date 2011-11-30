function out = glmnetWrapper(xTrain,yTrain,xTest,yTest,opt)

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

fit = glmnet(xTrain,yTrain,family,options);
options.lambda = fit.lambda;

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
p = glmnetPredict(fit,'class',xTest); % make predictions
nl = length(fit.lambda);
acc = zeros(1,nl);
for ii = 1:nl
     acc = mean( yTest == p(:,ii));
end
out = acc(end);

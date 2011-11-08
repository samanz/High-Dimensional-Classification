clear all;
%load data in sparse format
[all_lab all] = load_sparse('../../data/all.dat');
all_lab = full(all_lab);
all = full(all);
L = length(all_lab);
ntrain = L - 600; 
trainlab = all_lab(1:ntrain);
testlab = all_lab((ntrain+1):end);
train = all(1:ntrain,:);
test = all((ntrain+1):end,:);

%the family variable specifies which type of response variables we're using
family = 'gaussian'; % refers to quantitiative y.  
%family = 'binomial'; % refers to 0-1 y
%family = 'multinomial'; % refers to multiclass y

%the options object has most of the specifications for the model we're
%estimating

options = glmnetSet;
options.alpha = .5; %this is the mixing parameter for L1 vs. L2 in the 
%elastic net. 1 refers to Lasso
options.nlambda = 100; %number of lambda values to try
options.standardize = 'true'; % boolean for whether you standardize input data
%options.type = 'naive'; %uncomment this if you want a naive bayes assumption
%when using 'gaussian' classification family


fit = glmnet(train,trainlab,family,options);
%glmnetPlot(fit); %this makes an awesome figure
%glmnetCoef(fit,0.01); % print coefficients at a single value of lambda

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
p = glmnetPredict(fit,'response',test); % make predictions

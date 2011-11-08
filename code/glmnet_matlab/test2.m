 x=randn(100,20);
     y=randn(100,1);
     g2=randsample(2,100,true);
     g4=randsample(4,100,true);
     fit1=glmnet(x,y);
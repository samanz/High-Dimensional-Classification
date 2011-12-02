function err  = naiveBayesWrapper(xTrain,yTrain,xDev,yDev,xTest,yTest,opt)
    if(opt.smooth)
        err = naiveBayesWrapperNoSmooth(xTrain,yTrain,xDev,yDev,xTest,yTest,opt);
    else
        err = naiveBayesWrapperSmooth(xTrain,yTrain,xDev,yDev,xTest,yTest,opt);
    end
end

function err  = naiveBayesWrapperNoSmooth(xTrain,yTrain,xDev,yDev,xTest,yTest,opt)
 nb = NaiveBayes.fit(xTrain,yTrain); 
 err = mean(yTest == predict(nb,xTest));
end


function err  = naiveBayesWrapperSmooth(xTrain,yTrain,xDev,yDev,xTest,yTest,opt)

 yTrainSmooth = [yTrain; 1; 2];
 Os = ones(1,size(xTrain,2));
 
 sweights = [.01 .1 1 10];
 devAcc = zeros(1,length(sweights));
 for i = 1:length(sweights) 
    sw = sweights(i);
    xTrainSmooth = [xTrain; Os; Os];
    nb = NaiveBayes.fit(xTrainSmooth,yTrainSmooth); 
    devAcc(i) = mean(yDev == predict(nb,xDev));
 end
 [m ii] = max(devAcc);
 sw = sweights(ii);
 xTrainSmooth = [xTrain; sw*Os; sw*Os];
 nb = NaiveBayes.fit(xTrainSmooth,yTrainSmooth); 
 err = mean(yTest == predict(nb,xTest));
end
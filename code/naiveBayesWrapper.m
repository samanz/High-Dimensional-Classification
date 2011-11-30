function err  = naiveBayesWrapper(xTrain,yTrain,xTest,yTest,opt)
 yTrainSmooth = [yTrain; 1; 2];
 Os = ones(1,size(xTrain,2));
 xTrainSmooth = [xTrain; Os; Os];
 nb = NaiveBayes.fit(xTrainSmooth,yTrainSmooth);
 
 err = mean(yTest == predict(nb,xTest));
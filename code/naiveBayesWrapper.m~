function err  = naiveBayesWrapper(xTrain,yTrain,xTest,yTest,opt)
%todo: check to make sure mvmn option is correct.
 
 nb = NaiveBayes.fit(xTrain,yTrain);
 
 err = mean(yTest == predict(nb,xTest));
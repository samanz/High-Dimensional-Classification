function err  = matlabclassifierWrapper(xTrain,yTrain,xTest,yTest,opt)

c = classify(xTest,xTrain,yTrain,opt);
err = mean(c == yTest);
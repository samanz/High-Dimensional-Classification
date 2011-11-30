function out = libsvmWrapper(xTrain,yTrain,xTest,yTest,opt)

[a b c] = svmpredict(yTest,xTest,svmtrain(yTrain,xTrain,opt));

correct = (a == yTest);
out = mean(correct);

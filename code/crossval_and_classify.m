function out = crossval_and_classify(train_set_size,classifier,reduce,xTrain,yTrain,xDev,yDev,xTest,yTest,options)

trainsize = train_set_size;
xAll = [xTrain; xTest];
yAll = [yTrain; yTest];
r = randperm(size(xAll,1));
xAlls = xAll(r,:);
yAlls = yAll(r,:);
xTrain = xAlls(1:trainsize,:);
xTest = xAlls((trainsize + 1):end,:);
yTrain = yAlls(1:trainsize);
yTest = yAlls((trainsize + 1):end);

%reduce dimensionality of sets
[RxTrain RxDev RxTest] = reduce(xTrain,xDev,xTest,options);
out = classifier(RxTrain,yTrain,RxDev,yDev,RxTest,yTest,options);
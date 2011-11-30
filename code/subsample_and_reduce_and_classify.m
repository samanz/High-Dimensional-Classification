function out = subsample_and_reduce_and_classify(train_set_size,classifier,reduce,xTrain,yTrain,xTest,yTest,options)

%subsample train set
r = randperm(length(yTrain));
r = r(1:train_set_size);    
SmallxTrain = xTrain(r,:);
SmallyTrain = yTrain(r,:);

%reduce dimensionality of sets
[RxTrain RxTest] = reduce(SmallxTrain,xTest);
out = classifier(RxTrain,SmallyTrain,RxTest,yTest,options);
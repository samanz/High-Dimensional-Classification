function out  = knnWrapper(xTrain,yTrain,xDev,yDev,xTest,yTest,opt)

    IDX = knnsearch(xTrain,xTest,'K', opt.k, 'Distance', opt.distance);
    a = zeros(length(IDX),1);
    for i=1:length(IDX)
        a(i) = mode(yTrain(IDX(i)));
    end;
    
    correct = (a == yTest);
    out = mean(correct);
end
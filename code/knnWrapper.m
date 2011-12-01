function out  = knnWrapper(xTrain,yTrain,xTest,yTest,opt)

    NS.X = xTrain;
    tic
    IDX = knnsearch(xTrain,xTest,'K', opt.k, 'Distance', opt.distance);
    toc
    a = zeros(length(IDX),1);
    for i=1:length(IDX)
        a(i) = mode(yTrain(IDX(i)));
    end;
    
    correct = (a == yTest);
    out = mean(correct);
end
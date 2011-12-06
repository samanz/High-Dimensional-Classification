    
    d = size(xTrain, 2); % Orginal dimentionality of data
    k = 12000;

    
    R = {};
    for i=1:10
        Rs = spalloc(d,k,.5*d*k);
        R_temp = rand(d, k);

        Rs(find (R_temp < 2/6)) = 1;
        Rs(find (R_temp < 1/6)) = -1;
%    R(find (R_temp >= 2/6)) = 0;
        R{i} = Rs;
    end
    save('../data/RAND.mat','R');
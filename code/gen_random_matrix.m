    
    d = size(xTrain, 2); % Orginal dimentionality of data
    k = 1200;

    R_temp = rand(10,d, k);
    R = zeros(d, k);

    R(find (R_temp < 2/6)) = 1;
    R(find (R_temp < 1/6)) = -1;
    R(find (R_temp >= 2/6)) = 0;
    save('../data/RAND.mat','R');
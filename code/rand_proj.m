function [OTrain ODev OTest] = rand_proj(Train,Dev,Test)
    k = 2000; % Dimentions to project to
    
    d = size(Train, 2); % Orginal dimentionality of data
    
    R_temp = rand(d, k);
    R = zeros(d, k);

    R(find (R_temp < 2/6)) = sqrt(3);
    R(find (R_temp < 1/6)) = -sqrt(3);
    R(find (R_temp >= 2/6)) = 0;
       
    
    OTrain = Train*R;
    ODev = Dev*R;
    OTest = Test*R;
function  [Z Me V ] = myzscore(M)
Ms = sum(M,1);
Ms2 = sum(M.^2,1);
n = size(M,1);
denom = n-1;
Me = Ms/n;
V = (1/(n-1))*sum((M - repmat(Me,n,1)).^2,1);
Z = (M - repmat(Me,n,1))./repmat(sqrt(V),n,1);
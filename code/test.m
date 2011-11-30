A = rand(100,5);
[A1 A1M A1S] = zscore(A);
[A2 A2M A2S] = myzscore(A);
all(A1(:) == A2(:))
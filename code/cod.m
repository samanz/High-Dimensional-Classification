clear all;
close all;
np = 100;
md = 500;
Ds = 1:5:md;
nd = length(Ds);
M = zeros(1,nd);
S = zeros(1,nd);
di = 0;
for d = Ds
    di = di+1;
   pts =  rand(d,np);
    count = 0;
    total = 0;
    lengths = [];
   for i = 1:np
       for j = (i+1):np
           L = norm(pts(:,i)- pts(:,j));
           total = total + L;
           count = count+1;
           lengths = [lengths L];
       end
   end
   %hist(lengths,linspace(0,10,30));
   %pause(.25);
   M(di) = total/count;
   S(di) = std(lengths);
end
plot(Ds,M,'r',Ds,S,'b');
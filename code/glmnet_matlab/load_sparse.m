function [svm_lbl, svm_data] = load_sparse(fname)
row = []; 
col = []; 
value = []; 

fid = fopen(fname); 
line=0; 
svm_lbl=[]; 
svm_data=[]; 
while 1 
tline = fgetl(fid); 
line = line + 1; 
if ~ischar(tline), break, end 
[lbl, data] = strtok(tline); 

svm_lbl(line, 1) = sscanf(lbl, '%g'); 
[c, v] = strread(data, '%d:%f'); 
row = [row; line * ones(length(c), 1)]; 
col = [col; c]; 
value = [value; v]; 
end 

svm_data = sparse(row, col, value); 

fclose(fid); 
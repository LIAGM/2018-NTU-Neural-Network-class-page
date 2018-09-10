%a function use self relaxation algorithm to training
%parameter "data" is a matrix with input and real output
%data = [i1 i2 i3 ; i1 i2 i3 ; o1 o2 o3]
%return single node weight "w"
%w = [w1 w2 w3]

function w = self_relaxation(data)
[dimension, pattern_num] = size(data);

max_constant = 100;
train_para = 0.0001;
w = 2*rand(1, dimension)-1;

real_output = data(dimension, :);
data(dimension, :) = -1;

output = sign(w*data);
output(find(output==0)) = -1;
count = sum(abs(output-real_output))/2;
   
if pattern_num-count < count
   w = -w;
end

a = w;
a(dimension) = 0;
a(dimension) = sum(a.^2);
a(1:dimension-1) = a(1:dimension-1)*w(dimension)/a(dimension);
a(dimension) = 0;
a(dimension) = sum(a.^2);


tempw = w;
error_num = pattern_num;
iter = 1;
const_iter = 0;
while (const_iter <= max_constant) & (error_num ~= 0)
	matrixa = -2*a(1:dimension-1)'*a(1:dimension-1);
   matrixa = matrixa+diag(a(dimension)*ones(1, dimension-1));
   matrixa = matrixa/(a(dimension)^2);
      
   output = bipolar_con_fun(w*data);
   gradient = diag(0.5*(real_output-output).*(ones(size(output))-output.^2))*(matrixa*data(1:dimension-1, :))';
	a(1:dimension-1) = a(1:dimension-1) + train_para*sum(gradient);
   a(dimension) = 0;
   a(dimension) = sum(a.^2);
   
   w = a;
	w = w / a(dimension);

   
  	%pocket algorithm
	output = sign(w*data);
   output(find(output==0)) = -1;
   count = sum(abs(output-real_output))/2;
   
      
   if pattern_num-count < count
      a = -a;
      w = -w;
      count = pattern_num-count;
   end
   
   if count < error_num
      tempw = w;
      error_num = count;
     	const_iter = 0;
   else
     	const_iter = const_iter + 1;
   end
   
   iter = iter + 1;
end
w = tempw;
w(dimension) = -w(dimension);
      
   





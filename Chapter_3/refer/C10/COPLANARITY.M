%a function use the coplanarity of the data to find the weights
%parameter "data" is a matrix with input and real output
%data = [i1 i2 i3 ; i1 i2 i3 ; o1 o2 o3]
%return the weights "w" 
%w = [w1 w2 w3]

function w = coplanarity(data)
[dimension, pattern_num] = size(data);
temp_data = data;
real_output = data(dimension, :);
data(dimension, :) = [];

%find the patterns which class is "1"
data = data(:, find(real_output == 1));
number = size(data, 2);
%randomize
for i = 1 : number
	index1 = fix(number * rand(1)) + 1;
	index2 = fix(number * rand(1)) + 1;
	temp = data(:, index1);
	data(:, index1) = data(:, index2);
   data(:, index2) = temp;
end

vector_num = 1;
A = data(:, 1);
for i = 2 : number
   if rank([A data(:, i)]) == vector_num+1
      A = [A data(:, i)];
      vector_num = vector_num + 1;
   end
   if vector_num == dimension - 1
      break;
   end
end

while vector_num < dimension-1
   new_vector = 2*rand(dimension-1, 1)-1;
   temp = sum(diag(diag(1./(diag(A'*A)*(new_vector'*A))))*A');
   new_vector = new_vector - temp';
   if norm(new_vector) ~= 0
      A = [A new_vector];
      vector_num = vector_num + 1;
   end
end

B = -ones(1, dimension-1);
   
w = B/A;
w = [w 1];

   
      
      
      
      
   





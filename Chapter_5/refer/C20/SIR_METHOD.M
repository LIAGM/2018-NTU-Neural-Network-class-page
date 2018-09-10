%a function use Separable Internal Representation method training
%parameter "data" is a matrix with input and real output
%data = [i1 i2 i3 ; i1 i2 i3 ; o1 o2 o3]
%return single node weight "w"
%w = [w1 w2 w3]

function w = SIR_method(data)
[dimension, pattern_num] = size(data);

%initial weights
w = 2*rand(1, dimension)-1;

max_iter = 100;
train_para = 0.001;
iter = 1;

real_output = data(dimension, :);
data(dimension, :) = 1;

%separate data by class
data1 = [];
data2 = [];   
for i = 1 : pattern_num
  	if real_output(i) == 1
      data1 = [data1 data(:, i)];
  	else
     	data2 = [data2 data(:, i)];
   end
end

pattern_num1 = size(data1, 2);
pattern_num2 = size(data2, 2);


%compute distances matrix D
if pattern_num1 ~= 0 
	output1 = bipolar_con_fun(w*data1);
   temp1 = output1';
   D1 = abs(output1(ones(1, pattern_num1), :) - temp1(:, ones(1, pattern_num1)));
else
   D1 = [];
end

if pattern_num2 ~= 0 
	output2 = bipolar_con_fun(w*data2);
   temp2 = output2';
   D2 = abs(output2(ones(1, pattern_num2), :) - temp2(:, ones(1, pattern_num2)));
else
   D2 = [];
end

if pattern_num1 ~= 0 & pattern_num2 ~= 0
	D12 = abs(output1(ones(1, pattern_num2), :) - temp2(:, ones(1, pattern_num1)));
else
   D12 = [];
end

max_d1 = max(max(D1));
max_d2 = max(max(D2));
min_d = min(min(D12));

while iter <= max_iter 
   temp_w = w;		%save previous weight
   for i = 1 : pattern_num*10
	   index1 = fix(pattern_num * rand(1)) + 1;
		index2 = fix(pattern_num * rand(1)) + 1;
	   y1 = bipolar_con_fun(w * data(:, index1));
   	y2 = bipolar_con_fun(w * data(:, index2));
   
   	if real_output(index1) == real_output(index2)
   		temp_w = temp_w - train_para*0.5*(y1 - y2)*((1-y1^2)*data(:, index1)' - (1-y2^2)*data(:, index2)');
	   else
   	   temp_w = temp_w + train_para*0.5*(y1 - y2)*((1-y1^2)*data(:, index1)' - (1-y2^2)*data(:, index2)');
		end
   end
   
	output1 = [];
   output2 = [];
   
   for i = 1 : pattern_num1
      output1 = [output1 bipolar_con_fun(w*data1(:, i))];
   end
   for i = 1 : pattern_num2
      output2 = [output2 bipolar_con_fun(w*data2(:, i))];
   end

	%compute distances matrix D
	if pattern_num1 ~= 0 
   	temp1 = output1';
	   D1 = abs(output1(ones(1, pattern_num1), :) - temp1(:, ones(1, pattern_num1)));
	else
   	D1 = [];
	end

	if pattern_num2 ~= 0 
   	temp2 = output2';
	   D2 = abs(output2(ones(1, pattern_num2), :) - temp2(:, ones(1, pattern_num2)));
	else
   	D2 = [];
	end

	if pattern_num1 ~= 0 & pattern_num2 ~= 0
		D12 = abs(output1(ones(1, pattern_num2), :) - temp2(:, ones(1, pattern_num1)));
	else
   	D12 = [];
	end
   	
	temp_d1 = max(max(D1));
	temp_d2 = max(max(D2));
	temp_d = min(min(D12));
   
   flag = 0;
   if pattern_num1 ~= 0 & pattern_num2 ~= 0
      if min_d <= temp_d | max_d1 >= temp_d1 | max_d2 >= temp_d2
         flag = 1;
      end
   elseif pattern_num1 ~= 0 
      if max_d1 >= temp_d1
         flag = 1;
      end
   else 
      if max_d2 >= temp_d2
         flag = 1;
      end
   end
   if flag == 0
      iter = 0;  
   	w = temp_w;   
      max_d1 = temp_d1;
		max_d2 = temp_d2;
	   min_d = temp_d;   
   else
      iter = iter + 1;
   end
   
end

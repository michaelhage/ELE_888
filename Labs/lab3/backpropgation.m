function [w, g] = backpropgation(input1, input2, expected)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% set paramters
nu = 0.1;
theta = 0.001;
a = 1;
b = 1;

% set sigmoid functions
f = @(x) a * tanh(b * x);
f_d = @(x) a * b * sech( b * x );

% set max iterations
epoch_max = 1000;

% initialize weights 
% row 1 - bias (input always 1)
% row 2 - first feature (input 1)
% row 3 - second feature (input 2)
% 
% w1 - weight for neuron 1
% w2 - weight for neuron 2
% w3 - weight for determining neuron
w1 = rand(1,3);
w2 = rand(1,3);
wz = rand(1,3);

% output of neural function
output = zeros(1,size(input1,2));

% initialize temporary variables
delta(3) = 0;
delta_w(3,3) = 0;
gradient(1) = 0;

for i = 1:epoch_max
    
    output(:) = 0;
    delta_w(:,:) = 0;
    
    for j = 1:size(input1,2)
        
%         creating input
        input = [1 input1(j) input2(j)];
        
%         computing first two neurons
        net1 = dot(w1 , input);
        net2 = dot(w2 , input);
        
%         calculating sigmoid function
        f1 = f(net1);
        f2 = f(net2);
        
%         computing final neuron
        netz = dot(wz , [1 f1 f2]);
%         netz = dot(wz , [expected(j) f1 f2]);
        
%         calculating sigmoid function
        output(j) = f(netz);
        
%        computing error
        delta(3) = ( expected(j) - output(j) ) * f_d(netz);
        delta(1) = f_d(net1) * wz(2) * delta(3);
        delta(2) = f_d(net2) * wz(3) * delta(3);
        
%         adding to overall error
        delta_w(1,:) = delta_w(1,:) + nu * delta(1) * input;
        delta_w(2,:) = delta_w(2,:) + nu * delta(2) * input;
        delta_w(3,:) = delta_w(3,:) + nu * delta(3) * [1 f1 f2];
%         delta_w(3,:) = delta_w(3,:) + nu * delta(3) * [expected(j) f1 f2];
        
    end
    
    w1 = w1 + delta_w(1,:);
    w2 = w2 + delta_w(2,:);
    wz = wz + delta_w(3,:);
    
    gradient(i) = sum( (expected - output).^2 ) / 2;
%     output_z(i,:) = output(:);
    if(gradient(i) < theta)
%         disp(gradient)
        disp("number of epoches: " + num2str(i))
%         disp(output)
%         size(gradient)
%         display('final weights')
%         display('w1: '  num2str(w1))
%         display('w2: ' + num2str(w2))
%         display('wz: ' + num2str(wz))
        break
    end
end
g = gradient(:);
w = [w1 ; w2 ; wz];

end


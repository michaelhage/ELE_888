function [w, g] = backpropagation(input1, input2, expected)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% inputs to function
% input1 = [-1 -1 1 1];
% input2 = [-1 1 -1 1];
% expected = [-1 1 1 -1];
n = length(input1);


% set paramters
nu = 1;
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
w1 = rand(3,1);
w2 = rand(3,1);
wz = rand(3,1);


for i = 1:epoch_max
%     for j = 1:size(input1,2)
        
%   creating input
    input = [ones(n,1) input1 input2];
%   computing first two neurons
    net1 = input*w1;
    net2 = input*w2;
       
%   calculating sigmoid function
    f1 = f(net1);
    f2 = f(net2);
        
%   computing final neuron
    netz = [ones(n,1) f1 f2]*wz;
    
%   calculating sigmoid function
    output = f(netz);    
%   computing error
    delta_z = ( expected - output ) .* f_d(netz);
    delta_1 = wz(2) * delta_z .* f_d(net1);
    delta_2 = wz(3) * delta_z .* f_d(net2);
        
%   adding to overall error
    delta_w1 = nu * delta_1 .* input;
    delta_w2 = nu * delta_2 .* input;
    delta_wz = nu * delta_z .* [ones(n,1) f1 f2];
%     delta_wz = nu * delta_z .* [expected f1 f2];

    w1 = w1 + sum(delta_w1,1).';
    w2 = w2 + sum(delta_w2,1).';
    wz = wz + sum(delta_wz,1).';
    
    gradient(i) = sum( (expected - output).^2 ) / 2;
    display(gradient(i))
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


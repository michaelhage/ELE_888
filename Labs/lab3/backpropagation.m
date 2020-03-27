function [w, g, y_bar] = backpropagation(input1, input2, expected)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

n = length(input1);

% set paramters
eta = 0.1;
theta = 0.001;
a = 1;
b = 1;

f = @(x) a * tanh(b * x);
f_d = @(x) a *( 1 - b * tanh(b * x).^2 );

% set max iterations
epoch_max = 1000;
gradient = zeros(epoch_max,1);

% initialize weights 
% row 1 - bias (input always 1)
% row 2 - first feature (input 1)
% row 3 - second feature (input 2)
% 
% w1 - weight for neuron 1
% w2 - weight for neuron 2
% w3 - weight for determining neuron
w1 = 2*rand(3,2) -1;
wz = 2*rand(3,1) -1;


for i = 1:epoch_max
        
%   creating input
    X = [ones(n,1) input1 input2];
%   computing first two neurons
    net1 = X*w1;
       
%   calculating sigmoid function
    Z = f(net1);
    Z_bar = [ones(n,1) , Z];
        
%   computing final neuron
    netz = Z_bar*wz;
    
%   calculating sigmoid function which decides the class
    y_bar = f(netz);
    
%   computing error
    ez = (expected-y_bar);
    delta_z = ez .* f_d(netz);
%     delta = wz(2) * delta_z .* f_d(net1);
%     delta_2 = wz(3) * delta_z .* f_d(net2);
     e1 = delta_z .* wz(2,:).';
     delta1 = e1 .* f_d(Z);
     
%   adding to overall error
    w1 = w1 + eta*X.'*delta1;
    wz = wz + eta*Z_bar.'*delta_z;
%     delta_w1 = nu * delta_1 .* input;
%     delta_w2 = nu * delta_2 .* input;
%     delta_wz = nu * delta_z .* [ones(n,1) f1 f2];
%     delta_wz = nu * delta_z .* [expected f1 f2];

%     w1 = w1 + sum(delta_w1,1).';
%     w2 = w2 + sum(delta_w2,1).';
%     wz = wz + sum(delta_wz,1).';
    
    gradient(i) = sum( (expected - y_bar).^2 ) / 2;
   
    if(gradient(i) < theta)
%         disp(gradient)
        disp("number of epoches: " + num2str(i))
        break
    end
end

g = gradient(1:i);
w = [w1 wz];

end


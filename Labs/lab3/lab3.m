%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LAB 3, Multilayer Neural Networks
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% XOR problem
clear
clc
% inputs to function
input1 = [-1 -1 1 1].';
input2 = [-1 1 -1 1].';
expected = [-1 1 1 -1].';
eta = 0.1;

% perform gradient descent
[w, gradient] = backpropgation(input1, input2, expected, eta);

% initialize paramters
a = 1;
b = 1;
f = @(x) a * tanh(b * x);
n = length(input1.');

w1 = w(:,1);
w2 = w(:,2);
wz = w(:,3);

% creating input
input = [ones(n,1) input1 input2];
% computing first two neurons
net1 = input*w1;
net2 = input*w2;

% calculating sigmoid function
f1 = f(net1);
f2 = f(net2);

% computing final neuron
netz = [ones(n,1) f1 f2]*wz;

% calculating sigmoid function which decides the class
output = f(netz);

disp('Expected Output')
[expected output]

% plot the data
figure
scatter(input1, input2, 'b*')
hold on
min_val = -2;
max_val = 2;
% min_val = min([setA(:,2); setB(:,2)]);
% max_val = max([setA(:,2); setB(:,2)]);
r = linspace(min_val, max_val, 50);
d1 = w(1,1) / w(3,1) * r + w(1,1) / w(3,1);
d2 = w(2,2) / w(3,2) * r + w(1,2) / w(3,2);
plot(r, d1)
hold on
plot(r, d2)

% plot gradient function over epoches
figure
plot(gradient)

%% Wine Dataset 
% Attribute Information for WINE data:
%
% 1.  Alcohol
% 2.  Malic acid
% 3.  Ash
% 4.  Alcalinity of ash
% 5.  Magnesium
% 6.  Total phenols
% 7.  Flavanoids
% 8.  Nonflavanoid phenols
% 9.  Proanthocyanins
% 10. Color intensity
% 11. Hue
% 12. OD280/OD315 of diluted wines
% 13. Proline

clc

% read table into array
T = table2array(readtable('wine.csv'));

% dividing table into section
wine = [T((T(:,1)==1),1:3); T((T(:,1)==3),1:3)];

% replace class 3 with -1 value
wine( wine(:,1)==3 ) = -1;

% seperate labels and features
wine_labels = wine(:,1);
wine_features = wine(:,2:3);

wine_features = normalize(wine_features);

labels = unique(wine_labels);

wine = [wine_labels wine_features];

% seperate the two sets
setA = wine(wine_labels(:) == labels(1),:);
setB = wine(wine_labels(:) == labels(2),:);

% setA(:,2:3) = normalize(setA(:,2:3));
% setB(:,2:3) = normalize(setB(:,2:3));

% create training set and testing set split ratios for each set
split_ratio = 0.6;
m1 = floor((size(setA,1)*split_ratio));
m2 = floor((size(setB,1)*split_ratio));

% creating the testinfg and training sets
trainingSet = [setA(1:m1,:) ; setB(1:m2,:)];
testingSet = [setA(m1:end,:) ; setB(m2:end,:)];

% creating training set inputs
input1 = trainingSet(:,2);
input2 = trainingSet(:,3);
expected = trainingSet(:,1);
eta = 0.1;

% computing the back propagation algorithm
% [w, gradient, y_bar] = backpropagation(input1, input2, expected);
[w, gradient] = backpropgation(input1, input2, expected, eta);

% creating testing set inputs
input1 = testingSet(:,2);
input2 = testingSet(:,3);
expected = testingSet(:,1);

% initializing variables
a = 1;
b = 1;
f = @(x) a * tanh(b * x);

n = length(input1);

% creating input
input = [ones(n,1) input1 input2];

% computing first two neurons
net1 = input*w1;
net2 = input*w2;

% calculating sigmoid function
f1 = f(net1);
f2 = f(net2);

% computing final neuron
netz = [ones(n,1) f1 f2]*wz;

% calculating sigmoid function which decides the class
output = f(netz);

disp("this is the result")
result = heaviside(output)

disp('Expected Output')
[expected output]

disp("Error is approximately: " + num2str(gradient( length(gradient) )))

% thresholding output to either -1 or 1
output_t = 2 * heaviside(output) - 1;

disp("Expected Output After Thresholding")
[expected output_t]

disp('Items Misclassified are: ')
misclassified = sum(abs(expected - output_t))/2

% plot the data
figure
scatter(setA(:,2), setA(:,3),'ro')
hold on
scatter(setB(:,2), setB(:,3),'b*')
hold on
% min_val = -5;
% max_val = 5;
min_val = min([setA(:,2); setB(:,2)]);
max_val = max([setA(:,2); setB(:,2)]);
r = linspace(min_val, max_val, 50);
d1 = w(1,1) / w(3,1) * r + w(1,1) / w(3,1);
d2 = w(2,2) / w(3,2) * r + w(1,2) / w(3,2);
plot(r, d1)
hold on
plot(r, d2)

% plot the gradient function
figure
plot(gradient)
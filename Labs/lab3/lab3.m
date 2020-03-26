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

% perform gradient descent
[w, gradient, y_bar] = backpropagation(input1, input2, expected);

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

labels = unique(wine_labels);

% seperate the two sets
setA = wine(wine_labels(:) == labels(1),:);
setB = wine(wine_labels(:) == labels(2),:);

% create training set and testing set split ratios for each set
split_ratio = 0.7;
m1 = floor((size(setA,1)*split_ratio));
m2 = floor((size(setB,1)*split_ratio));

% creating the testinfg and training sets
trainingSet = [setA(1:m1,:) ; setB(1:m2,:)];
testingSet = [setA(m1:end,:) ; setB(m2:end,:)];

% creating training set inputs
input1 = trainingSet(:,2);
input2 = trainingSet(:,3);
expected = trainingSet(:,1);

% computing the back propagation algorithm
[w, gradient, y_bar] = backpropagation(input1, input2, expected);

% % creating testing set inputs
% input1 = testingSet(:,2);
% input2 = testingSet(:,3);
% expected = testingSet(:,1);
% 
% % initializing variables
% a = 1;
% b = 1;
% f = @(x) a * tanh(b * x);
% 
% n = length(input1);
% 
% % creating input
% input = [ones(n,1) input1 input2];
% 
% % computing first two neurons
% net1 = input*w1;
% net2 = input*w2;
% 
% % calculating sigmoid function
% f1 = f(net1);
% f2 = f(net2);
% 
% % computing final neuron
% netz = [ones(n,1) f1 f2]*wz;
% 
% % calculating sigmoid function which decides the class
% output = f(netz);
% 
% disp("this is the result")
% result = output - expected;

% plot the data
figure
scatter(setA(:,2), setA(:,3),'ro')
hold on
scatter(setB(:,2), setB(:,3),'b*')

% plot the gradient function
figure
plot(gradient)
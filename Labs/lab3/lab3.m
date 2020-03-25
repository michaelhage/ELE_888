%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LAB 3, Multilayer Neural Networks
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% XOR problem
clear
% inputs to function
input1 = [-1 -1 1 1];
input2 = [-1 1 -1 1];
expected = [-1 1 1 -1];

% perform gradient descent
[w, gradient] = backpropgation(input1, input2, expected);

% initialize paramters
a = 1;
b = 1;
f = @(x) a * tanh(b * x);

% check 
for i = 1:size(input1,2)
     input = [1 input1(i) input2(i)];
        
%    computing first two neurons
    net1 = dot(w(1,:) , input);
    net2 = dot(w(2,:) , input);

%    calculating sigmoid function
    f1 = f(net1);
    f2 = f(net2);

%    computing final neuron
    netz = dot(w(3,:) , [1 f1 f2]);

%    calculating sigmoid function
    result(i) = f(netz);
end

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

% create training set and testing set
trainingSet = [setA(1:size(setA,1)/2,:) ; setB(1:size(setB,1)/2,:)];
testingSet = [setA(size(setA,1)/2:end,:) ; setB(size(setB,1)/2:end,:)];

input1 = trainingSet(:,2).';
input2 = trainingSet(:,3).';
expected = trainingSet(:,1).';

[w, gradient] = backpropgation(input1, input2, expected);

input1 = testingSet(:,2).';
input2 = testingSet(:,3).';
expected = testingSet(:,1).';

for i = 1:size(input1,2)
     input = [1 input1(i) input2(i)];
        
%    computing first two neurons
    net1 = dot(w(1,:) , input);
    net2 = dot(w(2,:) , input);

%    calculating sigmoid function
    f1 = f(net1);
    f2 = f(net2);

%    computing final neuron
    netz = dot(w(3,:) , [1 f1 f2]);

%    calculating sigmoid function
    result(i) = f(netz);
end

% plot gradient function over epoches
figure
plot(gradient)
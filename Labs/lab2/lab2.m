%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LAB 2, Linear Discriminant Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Attribute Information for IRIS data:
%    1. sepal length in cm
%    2. sepal width in cm
%    3. petal length in cm
%    4. petal width in cm

%    class label/numeric label: 
%       -- Iris Setosa / 1 
%       -- Iris Versicolour / 2
%       -- Iris Virginica / 3


%% this script will run lab1 experiments..
clear
load irisdata.mat

%% extract unique labels (class names)
labels = unique(irisdata_labels);

%% generate numeric labels
numericLabels = zeros(size(irisdata_features,1),1);
for i = 1:size(labels,1)
    numericLabels(find(strcmp(labels{i},irisdata_labels)),:)= i;
end

%% build training data set for two class comparison
% merge feature samples with numeric labels for two class comparison (Iris
% Setosa vs. Iris Veriscolour
trainingSet = [irisdata_features(:,:) numericLabels(:,1) ];
setA = irisdata_features(numericLabels(:)==1,1:2);
setB = irisdata_features(numericLabels(:)==2,1:2);
setC = irisdata_features(numericLabels(:)==3,1:2);

%% Question 1

% Training Data (30% of the data)
n = 0.3;

% Set A
setA_training = setA(1:size(setA,1)*n, 1:2);
setA_testing = setA(size(setA,1)*n + 1:end , 1:2);

% Set B
setB_training = setB(1:size(setB,1)*n, 1:2);
setB_testing = setB(size(setB,1)*n + 1:end , 1:2);

% Criterion Function
a = criterion_function(setA_training, setB_training);

%% Question 2

% Testing funciton
testing_function(setA_testing, setB_testing, a);

% Question 6 - plot function
plot_function(setA_testing, setB_testing, a)

%% Question 3
% Training Data (70% of the data)
n = 0.7;

% Set A
setA_training = setA(1:size(setA,1)*n, 1:2);
setA_testing = setA(size(setA,1)*n + 1:end , 1:2);

% Set B
setB_training = setB(1:size(setB,1)*n, 1:2);
setB_testing = setB(size(setB,1)*n + 1:end , 1:2);

% Criterion Function
a = criterion_function(setA_training, setB_training);

% Testing funciton
testing_function(setA_testing, setB_testing, a);

% Question 6 - plot function
plot_function(setA_testing, setB_testing, a)
%% Question 4

%%% Training Data (30% of the data)
n = 0.3;

% Set A
setC_training = setC(1:size(setC,1)*n, 1:2);
setC_testing = setC(size(setC,1)*n + 1:end , 1:2);

% Set B
setB_training = setB(1:size(setB,1)*n, 1:2);
setB_testing = setB(size(setB,1)*n + 1:end , 1:2);

% Criterion Function
a = criterion_function(setC_training, setB_training);

% Testing funciton
testing_function(setC_testing, setB_testing, a);

% Question 6 - plot function
plot_function(setC_testing, setB_testing, a)

%%% Training Data (70% of the data)
n = 0.7;

% Set A
setC_training = setC(1:size(setC,1)*n, 1:2);
setC_testing = setA(size(setC,1)*n + 1:end , 1:2);

% Set B
setB_training = setB(1:size(setB,1)*n, 1:2);
setB_testing = setB(size(setB,1)*n + 1:end , 1:2);

% Criterion Function
a = criterion_function(setC_training, setB_training);

% Testing funciton
testing_function(setC_testing, setB_testing, a);

% Question 6 - plot function
plot_function(setC_testing, setB_testing, a)

%% Question 5
% Training Data (30% of the data)
n = 0.3;

% Set A
setA_training = setA(1:size(setA,1)*n, 1:2);
setA_testing = setA(size(setA,1)*n + 1:end , 1:2);

% Set B
setB_training = setB(1:size(setB,1)*n, 1:2);
setB_testing = setB(size(setB,1)*n + 1:end , 1:2);

%%% Nu value manipulation

% Original
disp("original")
a = criterion_function(setA_training, setB_training);
testing_function(setA_testing, setB_testing, a);

% Question 6 - plot function
plot_function(setA_testing, setB_testing, a)

% Changing nu values 
nu = 0.1;
display("Nu value: " + num2str(nu))
a = criterion_function(setA_training, setB_training, 0, nu);
testing_function(setA_testing, setB_testing, a);

% Question 6 - plot function
plot_function(setA_testing, setB_testing, a)

nu = 0.001;
display("Nu value: " + num2str(nu))
a = criterion_function(setA_training, setB_training, 0, nu);
testing_function(setA_testing, setB_testing, a);

% Question 6 - plot function
plot_function(setA_testing, setB_testing, a)

%%% Inital weight vector, a, manipulation
a = [10 0 1];
display("Initial weight vector: " + num2str(a))
a = criterion_function(setA_training, setB_training, 0, nu);
testing_function(setA_testing, setB_testing, a);

% Question 6 - plot function
plot_function(setA_testing, setB_testing, a)

a = [0 10 1];
display("Initial weight vector: " + num2str(a))
a = criterion_function(setA_training, setB_training, 0, nu);
testing_function(setA_testing, setB_testing, a);

% Question 6 - plot function
plot_function(setA_testing, setB_testing, a)
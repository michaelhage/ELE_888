function testing_function(set_A, set_B, a)
%TESTING_FUNCTION Summary of this function goes here
%   Detailed explanation goes here
sizeA = size(set_A,1);
sizeB = size(set_B,1);

% Creatin the test array
test = [ones(sizeA+sizeB,1) [set_A; set_B]];

% Creating the labels array
test_labels_A(1:sizeA) = 1;
test_labels_B(1:sizeB) = 2;
test_labels = [test_labels_A test_labels_B];

% get the classification values
% positive feedback = class 1
% negative feedback = class 2
classification = a * test.';

% calculate the accuracy of the model
acc = 0;
for i = 1:size(classification,2)
    if(classification(i) > 0) && (test_labels(i) == 1)
        acc = acc+1;
    elseif(classification(i) < 0) && (test_labels(i) == 2)
        acc = acc+1;
    end
end

% display the accuracy
accuracy = acc / (sizeA+sizeB);
disp("accuracy: " + num2str(accuracy))

end


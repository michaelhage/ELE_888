%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LAB 4, Unsupervised Learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Image Properties
clc
clear

% load image and display
I = imread('house.tiff');
figure 
title("Original Image")
imshow(I);

% reshape image
dim = size(I);
X = reshape(I, dim(1) * dim(2), dim(3));
X = double(X);

% calcultate mean of cluster
mn = mean( X(:,:));

% plot colorspace
figure
plot3(X(:,1), X(:,2), X(:,3),'.', 'Color','b');
hold on;
plot3(mn(1), mn(2), mn(3), '*', 'Color', 'r')
xlabel("Red Channel")
ylabel("Green Channel")
zlabel("Blue Channel")

%% Question A
clc

% initialize number of classes
c = 2;

% use algorithm
[mu, idx, error] = kmeans(X, c, 1);

% select two iterations
n = size(idx,2);
iter = ceil((n-1) * rand(1,2));

% earliest iteration portrayed
figure
plot3(X(idx(:,min(iter)) == 1,1), X(idx(:,min(iter)) == 1,2), X(idx(:,min(iter)) == 1,3),'.', 'Color',mu(:,1)/255);
hold on
plot3(X( idx(:,min(iter)) == 2 ,1), X(idx(:,min(iter)) == 2,2), X(idx(:,min(iter)) == 2,3),'.', 'Color',mu(:,2)/255)
hold on
plot3(mu(1,:), mu(2,:), mu(3,:), '*', 'Color', 'r')
xlabel("Red Channel")
ylabel("Green Channel")
zlabel("Blue Channel")
title("Iteration " + min(iter) + ": 1.2")

% middle iteration portrayed
figure
plot3(X(idx(:,max(iter)) == 1,1), X(idx(:,max(iter)) == 1,2), X(idx(:,max(iter)) == 1,3),'.', 'Color',mu(:,1)/255);
hold on
plot3(X( idx(:,max(iter)) == 2 ,1), X(idx(:,max(iter)) == 2,2), X(idx(:,max(iter)) == 2,3),'.', 'Color',mu(:,2)/255)
hold on
plot3(mu(1,:), mu(2,:), mu(3,:), '*', 'Color', 'r')
xlabel("Red Channel")
ylabel("Green Channel")
zlabel("Blue Channel")
title("Iteration " + max(iter) + ": 1.2")

% final iteration portrayed
figure
plot3(X(idx(:,n) == 1,1), X(idx(:,n) == 1,2), X(idx(:,n) == 1,3),'.', 'Color',mu(:,1)/255);
hold on
plot3(X( idx(:,n) == 2 ,1), X(idx(:,n) == 2,2), X(idx(:,n) == 2,3),'.', 'Color',mu(:,2)/255)
hold on
plot3(mu(1,:), mu(2,:), mu(3,:), '*', 'Color', 'r')
xlabel("Red Channel")
ylabel("Green Channel")
zlabel("Blue Channel")
title("Final Iteration - Iteration " + n + ": 1.3")


% 1.4
% convert to unsigned 8-bit integer
mu = uint8(mu);

% reshape index vector to index matrix
[x,y,z] = size(I);
M = reshape(idx(:,n),x,y);

% create new image
img = uint8(zeros(size(I)));

% map each pixel to their domain counterpart
for i = 1:x
    for j = 1:y
        img(i,j,:) = mu(:,M(i,j)).';
    end
end

figure, imshow(img);

% plot error
figure
plot(error, '.-')
title("Error Across Iterations")
xlabel("Iterations")
ylabel("Error: 1.1")
%% Question B
clc

% use kmeans algorithm
c = 5;
[mu1, idx1, error1] = kmeans(X, c);
[mu2, idx2, error2] = kmeans(X, c);

% plot domain RGB plot
figure
for i = 1:c
    plot3(X(idx1(:) == i,1), X(idx1(:) == i,2), X(idx1(:) == i,3),'.', 'Color',mu1(:,i)/255);
    hold on
end
plot3(mu1(1,:), mu1(2,:), mu1(3,:), '*', 'Color', 'r')
xlabel("Red Channel")
ylabel("Green Channel")
zlabel("Blue Channel")
title("First Run of Algorithm")

figure
for i = 1:c
    plot3(X(idx2(:) == i,1), X(idx2(:) == i,2), X(idx2(:) == i,3),'.', 'Color',mu2(:,i)/255);
    hold on
end
plot3(mu2(1,:), mu2(2,:), mu2(3,:), '*', 'Color', 'r')
xlabel("Red Channel")
ylabel("Green Channel")
zlabel("Blue Channel")
title("Second Run of Algorithm")
hold off

% Question C
% Compute Xie-Beni algorithm
Y1 = xiebeni(idx1, X, mu1, c);
Y2 = xiebeni(idx2, X, mu2, c);
disp("The Xie-Beni Difference for the first cluster is: " + Y1);
disp("The Xie-Beni Difference for the second cluster is: " + Y2);

% cluster 1
% create domain image
mu1 = uint8(mu1);
[x,y,z] = size(I);
M = reshape(idx1(:),x,y);
img = uint8(zeros(size(I)));

% map each pixel to its colour space
for i = 1:x
    for j = 1:y
        img(i,j,:) = mu1(:,M(i,j)).';
    end
end

figure, imshow(img);

% cluster 2
% create domain image
mu2 = uint8(mu2);
[x,y,z] = size(I);
M = reshape(idx2(:),x,y);
img = uint8(zeros(size(I)));

% map each pixel to its colour space
for i = 1:x
    for j = 1:y
        img(i,j,:) = mu2(:,M(i,j)).';
    end
end

figure, imshow(img);

% plot error
figure
plot(error1, '.-')
title("Error Across Iterations")
xlabel("Iterations")
ylabel("Error")

% plot error
figure
plot(error2, '.-')
title("Error Across Iterations")
xlabel("Iterations")
ylabel("Error")

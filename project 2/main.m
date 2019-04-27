%% Load path
clear all
close all
clc
path('../TP2/boosting/', path);

%% Load data

randn('seed', 0); 
nsamples = 200;
[data1, data2, ~] = construct_training_set_2_2(nsamples);

k1 = 3;
k2 = 3;

%% Train models

model1 = em(data1, k1);
model2 = em(data2, k2);

m = min(min(data1, [], 2), min(data2, [], 2));
M = max(max(data1, [], 2), max(data2, [], 2));

[XI, YI] = meshgrid(m(1):0.01:M(1), m(2):0.01:M(2));
oddMap = emOddRatio(model1, model2, [XI(:)'; YI(:)']);
oddMap = reshape(oddMap', size(XI));

figure, contour(XI, YI, oddMap, [1, 1], 'k'), hold on
scatter(data1(1, :), data1(2, :), '.r'), hold on
scatter(data2(1, :), data2(2, :), '.g'), hold on
title 'Classification boundary';

%% Classification error on training data

X = [data1, data2];
Y = [zeros(1, size(data1, 2)), ones(1, size(data1, 2))];

nMissclassify = sum((emOddRatio(model1, model2, X) < 1) ~= Y');

missclassifyRate = nMissclassify / size(X, 2)







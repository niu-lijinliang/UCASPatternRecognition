clc;
clear;
close all;

load('data1');
plot(x1(:,1), x1(:,2), 'r.'); hold on;
plot(x2(:,1), x2(:,2), 'b.');
plot(x3(:,1), x3(:,2), 'k.');
plot(x4(:,1), x4(:,2), 'g.');
plot(x5(:,1), x5(:,2), 'm.');
title('Data Point');


N = size(X, 1);
ITERATIONS = 100;
NUM_CLASS = 5;

mu = [randi([-2, 12]), randi([-8, 8]);
    randi([-2, 12]), randi([-8, 8]);
    randi([-2, 12]), randi([-8, 8]);
    randi([-2, 12]), randi([-8, 8]);
    randi([-2, 12]), randi([-8, 8])];

for i = 1:ITERATIONS  
    distance = zeros(N, NUM_CLASS);
    for j = 1:NUM_CLASS
        tmp = X - ones([N, 1]) * mu(j,:);
        dis = tmp(:, 1).^2 + tmp(:, 2).^2;
        distance(:, j) = dis;
    end
    
    [dis, label] = min(distance, [], 2);
    new_mu = mu;
    for j = 1:NUM_CLASS
        new_mu(j, :) = mean(X(label == j, :));
    end
    
    if all(mu == new_mu)
        break;
    else
        mu = new_mu;
    end
end

figure(2)
plot(X(label == 1,1), X(label == 1,2), 'r.'); hold on;
plot(X(label == 2,1), X(label == 2,2), 'b.');
plot(X(label == 3,1), X(label == 3,2), 'k.');
plot(X(label == 4,1), X(label == 4,2), 'g.');
plot(X(label == 5,1), X(label == 5,2), 'm.');
title('Cluster');
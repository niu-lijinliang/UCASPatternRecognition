clc;
clear;
close all;

X = csvread('data2.csv');
N = size(X, 1);
NUM_CLASS = 2;

iterations = 100;

k = 3;
sigma = 0.5;
W = zeros(N);
U = zeros(size(X));
mu = randn(NUM_CLASS, size(X, 2));  

distance = zeros(N, NUM_CLASS);
for i = 1:N
    tmp = X - ones([N, 1]) * X(i,:);
    dis = tmp(:, 1).^2 + tmp(:, 2).^2;
    dis(i) = 1000;
    for j = 1:k
        [value, index] = min(dis);
        W(i, index) = exp(-value / (2 * sigma^2));
        dis(index) = 1000;
    end
end
W = (W + W') / 2;
D = 5 * eye(N);
L = D - W;
Lsym = inv(D).^0.5 * L * inv(D).^0.5;

[V, D] = eig(Lsym);
D = diag(D);
for i = 1:NUM_CLASS
    [value, index] = min(D);
    U(:, i) = V(:, index);
    D(index) = 1000;
end
T = U ./ repmat(sum(U.^2, 2).^0.5, 1, 2);

for i = 1:iterations
    for j = 1:NUM_CLASS
        tmp = T - ones([N, 1]) * mu(j,:);
        dis = tmp(:, 1).^2 + tmp(:, 2).^2;
        distance(:, j) = dis;
    end
    [dis, label] = min(distance, [], 2);
    new_mu = mu;
    for j = 1:NUM_CLASS
        new_mu(j, :) = mean(T(label == j, :));
    end
    if all(mu == new_mu)
        break;
    else  
        mu = new_mu;
    end
end

acc1 = (sum(label(1:100) == 1) + sum(label(101:end) == 2)) / N;
acc2 = (sum(label(1:100) == 2) + sum(label(101:end) == 1)) / N;
acc = max(acc1, acc2);

figure(1)
plot(X(label == 1,1), X(label == 1,2), 'r.'); hold on;
plot(X(label == 2,1), X(label == 2,2), 'b.');


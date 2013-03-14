samples = importdata('data/sincTrain25.dt');

%% Parameters
alpha = 0.3;    % Learning rate
iter = 5000;   % Number of iterations for steepest descent
mu = 0;
sigma = 1e-2;
n = 10;
m = length(samples);


%% Check of numerical vs analytical gradient
% Initial configuration
W1 = normrnd(mu, sigma, n, 1);
W2 = normrnd(mu, sigma, 1, n);
b1 = normrnd(mu, sigma, n, 1);
b2 = normrnd(mu, sigma);
% Calculate numerical gradient
g1 = numgrads(W1, W2, b1, b2, samples);
% Calculate analytical gradient with backprop
g2 = zeros(3 * n + 1, 1);
for j = 1:m
    xj = samples(j, 1);
    tj = samples(j, 2);
    [a3, z3, a2, z2 ] = ffnet(W1, W2, b1, b2, xj);
    g_ = backprop(xj, z2, a2, W2, z3, a3, tj);

    g2 = g2 + g_;
end
g2 = g2/m * 2;
% The difference between the gradients is:
g2 - g1

[W1, W2, b1, b2, errors, norms] = train(alpha, 2, samples, iter);
[W1_, W2_, b1_, b2_, errors_, norms_] = train(alpha, 20, samples, iter);

%% Plot the results
f = @(xi) ffnet(W1, W2, b1, b2, xi);
f2 = @(xi) ffnet(W1_, W2_, b1_, b2_, xi);
sinc = @(x) sin(x)/x;

x = -10:0.01:10;
y = arrayfun(f, x);
y2 = arrayfun(f2, x);
y3 = arrayfun(sinc, x);
x2 = 1:iter;

figure(1), clf
plot(x, y, 'LineWidth', 2);
hold on
plot(x, y2, 'g', 'LineWidth', 2);
plot(x, y3, 'm', 'LineWidth', 2);
plot(samples(:,1), samples(:,2), 'ro', 'MarkerSize', 4, 'LineWidth', 4);
hold off

figure(2), clf
plot(x2, log(norms));
hold on
plot(x2, log(norms_), 'g');
hold off
title('Gradient norms per iteration');
xlabel('Iterations');
ylabel('log(Gradient norm)');
figure(3), clf
plot(x2, log(errors));
hold on
plot(x2, log(errors_), 'g');
hold off
title('Network error per iteration');
xlabel('Iterations');
ylabel('log(MSE)');
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




% Question 2.1
data = importdata('data/parkinsonsTrainStatML.dt');
labels = data(:,end);
data = data(:,1:end-1);

% Mean and variance of all coordinates
mean = sum(data,1) / size(data,1);
variance = var(data,0,1);

% Affine normalization map (f(x) = (x - mean) / variance)
norm_data = bsxfun(@rdivide, bsxfun(@minus,data, mean), sqrt(variance));
test_mean = sum(norm_data,1) / size(norm_data,1);
test_variance = var(norm_data,0,1);



% Question 2.2
cs = [0.01, 0.1, 1, 5, 10, 100, 1000];
gammas = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1];

indices = crossvalind('Kfold', size(data,1), 5);
cp = classperf(labels);

best_model = [0, 0, 0];
for i = 1:7
    for j = 1:7
        c = cs(i);
        gamma = gammas(j);
        kernel = @(x,z) exp(-gamma*norm(x-z)^2);
        for k = 1:5
            test = (indices == k); train = ~test;
            % Raw data
            m1 = svmtrain(norm_data(train,:),labels(train), ...
                          'boxconstraint', c,  ...
                          'kernel_function', 'rbf', ...
                          'rbf_sigma', sqrt(1/(2*gamma)));
            % Normalized data
            %svmtrain('kernel_function', @kernel);
            prediction = svmclassify(m1, norm_data(test,:));
            cp = classperf(cp, prediction, test);
        end
        if cp.CorrectRate > best_model(1)
            best_model = [cp.CorrectRate, c, gamma];
        end
        [cp.CorrectRate, c, gamma]
    end
end

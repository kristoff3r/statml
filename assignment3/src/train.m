function [ W1, W2, b1, b2, errors, norms ] = train( alpha, n, samples, iter )

%% Parameters
mu = 0;                     % for Initialization gauss
sigma = 1e-2;               % for Initialization gauss
m = length(samples);          % Number of samples


%% Initialization
% We should not initialize to all zeros or the network will not learn
% the right thing.
W1 = normrnd(mu, sigma, n, 1);
W2 = normrnd(mu, sigma, 1, n);
b1 = normrnd(mu, sigma, n, 1);
b2 = normrnd(mu, sigma);


%% Steepest descent optimization of neural network
minnorm = inf;
errors = zeros(iter, 1);
norms = zeros(iter, 1);
% Steepest descent optimization.
for i = 1:iter
    % Accumulate the gradient over this batch.
    g = zeros(3 * n + 1, 1);
    for j = 1:m
        xj = samples(j, 1);
        tj = samples(j, 2);
        [a3, z3, a2, z2 ] = ffnet(W1, W2, b1, b2, xj);
        g_ = backprop(xj, z2, a2, W2, z3, a3, tj);
        
        g = g + g_;
    end
    g = g/m * 2;
    minnorm = min(minnorm, norm(g));
    errors(i) = ffneterror(W1, W2, b1, b2, samples);
    norms(i) = norm(g);
    % Do batch update
    W1 = W1 - alpha * g(1:n);
    W2 = W2 - alpha * g(n+1 : 2*n)';
    b1 = b1 - alpha * g(2*n+1 : 3*n);
    b2 = b2 - alpha * g(3*n+1);
end


end


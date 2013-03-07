train = importdata('data/sincTrain25.dt');

% Parameters
alpha = 2;                  % Learning rate
n = 20;                      % Number of neurons in hidden layer
mu = 0;                     % for Initialization gauss
sigma = 1e-2;               % for Initialization gauss
samples = length(train);    % Number of samples
iter = 1000;

% Initialization
% We should not initialize to all zeros or the network will not learn
% the right thing.
W1 = normrnd(mu, sigma, n, 1);
W2 = normrnd(mu, sigma, 1, n);
b1 = normrnd(mu, sigma, n, 1);
b2 = normrnd(mu, sigma);

for i = 1:iter
    % Accumulate the gradient over this batch.
    gW1 = zeros(n,1);
    gW2 = zeros(1,n);
    gb1 = zeros(n,1);
    gb2 = 0;
    for j = 1:samples
        xj = train(j, 1);
        tj = train(j, 2);
        [a3, z3, a2, z2 ] = ffnet(W1, W2, b1, b2, xj);
        [gW1_, gb1_, gW2_, gb2_] = backprop(xj, z2, a2, W2, z3, a3, tj);
        
        gW1 = gW1 + gW1_;
        gb1 = gb1 + gb1_;
        gW2 = gW2 + gW2_;
        gb2 = gb2 + gb2_;
    end
    
    norm([gW1(:) ; gb1(:) ; gW2(:) ; gb2(:)]);
    % Do batch update
    W1 = W1 - alpha/samples * gW1;
    W2 = W2 - alpha/samples * gW2;
    b1 = b1 - alpha/samples * gb1;
    b2 = b2 - alpha/samples * gb2;
end

f = @(xi) ffnet(W1, W2, b1, b2, xi);

x = -10:0.01:10;
y = arrayfun(f, x);

plot(x, y, 'LineWidth', 2);
hold on
plot(train(:,1), train(:,2), 'ro', 'MarkerSize', 4, 'LineWidth', 4);
hold off
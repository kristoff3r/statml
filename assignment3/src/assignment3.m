% % Question 1
% train = importdata('data/sincTrain25.dt');
% 
% % Parameters
% alpha = 2;                  % Learning rate
% n = 20;                      % Number of neurons in hidden layer
% mu = 0;                     % for Initialization gauss
% sigma = 1e-2;               % for Initialization gauss
% samples = length(train);    % Number of samples
% iter = 1000;
% 
% % Initialization
% % We should not initialize to all zeros or the network will not learn
% % the right thing.
% W1 = normrnd(mu, sigma, n, 1);
% W2 = normrnd(mu, sigma, 1, n);
% b1 = normrnd(mu, sigma, n, 1);
% b2 = normrnd(mu, sigma);
% 
% for i = 1:iter
%     % Accumulate the gradient over this batch.
%     gW1 = zeros(n,1);
%     gW2 = zeros(1,n);
%     gb1 = zeros(n,1);
%     gb2 = 0;
%     for j = 1:samples
%         xj = train(j, 1);
%         tj = train(j, 2);
%         [a3, z3, a2, z2 ] = ffnet(W1, W2, b1, b2, xj);
%         [gW1_, gb1_, gW2_, gb2_] = backprop(xj, z2, a2, W2, z3, a3, tj);
%         
%         gW1 = gW1 + gW1_;
%         gb1 = gb1 + gb1_;
%         gW2 = gW2 + gW2_;
%         gb2 = gb2 + gb2_;
%     end
%     
%     norm([gW1(:) ; gb1(:) ; gW2(:) ; gb2(:)]);
%     % Do batch update
%     W1 = W1 - alpha/samples * gW1;
%     W2 = W2 - alpha/samples * gW2;
%     b1 = b1 - alpha/samples * gb1;
%     b2 = b2 - alpha/samples * gb2;
% end
% 
% f = @(xi) ffnet(W1, W2, b1, b2, xi);
% 
% x = -10:0.01:10;
% y = arrayfun(f, x);
% 
% plot(x, y, 'LineWidth', 2);
% hold on
% plot(train(:,1), train(:,2), 'ro', 'MarkerSize', 4, 'LineWidth', 4);
% hold off
% 

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
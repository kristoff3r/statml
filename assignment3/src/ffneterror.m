function [ e ] = ffneterror( W1, W2, b1, b2, xy)

f = @(x) ffnet(W1, W2, b1, b2, x);
estimates = arrayfun(f, xy(:, 1));
difs = estimates - xy(:,2);

e = 1/length(xy) * sum(difs.^2);




end


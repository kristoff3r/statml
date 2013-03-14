function [ g ] = numgrads( W1, W2, b1, b2, xy )

eps = 1e-6;
A = eye(length(W1));

% Get error of initial setting
e = ffneterror(W1, W2, b1, b2, xy);

gW1 = zeros(size(W1));
gW2 = zeros(size(W2));
gb1 = zeros(size(b1));

for i = 1 : length(W1)
    gW1(i) = (ffneterror(W1 + eps * A(:,i), W2, b1, b2, xy) - e)/eps;
end
for i = 1 : length(W2)
    gW2(i) = (ffneterror(W1, W2 + eps * A(i,:), b1, b2, xy) - e)/eps;
end
for i = 1 : length(b1)
    gb1(i) = (ffneterror(W1, W2, b1 + eps * A(:,i), b2, xy) - e)/eps;
end
gb2 = (ffneterror(W1, W2, b1, b2 + eps, xy) - e)/eps;

g = [gW1(:) ; gW2(:) ; gb1(:) ; gb2(:)];

end


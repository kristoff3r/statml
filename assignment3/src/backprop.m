function [ g ] = backprop( a1, z2, a2, W2, z3, a3, t )

h_ = @(x) 1/(1 + abs(x))^2;

delta3 = -(t - a3) .* arrayfun(h_, z3);
delta2 = (W2' * delta3) .* arrayfun(h_, z2);

gW1 = delta2 * a1';
gW2 = delta3 * a2';
gb1 = delta2;
gb2 = delta3;

g = [gW1(:) ; gW2(:) ; gb1(:) ; gb2(:)];

end


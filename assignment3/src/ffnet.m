function [ a3, z3, a2, z2 ] = ffnet( W1, W2, b1, b2, x)

h = @(x) x/(1 + abs(x));

z2 = W1 * x + b1;
a2 = arrayfun(h, z2);
z3 = W2 * a2 + b2;
a3 = arrayfun(h, z3);


end


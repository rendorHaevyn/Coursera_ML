function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

% h(theta)(x) = g(z) = 1 / (1 + e^-z) = 1 / (1 + e^-theta'X)
g = 1 ./ (1 + exp(-z)); % Period for element-wise multiplication


% =============================================================

end

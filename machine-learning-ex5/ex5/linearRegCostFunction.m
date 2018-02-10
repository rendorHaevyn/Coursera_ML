function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% ==  Minimise for J ==
J = sum(((X * theta) - y).^2);
J = J / (2 * m);

% Add regularization
Jreg = sum(theta(2:end) .^2);
Jreg = Jreg * (lambda / (2 * m));
J += Jreg;

% == Calculate gradient ==
D = sum(((X * theta) - y) .* X); % 1 X 2 (# features)
D = D / m;

% Add regularization (but not for bias term, theta 0)
r_vec = ones(length(theta),1); % 2 X 1
r_vec(1) = 0;
Dreg = (r_vec .* theta) * (lambda / m);
D += Dreg';
grad = D; % 1 X 2;

% =========================================================================

grad = grad(:);

end

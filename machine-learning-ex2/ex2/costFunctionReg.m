function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%vectorised cost function: 1/m·(-y'log(h)-(1-y)'log(1-h)) +  lambda/2m * sum{j=1:n}theta^2
ht = sigmoid((X * theta)); % a vector of m * 1
J = ( 1/m * ( -y' * log(ht) - (1-y)' * log(1 - ht)) ) + ...
    ( lambda/(2*m) * sum(theta(2:end,:) .^ 2) ); % constant term theta is not penalised (regularized)

%vectorised grad desc: ?:=?-a/m.X'(g(X?)-y(hat)), gradient:= 1/m.X'(g(X?)-y(hat))
gvals = zeros(size(X,2),1); % temp holder for **simultaneous** partial derivatives
for i = 1:size(X,2)
  if i == 1 % non-regularised constant
    gvals(i) = 1/m *  X(:,i)' * (ht - y);
  else % add lambda/m * theta for non-contant regularization
    gvals(i) = 1/m *  X(:,i)' * (ht - y) + ( lambda/m * theta(i) );
  end  
end
grad = gvals;



% =============================================================

end

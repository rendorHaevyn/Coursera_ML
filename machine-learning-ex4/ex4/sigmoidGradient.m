function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. This should work regardless if z is a matrix or a
%   vector. In particular, if z is a vector or matrix, you should return
%   the gradient for each element.

g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the gradient of the sigmoid function evaluated at
%               each value of z (z can be a matrix, vector or scalar).

g = sigmoid(z) .* (1 - sigmoid(z));

%gt = z(:); % rollout and create row vector
%for i = 1:numel(gt)
%  z_temp = gt(i);
%  gt(i) = sigmoid(z_temp) * (1 - sigmoid(z_temp));  
%end

%% unroll sigmoid gradient for each z
%g = reshape(gt,size(z));



% =============================================================




end

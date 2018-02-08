function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
holder = zeros(length(theta),1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    % update simultaneously - send updated coefficients to holding vector until updated
    for icf = 1:length(theta) % icf:= # theta coefficients == # features of X
      holder(icf) = theta(icf) - alpha * m^-1 * sum((X*theta - y) .* X(:,icf));  % cost function derivative   
    end
    theta = holder;    
  



    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

    % Every 50 iterations, print theta and J cost function
    if and(!mod(iter,100), 0) % set second param to 1 to print rolling results
      fprintf('iter %.0f of %.0f: J = %0.f, theta = [%.0f %.0f %.0f]\n', [iter num_iters J_history(iter) theta(:,1)']');
%    disp(theta)
    end  
    
end

end

function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Add ones to the X data matrix for bias node (input layer a1)
X = [ones(m, 1) X];

pred_a2 = sigmoid(X * Theta1'); % array with X rows and all_theta cols with preds for each class label in row

% Add ones to the pred_a2 data matrix for the bias node (input layer a2)
pred_a2 = [ones(m,1) pred_a2];

pred_a3 = sigmoid(pred_a2 * Theta2');

[x ix] = max(pred_a3, [], 2); % find max prediction along label cols (DIM=2) and return value (x) and index (ix)
p = ix;



% =========================================================================


end

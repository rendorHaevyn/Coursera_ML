%% Machine Learning Online Class
%  Exercise 1: Linear regression with multiple variables
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear regression exercise. 
%
%  You will need to complete the following functions in this 
%  exericse:
%
%     warmUpExercise.m
%     plotData.m
%     gradientDescent.m
%     computeCost.m
%     gradientDescentMulti.m
%     computeCostMulti.m
%     featureNormalize.m
%     normalEqn.m
%
%  For this part of the exercise, you will need to change some
%  parts of the code below for various experiments (e.g., changing
%  learning rates).
%

%% Initialization

%% ================ Part 1: Feature Normalization ================

%%%%%%%%%%  TO DO:  %%%%%%%%%%
% make dynamic to number of features and other feature transforms ie polynomial, log, cube


%% Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');

%% Load Data
data = load('ex1data2_Dynamic.txt');
% Add two columns...
A1 = log(data(:,1)); % ... a log of the house square feet (col 1) 
A2 = data(:,2) .^3;  % ... and a cube of the # BRs (col 2)
X = data(:,1:size(data,2)-1);
X = [X(:,1) A1 X(:,2) A2 X(:,3:end)];

y = data(:,size(data,2));
assert(length(X) == length(y)) % assert that X and Y vector length is equivalent

global m = length(y); % number of training examples (training set size)
global n = size(X,2); % number of features

% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f %.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

fprintf('Program paused. Press enter to continue.\n');
pause;

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X mu sigma] = _featureNormalizeDynamic(X, n);

% Print post-normalised data
fprintf('First 10 normalised examples from the dataset: \n');
fprintf(' x = [%.0f %.0f %.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

fprintf('Program paused. Press enter to continue.\n');
pause;

% Add intercept term to X
X = [ones(m, 1), X];


%% ================ Part 2: Gradient Descent ================

% ====================== YOUR CODE HERE ======================
% Instructions: We have provided you with the following starter
%               code that runs gradient descent with a particular
%               learning rate (alpha). 
%
%               Your task is to first make sure that your functions - 
%               computeCost and gradientDescent already work with 
%               this starter code and support multiple variables.
%
%               After that, try running gradient descent with 
%               different values of alpha and see which one gives
%               you the best result.
%
%               Finally, you should complete the code at the end
%               to predict the price of a 1650 sq-ft, 3 br house.
%
% Hint: By using the 'hold on' command, you can plot multiple
%       graphs on the same figure.
%
% Hint: At prediction, make sure you do the same feature normalization.
%

fprintf('Running gradient descent ...\n');

% Choose some alpha values and stroke/color combinations for plotting
aX = [0.003, 0.01, 0.03, 0.10];
num_iters = 500;
figure(1);
clf;
cc = hsv(12);  % colormap
for idx = 1:size(aX,2)
  alpha = aX(1,idx)
  % Init Theta and Run Gradient Descent 
  theta = zeros(n+1, 1); % n features + 1 for the intercept
  [theta, J_history] = _gradientDescentDynamic(X, y, theta, alpha, num_iters, m);

  % Plot the convergence graph
  figure (1);
  hold on;
  plot(1:numel(J_history), J_history, 'Color', cc(idx,:), 'LineWidth', 2);  % numel/length/size
  xlabel('Number of iterations');
  ylabel('Cost J');
    
  
  % Display gradient descent's result
  fprintf('Theta computed from gradient descent: \n');
  fprintf(' %f \n', theta);
  fprintf('\n');

  % Estimate the price of a 1650 sq-ft, 3 br house
  % ====================== YOUR CODE HERE ======================
  % Recall that the first column of X is all-ones. Thus, it does
  % not need to be normalized.

  Vx = [1650 log(1650) 3 3^3]; % test data
  Vx = (Vx - mu) ./ sigma; % normalise test data
  Vx = [ones(1) Vx]; % add coefficeint 1 for constant

  pred_price = Vx * theta; % predict using final theta


  % ============================================================

  fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
           '(using gradient descent):\n $%f\n'], pred_price);

end  
legend ('\alpha: 0.003', '\alpha: 0.01', '\alpha: 0.03', '\alpha: 0.1');
title ('Cost function descent (J\theta) by learning rate (\alpha)');

fprintf('Program paused. Press enter to continue.\n');
pause;
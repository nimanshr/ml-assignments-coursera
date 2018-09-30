function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

values = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0];
[V, U] = meshgrid(values, values);
combis = reshape(cat(2, V, U), [], 2);

n_combis = size(combis, 1);
errors = zeros(n_combis, 1);

for i_row=1:n_combis
  C_dummy = combis(i_row, 1);
  sigma_dummy = combis(i_row, 2);
  
  model = svmTrain(X, y, C_dummy, @(x1, x2) gaussianKernel(x1, x2, sigma_dummy));
  
  predictions = svmPredict(model, Xval);
  error = mean(double(predictions ~= yval));
  errors(i_row, 1) = error;
  
  fprintf('Done with test %02d from %d\n', i_row, n_combis);
  fprintf('[C, sigma] = [%6.2f, %6.2f]; prediction error: %8.4f\n', C_dummy, sigma_dummy, error);
end

i_row_best = find(errors == min(errors));
error_min = errors(i_row_best, 1);
C = combis(i_row_best, 1);
sigma = combis(i_row_best, 2);

% =========================================================================

end

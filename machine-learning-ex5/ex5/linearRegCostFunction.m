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
[m,n] = size(X);
h = X*theta;
theta_tem = [[0]; theta([2:length(theta)])];
J = 1/(2*m)*(sum((h-y).^2))+(lambda/(2*m))*sum(theta_tem.^2);
L = (h-y);
K = (lambda/m)*theta_tem';
grad(1) = (1/m)*sum(L.*X(:,1));
for ii = 2:n
grad(ii) = (1/m)*sum(L.*X(:,ii))+ K(ii);
end


% =========================================================================

grad = grad(:);

end

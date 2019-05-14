function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
gradient = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h=sigmoid(X*theta);

costJ = (-1/m) * sum( y .* log(h) + (1 - y) .* log(1 - h) );
costRegularizationTerm = lambda/(2*m) * sum( theta(2:end).^2 );
J = costJ + costRegularizationTerm;

for i = 1:m
		gradient = gradient + ( h(i) - y(i) ) * X(i, :)';
end
regularizationTerm = lambda/m * [0; theta(2:end)];
grad = (1/m) * gradient + regularizationTerm; 

%probOneTerm=((y')*log(h));
%probZeroTerm=((1-y)')*log(1-h);
%J = ((1/m)*(-probOneTerm-probZeroTerm))+(lambda/2*m)*theta(2:end).^2;
%grad = (((1/m)*X')*((sigmoid(X*theta))-y))+[0; theta(2:end)];%(lambda/m)*theta;




% =============================================================

end

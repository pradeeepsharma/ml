function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%https://www.coursera.org/learn/machine-learning/programming/AiHgN/neural-network-learning/discussions/threads/QFnrpQckEeWv5yIAC00Eog
%Size(y_matrix):5000*10
y_matrix=eye(num_labels)(y,:);

%Size(a1):5000*401
a1 = [ones(size(X,1),1) X];

%Size(Z1):5000*25
Z1 = a1*Theta1';

%Size(a):5000*25
a2=sigmoid(Z1);

%Size(a2WithBias):5000*26
a2WithBias = [ones(size(a2,1),1) a2];

%Size(Z2):5000*10
Z2 = a2WithBias*Theta2';

%Size(a3):5000*10
a3=sigmoid(Z2);


probOneTerm=(y_matrix.*log(a3));
probZeroTerm=(1-y_matrix).*log(1-a3);


J = -(1/m)*sum(sum(probOneTerm+probZeroTerm))+ (lambda/(2*m)) *(sum(sum(Theta1(:,2:end).^2))'+sum(sum(Theta2(:,2:end).^2)));

%Size(D3):5000*10
D3 = a3-y_matrix;

tempDel2 = (D3*Theta2);
tempDel2=tempDel2(:,2:end);

%Size(D2)=5000*25
D2=tempDel2.*sigmoidGradient(Z1);

Delta1=0;
%Size(Delta1) :25*401
Delta1=Delta1+((D2')*a1);

Delta2=0;
%Size(Delta2):
Delta2=Delta2+((D3') *a2WithBias);

Theta1_grad = (1/m)* Delta1;
Theta2_grad = (1/m)*Delta2;
















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

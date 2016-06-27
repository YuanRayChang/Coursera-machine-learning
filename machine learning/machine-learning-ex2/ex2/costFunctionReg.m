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
J_back=0;
for i=1:m
    h=sigmoid(X(i,:)*theta);
    temp=-y(i)*log(h)-(1-y(i))*log(1-h);
    J=J+temp;
end
J=J/m;
for i=2:size(theta)
    temp3=theta(i).^2;
    J_back=J_back+temp3;
end
J_back=J_back*lambda/2/m;
J=J+J_back;

g=0;
for i=1:m
    h=sigmoid(X(i,:)*theta);
    temp2=(h-y(i))*X(i,1);
    g=g+temp2;
end
grad(1)=g/m;

for j=2:size(theta)
    g=0;
for i=1:m
    h=sigmoid(X(i,:)*theta);
    temp2=(h-y(i))*X(i,j);
    g=g+temp2;
end
grad(j)=g/m+lambda/m*theta(j);

end




% =============================================================

end

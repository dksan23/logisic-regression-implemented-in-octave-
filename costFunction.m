function [J, grad] = costFunction(theta, X, y)


% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));


h=sigmoid((theta'*X')');
J=1/m*(sum(-y.*log(h)-(1-y).*log(1-h)));
grad=1/m*(sum((h-y).*X));







% =============================================================

end

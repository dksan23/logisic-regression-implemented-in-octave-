function p = predict(theta, X)


m = size(X, 1); % Number of training examples

% You need to return the following variables correctly
p = zeros(m, 1);


p=(sigmoid((theta'*X')')>=0.5);





% =========================================================================


end

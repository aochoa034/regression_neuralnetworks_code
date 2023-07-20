%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% Images interpolation %%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [coefficients,images] = regression(hidden_layer_neurons,x_training,training_set,sigma,eta,maxiter)
% [coefficients, images] = regression(hidden_layer_neurons,x_training,training_set,sigma,eta,maxiter) interpolates the images
% produced by a neural network with a polynomial. These images are obtained freom the points  of the training set.
% sigma is the activation function utilised in the layers of the network.
% eta and maxiter are parameters of the Gradient Descent, learning rate and iterations maximum, respectively.

m1 = length(training_set);

%  initialize the column vector of images
images = zeros(1,m1);

% Compute the network images
for i = 1 : m1
    images(i) =  NeuralNetwork(hidden_layer_neurons,x_training(i),training_set(i),m1+1,sigma,eta,maxiter);
end
% Get the polynomial to interpolate all the sample
X = zeros(m1,hidden_layer_neurons-1);
for i= 0 : hidden_layer_neurons-1
    X(:,i+1) = x_training.^i;
end
coefficients = flip(X\images');
end
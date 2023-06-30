function [coefficients,images] = regression(hidden_layer_neurons,x_training,training_set,sigma,eta,maxiter)
m1 = length(training_set);
images = zeros(1,m1);
for i = 1 : m1
    images(i) =  NeuralNetwork(hidden_layer_neurons,x_training(i),training_set(i),m1+1,sigma,eta,maxiter);
end
X = zeros(m1,hidden_layer_neurons-1);
for i= 0 : hidden_layer_neurons-1
    X(:,i+1) = x_training.^i;
end
coefficients = flip(X\images');
end
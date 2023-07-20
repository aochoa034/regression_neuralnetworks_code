function [coefficients, net, tr] = matlab_NN(neurons,x_training,training_set,sigma,maxiter)
    % [coefficients, net, tr] = matlab_NN(neurons,x_training,training_set,sigma,maxiter) returns a polynomial
    % interpolating the images produced by Matlab's neural network command.
    % We give the number of neurons of the hidden layer of the network, a training set, an activation function sigma, and 
    % a number of iterations to perform.

    % Define the network
    net = fitnet(neurons); % neural network of the indicated neuron number

    % We choose the training set to be all the sample, this way, instead of 
    % making a different random sample for each number of neurons, we work with
    % the same sample in all cases
    net.divideParam.trainRatio = 1;              % set proportion of data for training
    net.divideParam.valRatio = 0.0;                % set proportion of data for validation
    net.divideParam.testRatio = 0.0;               % set proportion of data for test
    
    % Choose the activation function

    % fitnet has a predefined activation function for the layers, howerver, we will work with the 
    % logistic function, so we have to change that parameter
    if sigma == "logistic"
        net.layers{1}.transferFcn = "logsig";
    end

    % Set another number of iterations if the default is no enough
    if nargin == 5
        net.trainParam.epochs = maxiter;   
    end

    % Select the loss function we are going to use, if we haven't selected one, fitnet has a default one.
    net.performFcn = "mse";

    % Train the network
    [net,tr] = train(net,x_training,training_set);

    % compute the images of the training set
    images = net(x_training);

    % Interpolate the images of the network
    X = zeros(length(training_set),neurons-1);
    for i= 0 : neurons-1
        X(:,i+1) = x_training.^i;
    end
    coefficients = flip(X\images');
end
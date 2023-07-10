function [coefficients, net, tr] = matlab_NN(neurons,x_training,training_set,sigma,maxiter)
    % Define the network
    net = fitnet(neurons);

    % We choose the training set to be all the sample, thuis way, instead of 
    % making a different random sample for each number of neurons, we work with
    % the same sample in all cases
    net.divideParam.trainRatio = 1;              % set proportion of data for training
    net.divideParam.valRatio = 0.0;                % set proportion of data for validation
    net.divideParam.testRatio = 0.0;               % set proportion of data for test
    
    % Choose the activation function
    if sigma == "logistic"
        net.layers{1}.transferFcn = "logsig";
    end

    % Set another number of iterations if the default is no enough
    if nargin == 5
        net.trainParam.epochs = maxiter;   
    end

    % Select the loss function
    net.performFcn = "mse";

    % Train the network
    [net,tr] = train(net,x_training,training_set);
    images = net(x_training);

    % Interpolate the images of the network
    X = zeros(length(training_set),neurons-1);
    for i= 0 : neurons-1
        X(:,i+1) = x_training.^i;
    end
    coefficients = flip(X\images');
end
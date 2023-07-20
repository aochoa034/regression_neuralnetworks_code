%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% Neural Network %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function G = NeuralNetwork(hidden_layer_neurons,x,y,set_size,sigma,eta,maxiter)

    % G = NeuralNetwork(hidden_layer_neurons,x,y,set_size,sigma,eta,maxiter) computes the neural network with an input point
    % x and a image y from a training set. 
    % set_size refers to the size of the training set. sigma is the activation function utilised in the layers of the network.
    % eta and maxiter are parameters of the Gradient Descent, learning rate and iterations maximum, respectively.

    % We randomly generate the weights of the neuron connections; we will take null biases
    w = randn(hidden_layer_neurons,1);
    alpha = randn(hidden_layer_neurons,1);
    theta = randn(hidden_layer_neurons,1);
    % We compute the feedworward of the network, product of the first weight ---> apply the logistic function ---> multiply the 
    % result with the second weight
    if sigma == "relu"
        % The network will conpute a first sum 
        G = sum(alpha'*relu(w.*x + theta));
        % We now need to minimize the weights values in order to get the minimum risk between our data points and the ouptut of the network
        % To do so, we implement the Gradient Descent method
        for iterations = 1:maxiter
            % The partial derivatives of alpha, the second weights and the bias
            for j = 1 :hidden_layer_neurons
                    alpha(j) = alpha(j) - eta * (2/set_size) * (G - y) * relu(w(j)*x + theta(j));
                    w(j) = w(j) - eta * (2/set_size) * (G - y) * alpha(j) *Drelu(w(j)*x + theta(j))*x;
                    theta(j) = theta(j) - eta * (2/set_size) * (G - y) * alpha(j) *Drelu(w(j)*x + theta(j));
             end 
            % We update the output of the network with the new weights 
            G = sum(alpha'*relu(w.*x + theta));
        end
    else   % we will consider the logistic function by default
        G = sum(alpha'*logistic(w.*x + theta));
        % We now need to minimize the weights values in order to get the minimum risk between our data points and the ouptut of the network
        % To do so, we implement the Gradient Descent method
        for iterations = 1:maxiter
            % The partial derivatives of alpha, the second weights and the bias
            % The partial derivatives of w, the first weights 
             for j = 1 :hidden_layer_neurons
                    alpha(j) = alpha(j) - eta * (2/set_size) * (G - y) * logistic(w(j)*x + theta(j));
                    w(j) = w(j) - eta * (2/set_size) * (G - y) * alpha(j) *Dlogistic(w(j)*x + theta(j))*x;
                    theta(j) = theta(j) - eta * (2/set_size) * (G - y) * alpha(j) *Dlogistic(w(j)*x + theta(j));
             end
            % We update the output of the network with the new weights 
            G = sum(alpha'*logistic(w.*x + theta));
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% Activation function %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Logistic activation function
function y = logistic(x)
    y = 1 ./ (1 + exp(-x));
end
%  Derivative of the logistic function
function y = Dlogistic(x)
    y = logistic(x) .* (1 - logistic(x));
end

% ReLu activation function
function y = relu(x)
    y = max(0.01*x,x);
end
%  Derivative of ReLu
function y = Drelu(x)
    if x > 0
        y = 1;
    else
        y = 0.01;
    end
end
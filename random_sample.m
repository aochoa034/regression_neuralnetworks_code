%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%% Random sampling %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [x_training,training_set,x_test,test_set,x_valid,valid_set] = random_sample(x,data,training_set_percentaje,test_set_percentaje)
    % The function receives a data sample and returns a randomly ordered sample divided in three sets, training, validation and test sets.
    % The size of these sets is decided according to the specified percentajes.

    % We make a copy of x and the data points to work with it instead of taking values from the data vector directly
    data_copy = data;
    x_copy = x;
    % Let's build the training set
    training_set_obs_quantity = ceil(length(data)*training_set_percentaje); 
    training_set = zeros(1,training_set_obs_quantity);
    x_training = zeros(1,training_set_obs_quantity);        % the values from the x-axis corresponding the training set observations
    
    V =(1:length(data));
    
    % We now take aleatory values of the dataset
    V1 = randperm(length(V),training_set_obs_quantity); %get an specified amount of random integers in a range wihtout repetitions
    V1 = sort(V1); %we sort the taken values so the observations are ordered
    for i = 1:length(V1)
        training_set(i) = data(V1(i));
        x_training(i) = x(V1(i));
    end
    
    % We need to get rid of the used indexes
    for i = 1:length(V1)
        V = V(V~=V1(i));
        data_copy = data_copy(data_copy ~= data(V1(i))); %we eliminate from the data copy the values we have already taken for thetraining set
        x_copy = x_copy(x_copy ~= x(V1(i)));
    end
    
    test_set_obs_quantity = round(length(data)*test_set_percentaje); 
    test_set= zeros(1,test_set_obs_quantity);
    x_test = zeros(1,test_set_obs_quantity); 
    
    % We now take aleatory values of the remaining observations for the test set
    V2 = randsample(V,test_set_obs_quantity);
    V2 = sort(V2);
    for i = 1:length(V2)
        test_set(i) = data(V2(i));
        x_test(i) = x(V2(i));
    end
    
    for i = 1:length(V2)
        V = V(V~=V2(i));
        data_copy = data_copy(data_copy ~= data(V2(i))); %we eliminate from the data copy the values we have already taken for thetraining set
        x_copy = x_copy(x_copy ~= x(V2(i)));
    end
    
    % We assign the 10% of the dataset to the validation set
    valid_set_obs_quantity = length(data) - training_set_obs_quantity - test_set_obs_quantity; 
    valid_set= zeros(1,valid_set_obs_quantity);
    x_valid = zeros(1,valid_set_obs_quantity); 
    
    % The remaining values from the dataset will be the validation set values
    for i = 1:length(V)
        valid_set(i) = data(V(i));
        x_valid(i) = x(V(i));
    end
end
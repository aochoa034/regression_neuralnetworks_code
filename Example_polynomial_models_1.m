clear
clf % We clear the current figure so that on each execution the plots don't overlap
rng(2023) % We set a seed in order to make the data reproducible, since Matlab doesn't allow letters, we set, for instance, the current year

% We will consider the following function in the interval [0,2pi]
h = 0.05; % step
x = 0:h:2*pi; % we discretize the interval with a step h 
f = @(x) x.*sin(x); 

% We now add a small gaussian noise to the images of our f function. Thus, we displace the points a bit 
% and we make the simulation a bit more realistic.
mu = 0; sigma = 0.1; % make it aleatory if wanted
noise = sigma.*randn(1,length(x)) + mu;
data = f(x) + noise;  

% Divide randomly the sample in three sets
[x_training,training_set,x_test,test_set,x_valid,valid_set] = random_sample(x,data,0.8,0.1);

%%%%%%%%%%%% EXAMPLE 1  %%%%%%%%%%%%%%
%%%% TAKE h=0.05 AT THE BEGGINING %%%%

% We will make the simulation for the following degrees.
n1 = 3; n2 = 5; n3 = 8;

% Models and risks of the training test
[P1,R1_training] = polynomial_fitting(n1,x_training,training_set);
[P2,R2_training] = polynomial_fitting(n2,x_training,training_set);
[P3,R3_training] = polynomial_fitting(n3,x_training,training_set);

% %%%%%%%%%%%% EXAMPLE 2  %%%%%%%%%%%%%%
% %%%% TAKE h=0.4 AT THE BEGGINING %%%%
% 
% % We will make the simulation for the following degrees.
% n1 = 3; n2 = 5; n3 = 12;
% 
% % Models and risks of the training test
% [P1,R1_training] = polynomial_fitting(n1,x_training,training_set);
% [P2,R2_training] = polynomial_fitting(n2,x_training,training_set);
% [P3,R3_training] = polynomial_fitting(n3,x_training,training_set);

% Risks of the test test
R1_test = sum((test_set - polyval(P1,x_test)).^2)/length(test_set);
R2_test = sum((test_set - polyval(P2,x_test)).^2)/length(test_set); 
R3_test = sum((test_set - polyval(P3,x_test)).^2)/length(test_set); 

% Risks of the validation test
R1_valid = sum((valid_set - polyval(P1,x_valid)).^2)/length(valid_set); 
R2_valid = sum((valid_set - polyval(P2,x_valid)).^2)/length(valid_set); 
R3_valid = sum((valid_set - polyval(P3,x_valid)).^2)/length(valid_set); 

% If we analise them
one = [R1_training; R1_test; R1_valid]; 
two = [R2_training; R2_test; R2_valid];
three =[R3_training; R3_test; R3_valid];

% We now plot all the models we have made so far
I = 0:0.01:2*pi;
figure(1)
plot(I,polyval(P1,I),"red","DisplayName",strcat("Polynomial of ",num2str(n1),"-th degree" ))
hold on
plot(I,polyval(P2,I),"cyan","DisplayName",strcat("Polynomial of ",num2str(n2),"-th degree" ))
plot(I,polyval(P3,I),"magenta","DisplayName",strcat("Polynomial of ",num2str(n3),"-th degree" ))
plot(x_training,training_set,"blue .","DisplayName","Training set observations",MarkerSize=10)
plot(x_test,test_set,"black .","DisplayName","Test set observations",MarkerSize=10)
plot(x_valid,valid_set,"green .","DisplayName","Validation set observations",MarkerSize=10)
fplot(f,[0 2*pi],"black-.", "DisplayName",strcat("Original function ", "$f(x)$" ))
legend("Interpreter","latex");
xlabel("$x$","Interpreter","latex");
ylabel("$y$","Interpreter","latex");
hold off

figure(2)
plot((polyval(P1,x_training)-training_set).^2,"red", "DisplayName",strcat(num2str(n1),"-th degree" ))
hold on
plot((polyval(P2,x_training)-training_set).^2, "cyan", "DisplayName",strcat(num2str(n2),"-th degree" ))
plot((polyval(P3,x_training)-training_set).^2, "magenta", "DisplayName",strcat(num2str(n3),"-th degree" ))
hold off
xlabel("$i$","Interpreter","latex");
ylabel("$L(x^i,y^i)$","Interpreter","latex");
ylim([0 2])
legend()
title("Training set loss")

figure(3)
plot((polyval(P1,x_valid)-valid_set).^2,"red", "DisplayName",strcat(num2str(n1),"-th degree" ))
hold on
plot((polyval(P2,x_valid)-valid_set).^2, "cyan", "DisplayName",strcat(num2str(n2),"-th degree" ))
plot((polyval(P3,x_valid)-valid_set).^2, "magenta", "DisplayName",strcat(num2str(n3),"-th degree" ))
hold off
xlabel("$i$","Interpreter","latex");
ylabel("$L(x^i,y^i)$","Interpreter","latex");
ylim([0 2])
legend()
title("Validation set loss")

figure(4)
plot((polyval(P1,x_test)-test_set).^2,"red", "DisplayName",strcat(num2str(n1),"-th degree"))
hold on
plot((polyval(P2,x_test)-test_set).^2, "cyan", "DisplayName",strcat(num2str(n2),"-th degree"))
plot((polyval(P3,x_test)-test_set).^2, "magenta", "DisplayName",strcat(num2str(n3),"-th degree"))
hold off
xlabel("$i$","Interpreter","latex");
ylabel("$L(x^i,y^i)$","Interpreter","latex");
ylim([0 2])
legend()
title("Test set loss")
hold off

function [p,risk] = polynomial_fitting(n,Interval,data)
    p = polyfit(Interval,data,n);  %look for the polinomial who minimizes the error between the loss function and the data
    polynomialValues = @(p) polyval(p,Interval); % evaluate the polynomial on the points at the wanted interval
    risk = sum((data - polynomialValues(p)).^2)/length(Interval); % the average of the error using the loss function |x-y|^2
end
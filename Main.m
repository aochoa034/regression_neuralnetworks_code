clear % Clear all variables
clf   % We clear the current figure
rng(2023)

% We will consider the following function in the interval [0,2pi]
h = 0.05; % step
%%%%%  (For the second example take h=0.4) %%%%%%%

x = 0:h:2*pi; % we discretize the interval with a step h 
f = @(x) x.*sin(x); 

% We now add a small gaussian noise to the images of our f function. Thus, we displace the points a bit 
% and we make the simulation a bit more realistic.
sigma = 0.1;
noise = sigma.*randn(1,length(x)) ;
data = f(x) + noise; 

% Divide randomly the sample in three sets
[x_training,training_set,x_test,test_set,x_valid,valid_set] = random_sample(x,data,0.8,0.1);

%%%%%%%%% EXAMPLE 1 (Logistic function) %%%%%%%%%%%%%%
%% TAKE h=0.05 AT THE BEGGINING %%%%
% We will make the simulation for the following degrees.
n1 = 4; n2 = 6; n3 = 9;

% Models
P1 = regression(n1,x_training,training_set,"logistic",0.1,500);
P2 = regression(n2,x_training,training_set,"logistic",0.1,500);
P3 = regression(n3,x_training,training_set,"logistic",0.1,500);

% %%%%%%%%%% EXAMPLE 1 (Relu function) %%%%%%%%%%%%%%
% % We will make the simulation for the following degrees.
% n1 = 4; n2 = 6; n3 = 9;
%  
% % Models
% [P1,output1] = regression(n1,x_training,training_set,"relu",0.1,500);  %%MIRAR CON 400
% [P2,output2]= regression(n2,x_training,training_set,"relu",0.1,500);
% [P3,output3] = regression(n3,x_training,training_set,"relu",0.1,500);

%%%%%%%%% EXAMPLE 2 (Logistic function) %%%%%%%%%%%%%%
%%% TAKE h=0.4 AT THE BEGGINING %%%%
% % We will make the simulation for the following degrees.
% n1 = 4; n2 = 6; n3 = 13;
% 
% % Models 
% P1 = regression(n1,x_training,training_set,"logistic",0.1,100);
% P2 = regression(n2,x_training,training_set,"logistic",0.1,100);
% P3 = regression(n3,x_training,training_set,"logistic",0.1,100);

%%%%%%%%% EXAMPLE 2 (Relu function) %%%%%%%%%%%%%%
% % % We will make the simulation for the following degrees.
% n1 = 4; n2 = 6; n3 = 13;
% % 
% % % Models 
% [P1,output1] = NeuralNetwork(n1,x_training,training_set,"relu",0.01,300);
% [P2,output2] = NeuralNetwork(n2,x_training,training_set,"relu",0.01,300);
% [P3,output3] = NeuralNetwork(n3,x_training,training_set,"relu",0.001,3000);

% Risks of the training test
R1_training = sum((training_set - polyval(P1',x_training)).^2)/length(training_set); 
R2_training = sum((training_set - polyval(P2',x_training)).^2)/length(training_set); 
R3_training = sum((training_set - polyval(P3',x_training)).^2)/length(training_set); 

% Risks of the validation test
R1_valid = sum((valid_set - polyval(P1',x_valid)).^2)/length(valid_set); 
R2_valid = sum((valid_set - polyval(P2',x_valid)).^2)/length(valid_set); 
R3_valid = sum((valid_set - polyval(P3',x_valid)).^2)/length(valid_set); 

% Risks of the test test
R1_test = sum((test_set - polyval(P1',x_test)).^2)/length(test_set);
R2_test = sum((test_set - polyval(P2',x_test)).^2)/length(test_set); 
R3_test = sum((test_set - polyval(P3',x_test)).^2)/length(test_set); 

% If we analise them
one = [R1_training; R1_test; R1_valid]; 
two = [R2_training; R2_test; R2_valid];
three =[R3_training; R3_test; R3_valid];

figure(1)
I = 0:0.01:2*pi;
plot(I,polyval(P1',I),"red","DisplayName",strcat(num2str(n1)," neurons" ))
hold on
plot(I,polyval(P2',I),"cyan","DisplayName",strcat(num2str(n2)," neurons" ))
plot(0:0.01:2*pi-0.25,polyval(P3',0:0.01:2*pi-0.25),"magenta","DisplayName",strcat(num2str(n3)," neurons" ))
plot(x_training,training_set,"blue .","DisplayName","Training set observations",MarkerSize=10)
plot(x_test,test_set,"black .","DisplayName","Test set observations",MarkerSize=10)
plot(x_valid,valid_set,"green .","DisplayName","Validation set observations",MarkerSize=10)
fplot(f,[0 2*pi],"black-.", "DisplayName",strcat("Original function ", "$f(x)$" ))
legend("Interpreter","latex");
xlabel("$x$","Interpreter","latex");
ylabel("$y$","Interpreter","latex");
ylim([-5 2])
hold off

figure(2)
plot((polyval(P1',x_training)-training_set).^2,"red", "DisplayName",strcat(num2str(n1)," neurons" ))
hold on
plot((polyval(P2',x_training)-training_set).^2, "cyan", "DisplayName",strcat(num2str(n2)," neurons" ))
plot((polyval(P3',x_training)-training_set).^2, "magenta", "DisplayName",strcat(num2str(n3)," neurons" ))
hold off
xlabel("$i$","Interpreter","latex");
ylabel("$L(x^i,y^i)$","Interpreter","latex");
ylim([0 2])
legend()
title("Training set loss")

figure(3)
plot((polyval(P1',x_valid)-valid_set).^2,"red", "DisplayName",strcat(num2str(n1)," neurons" ))
hold on
plot((polyval(P2',x_valid)-valid_set).^2, "cyan", "DisplayName",strcat(num2str(n2)," neurons" ))
plot((polyval(P3',x_valid)-valid_set).^2, "magenta", "DisplayName",strcat(num2str(n3)," neurons" ))
hold off
xlabel("$i$","Interpreter","latex");
ylabel("$L(x^i,y^i)$","Interpreter","latex");
ylim([0 2])
legend()
title("Validation set loss")

figure(4)
plot((polyval(P1',x_test)-test_set).^2,"red", "DisplayName",strcat(num2str(n1)," neurons" ))
hold on
plot((polyval(P2',x_test)-test_set).^2, "cyan", "DisplayName",strcat(num2str(n2)," neurons" ))
plot((polyval(P3',x_test)-test_set).^2, "magenta", "DisplayName",strcat(num2str(n3)," neurons" ))
hold off
xlabel("$i$","Interpreter","latex");
ylabel("$L(x^i,y^i)$","Interpreter","latex");
ylim([0 2])
legend()
title("Test set loss")
hold off
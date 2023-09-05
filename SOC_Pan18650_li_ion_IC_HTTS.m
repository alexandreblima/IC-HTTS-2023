%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                       %
% Code for pre-processing the training data of the      %
% 2.9Ah Panasonic 18650 lithium-ion cell - Aug/2023     %
%                                                       %
% Training data (drive cycles) applied on the battery:  %
% https://www.mendeley.com/profiles/phillip-kollmeyer/  % 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all 
close all

% =========================================================================
% INPUT DATA

% M-point moving average filter
M = 400;
B = (1/M)*ones(M,1);

% EWMA 
% https://www.mathworks.com/help/signal/ug/signal-smoothing.html 
alpha = 0.01;

% DRIVE CYCLE: NEURAL NETWORKS (NN)
% 25degC_NN_Pan18650PF.mat
data = input('file with the data: ','s');
container = load(data);

t = container.meas.Time;

% Feature 1: Voltage
Voltage = container.meas.Voltage;
N = size(Voltage,1);

% Filtering out high frequencies
VoltageFilt = filter(B,1,Voltage); % moving average
VoltageFilt1 = filter(alpha, [1 alpha-1], Voltage); % EWMA

% Feature scaling: normalization range [0-1]
Voltage_scaling = (Voltage - min(Voltage))/(max(Voltage)-min(Voltage));
% Voltage_scaling = (Voltage - mean(Voltage))/std(Voltage);
% Voltage_scaling1 = (VoltageFilt - mean(VoltageFilt))/std(VoltageFilt);
% Voltage_scaling1 = (VoltageFilt - min(VoltageFilt))/(max(VoltageFilt)-min(VoltageFilt));
% Voltage_scaling2 = (VoltageFilt1 - mean(VoltageFilt1))/std(VoltageFilt1);

if all(Voltage_scaling >= 0) && all(Voltage_scaling <= 1)
    disp('Voltage_scaling está entre 0 e 1.');
else
    disp('Voltage_scaling não está entre 0 e 1.');
end

% Feature 2: Current
Current = container.meas.Current;

% Filtering out high frequencies
CurrentFilt = filter(B,1,Current); % moving average
CurrentFilt1 = filter(alpha, [1 alpha-1],Current); % EWMA

% % Feature scaling: normalization range [0-1]
Current_scaling = (Current - min(Current))/(max(Current)-min(Current));
%Current_scaling1 = (CurrentFilt - mean(CurrentFilt))/std(CurrentFilt);
%Current_scaling1 = (CurrentFilt - min(CurrentFilt))/(max(CurrentFilt)-min(CurrentFilt));
%Current_scaling2 = (CurrentFilt1 - mean(CurrentFilt1))/std(CurrentFilt1);

if all(Current_scaling >= 0) && all(Current_scaling <= 1)
    disp('Current_scaling está entre 0 e 1.');
else
    disp('Current_scaling não está entre 0 e 1.');
end

% Feature 3: Temperature
Temperature = container.meas.Battery_Temp_degC;

% Filtering out high frequencies
TemperatureFilt = filter(B,1,Temperature); % moving average
TemperatureFilt1 = filter(alpha, [1 alpha-1],Temperature); % EWMA

% % Feature scaling: normalization range [0-1]
Temperature_scaling = (Temperature - min(Temperature))/(max(Temperature)-min(Temperature));
%Temperature_scaling1 = (TemperatureFilt - mean(TemperatureFilt))/std(TemperatureFilt);
%Temperature_scaling1 = (TemperatureFilt - min(TemperatureFilt))/(max(TemperatureFilt)-min(TemperatureFilt));
%Temperature_scaling2 = (TemperatureFilt1 - mean(TemperatureFilt1))/std(TemperatureFilt1);

if all(Temperature_scaling >= 0) && all(Temperature_scaling <= 1)
    disp('Temperature_scaling está entre 0 e 1.');
else
    disp('Temperature_scaling não está entre 0 e 1.');
end

% Label: State of Charge
Ah_discharge = container.meas.Ah;
pAh = -1*Ah_discharge;
bias = (-1)*min(Ah_discharge); % 
SOC = 100*((Ah_discharge + bias)/bias);

% Saves all training data and labels (targets)
% The validation split is done by the Python code
% In this case, we will not with test data and test labels
% train_data = [Voltage_scaling Current_scaling Temperature_scaling];
% save('train_data.mat', 'train_data');
% save('train_label.mat', 'SOC');

% Test data and test labels 

% generate random integers
rng('shuffle')
r = randperm(N,round(0.2*N));% vector 9.612 indexes
r = r';
r = sort(r); % sorts elements in ascending order

% Test data and test labels
Voltage_test = Voltage_scaling(r); % voltage
%Voltage_test1 = Voltage_scaling1(r); % moving average voltage 
%Voltage_test2 = Voltage_scaling2(r); % EWMA voltage

Current_test = Current_scaling(r); % current
%Current_test1 = Current_scaling1(r); % moving average current
%Current_test2 = Current_scaling2(r); % EWMA current

Temperature_test = Temperature_scaling(r); % temperature
%Temperature_test1 = Temperature_scaling1(r); % moving average temperature
%Temperature_test2 = Temperature_scaling2(r); % EWMA temperature

%test_data = [Voltage_test Voltage_test1 Current_test1 Temperature_test1];
test_data = [Voltage_test Current_test Temperature_test];
save('test_data.mat', 'test_data');

SOC_label_test = SOC(r);
save('test_label.mat', 'SOC_label_test');

% Training data and labels
a = [1:N];
a(r)=[];
a = a';
Voltage_train = Voltage_scaling(a); % voltage
%Voltage_train1 = Voltage_scaling1(a); % moving average voltage 
%Voltage_train2 = Voltage_scaling2(a); % EWMA voltage

Current_train = Current_scaling(a); % current
%Current_train1 = Current_scaling1(a); % moving average current
%Current_train2 = Current_scaling2(a); % EWMA current

Temperature_train = Temperature_scaling(a); % temperature
%Temperature_train1 = Temperature_scaling1(a); % moving average temperature
%Temperature_train2 = Temperature_scaling2(a); % EWMA temperature

%train_data = [Voltage_train Voltage_train1 Current_train1 Temperature_train1];
train_data = [Voltage_train Current_train Temperature_train];
save('train_data.mat', 'train_data');
save('train_data.csv', 'train_data')

SOC_label_train = SOC(a);
save('train_label.mat', 'SOC_label_train');
save('train_label.csv', 'SOC_label_train');

% A few plots to visualize the Drive Cycle
figure(1)
plot(SOC,Temperature,'r')  
xlabel('SOC (%)', 'FontSize', 14)
ylabel('Temperature (Celsius)', 'FontSize', 14)
% Set x and y font sizes.
set(gca,'FontSize',14)
grid
saveas(figure(1), 'figuras/tempsoc.png');
hold on
%pause

figure(2)
plot(t/60,Ah_discharge,'m')
xlabel('Time (minutes)', 'FontSize', 14)
ylabel('Amp-hours discharged', 'FontSize', 14)
xlim([0 round(max(t/60))])
% Set x and y font sizes.
set(gca,'FontSize',14)
grid
saveas(figure(2), 'figuras/amphourstime.png');
hold on
%pause

figure(3)
plot(t/60,Voltage,'b')
xlabel('Time (minutes)', 'FontSize', 14)
ylabel('Voltage (V)', 'FontSize', 14)
xlim([0 round(max(t/60))])
% Set x and y font sizes.
set(gca,'FontSize',14)
grid
saveas(figure(3), 'figuras/voltagetime.png');
hold on
%pause

figure(4)
plot(t/60,Current,'k')
xlabel('Time (minutes)', 'FontSize', 14)
ylabel('Current (A)', 'FontSize', 14)
xlim([0 round(max(t/60))])
% Set x and y font sizes.
set(gca,'FontSize',14)
grid
saveas(figure(4), 'figuras/currenttime.png');
hold on
%pause

figure(5)
%plot(t,Temperature,'g')
plot(t/60,Temperature,'g')
xlabel('Time (minutes)', 'FontSize', 14)
ylabel('Temperature (Celsius)', 'FontSize', 14)
xlim([0 round(max(t/60))])
ylim([25.5 30.0])
% Set x and y font sizes.
set(gca,'FontSize',14)
grid
saveas(figure(5), 'figuras/temperaturetime.png');
hold on
%pause

figure(6)
plot(SOC,Voltage,'r')
xlabel('SOC (%)', 'FontSize', 14)
ylabel('Voltage (V)', 'FontSize', 14)
% Set x and y font sizes.
set(gca,'FontSize',14)
grid
saveas(figure(6), 'figuras/voltagesoc.png');
hold on
pause

close all
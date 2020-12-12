clc
clear
close all

%Linear and Gaussian are taking more time to run so they are commenting

%% Importing Data
file = readtable("OnlineNewsPopularity.csv");
urls = file.url;
urlIndex = 1:length(urls);
numData = file{:,3:end};
numData1 = file{:,3:end-1};

%% Undestanding goal variable - shares
shares = numData(:,end);
meanOfShares = mean(shares);
medianOfShares = median(shares);
sigmaOfShares = sqrt(var(shares));

%% Data Cleaning

% 1) No Null values - mentioning in citation of dataset

% 2) Outliers removal
cleanedData = numData(:,:);
for i = 1:length(numData(1,:))
    vecData = cleanedData(:,i);
    cleanedData = cleanedData((vecData<=(mean(vecData)+3*sqrt(var(vecData))))&(vecData>=(mean(vecData-3*sqrt(var(vecData))))),:);
end
cleanedShares = cleanedData(:,end);
cleanedData = cleanedData(:,1:end-1);

clear vecData
%% Raw Data
t = zeros(length(cleanedShares),1);
for i = 1:length(t)
    if numData(i,end)-1400 > 0
        t(i)=1;
    end
end
responseData = t;

t1 = zeros(length(shares),1);
for i = 1:length(t1)
    if numData(i,end)-1400 > 0
        t1(i)=1;
    end
end

z_raw_outliers = [numData1 t1];

trainingData1 = cleanedData;
z_raw_no_outliers = [trainingData1 responseData];

tic
[model_1, acc_1] = trainClassifier_logistic_Raw_without_Outliers(z_raw_no_outliers);
toc
time = toc;
disp('Accuracy of Logistic_Rawdata - No Outliers')
disp(acc_1)
disp('Time taken - Logistic_Rawdata - No Outliers')
disp(time)
% 
tic
[model_2, acc_2] = trainClassifier_KNN_Raw_without_Outliers(z_raw_no_outliers);
toc
time = toc;
disp('Accuracy of KNN_Rawdata - No Outliers')
disp(acc_2)
disp('Time taken - KNN_Rawdata - No Outliers')
disp(time)

tic
[model_3, acc_3] = trainClassifier_linearSVM_Raw_without_Outliers(z_raw_no_outliers);
toc
time = toc;
disp('Accuracy of LinearSVM_Rawdata - No Outliers')
disp(acc_3)
disp('Time taken - LinearSVM_Rawadata - No Outliers')
disp(time)

tic
[model_4, acc_4] = trainClassifier_GaussianSVM_Raw_without_Outliers(z_raw_no_outliers);
toc
time = toc;
disp('Accuracy of GaussianSVM_ Rawdata - No Outliers')
disp(acc_4)
disp('Time taken - GaussianSVM_ Rawdata - No Outliers')
disp(time)

tic
[model_5, acc_5] = trainClassifier_Logistic_Raw(z_raw_outliers);
toc
time = toc;
disp('Accuracy of Logistic_Rawdata')
disp(acc_5)
disp('Time taken - Logistic_Rawdata')
disp(time)

tic
[model_6, acc_6] = trainClassifier_KNN_Raw(z_raw_outliers);
toc
time = toc;
disp('Accuracy of KNN_Rawdata')
disp(acc_6)
disp('Time taken - KNN  _Rawdata')
disp(time)
% 
% tic
% [model_7, acc_7] = trainClassifier_LinearSVM_Raw(z_raw_outliers);
% toc
% time = toc;
% disp('Accuracy of LinearSVM_Rawdata')
% disp(acc_7)
% disp('Time taken - LinearSVM_Rawdata')
% disp(time)
% 
% tic
% [model_8, acc_8] = trainClassifier_GaussianSVM_Raw(z_raw_outliers);
% toc
% time = toc;
% disp('Accuracy of GaussianSVM_Rawdata')
% disp(acc_8)
% disp('Time taken - GaussianSVM_Rawdata')
% disp(time)

%% Feature Extraction Method 2 : PCA - Unsupervised - With outliers
normCleanedData = normalize(numData(:,1:end-1));
out = normCleanedData(:,all(~isnan(normCleanedData)));
[coeff,score,latent,tsquared,explained,mu] = pca(out);
dataInPrincipalComponentSpace = score;
sumOfExplanation=0;
for i = 1:length(explained)
    sumOfExplanation = sumOfExplanation + explained(i);
    if sumOfExplanation > 95
       break;
    end
end
pcaData = dataInPrincipalComponentSpace(:,1:i);
figure
plot(0:length(explained)-1,cumsum(explained), 's-', 'Linewidth',1);hold('on');
title('Explanation of principal componenets - Cumilative correlation');
xlim([0 60]); hold('on')

%% Training
trainingData = pcaData;
t = zeros(length(shares),1);
for i = 1:length(t)
    if numData(i,end)-1400 > 0
        t(i)=1;
    end
end
responseData = t;
tic
[model1, acc1] = trainClassifierLogisticOutliers(trainingData, responseData);
toc
time = toc;
disp('Accuracy of Logistic - Outliers')
disp(acc1)
disp('Time taken - Logistic - Outliers')
disp(time)
tic
[model2, acc2] = trainClassifierKNNOutliers(trainingData, responseData);
toc
time = toc;
disp('Accuracy of KNN - Outliers')
disp(acc2)
disp('Time taken - KNN- Outliers')
disp(time)

%% Feature Extraction Method 2 : PCA - Unsupervised - No outliers
normCleanedData = normalize(cleanedData);
out = normCleanedData(:,all(~isnan(normCleanedData)));
[coeff,score,latent,tsquared,explained,mu] = pca(out);
dataInPrincipalComponentSpace = score;
sumOfExplanation=0;
for i = 1:length(explained)
    sumOfExplanation = sumOfExplanation + explained(i);
    if sumOfExplanation > 95
       break;
    end
end
pcaData = dataInPrincipalComponentSpace(:,1:i);
plot(0:length(explained)-1,cumsum(explained), 's-', 'Linewidth',1);hold('on');
yline(95, 'g--', 'Linewidth', 1);
xlim([0 60])
legend('With outliers','Without Outliers','Explanation limit');

%% Training
trainingData = pcaData;
t = zeros(length(cleanedShares),1);
for i = 1:length(t)
    if numData(i,end)-1400 > 0
        t(i)=1;
    end
end
responseData = t;
tic
[model3, acc3] = trainClassifierLinearSVM(trainingData, responseData);
toc
time = toc;
disp('Accuracy of LinearSVM - No Outliers')
disp(acc3)
disp('Time taken - LinearSVM - No Outliers')
disp(time)
tic
[model4, acc4] = trainClassifierGuassianSVM(trainingData, responseData);
toc
time = toc;
disp('Accuracy of GaussianSVM - No Outliers')
disp(acc4)
disp('Time taken - GAussianSVM - No Outliers')
disp(time)
tic
[model5, acc5] = trainClassifierLogistic(trainingData, responseData);
toc
time = toc;
disp('Accuracy of Logistic - No Outliers')
disp(acc5)
disp('Time taken - Logistic - No Outliers')
disp(time)
tic
[model6, acc6] = trainClassifierKNN(trainingData, responseData);
toc
time = toc;
disp('Accuracy of KNN - No Outliers')
disp(acc6)
disp('Time taken - KNN - No Outliers')
disp(time)

%% Log Transform on Raw Data
log_raw_outliers = zeros(size(z_raw_outliers));
no_features = size(z_raw_outliers,2)-1;
for i = 1:no_features
    feature_vec = z_raw_outliers(:,i);
    neg_zeros = (feature_vec(feature_vec<=0));
    if size(neg_zeros,1) ==0
        new_feature_vec = log(feature_vec);
    else
        new_feature_vec = feature_vec;
    end
    log_raw_outliers(:,i) = new_feature_vec;
end
log_raw_outliers(:,no_features+1) = t1;

%% Log Transform on outlier less Data
log_clean_outliers = zeros(size(z_raw_no_outliers));
no_features = size(z_raw_no_outliers,2)-1;
for i = 1:no_features
    feature_vec = z_raw_no_outliers(:,i);
    neg_zeros = (feature_vec(feature_vec<=0));
    if size(neg_zeros,1) ==0
        new_feature_vec = log(feature_vec);
    else
        new_feature_vec = feature_vec;
    end
    log_clean_outliers(:,i) = new_feature_vec;
end
log_clean_outliers(:,no_features+1) = t;

trainClassifier_Logistic_raw_log(log_raw_outliers)
trainClassifier_KNN_raw_log(log_raw_outliers)

trainClassifier_Logistic_raw_outliers_log(log_clean_outliers)
trainClassifier_KNN_raw_outliers_log(log_clean_outliers)
trainClassifier_LinearSVM_raw_outliers_log(log_clean_outliers)
trainClassifier_GaussianSVM_raw_outliers_log(log_clean_outliers)

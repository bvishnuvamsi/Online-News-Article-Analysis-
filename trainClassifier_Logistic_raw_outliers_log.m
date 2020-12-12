function [] = trainClassifier_Logistic_raw_outliers_log(trainingData)

tic
inputTable = array2table(trainingData, 'VariableNames', {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10', 'column_11', 'column_12', 'column_13', 'column_14', 'column_15', 'column_16', 'column_17', 'column_18', 'column_19', 'column_20', 'column_21', 'column_22', 'column_23', 'column_24', 'column_25', 'column_26', 'column_27', 'column_28', 'column_29', 'column_30', 'column_31', 'column_32', 'column_33', 'column_34', 'column_35', 'column_36', 'column_37', 'column_38', 'column_39', 'column_40', 'column_41', 'column_42', 'column_43', 'column_44', 'column_45', 'column_46', 'column_47', 'column_48', 'column_49', 'column_50', 'column_51', 'column_52', 'column_53', 'column_54', 'column_55', 'column_56', 'column_57', 'column_58', 'column_59'});

predictorNames = {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10', 'column_11', 'column_12', 'column_13', 'column_14', 'column_15', 'column_16', 'column_17', 'column_18', 'column_19', 'column_20', 'column_21', 'column_22', 'column_23', 'column_24', 'column_25', 'column_26', 'column_27', 'column_28', 'column_29', 'column_30', 'column_31', 'column_32', 'column_33', 'column_34', 'column_35', 'column_36', 'column_37', 'column_38', 'column_39', 'column_40', 'column_41', 'column_42', 'column_43', 'column_44', 'column_45', 'column_46', 'column_47', 'column_48', 'column_49', 'column_50', 'column_51', 'column_52', 'column_53', 'column_54', 'column_55', 'column_56', 'column_57', 'column_58'};
predictors = inputTable(:, predictorNames);
response = inputTable.column_59;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false];

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
% For logistic regression, the response values must be converted to zeros
% and ones because the responses are assumed to follow a binomial
% distribution.
% 1 or true = 'successful' class
% 0 or false = 'failure' class
% NaN - missing response.
successClass = double(1);
failureClass = double(0);
% Compute the majority response class. If there is a NaN-prediction from
% fitglm, convert NaN to this majority class label.
numSuccess = sum(response == successClass);
numFailure = sum(response == failureClass);
if numSuccess > numFailure
    missingClass = successClass;
else
    missingClass = failureClass;
end
successFailureAndMissingClasses = [successClass; failureClass; missingClass];
isMissing = isnan(response);
zeroOneResponse = double(ismember(response, successClass));
zeroOneResponse(isMissing) = NaN;
% Prepare input arguments to fitglm.
concatenatedPredictorsAndResponse = [predictors, table(zeroOneResponse)];
% Train using fitglm.
GeneralizedLinearModel = fitglm(...
    concatenatedPredictorsAndResponse, ...
    'Distribution', 'binomial', ...
    'link', 'logit');

% Convert predicted probabilities to predicted class labels and scores.
convertSuccessProbsToPredictions = @(p) successFailureAndMissingClasses( ~isnan(p).*( (p<0.5) + 1 ) + isnan(p)*3 );
returnMultipleValuesFcn = @(varargin) varargin{1:max(1,nargout)};
scoresFcn = @(p) [1-p, p];
predictionsAndScoresFcn = @(p) returnMultipleValuesFcn( convertSuccessProbsToPredictions(p), scoresFcn(p) );

% Create the result struct with predict function
predictorExtractionFcn = @(x) array2table(x, 'VariableNames', predictorNames);
logisticRegressionPredictFcn = @(x) predictionsAndScoresFcn( predict(GeneralizedLinearModel, x) );
trainedClassifier.predictFcn = @(x) logisticRegressionPredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.GeneralizedLinearModel = GeneralizedLinearModel;
trainedClassifier.SuccessClass = successClass;
trainedClassifier.FailureClass = failureClass;
trainedClassifier.MissingClass = missingClass;
trainedClassifier.ClassNames = {successClass; failureClass};
trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2020a.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new predictor column matrix, X, use: \n  yfit = c.predictFcn(X) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nX must contain exactly 58 columns because this model was trained using 58 predictors. \nX must contain only predictor columns in exactly the same order and format as your training \ndata. Do not include the response column or any columns you did not import into the app. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
% Convert input to table
inputTable = array2table(trainingData, 'VariableNames', {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10', 'column_11', 'column_12', 'column_13', 'column_14', 'column_15', 'column_16', 'column_17', 'column_18', 'column_19', 'column_20', 'column_21', 'column_22', 'column_23', 'column_24', 'column_25', 'column_26', 'column_27', 'column_28', 'column_29', 'column_30', 'column_31', 'column_32', 'column_33', 'column_34', 'column_35', 'column_36', 'column_37', 'column_38', 'column_39', 'column_40', 'column_41', 'column_42', 'column_43', 'column_44', 'column_45', 'column_46', 'column_47', 'column_48', 'column_49', 'column_50', 'column_51', 'column_52', 'column_53', 'column_54', 'column_55', 'column_56', 'column_57', 'column_58', 'column_59'});

predictorNames = {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10', 'column_11', 'column_12', 'column_13', 'column_14', 'column_15', 'column_16', 'column_17', 'column_18', 'column_19', 'column_20', 'column_21', 'column_22', 'column_23', 'column_24', 'column_25', 'column_26', 'column_27', 'column_28', 'column_29', 'column_30', 'column_31', 'column_32', 'column_33', 'column_34', 'column_35', 'column_36', 'column_37', 'column_38', 'column_39', 'column_40', 'column_41', 'column_42', 'column_43', 'column_44', 'column_45', 'column_46', 'column_47', 'column_48', 'column_49', 'column_50', 'column_51', 'column_52', 'column_53', 'column_54', 'column_55', 'column_56', 'column_57', 'column_58'};
predictors = inputTable(:, predictorNames);
response = inputTable.column_59;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false];

% Perform cross-validation
KFolds = 5;
cvp = cvpartition(response, 'KFold', KFolds);
% Initialize the predictions to the proper sizes
validationPredictions = response;
numObservations = size(predictors, 1);
numClasses = 2;
validationScores = NaN(numObservations, numClasses);
for fold = 1:KFolds
    trainingPredictors = predictors(cvp.training(fold), :);
    trainingResponse = response(cvp.training(fold), :);
    foldIsCategoricalPredictor = isCategoricalPredictor;
    
    % Train a classifier
    % This code specifies all the classifier options and trains the classifier.
    % For logistic regression, the response values must be converted to zeros
    % and ones because the responses are assumed to follow a binomial
    % distribution.
    % 1 or true = 'successful' class
    % 0 or false = 'failure' class
    % NaN - missing response.
    successClass = double(1);
    failureClass = double(0);
    % Compute the majority response class. If there is a NaN-prediction from
    % fitglm, convert NaN to this majority class label.
    numSuccess = sum(trainingResponse == successClass);
    numFailure = sum(trainingResponse == failureClass);
    if numSuccess > numFailure
        missingClass = successClass;
    else
        missingClass = failureClass;
    end
    successFailureAndMissingClasses = [successClass; failureClass; missingClass];
    isMissing = isnan(trainingResponse);
    zeroOneResponse = double(ismember(trainingResponse, successClass));
    zeroOneResponse(isMissing) = NaN;
    % Prepare input arguments to fitglm.
    concatenatedPredictorsAndResponse = [trainingPredictors, table(zeroOneResponse)];
    % Train using fitglm.
    GeneralizedLinearModel = fitglm(...
        concatenatedPredictorsAndResponse, ...
        'Distribution', 'binomial', ...
        'link', 'logit');
    
    % Convert predicted probabilities to predicted class labels and scores.
    convertSuccessProbsToPredictions = @(p) successFailureAndMissingClasses( ~isnan(p).*( (p<0.5) + 1 ) + isnan(p)*3 );
    returnMultipleValuesFcn = @(varargin) varargin{1:max(1,nargout)};
    scoresFcn = @(p) [1-p, p];
    predictionsAndScoresFcn = @(p) returnMultipleValuesFcn( convertSuccessProbsToPredictions(p), scoresFcn(p) );
    
    % Create the result struct with predict function
    logisticRegressionPredictFcn = @(x) predictionsAndScoresFcn( predict(GeneralizedLinearModel, x) );
    validationPredictFcn = @(x) logisticRegressionPredictFcn(x);
    
    % Add additional fields to the result struct
    
    % Compute validation predictions
    validationPredictors = predictors(cvp.test(fold), :);
    [foldPredictions, foldScores] = validationPredictFcn(validationPredictors);
    
    % Store predictions in the original order
    validationPredictions(cvp.test(fold), :) = foldPredictions;
    validationScores(cvp.test(fold), :) = foldScores;
end

% Compute validation accuracy
correctPredictions = (validationPredictions == response);
isMissing = isnan(response);
correctPredictions = correctPredictions(~isMissing);
validationAccuracy = sum(correctPredictions)/length(correctPredictions);

toc;

time = toc;
disp('Logistic Regression Without Outliers')
disp('_________________________________')


disp('Accuracy')
disp(validationAccuracy)
disp('Time taken')
disp(time)

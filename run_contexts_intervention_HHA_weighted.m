function [net_Context,info_Context,TP,FP,FN,TN,resultTable] = run_contexts_intervention_HHA(imdsTrain, imdsTest, foldNum) 
disp(foldNum);

% Load pretrained network
net = vgg16;
 
% % First image input size
inputSize = net.Layers(1).InputSize;
% 
% % Modify numchannels 3-->1
% inputSize(3) = 1;
% 
% W = net.Layers(2, 1).Weights;   W(:,:,3,:) = []; W(:,:,2,:) = [];
% B = net.Layers(2, 1).Bias;      



% Modify last 3 layers
layersTransfer = net.Layers(1:end-3);

weight_for_0 = 0.5462147887323944;
weight_for_1 = 5.90952380952381;
CLASS_WEIGHT = [ weight_for_0, weight_for_1 ];
CLASSES = ['noone', 'nurse'];

numClasses = 2;
layers = [
%     imageInputLayer(inputSize,'Name','input')
%     convolution2dLayer(3,64,'NumChannels',1,'Stride',1,'Padding',1,'Weights',W,'Bias',B,'Name','conv1')
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    softmaxLayer
    classificationLayer('Classes', CLASSES, 'ClassWeights', CLASS_WEIGHT)
    %weightedClassificationLayer(CLASS_WEIGHT)
    ];

% Augment data
% imageAugmenter = imageDataAugmenter( ...
%     'RandXReflection',true, ...
%     'RandYReflection',true, ...
%     'RandRotation',[0 360],...
%     'RandScale',[0.5 2],...
%     'RandXTranslation',[-5 5],...
%     'RandYTranslation',[-5 5]);

    imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandYReflection',true, ...
    'RandRotation',[0 360]);

% Resize train and test images
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest,...
    'DataAugmentation',imageAugmenter);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain,...
    'DataAugmentation',imageAugmenter);

% Freeze initial weights
%lay = layers.Layers;
%connections = layers.Connections;

%layers(1:31) = freezeWeights(layers(1:31));
%layers = createLgraphUsingConnections(lay,connections);


% Training options
options = trainingOptions('sgdm', ...
    'MiniBatchSize',20, ...
    'MaxEpochs',1, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsTest, ...
    'ValidationFrequency',5, ...
    'Verbose',false);

% Train network
%%% Update network name
tic
[net_Context,info_Context] = trainNetwork(augimdsTrain,layers,options);
toc

%%% Testing predictions
%%% Select new network for testing
tic
[predictedLabels, scores] = classify(net_Context,augimdsTest);
toc

maxScore = max(scores.');
resultTable = [cellstr(predictedLabels) num2cell(maxScore.')];

% accuracy = mean(predictedLabels == imdsTest.Labels);
% cm = confusionchart(imdsTest.Labels,predictedLabels);

labelsTable = table(imdsTest.Labels,predictedLabels);
TP = 0; FP = 0;
FN = 0; TN = 0; 

for i = 1:height(labelsTable)
    a = imdsTest.Labels(i);
    b = predictedLabels(i);
    if (a == 'nurse')&&(b == 'nurse')
        TP = TP + 1;
    elseif (a == 'noone')&&(b == 'nurse')
        FP = FP + 1;
    elseif (a == 'nurse')&&(b == 'noone')
        FN = FN + 1;
    elseif (a == 'noone')&&(b == 'noone')
        TN = TN + 1;
    end
end
        
confMatrix = [TP FP;FN TN]

sens = TP/(TP+FN)
spec = TN/(TN+FP)
acc = (TP+TN)/sum(sum(confMatrix))

% Plot and save training accuracy and loss graphs
numPoints = length(info_Context.TrainingAccuracy);

figure;
plot(1:numPoints, info_Context.TrainingAccuracy, 'Color', [0 0.4470 0.7410])
hold on
scatter(1:numPoints, info_Context.ValidationAccuracy,'black', '.')
legend('TrainingAccuracy' , 'ValidationAccuracy')
hold off
figName_Acc = strcat('TrainingAccuracy_',string(foldNum),'.png');
saveas(gcf , figName_Acc)

figure;
plot(1:numPoints, info_Context.TrainingLoss, 'Color', [0.8500 0.3250 0.0980])
hold on
scatter(1:numPoints, info_Context.ValidationLoss,'black', '.')
legend('TrainingLoss' , 'ValidationLoss')
hold off
figName_Loss = strcat('TrainingLoss_', string(foldNum),'.png');
saveas(gcf , figName_Loss)

end


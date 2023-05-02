filepath = 'C:\Users\Zalamaan\Documents\Repos\depthSliceTool\bagmerge\InterventionDetectionFiles\FirstFrameDepthRGB_origData\';
allPts = [1, 2, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16, 17, 21, 22, 23:28, 29:34];
cvFolds = [0;0;0;0;0;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1];
cvFolds = [cvFolds, [1;1;1;1;1;0;0;0;0;0;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1]];
cvFolds = [cvFolds, [1;1;1;1;1;1;1;1;1;1;0;0;0;0;0;1;1;1;1;1;1;1;1;1;1;1;1]];
cvFolds = [cvFolds, [1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;0;0;0;0;0;0;1;1;1;1;1;1]];
cvFolds = [cvFolds, [1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;0;0;0;0;0;0]];

netAry = {5, 1};
perfAry = zeros(5, 1);

for fold = cvFolds
    trainPts = allPts(~~fold);
    testPts = allPts(~fold);
    
    allFiles = [];
    allLabels = [];
    trainIdx = [];
    testIdx = [];
    
    for i = trainPts
        currFiles = dir(strcat(filepath, 'p', num2str(i), '\noone\*.jpg'));
        allFiles = [allFiles; currFiles];
        allLabels = [allLabels; zeros(length(currFiles), 1)];
        trainIdx = [trainIdx; ones(length(currFiles), 1)];
        testIdx = [testIdx; zeros(length(currFiles), 1)];
        currFiles = dir(strcat(filepath, 'p', num2str(i), '\nurse\*.jpg'));
        allFiles = [allFiles; currFiles];
        allLabels = [allLabels; ones(length(currFiles), 1)];
        trainIdx = [trainIdx; ones(length(currFiles), 1)];
        testIdx = [testIdx; zeros(length(currFiles), 1)];
    end
    for i = testPts
        currFiles = dir(strcat(filepath, 'p', num2str(i), '\noone\*.jpg'));
        allFiles = [allFiles; currFiles];
        allLabels = [allLabels; zeros(length(currFiles), 1)];
        trainIdx = [trainIdx; zeros(length(currFiles), 1)];
        testIdx = [testIdx; ones(length(currFiles), 1)];
        currFiles = dir(strcat(filepath, 'p', num2str(i), '\nurse\*.jpg'));
        allFiles = [allFiles; currFiles];
        allLabels = [allLabels; ones(length(currFiles), 1)];
        trainIdx = [trainIdx; zeros(length(currFiles), 1)];
        testIdx = [testIdx; ones(length(currFiles), 1)];
    end

end
    
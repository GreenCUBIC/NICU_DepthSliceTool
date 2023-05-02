allPts = [1, 2, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16, 17, 21, 22, 23:34, 90:94];
cvFolds = [0;0;0;0;0;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1];
cvFolds = [cvFolds, [1;1;1;1;1;0;0;0;0;0;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1]];
cvFolds = [cvFolds, [1;1;1;1;1;1;1;1;1;1;0;0;0;0;0;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1]];
cvFolds = [cvFolds, [1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;0;0;0;0;0;0;1;1;1;1;1;1;1;1;1;1;1]];
cvFolds = [cvFolds, [1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;0;0;0;0;0;0;1;1;1;1;1]];

for runNum = 1:5
    netAry = {5, 1};
    perfAry = zeros(5, 1);
    foldNum = 1;
    
    TPs = zeros(5, 1);
    FPs = zeros(5, 1);
    FNs = zeros(5, 1);
    TNs = zeros(5, 1);
    
    net_Context = [];
    info_Context = [];
    TP = 0;
    FP = 0;
    FN = 0;
    TN = 0;
    resultTable = [];
    
    
    for fold = cvFolds
        trainPts = allPts(~~fold);
        testPts = allPts(~fold);
        
        allFiles = [];
        allLabels = [];
        trainIdx = [];
        testIdx = [];
        trainFullFiles = cell(length(trainPts), 1);
        testFullFiles = cell(length(testPts), 1);
    
        k = 1;
        for i = trainPts
            fullFile = fullfile('C:', 'Users', 'Zalamaan', 'Documents', 'Repos', 'NICU_DATA', 'DepthFrameFullPrec_prePT', strcat('p', num2str(i)));
            trainFullFiles{k} = fullFile;
            k = k+1;
        end
        train_imds = imageDatastore(trainFullFiles, 'IncludeSubfolders', true, 'FileExtensions', ['.png'], 'LabelSource', 'foldernames');
        k = 1;
        for i = testPts
            fullFile = fullfile('C:', 'Users', 'Zalamaan', 'Documents', 'Repos', 'NICU_DATA', 'DepthFrameFullPrec_prePT', strcat('p', num2str(i)));
            testFullFiles{k} = fullFile;
            k = k+1;
        end
        test_imds = imageDatastore(testFullFiles, 'IncludeSubfolders', true, 'FileExtensions', ['.png'], 'LabelSource', 'foldernames');
        
        [net_Context, info_Context,TP,FP,FN,TN,resultTable] = run_contexts_intervention_D(train_imds, test_imds, foldNum);
    
        TPs(foldNum) = TP;
        TNs(foldNum) = TN;
        FPs(foldNum) = FP;
        FNs(foldNum) = FN;
    
        
        save(strcat('resultTable_FullPrec_prePT_run', string(runNum), '_fold', string(foldNum), '.mat'), 'resultTable');
        foldNum = foldNum + 1;
    end
    
    mat_filename = strcat('run', string(runNum), '_FullPrec_prePT_metrics.mat');
    save(mat_filename, 'TNs', 'TPs', 'FNs', 'FPs');
end
    
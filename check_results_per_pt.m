allPts = [1, 2, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16, 17, 21, 22, 23:28, 29:34];
load('ResultTables\actual_resultTable_pts.mat');

finalTable = zeros(135, 11);
finalTable_counter = 1;

for runNum = 1:5

    resultTable_run = [];
    for foldNum = 1:5
        load(strcat('ResultTables\resultTable_FullPrec_prePT_HHA_run', string(runNum), '_fold', string(foldNum), '.mat'))
        resultTable_run = [resultTable_run; resultTable];
    end
    resultTable_run = [resultTable_run, actual];

    for ptNum = 1:length(allPts)
        idx = [resultTable_run{:, 4}] == allPts(ptNum);
        idx = idx';

        segmented_RT = resultTable_run(idx, :);

        pred_T_idx = [segmented_RT(:, 1)] == "nurse";
        pred_T = segmented_RT(pred_T_idx, :);
        TP_idx = [pred_T(:, 3)] == "nurse";
        FP_idx = [pred_T(:, 3)] == "noone";
        TP = size(pred_T(TP_idx, :));
        TP = TP(1);
        FP = size(pred_T(FP_idx, :));
        FP = FP(1);

        pred_F_idx = [segmented_RT(:, 1)] == "noone";
        pred_F = segmented_RT(pred_F_idx, :);
        TN_idx = [pred_F(:, 3)] == "noone";
        FN_idx = [pred_F(:, 3)] == "nurse";
        TN = size(pred_F(TN_idx, :));
        TN = TN(1);
        FN = size(pred_F(FN_idx, :));
        FN = FN(1);

        sens = TP / (TP + FN);
        spec = TN / (TN + FP);
        prec = TP / (TP + FP);
        acc = (TP + TN) / (TP + TN + FP + FN);
        F1 = (2*TP) / ((2*TP) + FP + FN);

        finalTable(finalTable_counter, 1) = runNum;
        finalTable(finalTable_counter, 2) = allPts(ptNum);
        finalTable(finalTable_counter, 3) = sens;
        finalTable(finalTable_counter, 4) = spec;
        finalTable(finalTable_counter, 5) = prec;
        finalTable(finalTable_counter, 6) = acc;
        finalTable(finalTable_counter, 7) = F1;
        finalTable(finalTable_counter, 8) = TP;
        finalTable(finalTable_counter, 9) = FP;
        finalTable(finalTable_counter, 10) = FN;
        finalTable(finalTable_counter, 11) = TN;
        finalTable_counter = finalTable_counter + 1;
    end
end

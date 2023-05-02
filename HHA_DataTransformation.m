source_path = 'C:\Users\Zalamaan\Documents\Repos\NICU_Data\DepthFrameFullPrec_PT\';
target_path = 'C:\Users\Zalamaan\Documents\Repos\NICU_Data\HHA_Depth_PT\';
intrinsics_path = 'C:\Users\Zalamaan\Documents\Repos\NICU_Data\DepthFrameFullPrec_prePT\';

% allPts = [1, 2, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16, 17, 21:34];
allPts = [90:94];
Pts_data = cell(length(allPts)*2, 6);
cell_num = 0;

addpath('./Depth2HHA');

for i = 1:length(allPts)
    uniqueFileNames = find_unique_file_names(source_path, allPts(i));
    CMatrix = getCameraMatrix(intrinsics_path, allPts(i));
    disp(strcat('Patient ', num2str(allPts(i)), ':'));
    tic
    for file_num = 1:length(uniqueFileNames)
        transformHHA_allFiles(source_path, target_path, allPts(i), uniqueFileNames(file_num), CMatrix);
    end
    toc
end

function [] = transformHHA_allFiles(srcPath, dstPath, patientID, filename, CMatrix)
    noone_folder_path = strcat(srcPath, 'p', num2str(patientID), '\noone\');
    noone_part_files = dir(strcat(noone_folder_path, filename, '_*.png'));
    % noone_main_file = dir(strcat(noone_folder_path, filename, '.png'));
    nurse_folder_path = strcat(srcPath, 'p', num2str(patientID), '\nurse\');
    nurse_part_files = dir(strcat(nurse_folder_path, filename, '_*.png'));
    % nurse_main_file = dir(strcat(nurse_folder_path, filename, '.png'));
    % noone_files = [noone_main_file; noone_part_files];
    % nurse_files = [nurse_main_file; nurse_part_files];
    noone_files = noone_part_files;
    nurse_files = nurse_part_files;

    parfor i = 1:length(noone_files)
        curr_nooneFile = noone_files(i);
        preHHA_filename = char(strcat(curr_nooneFile.folder, '\', curr_nooneFile.name));
        DepthImage = imread(preHHA_filename);
        RawDepthImage = DepthImage == 0;
        RawDepthImage = ~RawDepthImage;
        outDir = char(strcat(dstPath, 'p', num2str(patientID), '\noone'));
        filename = erase(curr_nooneFile.name,'.png')
    
        saveHHA(filename, CMatrix, outDir, DepthImage, RawDepthImage);
    end
    parfor i = 1:length(nurse_files)
        curr_nurseFile = nurse_files(i);
        preHHA_filename = char(strcat(curr_nurseFile.folder, '\', curr_nurseFile.name));
        DepthImage = imread(preHHA_filename);
        RawDepthImage = DepthImage == 0;
        RawDepthImage = ~RawDepthImage;
        outDir = char(strcat(dstPath, 'p', num2str(patientID), '\nurse'));
        filename = erase(curr_nurseFile.name,'.png')
    
        saveHHA(filename, CMatrix, outDir, DepthImage, RawDepthImage);
    end
end

function [unique_file_names] = find_unique_file_names(filepath, patient_id)
%     noone_files = dir(strcat(filepath, 'p', num2str(patient_id), '\noone\*.png'));
%     noone_file_names =  string(extractfield(noone_files, 'name'));
%     noone_file_names_with_ext = noone_file_names(arrayfun(@(x) contains(x,'_0.png'), noone_file_names));
%     unique_noone_file_names = arrayfun(@(x) erase(x,'_0.png'),noone_file_names_with_ext);
    nurse_files = dir(strcat(filepath, 'p', num2str(patient_id), '\nurse\*.png'));
    nurse_file_names =  string(extractfield(nurse_files, 'name'));
    nurse_file_names_with_ext = nurse_file_names(arrayfun(@(x) contains(x,'_0.png'), nurse_file_names));
    unique_nurse_file_names = arrayfun(@(x) erase(x,'_0.png'),nurse_file_names_with_ext);
    unique_file_names = unique_nurse_file_names

%     unique_file_names = unique([unique_noone_file_names, unique_nurse_file_names])
end

function [CMatrix] = getCameraMatrix(srcPath, patientID)
    intrinsicsFile = dir(strcat(srcPath, 'p', num2str(patientID), '\*.txt'));
    intrinsicsFilename = char(strcat(intrinsicsFile.folder, "\", intrinsicsFile.name));
    [scalingFactor, K, distortionModel, D] = parseIntrinsics(intrinsicsFilename);

    ppx = K(3);
    ppy = K(6);
    fx = K(1);
    fy = K(5);

    CMatrix = [fx,  0,   ppx; ...
                0,  fy,  ppy; ...
                0,  0,   1];
end

function [scalingFactor, K, distortionModel, D] = parseIntrinsics(filename)
    intrinsics = fileread(filename);
    newstr = split(intrinsics, '=');
    scalingFactor = splitlines(newstr{2});
    scalingFactor = str2num(scalingFactor{1});
    K = splitlines(newstr{3});
    K = str2num (K{1});
    distortionModel = split(newstr{4}, '"');
    distortionModel = distortionModel{2};
    D = splitlines(newstr{5});
    D = str2num(D{1});
end
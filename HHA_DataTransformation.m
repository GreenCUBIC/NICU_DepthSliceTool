prePT_path = 'C:\Users\Zalamaan\Documents\Repos\depthSliceTool\bagmerge\InterventionDetectionFiles\FirstFrameDepthRGB_origData_prePT\';
PT_path = 'C:\Users\Zalamaan\Documents\Repos\depthSliceTool\bagmerge\InterventionDetectionFiles\FirstFrameDepthRGB_origData_PT\';
allPts = [1, 2, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16, 17, 21:34];
Pts_data = cell(length(allPts)*2, 6);
cell_num = 0;

for i = 1:length(allPts)
    uniqueFileNames = find_unique_file_names(prePT_path, allPts(i));
    for file_num = 1:length(uniqueFileNames)
        cell_num = cell_num + 1;
        Pts_data{cell_num, 1} = allPts(i);
        Pts_data{cell_num, 2} = uniqueFileNames(file_num);
        [Pts_data{cell_num, 3}, Pts_data{cell_num, 4}, Pts_data{cell_num, 5}, Pts_data{cell_num, 6}] = getPTVars(prePT_path, Pts_data{cell_num, 1}, Pts_data{cell_num, 2});
    end
end
% 
% for i = 1:cell_num
%     disp(strcat("Patient Number ", num2str(Pts_data{i, 1})))
%     tic
%     transformPerspective_allFiles(prePT_path, PT_path, Pts_data{i, 1}, Pts_data{i, 2}, Pts_data{i, 3}, Pts_data{i, 4}, Pts_data{i, 5}, Pts_data{i, 6});
%     toc
% end

function [] = transformPerspective_allFiles(srcPath, dstPath, patientID, filename, rotationMatrix, K, D, distortionModel)
    noone_folder_path = strcat(srcPath, 'p', num2str(patientID), '\noone\');
    noone_part_files = dir(strcat(noone_folder_path, filename, '_part_*.jpg'));
    noone_main_file = dir(strcat(noone_folder_path, filename, '.jpg'));
    nurse_folder_path = strcat(srcPath, 'p', num2str(patientID), '\nurse\');
    nurse_part_files = dir(strcat(nurse_folder_path, filename, '_part_*.jpg'));
    nurse_main_file = dir(strcat(nurse_folder_path, filename, '.jpg'));
    noone_files = [noone_main_file; noone_part_files];
    nurse_files = [nurse_main_file; nurse_part_files];

    parfor i = 1:length(noone_files)
        curr_nooneFile = noone_files(i);
        prePT_filename = char(strcat(curr_nooneFile.folder, '\', curr_nooneFile.name));
        PT_filename = char(strcat(dstPath, 'p', num2str(patientID), '\noone\', curr_nooneFile.name));
    
        transformPerspective(prePT_filename, PT_filename, rotationMatrix, K, D, distortionModel);
    end
    parfor i = 1:length(nurse_files)
        curr_nurseFile = nurse_files(i);
        prePT_filename = char(strcat(curr_nurseFile.folder, '\', curr_nurseFile.name));
        PT_filename = char(strcat(dstPath, 'p', num2str(patientID), '\nurse\', curr_nurseFile.name));
    
        transformPerspective(prePT_filename, PT_filename, rotationMatrix, K, D, distortionModel);
    end
end

function [unique_file_names] = find_unique_file_names(filepath, patient_id)
    noone_files = dir(strcat(filepath, 'p', num2str(patient_id), '\noone\*.jpg'));
    noone_file_names =  string(extractfield(noone_files, 'name'));
    noone_file_names_with_ext = noone_file_names(arrayfun(@(x) ~contains(x,'_part_'), noone_file_names));
    unique_noone_file_names = arrayfun(@(x) erase(x,'.jpg'),noone_file_names_with_ext);
    nurse_files = dir(strcat(filepath, 'p', num2str(patient_id), '\nurse\*.jpg'));
    nurse_file_names =  string(extractfield(nurse_files, 'name'));
    nurse_file_names_with_ext = nurse_file_names(arrayfun(@(x) ~contains(x,'_part_'), nurse_file_names));
    unique_file_names = arrayfun(@(x) erase(x,'.jpg'),nurse_file_names_with_ext);
    if ~isempty(unique_noone_file_names)
        for i=1:length(unique_noone_file_names)
            TF = matches(unique_file_names, unique_noone_file_names(i));
            if sum(TF) == 0
                unique_file_names = [unique_file_names, unique_noone_file_names(i)];
            end
        end
    end
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
% Taken from (https://github.com/GreenCUBIC/nicu_psm_research_files/blob/master/matlab_scripts/read_rosbag_files.m)

function [bag_selections, num_bag_selections, total_num_files] = read_rosbag_files(filepath, patient_id, main_file_name, part_num, limit)
%read_rosbag_files read multiple rosbag files from the data storage path
%   filepath is the root folder for all of patient data, eg: 'Z:\'
%   patient_id is the id of the patient
%   part_num is the number from part files to load
%   limit is the maximum number of file to load
%   Eg.  read_rosbag_files('Z:\',1,0,2); reads the first file and part 1
%   from patient 1
%       read_rosbag_files('Z:\',1,34,4); reads part files 34,35,36,37 from patient 1
    rosbag_folder_path = strcat(filepath, 'Patient_', num2str(patient_id), '\Video_Data\');
    part_files = dir(strcat(rosbag_folder_path,main_file_name,'_part_*.bag'));
    main_file = dir(strcat(rosbag_folder_path,main_file_name,'.bag'));
    rosbag_files = [main_file;part_files];
    total_num_files = length(rosbag_files);
    rosbag_file_names =  string(extractfield(rosbag_files, 'name'));
    [~, sorted_index] = sort(arrayfun(@get_part_num, rosbag_file_names));
    sorted_files= rosbag_files(sorted_index);

    
    limit = min(limit - 1, length(sorted_files));
    bag_selections = cell(1,limit);
    num_bag_selections = 0;
    for p = part_num:(part_num + limit)
        if p > length(sorted_files) - 1
            break;
        end
        rosbag_file = sorted_files(p + 1);
        file_path = char(strcat(rosbag_file.folder, "\", rosbag_file.name));
        bag_selections{1,p - part_num + 1} = rosbag(file_path);
        num_bag_selections = num_bag_selections + 1;
    end
    
end

function [part_num] = get_part_num(file_name)
    char_list = char(file_name);
    [start_index, end_index] = regexp(char_list, "part_\d*");
    if isempty(start_index)
        part_num = 0;
    else
        part_num = str2double(char_list(start_index+5:end_index));
    end
end
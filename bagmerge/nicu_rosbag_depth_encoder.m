function [] = nicu_rosbag_depth_encoder(filepath, output_path, PATIENT_IDS, part_start, part_limit)
%NICU_ROSBAG_CONVERTER Converts rosbag files to mp4
%  filepath --> File path of data files per patient, for eg. Z:\
%  output_path --> To write converted files
%  PATIENT_IDS --> List of patient ids
%  MAIN_FILE_NAMES --> List of main video file names

    % Creating the video

    batch_size = 10;
    tic
    parfor i=1:length(PATIENT_IDS)
        unique_file_names = find_unique_file_names(filepath, PATIENT_IDS(i));
        try
            for file_num = 1:length(unique_file_names)
                encode_video(filepath, PATIENT_IDS(i), unique_file_names(file_num), output_path, part_start, part_limit, batch_size, 30);
            end
        catch Error
            errorText = getReport(Error);
            disp(errorText);
        end
    end
    toc

end

function [unique_file_names] = find_unique_file_names(filepath, patient_id)
    rosbag_files = dir(strcat(filepath, 'Patient_', num2str(patient_id), '\Video_Data\*.bag'));
    rosbag_file_names =  string(extractfield(rosbag_files, 'name'));
    unique_file_names_with_ext = rosbag_file_names(arrayfun(@(x) ~contains(x,'_part_'), rosbag_file_names));
    unique_file_names = arrayfun(@(x) erase(x,'.bag'),unique_file_names_with_ext);
end

function [] = encode_video(filepath, patient_id, main_file_name, output_path, part_start, part_end, batch_size, frame_rate)
    video_writer = VideoWriter(char(strcat(output_path, main_file_name, "_depth",".mp4")),'MPEG-4');
    video_writer.FrameRate = frame_rate;
    open(video_writer);
    absolute_video_start_time = NaN;
    encoded_video_sec = 0;
    progress_msg = strcat("Encoding video for patient ", num2str(patient_id), " ... ");
    progress_bar = waitbar(0, progress_msg);
    batch_size = min(part_end - part_start, batch_size);
    [bag_selections, num_bag_selections, total_num_files] = read_rosbag_files(filepath, patient_id, main_file_name, part_start, batch_size);
    while num_bag_selections > 0
     
        for bag_num=1:num_bag_selections
            bag = bag_selections{1,bag_num};
            meta_topic = select(bag, 'Topic', ['/device_0/sensor_0/Depth_0/image/metadata']);  
            data_topic = select(bag, 'Topic', ['/device_0/sensor_0/Depth_0/image/data']);
            num_messages = meta_topic.NumMessages;
            
            %1st message and every 10th meta message is system timestamp
            sys_time_messages = readMessages(meta_topic, 1:10:num_messages);
            time_stamps = cellfun(@(x) str2double(x.Value),sys_time_messages);
           
            if isnan(absolute_video_start_time)
                % Convert unix time_stamp to EST timezone date string
                absolute_video_start_time = time_stamps(1);
                absolute_video_start_time_str = datestr((time_stamps(1) - 5*3600*1000)/(86400 * 1000) + datenum(1970,1,1),'yyyy-mm-dd HH:MM:SS.FFF');
                fileID = fopen(char(strcat(output_path,main_file_name,".txt")),'w');
                fprintf(fileID,'recordingStart=%s\n',absolute_video_start_time_str);
                fclose(fileID);
            end
            
            % downsample frames to frame rate in batches of 1 sec
            % Normalize time stamps by the absolute start time
            
            absolute_time_stamp = time_stamps - absolute_video_start_time;
            absolute_time_stamp_sec = round(absolute_time_stamp/1000);
            
            frame_indices = 1:length(absolute_time_stamp_sec);
            carried_messages = [];
            while encoded_video_sec <= max(absolute_time_stamp_sec)
                frames_for_sec = frame_indices(absolute_time_stamp_sec == encoded_video_sec);
                num_frames_in_sec = length(frames_for_sec);
                
                if ~isempty(carried_messages)
                    additional_messages = readMessages(data_topic, frames_for_sec);
                    merged_messages = [carried_messages;additional_messages];
                    num_merged_messages = length(merged_messages); 
                    merged_frames_indices = 1:num_merged_messages;
                    down_sampled_indices = interp1(merged_frames_indices, 1:num_merged_messages/frame_rate:num_merged_messages, 'nearest');
                    data_messages = merged_messages(down_sampled_indices);
                    carried_messages = [];
                elseif encoded_video_sec == max(absolute_time_stamp_sec) && num_frames_in_sec < frame_rate
                    carried_messages = readMessages(data_topic, frames_for_sec);
                    continue;
                else
                    down_sampled_indices = interp1(frames_for_sec, 1:num_frames_in_sec/frame_rate:num_frames_in_sec, 'nearest');
                    %Now read the actual frames from the data topic
                    data_messages = readMessages(data_topic, down_sampled_indices);
                end
           
                rgbImages = cell(1, length(data_messages));
                parfor j = 1:length(data_messages)
                    message = data_messages{j,1};
%                     disp(message)
                    % Combining all channels to obtain the rgb image
                    rgbImages{1,j} = message_to_rgbimage(message);               
                end
                
                for j = 1:length(rgbImages)
                    writeVideo(video_writer, rgbImages{1,j});               
                end
                
                encoded_video_sec = encoded_video_sec + 1;
            end
             
        end  
        part_start =  part_start + num_bag_selections;
        if part_start < part_end
            [bag_selections, num_bag_selections] = read_rosbag_files(filepath, patient_id, main_file_name, part_start, batch_size);
            waitbar(part_start/min(part_end,total_num_files), progress_bar, strcat(progress_msg, num2str(part_start),"/", num2str(total_num_files)));
        else
            break;
        end
    end
    close(progress_bar);
    close(video_writer);
end


function [rgbImage] = message_to_rgbimage(message)

    col = message.Width;
    row = message.Height;
       
    img = readImage(message);
    
    % Initializing all channels
    redChannel = zeros(row,col,'uint8');

    high = bitshift(img, -8);
    greenChannel = uint8(high);
    blueChannel = uint8(img - bitshift(high, 8));

    rgbImage = cat(3, redChannel, greenChannel, blueChannel);
end


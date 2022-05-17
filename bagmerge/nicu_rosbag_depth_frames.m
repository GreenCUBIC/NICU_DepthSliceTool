function [] = nicu_rosbag_depth_frames(filepath, output_folder_path, PATIENT_IDS, part_start)
    batch_size = 25;
    part_end = 1000;
    tic
    parfor i=1:length(PATIENT_IDS)
        unique_file_names = find_unique_file_names(filepath, PATIENT_IDS(i));
        disp(unique_file_names);
        try
            for file_num = 1:length(unique_file_names)
                save_rgb_images(filepath, PATIENT_IDS(i), unique_file_names(file_num), output_folder_path, part_start, part_end, batch_size);
            end
        catch Error
            errorText = getReport(Error);
            disp(errorText);
        end
    end
end

function [unique_file_names] = find_unique_file_names(filepath, patient_id)
    rosbag_files = dir(strcat(filepath, 'Patient_', num2str(patient_id), '\Video_Data\*.bag'));
    rosbag_file_names = string(extractfield(rosbag_files, 'name'));
    unique_file_names_with_ext = rosbag_file_names(arrayfun(@(x) ~contains(x, '_part_'), rosbag_file_names));
    unique_file_names = arrayfun(@(x) erase(x, '.bag'), unique_file_names_with_ext);
end

function [] = save_rgb_images(filepath, patient_id, main_file_name, output_folder_path, part_start, part_end, batch_size)
    disp("started save_rgb_images");
    batch_size = min(part_end - part_start, batch_size);
    [bag_selections, num_bag_selections, total_num_files] = read_rosbag_files(filepath, patient_id, main_file_name, part_start, batch_size);
    image_idx = 0;
    while(num_bag_selections) > 0
        
        for bag_num=1:num_bag_selections
            bag = bag_selections{1, bag_num};
            % meta_topic = select(bag, 'Topic', ['/device_0/sensor_0/Depth_0/image/metadata']);  
            data_topic = select(bag, 'Topic', ['/device_0/sensor_0/Depth_0/image/data']);
            % num_messages = meta_topic.NumMessages;

            third_frame_message = readMessages(data_topic, [3]);
            rgb_image = message_to_rgbimage(third_frame_message{1, 1});
            imshow(rgb_image);
            imwrite(rgb_image, strcat(output_folder_path, "p", num2str(patient_id), "\", main_file_name, "_", num2str(image_idx), ".png"));
            image_idx = image_idx + 1;
        end
        
        
        part_start =  part_start + num_bag_selections;
        if part_start < part_end
            [bag_selections, num_bag_selections] = read_rosbag_files(filepath, patient_id, main_file_name, part_start, batch_size);
        else
            break;
        end
    end
end

function [rgb_image] = message_to_rgbimage(message)
    col = message.Width;
    row = message.Height;
       
    img = readImage(message);
        
    % Initializing all channels
    red_channel = zeros(row,col,'uint8');

    high = bitshift(img, -8);
    green_channel = uint8(high);
    blue_channel = uint8(img - bitshift(high, 8));

    rgb_image = cat(3, red_channel, green_channel, blue_channel);
end

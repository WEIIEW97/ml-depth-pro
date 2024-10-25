function save_rgb_images_as_binary(path)
    %SAVE_RGB_IMAGES_AS_BINARY Saves red channel of RGB images to binary files.
    %   This function reads RGB images from specified paths, extracts the red
    %   channel, and saves it to a .binary file as uint8.
    
    % Define the paths for the input images
    left_path = "/home/william/extdisk/data/case1/left.png";
    right_path = "/home/william/extdisk/data/case1/right.png";
    
    % Read the RGB images
    left_img = imread(left_path);
    right_img = imread(right_path);
    
    % Extract the red channel from each image
    left_red = left_img(:,:,1);
    right_red = right_img(:,:,1);
    
    % Define the paths for saving the binary files
    save_left_path = "/home/william/extdisk/data/case1/left.raw";
    save_right_path = "/home/william/extdisk/data/case1/right.raw";
    
    % Save the red channel to binary files
    write_to_binary_file(save_left_path, left_red);
    write_to_binary_file(save_right_path, right_red);
    disp("done!");
end

function write_to_binary_file(filename, data)
    %WRITE_TO_BINARY_FILE Writes matrix data to a binary file.
    %   This function takes a filename and a matrix 'data', and writes the data
    %   to the file specified by 'filename' as type uint8.
    
    file_id = fopen(filename, 'w');
    
    % Check if the file was opened successfully
    if file_id == -1
        error('Error opening file %s for writing.', filename);
    end
    
    fwrite(file_id, data, 'uint8');
    
    fclose(file_id);
end

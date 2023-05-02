addpath('./Depth2HHA');
img = dir("conventionalMethod/8bitDepth.png");
CMat = getCameraMatrix('C:\Users\Zalamaan\Documents\Repos\NICU_Data\DepthFrameFullPrec_prePT\', 29);

preHHA_filename = char(strcat(img.folder, '\', img.name));
DepthImage = imread(preHHA_filename);
RawDepthImage = DepthImage == 0;
RawDepthImage = ~RawDepthImage;
filename = 'HHAFILE.png';

saveHHA(filename, CMat, './', DepthImage, RawDepthImage);

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





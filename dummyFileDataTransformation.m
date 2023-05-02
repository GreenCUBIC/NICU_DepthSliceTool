prePT_path = 'C:\Users\Zalamaan\Documents\Repos\depthSliceTool\bagmerge\InterventionDetectionFiles\FirstFrameDepthRGB_origData_prePT\';
PT_path = 'C:\Users\Zalamaan\Documents\Repos\depthSliceTool\bagmerge\InterventionDetectionFiles\FirstFrameDepthRGB_origData_PT\';
allPts = [1, 2, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16, 17, 21:34];

img = imread('nurse/Patient6_part_99_D.jpg');
intrinsicsFile = 'Patient6_intrinsics.txt';
[scalingFactor, K, distortionModel, D] = parseIntrinsics(intrinsicsFile);

fig = imshow(img);

PTpoints = zeros(4, 3);
PTpixels = zeros(4, 3);
for i = 1:4
    [x, y] = getpts;
    x = uint16(x);
    y = uint16(y);
    depth = uint16(img(y, x));
    PTpixels(i,:) = [x, y, depth];
    [point] = deprojectPixelToPoint(double(x), double(y), double(depth), K, D, distortionModel);
    PTpoints(i,:) = point;
end

close();

tic

[rotationMatrix, fulcrumPixel_idx] = calculateRotationMatrix(PTpoints);

maxCropX = max(PTpixels(:,1)) + 1;
maxCropY = max(PTpixels(:,2)) + 1;
minCropX = min(PTpixels(:,1));
minCropY = min(PTpixels(:,2));

vertsNum = length(img(:,1))*length(img(1,:));
verts = zeros(vertsNum, 3);
i = 1;

for y = 1:length(img(:,1))
    for x = 1:length(img(1,:))
        depth = img(y, x);
        [point] = deprojectPixelToPoint(double(x), double(y), double(depth), K, D, distortionModel);
        verts(i,:) = point;
        i = i + 1;
    end
end

verts_transformed = (rotationMatrix * verts')';
verts_transformed(~any(verts_transformed, 2), :) = [];

depthFrame_transformed = zeros(1080,1920);

for i=1:length(verts_transformed)
    pixel = projectPointToPixel(verts_transformed(i,:), K, D, distortionModel);
    if pixel(1) < 960 && pixel(2) < 540 && pixel(1) >= -960 && pixel(2) >= -540
        depthFrame_transformed(int16(fix(pixel(2)+540)+1),int16(fix(pixel(1)+960)+1)) = verts_transformed(i,3);
    end
end

depthFrame_transformed(~any(depthFrame_transformed, 2), :) = [];
depthFrame_transformed(:, ~any(depthFrame_transformed, 1)) = [];

toc

imshow(depthFrame_transformed)

function [] = transformPerspective(srcPath, dstPath, PatientID)
    
end

function [unique_file_names] = find_unique_file_names(filepath, patient_id)
    noone_files = dir(strcat(filepath, 'p', num2str(patient_id), '\noone\*.jpg'));
    noone_file_names =  string(extractfield(noone_files, 'name'));
    noone_file_names_with_ext = noone_file_names(arrayfun(@(x) ~contains(x,'_part_'), noone_file_names));
    unique_file_names = arrayfun(@(x) erase(x,'.jpg'),noone_file_names_with_ext);
end

function [rotationMatrix, fulcrumPixel_idx] = calculateRotationMatrix(PTpoints)
    rMatrices = cell(1, 4);
    tpDiffs = cell(1, 4);
    tpComparision = cell(1, 4);
    for pointIndex = 1:4
        vAB_idx = mod(pointIndex, 4) + 1;
        vAC_idx = mod(pointIndex + 2, 4) + 1;
        vAB = PTpoints(vAB_idx, :) - PTpoints(pointIndex, :);
        vAC = PTpoints(vAC_idx, :) - PTpoints(pointIndex, :);
        normalVector = cross(vAB, vAC);
        normalVector = normalVector / norm(normalVector);
        newNormal = [0, 0, -1];
        rAxis = cross(normalVector, newNormal);
        rAxis = rAxis / norm(rAxis);
        rAngle = acos(dot(normalVector, newNormal));
        rAxisCMatrix = [0, -rAxis(3), rAxis(2);...
                        rAxis(3), 0, -rAxis(1);...
                        -rAxis(2), rAxis(1), 0];
        rotationMatrix = (cos(rAngle)*eye(3)) + ((sin(rAngle)*rAxisCMatrix) +((1-cos(rAngle))*(rAxis' * rAxis)));
        rMatrices{pointIndex} = rotationMatrix;


        testPoints = (rotationMatrix * PTpoints')';
        testPointDiff = testPoints(mod(pointIndex + 1, 4) + 1, 2) - testPoints(pointIndex, 2);
        tpComparision{pointIndex} = testPoints(pointIndex, 2);
        tpDiffs{pointIndex} = abs(testPointDiff);
    end
        
    [tpDiff, tpDiff_idx] = min([tpDiffs{:}]);
    fulcrumPixel_idx = mod(tpDiff_idx + 2, 4) + 1;
    rotationMatrix = rMatrices{tpDiff_idx};
end

function [pixel] = projectPointToPixel(point, K, coeffs, model)
    ppx = K(3);
    ppy = K(6);
    fx = K(1);
    fy = K(5);
    pixel = [0, 0];

    x = point(1) / point(3);
    y = point(2) / point(3);

    if model == "Modified Brown Conrady"
        r2 = (x * x) + (y * y);
        f = 1 + (coeffs(1) * r2) + (coeffs(2) * r2 * r2) + (coeffs(5) * r2 * r2 * r2);
        x = f * x;
        y = f * y;
        dx = x + (2 * coeffs(3) * x * y) + (coeffs(4) * (r2 + 2 * x*x));
        dy = y + (2 * coeffs(4) * x * y) + (coeffs(3) * (r2 + 2 * y*y));
        x = dx;
        y = dy;
    elseif model == "F-Theta"
        r = sqrt(x*x + y*y);
        rd = (1.0 / coeffs(1) * atan(2 * r * tan(coeffs(1) / 2.0)));
        x = x * rd / r;
        y = y * rd / r;
    end
    pixel(1) = x * fx + ppx;
    pixel(2) = y * fy + ppy;
end

function [point] = deprojectPixelToPoint(pixelX, pixelY, pixelDepth, K, coeffs, model)
    ppx = K(3);
    ppy = K(6);
    fx = K(1);
    fy = K(5);
    point = [0, 0, 0];

    x = (pixelX - ppx) / fx;
    y = (pixelY - ppy) / fy;

    if model == "Inverse Brown Conrady"
        r2 = (x * x) + (y * y);
        f = 1 + (coeffs(1) * r2) + (coeffs(2) * r2 * r2) + (coeffs(5) * r2 * r2 * r2);
        ux = (x * f) + (2 * coeffs(3) * x * y) + (coeffs(4) * (r2 + 2 * x*x));
        uy = (y * f) + (2 * coeffs(4) * x * y) + (coeffs(3) * (r2 + 2 * y*y));
        x = ux;
        y = uy;
    end

    point(1) = pixelDepth * x;
    point(2) = pixelDepth * y;
    point(3) = pixelDepth;
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
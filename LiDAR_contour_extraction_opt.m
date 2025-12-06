function Edge_Lidar_contour3 = LiDAR_contour_extraction_opt(BW2, pt_2d__)
%LIDAR_CONTOUR_EXTRACTION_OPT
% Extract LiDAR projected contour points from a binary projection image.
%
% This function takes a binary projection map of LiDAR points and identifies
% which projected pixels lie on the contour of the object by combining Sobel
% and Canny edge detectors. It then returns only those LiDAR 2D projected
% points that coincide with detected image edges.
%
% ------------------------------ Corresponds to ------------------------------
% CASA-Calib paper, Algorithm 2 (CASA-Loss computation), Step:
%    “Extract projected LiDAR contour C_lidar in the image plane”
% ----------------------------------------------------------------------------
%
% INPUTS:
%   BW2       : H×W logical mask of LiDAR-projected pixels (true = projected)
%   pt_2d__   : N×2 integer pixel coordinates of LiDAR projections [u, v]
%
% OUTPUT:
%   Edge_Lidar_contour3 : M×2 array of LiDAR-projected pixels lying on edges
%
% Author: Yuan-Ting Fu
% CASA-Calib
% ---------------------------------------------------------------------------

%% ---- Morphological filtering to obtain cleaner blobs ----
se  = strel('disk', 20);
BW3 = imclose(BW2, se);
BW4 = imfill(BW3, 'holes');

%% ---- Edge detection: Sobel + Canny ----
edge_map1 = edge(BW4, 'sobel');
edge_map2 = edge(BW4, 'canny');
edge_combined = edge_map1 | edge_map2;

%% ---- Extract edge pixel coordinates ----
[edge_y, edge_x] = find(edge_combined);     % row = y, col = x
edge_pixels = [edge_x, edge_y];            % convert to [u, v]

%% ---- Match LiDAR-projected points to edge pixels ----
[is_in, ~] = ismember(pt_2d__, edge_pixels, 'rows');
Edge_Lidar_contour3 = pt_2d__(is_in, :);

%% ---- Remove duplicates ----
Edge_Lidar_contour3 = unique(Edge_Lidar_contour3, 'rows');

end

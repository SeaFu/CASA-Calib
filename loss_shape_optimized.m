function [iou, arc_similarity, dist_CC, weight_contour, area_ratio] = ...
    loss_shape_optimized(img_contour_seq, lidar_contour_seq)
%LOSS_SHAPE_OPTIMIZED
% Compute IoU, centroid consistency, and shape similarity between
% the image contour and LiDAR contour.
%
% The LiDAR and image contours are first aligned by matching their
% angular coordinate φ (third column), then represented as polygonal
% curves to compute IoU and centroid distance.
%
% ------------------------------ Corresponds to ------------------------------
% CASA-Calib paper, Algorithm 2 (CASA-Loss computation), Steps:
%   Step: “Align contours by angle (φ)”
%   Step: “Compute shape consistency terms: IoU, centroid distance”
% ----------------------------------------------------------------------------
%
% INPUTS:
%   img_contour_seq   : Nx4 ordered image contour [u, v, φ, r]
%   lidar_contour_seq : Mx4 ordered LiDAR contour [u, v, φ, r]
%
% OUTPUTS:
%   iou             : Intersection-over-Union between two polygons
%   arc_similarity  : Difference in polygon enclosed areas
%   dist_CC         : Euclidean distance between centroids
%   weight_contour  : auxiliary weight based on area ratio
%   area_ratio      : area(lidar polygon) / area(image polygon)
%
% Author: Yuan-Ting Fu
% CASA-Calib
% ---------------------------------------------------------------------------

%% ---- 1. Match image contour to LiDAR by φ (angular coordinate) ----
idx = knnsearch(img_contour_seq(:,3), lidar_contour_seq(:,3));
sampled = img_contour_seq(idx, :);

%% ---- 2. Build polygon loops (closed contours) ----
PI_cart = [sampled(:,1:2); sampled(1,1:2)];
PV_cart = [lidar_contour_seq(:,1:2); lidar_contour_seq(1,1:2)];

polyI = polyshape(PI_cart, 'Simplify', true);
polyV = polyshape(PV_cart, 'Simplify', true);

%% ---- 3. Compute area + centroid consistency ----
Ai = area(polyI);
[ci_x, ci_y] = centroid(polyI);
ci = [ci_x, ci_y];

Av = area(polyV);
[cv_x, cv_y] = centroid(polyV);
cv = [cv_x, cv_y];

dist_CC = norm(ci - cv);

%% ---- 4. Compute IoU of two polygons ----
inter_poly = intersect(polyI, polyV);
union_poly = union([polyI, polyV]);

iou = area(inter_poly) / max(area(union_poly), eps);

%% ---- 5. Shape area similarity ----
arc_similarity = abs(Ai - Av);
area_ratio     = Av / max(Ai, eps);

% simple contour weight (not used directly in final CASA-loss)
weight_contour = double(area_ratio > 1) * 0.1;

end

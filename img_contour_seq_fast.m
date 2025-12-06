function seq = img_contour_seq_fast(mask_index)
%IMG_CONTOUR_SEQ_FAST  Order unordered 2D contour pixels into a continuous sequence.
%
% This function converts a set of unordered contour pixels obtained from an
% image mask into an ordered 1D contour sequence. The ordered sequence is used
% by CASA-Loss to compute shape-alignment terms such as SDS-1D, SDS-2D,
% IoU-based weighting, and centroid consistency.
%
% The ordering is done in two stages:
%   (1) Coarse ordering by polar angle (phi) around the contour centroid.
%   (2) Refinement via greedy nearest-neighbor chaining to ensure spatial
%       continuity along the contour.
%
% This operation corresponds to the "SequenceContour(·)" step used in the
% CASA-Loss pipeline (Algorithm 2 in the paper).
%
% -------------------------------------------------------------------------
% INPUT
%   mask_index : Nx2 array
%       Pixel coordinates of contour points, each row = [u, v].
%       These points are typically extracted from a binary mask using
%       edge detection (e.g., MATLAB 'edge' function).
%
% OUTPUT
%   seq : Nx4 array
%       Ordered contour sequence, each row:
%           [u, v, phi, r]
%       where:
%           (u, v) = pixel coordinates,
%           phi    = polar angle around centroid (0–2π),
%           r      = radial distance from centroid.
%
% The output ensures that consecutive rows correspond to spatially adjacent
% contour points, forming a continuous curve suitable for shape comparison.
%
% -------------------------------------------------------------------------
% Author: Yuan-Ting Fu
% Project: CASA-Calib
% -------------------------------------------------------------------------


%% ----------------------- 1. Convert to polar coordinates -----------------------
% Compute contour centroid
cent = mean(mask_index, 1);

% Shift points to centroid and convert to polar
coords = mask_index - cent;
[phi, r] = cart2pol(coords(:,1), coords(:,2));

% Normalize phi to [0, 2π)
phi = mod(phi, 2*pi);

% Combine into augmented contour array
contour_aug = [mask_index, phi, r];


%% ----------------------- 2. Initial ordering by angular sorting ----------------
% Sorting by polar angle gives a first approximation of contour ordering.
[~, order] = sort(phi);
sorted = contour_aug(order, :);


%% ----------------------- 3. Greedy nearest-neighbor refinement -----------------
% After angle sorting, refine by building a nearest-neighbor chain.
N = size(sorted, 1);
seq = zeros(N, 4);

% Start with the first point
seq(1,:) = sorted(1,:);
remaining = sorted(2:end, :);
count = 1;

% Iteratively append the nearest remaining contour point
while ~isempty(remaining)
    last_pt = seq(count, 1:2);  % [u, v] of last point in the sequence

    % Compute Euclidean distance to all remaining points
    dists = hypot(remaining(:,1) - last_pt(1), remaining(:,2) - last_pt(2));

    % Choose the closest one
    [~, idx] = min(dists);

    count = count + 1;
    seq(count,:) = remaining(idx, :);

    % Remove the chosen point
    remaining(idx, :) = [];
end

end

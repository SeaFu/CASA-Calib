function [SDS_1d, SDS_2d, pt_1D, pt_2D] = loss_proj(pt_2d_seq, img_contour_seq, img_raw)
%LOSS_PROJ
% Compute 1D and 2D Signed Distance Scores (SDS) between the LiDAR-projected
% contour sequence and the image contour sequence.
%
% This function assigns each LiDAR contour point a local distribution of nearby
% image contour points, determines whether the local distribution resembles
% a 1D (linear) or 2D (compact) pattern, and computes SDS-1D or SDS-2D
% accordingly.
%
% ------------------------------ Corresponds to ------------------------------
% CASA-Calib paper, Algorithm 2 (CASA-Loss computation), Step:
%   “Compute local geometric consistency via SDS-1D and SDS-2D”
% ----------------------------------------------------------------------------
%
% INPUTS:
%   pt_2d_seq      : LiDAR contour sequence (N×2 or N×4, first 2 = [u, v])
%   img_contour_seq: Image contour sequence (N×4: [u, v, φ, r])
%   img_raw        : Raw grayscale/RGB image (optional, used for visualization)
%
% OUTPUTS:
%   SDS_1d : list of 1D distance scores (points that lie on line-like areas)
%   SDS_2d : list of 2D distribution scores (points in blob-like areas)
%   pt_1D  : LiDAR points classified as 1D structure
%   pt_2D  : LiDAR points classified as 2D structure
%
% Author: Yuan-Ting Fu
% CASA-Calib
% ---------------------------------------------------------------------------

pt_2d = pt_2d_seq(:,1:2);
match_list = img_contour_seq(:,1:2);  % [u, v]

phi_list = img_contour_seq(:,3);
r_list   = img_contour_seq(:,4);

SDS_1d = [];
SDS_2d = [];
pt_1D  = [];
pt_2D  = [];

% number of neighbors to sample around matched image contour point
sample_N = floor(length(img_contour_seq) / 70);

for i = 1:size(pt_2d,1)

    %% ---- (1) Find nearest image-contour point to this LiDAR point ----
    dxy = match_list - pt_2d(i,:);
    dL1 = abs(dxy(:,1)) + abs(dxy(:,2));
    [~, min_idx] = min(dL1);

    %% ---- (2) Extract a local window of image contour pixels ----
    win_idx = (-sample_N:sample_N) + min_idx;

    % periodic wrap-around (contour is a closed loop)
    win_idx(win_idx < 1) = win_idx(win_idx < 1) + length(match_list);
    win_idx(win_idx > length(match_list)) = win_idx(win_idx > length(match_list)) - length(match_list);

    neighborhood = match_list(win_idx, :);

    %% ---- (3) Statistical distribution of local neighborhood ----
    mu  = mean(neighborhood);
    C   = cov(neighborhood);
    [~, S, ~] = svd(C);
    eig_vals = diag(S);

    % check 1D-like distribution (one eigenvalue >> the other)
    is_1D = max(eig_vals(1)/eig_vals(2), eig_vals(2)/eig_vals(1)) > 1000;

    %% ---- (4) Compute SDS-1D or SDS-2D ----
    if is_1D
        % fit a line to the neighborhood
        line_eq = polyfit(neighborhood(:,1), neighborhood(:,2), 1);
        normal_vec = [line_eq(1), -1];                 % line normal vector
        d = ([pt_2d(i,:), 1] * [normal_vec, line_eq(2)]') / norm(normal_vec);

        SDS_1d = [SDS_1d; abs(d)];
        pt_1D  = [pt_1D; pt_2d(i,:)];
        SDS_2d = [SDS_2d; 0];

    else
        % 2D Mahalanobis distance
        diff   = pt_2d(i,:) - mu;
        d2     = 1 - exp(-0.0001 * diff / C * diff');
        SDS_2d = [SDS_2d; d2];
        pt_2D  = [pt_2D; pt_2d(i,:)];
        SDS_1d = [SDS_1d; 0];
    end

end
end

function Loss = CASA_Loss(theta_hat, t_hat, K, PL_features, M_features, img_raws)
% CASA_LOSS  Multi-frame CASA-Loss for camera–LiDAR calibration.
%
%   This function implements the full Context-Aware Semantic Alignment Loss
%   (CASA-Loss) described in the CASA-Calib paper:
%
%       CASA-Calib: Context-Aware Semantic Alignment for
%       Camera–LiDAR Calibration
%
%   CASA-Loss evaluates the geometric & semantic alignment between the
%   projected LiDAR contour and the image contour using:
%
%       • Global IoU similarity (L_IoU)
%       • Centroid consistency (L_CC)
%       • Local Semantic Distribution Similarity
%            - SDS_Linear (line-like local neighborhoods)
%            - SDS_Elliptical (area-like neighborhoods)
%       • IoU-guided coupling α(L_IoU)
%
%   Multi-frame CASA-Loss is computed by summing the per-frame similarity:
%
%       L_total = Σ_i L_CASA(frame_i)
%
%   NOTE:
%       CASA_Loss returns a *similarity* score (larger = better).
%       For minimization-based optimizers, define:
%
%           J = -CASA_Loss(...)
%
%       to convert it into a cost.
%
% -------------------------------------------------------------------------
% INPUTS:
%   theta_hat   : 1×3 rotation vector (rad), axis-angle, LiDAR→Camera
%   t_hat       : 3×1 translation vector (m), LiDAR→Camera
%   K           : 3×3 camera intrinsic matrix
%   PL_features : {N_i×3} LiDAR point clouds for each frame
%   M_features  : {H×W} image mask for each frame (vehicle segmentation)
%   img_raws    : {H×W×3 or H×W} raw images (optional)
%
% OUTPUT:
%   Loss        : scalar multi-frame CASA-Loss (larger is better)
%
% -------------------------------------------------------------------------
% Author: Yuan-Ting Fu
% Project: CASA-Calib — Context-Aware Semantic Alignment for Camera–LiDAR Calibration
% -------------------------------------------------------------------------
%
% DEPENDENCIES (internal functions):
%   • img_contour_seq_fast.m        – image contour sequencing
%   • LiDAR_contour_extraction_opt.m – LiDAR contour extraction in image
%   • loss_proj.m                    – SDS-Linear / SDS-Elliptical computation
%   • loss_shape_optimized.m         – IoU + centroid consistency
%
% -------------------------------------------------------------------------

%% ========================== Hyperparameters ================================
Z_EPS      = 1e-6;    % minimum positive depth
MARGIN     = 0.0;     % pixel boundary tolerance
OOB_THR    = 0.0;     % allowed out-of-bounds ratio
J_OOB      = 1e6;     % early-exit penalty
USE_SMOOTH = false;   % smooth penalty for OOB (optional)

%% ============================= Setup ======================================
R = rotationVectorToMatrix(theta_hat);
t = t_hat(:);

K33   = K;
K_aug = [K33, zeros(3,1)];

Loss_all = 0;
num_frames = numel(PL_features);

%% ============================ Main Loop ==================================
for i = 1:num_frames
    PL = PL_features{i};
    M  = M_features{i};
    img_raw = img_raws{i}; %#ok<NASGU>

    [H, W] = size(M);

    %% Step 1 — 3D→2D projection + filtering
    XcYcZc = R * PL.' + t;
    Zc     = XcYcZc(3,:);

    finite_mask = all(isfinite(XcYcZc), 1) & isfinite(Zc);
    zpos_mask   = Zc > Z_EPS;
    keep        = finite_mask & zpos_mask;

    if ~any(keep), continue; end

    Xc = XcYcZc(1, keep);
    Yc = XcYcZc(2, keep);
    Zk = Zc(keep);

    fx = K33(1,1); fy = K33(2,2);
    cx = K33(1,3); cy = K33(2,3);

    invZ = 1 ./ max(Zk, Z_EPS);
    u = fx * (Xc .* invZ) + cx;
    v = fy * (Yc .* invZ) + cy;

    %% Step 2 — Out-of-bounds rejection
    inW = (u >= 1-MARGIN) & (u <= W+MARGIN);
    inH = (v >= 1-MARGIN) & (v <= H+MARGIN);
    oob_ratio = 1 - mean(inW & inH);

    if ~USE_SMOOTH
        if oob_ratio > OOB_THR
            Loss = -J_OOB;   % stop and return a bad score
            return;
        end
    else
        Loss_all = Loss_all - J_OOB * max(0, oob_ratio-OOB_THR)^2;
    end

    %% Step 3 — Homogeneous projection (LiDAR→image)
    PL_h = [PL, ones(size(PL,1),1)];
    T_lr = [R, t; 0 0 0 1];
    pt = (K_aug * T_lr * PL_h.').';
    pt = [ pt(:,1)./max(pt(:,3),Z_EPS), ...
           pt(:,2)./max(pt(:,3),Z_EPS) ];

    %% Step 4 — Image contour extraction
    edge_map = edge(M);
    [cy, cx] = find(edge_map > 0);
    contour_pixels = [cx, cy];   % [u, v]

    C_img = img_contour_seq_fast(contour_pixels);
    if isempty(C_img), continue; end

    %% Step 5 — LiDAR contour extraction
    pt_pix = round(pt);

    valid = pt_pix(:,1)>=1 & pt_pix(:,1)<=W & ...
            pt_pix(:,2)>=1 & pt_pix(:,2)<=H;

    BW2 = false(H,W);
    if any(valid)
        ind = sub2ind([H,W], pt_pix(valid,2), pt_pix(valid,1));
        BW2(ind) = true;
    end

    C_lid_raw = LiDAR_contour_extraction_opt(BW2, pt_pix);
    if isempty(C_lid_raw), continue; end

    C_lid = img_contour_seq_fast(C_lid_raw);
    if isempty(C_lid), continue; end

    %% Step 6 — SDS Loss (local distribution similarity)
    [S1, S2] = loss_proj(C_lid, C_img, img_raw);

    S1 = min(abs(S1), 10);
    S2 = min(S2, 1.5);

    SDS_lin = mean(exp(-0.05 * S1));
    SDS_ell = mean(exp(-2.5  * S2));

    %% Step 7 — IoU, centroid consistency (global terms)
    [iou, arcS, dCC, wC, aR] = ...
        loss_shape_optimized(C_img, C_lid); %#ok<ASGLU>

    iou_score = exp(-0.5 * (iou - 0.95)^2 / (0.038^2));
    CC_score  = exp(-0.1 * abs(dCC));

    %% Step 8 — IoU-guided coupling
    tau = 0.8;  k = 10;
    alpha = max(0, 1 - exp(-k*(iou_score - tau)));

    %% Step 9 — Final per-frame CASA-Loss
    L_frame = iou_score + alpha*SDS_lin + alpha*SDS_ell + CC_score;
    if ~isfinite(L_frame), L_frame = 0; end

    Loss_all = Loss_all + L_frame;
end

%% Return total multi-frame CASA-Loss
Loss = Loss_all;

end

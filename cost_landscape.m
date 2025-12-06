% =======================================================================
% CASA-Calib: Cost Landscape Analysis for Camera–LiDAR Extrinsic Calibration
%
% This script computes the 2D cost landscape J(Δty, Δtz) around the
% ground-truth camera–LiDAR extrinsic translation for a selected
% tfrecord segment, and visualizes:
%   - 3D surface of the cost landscape
%   - 2D contour map with three landscape metrics:
%       * d*      : distance between ground truth and empirical minimizer
%       * FWHM_tz : full width at half maximum along Δtz
%       * FWHM_ty : full width at half maximum along Δty
%       * A_eps   : attraction-basin area under a relative threshold
%
% The landscape is explicitly computed as the *frame-wise average* of the
% per-frame cost across all valid frames of the selected tfrecord.
% This reproduces the averaged cost surfaces shown in Fig. 7 of the paper.
%
% -----------------------------------------------------------------------
% Usage
%   1. Set DATA_ROOT and SHEET_PATH to your prepared Waymo-based dataset.
%   2. Set METHOD  = 'CASA' or 'CHAO' (depending on the cost you want).
%   3. Set target_idx to select which tfrecord in the summary CSV to use.
%   4. Run this script. It will:
%       - load all frames belonging to the chosen tfrecord,
%       - compute the frame-averaged cost landscape on a (Δty, Δtz) grid,
%       - output metrics as CSV,
%       - optionally save the 3D and 2D plots as PNG and PDF.
%
% -----------------------------------------------------------------------
% Dependencies (expected to be on MATLAB path)
%   - multi_frame_my_cost(...)   : CASA-based multi-frame cost (user code)
%   - multi_frame_cost_chao(...) : CHAO-based multi-frame cost (user code)
%   - Preprocessed files under DATA_ROOT organized as:
%       DATA_ROOT / <seq> / <tfrecord> / <frame> / pixel_file
%       DATA_ROOT / <seq> / <tfrecord> / <frame> / lidar_file
%       DATA_ROOT / <seq> / <tfrecord> / <frame> / copyy_chao.mat
%       DATA_ROOT / <seq> / <tfrecord> / <frame> / calib.txt
%
% -----------------------------------------------------------------------
% Inputs (high-level)
%   - pair_summary_XX_YY_deduplicated.csv:
%       * columns include at least:
%           - tfrecord_name
%           - seq
%           - frame_id
%           - pixel_file_name
%           - lidar_file_name
%
% Outputs
%   - CSV: metrics_<method>_<mode>_<tfrecord>.csv
%   - Figures:
%       - landscape3D_<method>_<mode>_<tfrecord>.(png|pdf)
%       - landscape2D_<method>_<mode>_<tfrecord>.(png|pdf)
%
% This file is intended to be public (e.g., on GitHub) as part of the
% CASA-Calib reproducibility package.
% =======================================================================

clear; close all; clc; format long; warning('off');

%% [A] Paths and I/O configuration

addpath('../');                    % user cost / optimizer code
addpath('../../../matlab_code');   % shared utility code

DATA_ROOT  = '../../../waymo_segment_data/';
SHEET_PATH = fullfile(DATA_ROOT, 'pair_summary_00_03_deduplicated.csv');

OUT_ROOT   = 'CHAO_output';        % main output folder
METHOD     = 'CHAO';               % 'CASA' or 'CHAO'
target_idx = 32;                   % index into all_TF (unique tfrecords)

%% [B] Plot and computation settings

landscape_mode   = 'avg';      % 'avg' (multi-frame average) or 'single'
single_frame_idx = 1;          % only used when landscape_mode = 'single'

grid_range       = -1:0.1:1;   % scan range (m) for Δty / Δtz
levels_contour   = 20;         % number of contour levels
eps_percent      = 0.10;       % A_eps threshold: percentage of dynamic range

show_figure      = true;       % true: show figures; false: only save to disk
save_png_pdf     = false;      % true to export PNG and PDF

% =======================================================================
%                      Code below rarely needs to be changed
% =======================================================================

%% Output subfolder (per METHOD)

outdir = fullfile(OUT_ROOT, [lower(METHOD) '-landscape']);
if ~exist(outdir, 'dir')
    mkdir(outdir);
end

%% Read summary sheet

opts = detectImportOptions(SHEET_PATH,'NumHeaderLines',0);
opts = setvartype(opts,'seq','string');
DataSheet = readtable(SHEET_PATH, opts);

% Basic cleaning for this particular CSV (can be adapted if needed)
if height(DataSheet) >= 1
    DataSheet(1,:) = [];
end
if height(DataSheet) >= 69
    DataSheet(69,:) = [];
end

all_TF = unique(DataSheet.tfrecord_name);

% Select target tfrecord
this_TF = all_TF{target_idx};
indices = find(strcmp(DataSheet.tfrecord_name, this_TF));

fprintf('\n============ Cost landscape analysis: %s (%d frames) ============\n', ...
        this_TF, numel(indices));
assert(numel(indices) >= 1, 'Selected tfrecord has no valid frames.');

%% Accumulate per-frame data (mask / LiDAR / copyy_chao)

M_list     = {};   % image masks (semantic car region)
PL_list    = {};   % LiDAR semantic car points
copyy_list = {};   % CHAO-specific preprocessed data

for j = 1:numel(indices)
    dd = indices(j);

    seq_raw    = char(DataSheet.seq(dd));
    seq        = sprintf('%02d', str2double(seq_raw));
    frame      = num2str(DataSheet.frame_id(dd));
    pixel_file = char(DataSheet.pixel_file_name(dd));
    lidar_file = char(DataSheet.lidar_file_name(dd));

    % --- Image-domain car mask ---
    img_car = readmatrix(fullfile(DATA_ROOT, seq, this_TF, frame, pixel_file));
    BW = zeros(1280,1920,'uint8');
    ind = sub2ind([1280,1920], img_car(:,2), img_car(:,1));
    BW(ind) = 1;
    BW1 = mat2gray(BW);
    BW1 = imfill(BW1,'holes');
    M_list{end+1} = BW1;

    % --- LiDAR semantic points in car region ---
    pt_car = readmatrix(fullfile(DATA_ROOT, seq, this_TF, frame, lidar_file));
    PL_list{end+1} = pt_car;

    % --- CHAO-specific data (for the CHAO loss) ---
    chao_path  = fullfile(DATA_ROOT, seq, this_TF, frame, 'copyy_chao.mat');
    copyy_chao = load(chao_path, 'copyy_chao');
    copyy_list{end+1} = copyy_chao.copyy_chao;
end

%% Calibration (read from the last frame of this tfrecord)

seq_raw = char(DataSheet.seq(indices(end)));
seq     = sprintf('%02d', str2double(seq_raw));
frame   = num2str(DataSheet.frame_id(indices(end)));

calib = dlmread(fullfile(DATA_ROOT, seq, this_TF, frame, 'calib.txt'), ' ', 0, 1);
Tr_velo_to_cam  = reshape(calib(6,:), [4,3])';
Tr_velo_to_cam2 = [Tr_velo_to_cam; 0 0 0 1];
P2 = reshape(calib(3,:), [4,3])';
K  = P2(1:3,1:3);

%% Grid settings

range   = grid_range;    % shorthand for Δty / Δtz range
J_vals  = zeros(numel(range), numel(range));

theta_fix = rotationMatrixToVector(Tr_velo_to_cam2(1:3,1:3));
t_origin  = Tr_velo_to_cam2(1:3,4);

%% Decide which frames to use (avg vs. single-frame mode)

if strcmpi(landscape_mode, 'avg')
    use_PL   = PL_list;
    use_M    = M_list;
    use_copy = copyy_list;
    denom    = numel(M_list);

    mode_tag   = sprintf('multi-frame AVERAGE cost (N = %d)', denom);
    save_tag   = 'avg';
    zlabel_str = 'J (frame-averaged)';
else
    k = min(max(1, single_frame_idx), numel(M_list));
    use_PL   = {PL_list{k}};
    use_M    = {M_list{k}};
    use_copy = {copyy_list{k}};
    denom    = 1;

    mode_tag   = sprintf('single frame #%d', k);
    save_tag   = sprintf('single%02d', k);
    zlabel_str = 'J';
end

%% Choose base multi-frame cost function (CASA or CHAO)

% We wrap the existing cost functions in a unified interface:
%   base_cost_fn(theta, t, K, copy_cells, PL_cells, M_cells)
%
% Later we will pass only a single frame at a time (1x1 cell arrays),
% and explicitly average over all frames inside this script.

switch upper(METHOD)
    case 'CASA'
        % Note: CASA cost is negated here so that "smaller is better".
        base_cost_fn = @(th, tv, Kmat, copy_cells, PL_cells, M_cells) ...
            -multi_frame_my_cost(th, tv, Kmat, PL_cells, M_cells, M_cells);

    case 'CHAO'
        base_cost_fn = @(th, tv, Kmat, copy_cells, PL_cells, M_cells) ...
             multi_frame_cost_chao(th, tv, Kmat, copy_cells, PL_cells, M_cells);

    otherwise
        error('METHOD must be ''CASA'' or ''CHAO''.');
end

%% Compute J(Δty, Δtz) as explicit frame-wise average

for m = 1:numel(range)
    for n = 1:numel(range)
        % Current translation perturbation around the ground truth
        ty = t_origin(2) + range(m);
        tz = t_origin(3) + range(n);
        t_test = [t_origin(1); ty; tz];

        % Explicitly accumulate cost over all selected frames
        J_sum = 0;
        for f = 1:denom
            % Wrap each frame into 1x1 cell arrays so that the existing
            % multi-frame cost function can still be reused.
            this_copy = use_copy(f);   % 1x1 cell
            this_PL   = use_PL(f);     % 1x1 cell
            this_M    = use_M(f);      % 1x1 cell

            J_f = base_cost_fn(theta_fix, t_test, K, ...
                               this_copy, this_PL, this_M);
            J_sum = J_sum + J_f;
        end

        % Average over frames (this guarantees that the landscape is the
        % mean cost across all frames for this tfrecord)
        J_vals(m,n) = J_sum / denom;
    end
end

fprintf('J range (frame-averaged): [%.6f, %.6f]\n', ...
        min(J_vals(:)), max(J_vals(:)));

%% Compute three landscape metrics and save as CSV

[metrics, argmin_xy] = compute_landscape_metrics(J_vals, range, eps_percent);
disp(struct2table(metrics, 'AsArray', true));

metrics_tbl          = struct2table(metrics, 'AsArray', true);
metrics_tbl.tfrecord = string(this_TF);
metrics_tbl          = movevars(metrics_tbl, 'tfrecord', 'Before', 1);

out_csv = fullfile(outdir, ...
          sprintf('metrics_%s_%s_%s.csv', lower(METHOD), save_tag, this_TF));
writetable(metrics_tbl, out_csv);
fprintf('Saved landscape metrics to: %s\n', out_csv);

%% Visualization (3D surface & 2D contour)

vis = tern(show_figure, 'on', 'off');

% ---------- 3D surface plot ----------
fig3 = figure('Visible', vis, 'Color','w');
surf(range, range, J_vals, 'EdgeColor','none'); hold on;

xlabel('\Delta ty (m)');
ylabel('\Delta tz (m)');
zlabel(zlabel_str);

title(sprintf('3D cost landscape around GT (%s) [%s] (%s)', ...
      this_TF, mode_tag, METHOD), ...
      'Interpreter', 'none');

colorbar; view(45, 30);

% Mark ground-truth position (assumed at Δty = 0, Δtz = 0)
[~, ix] = min(abs(range - 0));
gt_val  = J_vals(ix, ix);
plot3(0, 0, gt_val, 'r*', 'MarkerSize', 10, 'LineWidth', 2);

legend('J surface', 'GT point', 'Location', 'best');

if save_png_pdf
    exportgraphics(fig3, fullfile(outdir, ...
        sprintf('landscape3D_%s_%s_%s.png', ...
        lower(METHOD), save_tag, this_TF)), 'Resolution', 300);

    exportgraphics(fig3, fullfile(outdir, ...
        sprintf('landscape3D_%s_%s_%s.pdf', ...
        lower(METHOD), save_tag, this_TF)), 'ContentType', 'vector');
end

% ---------- 2D contour plot ----------
fig2 = figure('Visible', vis, 'Color','w');

[~, hSurf] = contourf(range, range, J_vals, levels_contour, 'LineStyle','none');
hold on; axis xy;

xlabel('\Delta tz (m)');
ylabel('\Delta ty (m)');

title(sprintf('2D cost landscape around GT (%s) [%s] (%s)', ...
      this_TF, mode_tag, METHOD), ...
      'Interpreter','none');

cb = colorbar;
cb.Label.String = zlabel_str;

% Mark GT and argmin (we keep only symbols, no coordinate text on figure)
hGT  = plot(0, 0, 'r*', 'MarkerSize', 10, 'LineWidth', 2);
hMin = plot(argmin_xy(1), argmin_xy(2), 'kx', 'MarkerSize', 10, 'LineWidth', 2);

legend([hSurf, hGT, hMin], {'J surface', 'GT point', 'argmin'}, 'Location', 'best');

% Annotate landscape metrics in a small box (top-left corner)
txtStr = sprintf(['d* = %.3f m\n'       ...
                  'FWHM_{tz} = %.3f m\n' ...
                  'FWHM_{ty} = %.3f m\n' ...
                  'A_{\\epsilon} = %.3f m^2'], ...
                  metrics.d_star_m, ...
                  metrics.FWHM_tz_m, ...
                  metrics.FWHM_ty_m, ...
                  metrics.A_eps_m2);

x0 = range(1) + 0.02 * (range(end) - range(1));
y0 = range(end) - 0.08 * (range(end) - range(1));

hTxt = text(x0, y0, txtStr, ...
            'Color', 'w', ...
            'FontSize', 10, ...
            'FontWeight', 'bold', ...
            'VerticalAlignment', 'top', ...
            'Interpreter', 'tex');

% Add semi-transparent black background behind the text for readability
pad = 0.01 * (range(end) - range(1));
ext = get(hTxt, 'Extent');  % [x y w h]

rectangle('Position', [ext(1)-pad, ext(2)-pad, ext(3)+2*pad, ext(4)+2*pad], ...
          'FaceColor', [0 0 0 0.35], ...
          'EdgeColor', 'none');

uistack(hTxt, 'top');

if save_png_pdf
    exportgraphics(fig2, fullfile(outdir, ...
        sprintf('landscape2D_%s_%s_%s.png', ...
        lower(METHOD), save_tag, this_TF)), 'Resolution', 300);

    exportgraphics(fig2, fullfile(outdir, ...
        sprintf('landscape2D_%s_%s_%s.pdf', ...
        lower(METHOD), save_tag, this_TF)), 'ContentType', 'vector');
end

drawnow;

% Optional: print argmin coordinates to console (for debugging / analysis)
fprintf('argmin at (tz, ty) = (%.2f, %.2f) m\n', argmin_xy(1), argmin_xy(2));

%% ==================== Local helper functions ====================

function y = tern(cond, a, b)
% Simple ternary helper: y = cond ? a : b;
    if cond
        y = a;
    else
        y = b;
    end
end

function [metrics, argmin_xy] = compute_landscape_metrics(J, range, eps_percent)
% COMPUTE_LANDSCAPE_METRICS
%   Compute:
%       - d_star_m  : distance from GT (0,0) to global minimizer
%       - FWHM_tz_m : full width at half maximum along Δtz
%       - FWHM_ty_m : full width at half maximum along Δty
%       - A_eps_m2  : area of attraction basin within an epsilon threshold
%
%   J      : cost matrix (size: numel(range) x numel(range))
%   range  : vector of Δty or Δtz sample positions (uniform gap)
%   eps_percent : relative threshold percentage for A_eps

    % Global minimizer
    [~, linIdxMin] = min(J(:));
    [m_star, n_star] = ind2sub(size(J), linIdxMin);

    ty_star = range(m_star);
    tz_star = range(n_star);
    d_star  = hypot(ty_star, tz_star);

    % Profile along Δty (fix tz = tz_star)
    prof_ty = J(:, n_star);
    % Profile along Δtz (fix ty = ty_star)
    prof_tz = J(m_star, :);

    [fwhm_ty, ~, ~] = fwhm_around_min(range(:), prof_ty(:), m_star);
    [fwhm_tz, ~, ~] = fwhm_around_min(range(:), prof_tz(:), n_star);

    % Attraction-basin area A_eps
    Jmin = J(m_star, n_star);
    Jmax = max(J(:));
    eps_val = eps_percent * (Jmax - Jmin);

    mask = (J <= Jmin + eps_val);
    step = abs(range(2) - range(1));
    A_eps = nnz(mask) * (step^2);

    % Pack metrics
    metrics = struct( ...
        'd_star_m',   d_star, ...
        'FWHM_tz_m',  fwhm_tz, ...
        'FWHM_ty_m',  fwhm_ty, ...
        'A_eps_m2',   A_eps, ...
        'tz_star_m',  tz_star, ...
        'ty_star_m',  ty_star, ...
        'Jmin',       Jmin, ...
        'Jmax',       Jmax, ...
        'epsilon',    eps_val);

    % argmin for plotting (x = tz, y = ty)
    argmin_xy = [tz_star, ty_star];
end

function [w, xL, xR] = fwhm_around_min(x, y, idxMin)
% FWHM_AROUND_MIN
%   Approximate the "full width at half maximum" for a concave basin
%   around its minimum point.
%
%   x      : sample positions (vector)
%   y      : cost values (vector)
%   idxMin : index of minimum in y
%
%   Returns:
%       w  : FWHM width
%       xL : left crossing position
%       xR : right crossing position

    y = y(:);
    x = x(:);

    yMin = y(idxMin);
    yMax = max(y);
    lvl  = yMin + 0.5 * (yMax - yMin);   % half-depth threshold

    xL = x(1);
    xR = x(end);
    leftOK  = false;
    rightOK = false;

    % Search to the left of the minimum
    for k = idxMin:-1:2
        if y(k-1) > lvl && y(k) <= lvl
            xL = interp1([y(k-1) y(k)], [x(k-1) x(k)], lvl);
            leftOK = true;
            break;
        end
    end
    if ~leftOK
        xL = x(1);
    end

    % Search to the right of the minimum
    N = numel(x);
    for k = idxMin:N-1
        if y(k) <= lvl && y(k+1) > lvl
            xR = interp1([y(k) y(k+1)], [x(k) x(k+1)], lvl);
            rightOK = true;
            break;
        end
    end
    if ~rightOK
        xR = x(end);
    end

    w = max(0, xR - xL);
end

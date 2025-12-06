% =======================================================================
% perturbation_analysis.m
%
% CASA-Calib: Perturbation Robustness Experiment (Figure 5)
%
% This script evaluates the robustness of the CASA-Calib optimization
% under controlled perturbations of the camera–LiDAR extrinsic parameters.
% For each selected tfrecord (with at least 2 frames), we:
%   1) Apply synthetic perturbations to either rotation, translation,
%      or both, around the ground-truth extrinsic.
%   2) Run local optimization (fminunc or fminsearch) to recover the pose.
%   3) Measure the final rotation and translation errors.
%
% The aggregated results can be used to reproduce curves similar to
% Figure 5 in the CASA-Calib paper (average errors vs perturbation level).
%
% NOTE:
%   - This script uses only the CASA loss via CASA_Loss.m.
% 
%
% Dependencies:
%   - CASA_Loss(theta, t, K, PL_list, M_list, img_raws)
%   - rotationMatrixToVector, rotationVectorToMatrix
%   - Optimization Toolbox (optional, for fminunc; otherwise fminsearch)
%
% Data assumptions:
%   - Waymo-based preprocessed data under base_dir:
%       pair_summary_00_03_deduplicated.csv
%       base_dir / <seq> / <tfrecord> / <frame> / <pixel_file>
%       base_dir / <seq> / <tfrecord> / <frame> / <lidar_file>
%       base_dir / <seq> / <tfrecord> / <frame> / calib.txt
%
% Author: Yuan-Ting Fu
% Project: CASA-Calib — Context-Aware Semantic Alignment for Camera–LiDAR Calibration
% =======================================================================

clear; close all; clc;
format long; warning('off');
addpath('../');
addpath('../../../matlab_code');

rng(42);  % Fix random seed for reproducibility

%% ========= User settings =========

% Perturbation mode selection
fprintf('\n=== Perturbation mode ===\n');
fprintf('  1) Rotation only\n');
fprintf('  2) Translation only\n');
fprintf('  3) Rotation + Translation (default)\n');
mode_in = input('Please enter 1/2/3 (default = 3): ','s');

mode_choice = 3;
if ~isempty(mode_in)
    v = str2double(mode_in);
    if ismember(v, [1 2 3])
        mode_choice = v;
    end
end

% Whether to customize perturbation ranges
custom_in  = lower(strtrim(input('Customize perturbation ranges? (y/N): ','s')));
use_custom = ismember(custom_in, {'y','yes','1'});

% Default ranges
rot_values   = linspace(0, 4.0, 21);   % rotation perturbation (deg)
trans_values = linspace(0, 1.0, 11);   % translation perturbation (m)

if use_custom
    fprintf('\n--- Rotation perturbation (degrees) ---\n');
    rv = input('Enter [start, end, count] (e.g. [0, 4, 21]; Enter = default): ');
    if isnumeric(rv) && numel(rv) == 3
        rot_values = linspace(rv(1), rv(2), rv(3));
    end

    fprintf('\n--- Translation perturbation (meters) ---\n');
    tv = input('Enter [start, end, count] (e.g. [0, 1.0, 11]; Enter = default): ');
    if isnumeric(tv) && numel(tv) == 3
        trans_values = linspace(tv(1), tv(2), tv(3));
    end
end

fprintf('\nRotation perturbation range (deg): %s\n', mat2str(rot_values, 3));
fprintf('Translation perturbation range (m): %s\n\n', mat2str(trans_values, 3));

%% ========= Load datasheet & keep tfrecords with >= 2 frames =========

sheet_path = '../../../waymo_segment_data/pair_summary_00_03_deduplicated.csv';
opts = detectImportOptions(sheet_path, 'NumHeaderLines', 0);
opts = setvartype(opts, 'seq', 'string');
DataSheet = readtable(sheet_path, opts);

% Clean rows according to the original preprocessing
DataSheet(1,:)  = [];
DataSheet(69,:) = [];

base_dir = '../../../waymo_segment_data/';
all_TF   = unique(DataSheet.tfrecord_name);

selected_TF = {};
for iTF = 1:numel(all_TF)
    tf  = all_TF{iTF};
    idx = find(strcmp(DataSheet.tfrecord_name, tf));
    if numel(idx) >= 2
        selected_TF{end+1} = tf; %#ok<SAGROW>
    else
        fprintf('[Skip] %s has too few frames (only %d frame).\n', tf, numel(idx));
    end
end
selected_TF = selected_TF(:);
fprintf('Total %d tfrecords kept (with at least 2 frames).\n\n', numel(selected_TF));

%% ========= Mode combinations =========

all_modes  = {'rotation', 'translation'};
all_values = {rot_values, trans_values};  % rotation(°), translation(m)

if mode_choice == 1
    run_modes = 1;
elseif mode_choice == 2
    run_modes = 2;
else
    run_modes = [1, 2];  % both rotation & translation
end

%% ========= Solver selection (prefer fminunc, fallback to fminsearch) =========

fprintf('\n=== Choose optimizer ===\n');
fprintf('  1) fminunc  (Quasi-Newton, requires Optimization Toolbox)\n');
fprintf('  2) fminsearch (Nelder–Mead, no Toolbox)  [default]\n');
solver_in = strtrim(input('Please enter 1/2 (Enter = 2): ','s'));

solver_choice = 2;
if ~isempty(solver_in)
    v = str2double(solver_in);
    if ismember(v, [1 2])
        solver_choice = v;
    end
end

if solver_choice == 1
    use_fminunc = true;
else
    use_fminunc = false;
end

% Parameter scaling (avoid imbalance between rotation and translation)
scale_rot = 1.0;   % axis-angle in rad; keep 1.0 by default
scale_t   = 1.0;   % translation in m; adjust if needed for your scene

% Map unscaled x=[theta(3); t(3)] <-> scaled z
to_z = @(x) [x(1:3)/scale_rot; x(4:6)/scale_t];
to_x = @(z) [z(1:3)*scale_rot; z(4:6)*scale_t];

%% ========= Main loop over perturbation modes =========

for midx = run_modes

    perturb_mode   = all_modes{midx};       % 'rotation' or 'translation'
    perturb_values = all_values{midx};
    P = numel(perturb_values);

    trans_err_list = cell(P, 1);   % translation error (m)
    rot_err_list   = cell(P, 1);   % rotation error (deg)

    % Additional statistics (cost & keep ratio, optional visualization)
    J_hist_all    = cell(P, 1);    % final cost for each run (J = -CASA_Loss)
    keep_all      = cell(P, 1);    % whether the "best" solution was reused
    bestcurve_all = cell(P, 1);    % running best cost per run

    % Loop over all selected tfrecords
    for iTF = 1:numel(selected_TF)
        this_TF = selected_TF{iTF};
        indices = find(strcmp(DataSheet.tfrecord_name, this_TF));

        % Collect multi-frame data for this tfrecord
        M_list  = {};   % masks
        PL_list = {};   % LiDAR semantic points

        seq   = '';
        frame = '';

        for j = 1:numel(indices)
            dd = indices(j);

            seq_raw    = char(DataSheet.seq(dd));
            seq        = sprintf('%02d', str2double(seq_raw));
            frame      = num2str(DataSheet.frame_id(dd));
            pixel_file = char(DataSheet.pixel_file_name(dd));
            lidar_file = char(DataSheet.lidar_file_name(dd));

            % --- Image mask (1280x1920) ---
            img_car = readmatrix(fullfile(base_dir, seq, this_TF, frame, pixel_file));
            BW = zeros(1280,1920,'uint8');
            ind = sub2ind([1280,1920], img_car(:,2), img_car(:,1));
            BW(ind) = 1;
            BW1 = mat2gray(BW);
            BW1 = imfill(BW1, 'holes');
            M_list{end+1} = BW1;

            % --- LiDAR semantic car points ---
            pt_car = readmatrix(fullfile(base_dir, seq, this_TF, frame, lidar_file));
            PL_list{end+1} = pt_car;
        end

        % Ground-truth extrinsic (from calib.txt)
        calib_path = fullfile(base_dir, seq, this_TF, frame, 'calib.txt');
        try
            calib = readmatrix(calib_path);
            if isempty(calib) || any(isnan(calib(:)))
                error('Empty or invalid calib.');
            end
        catch
            calib = dlmread(calib_path, ' ', 0, 1);
        end

        Tr_velo_to_cam = reshape(calib(6,:), [4,3])';
        Tr_gt = [Tr_velo_to_cam; 0 0 0 1];

        P2 = reshape(calib(3,:), [4,3])';
        K  = P2(1:3,1:3);

        R_gt = Tr_gt(1:3,1:3);
        t_gt = Tr_gt(1:3,4);

        % Keep the best solution for this tfrecord (monotonic safeguard)
        best_J = inf;
        best_x = [];

        % Loop over perturbation magnitudes
        for p = 1:P
            val = perturb_values(p);

            % Generate initial pose around ground truth
            if strcmp(perturb_mode, 'rotation')
                % Random axis, scaled by perturbation magnitude (deg -> rad)
                v = randn(3,1);
                v = v / max(norm(v), 1e-12);
                theta_perturb = deg2rad(val) * v;
                R_init = rotationVectorToMatrix(theta_perturb) * R_gt;
                t_init = t_gt;
            else
                % Random direction in translation space
                v = randn(3,1);
                v = v / max(norm(v), 1e-12);
                t_init = t_gt + val * v;
                R_init = R_gt;
            end

            theta0 = rotationMatrixToVector(R_init);
            x0 = [theta0'; t_init];   % 6×1 unscaled
            z0 = to_z(x0);           % scaled for optimizer

            % ===== CASA_Loss (multi-frame), maximize -> minimize by negation =====
            % Note: CASA_Loss returns a similarity score (larger is better),
            % so we define the cost as J = -CASA_Loss(...)
            base_cost = @(x) -CASA_Loss( ...
                                x(1:3), x(4:6), K, PL_list, M_list, M_list);

            % Wrap scaling: z -> x -> cost(x)
            cost_fun_z = @(z) base_cost(to_x(z));

            % ===== Optimization =====
            if use_fminunc
                opts = optimoptions('fminunc', ...
                                    'Display', 'off', ...
                                    'Algorithm', 'quasi-newton', ...
                                    'MaxIterations', 200, ...
                                    'OptimalityTolerance', 1e-6, ...
                                    'StepTolerance', 1e-8);
                try
                    [z_opt, fval] = fminunc(cost_fun_z, z0, opts);
                catch
                    % Fallback to fminsearch if fminunc fails
                    opts2 = optimset('Display', 'off', ...
                                     'MaxIter', 200, ...
                                     'TolX', 1e-8, ...
                                     'TolFun', 1e-6);
                    [z_opt, fval] = fminsearch(cost_fun_z, z0, opts2);
                end
            else
                opts2 = optimset('Display', 'off', ...
                                 'MaxIter', 200, ...
                                 'TolX', 1e-8, ...
                                 'TolFun', 1e-6);
                [z_opt, fval] = fminsearch(cost_fun_z, z0, opts2);
            end

            x_opt = to_x(z_opt);   % convert back to unscaled parameters
            J_hist = fval;         % this is J = -CASA_Loss

            % Safeguard: if new solution is worse, reuse previous best
            if J_hist < best_J || isinf(best_J) || isempty(best_x)
                best_J = J_hist;
                best_x = x_opt;
                x_used = x_opt;
                use_keep = false;
            else
                x_used = best_x;
                use_keep = true;
            end

            % Collect statistics
            if isempty(J_hist_all{p})
                J_hist_all{p} = J_hist;
            else
                J_hist_all{p}(end+1) = J_hist;
            end

            if isempty(keep_all{p})
                keep_all{p} = use_keep;
            else
                keep_all{p}(end+1) = use_keep;
            end

            if isempty(bestcurve_all{p})
                bestcurve_all{p} = best_J;
            else
                bestcurve_all{p}(end+1) = best_J;
            end

            % Compute final errors (relative rotation angle + translation L2)
            theta_hat = x_used(1:3);
            t_hat     = x_used(4:6);
            R_hat     = rotationVectorToMatrix(theta_hat);

            % Relative rotation error (rad)
            c  = (trace(R_hat * R_gt') - 1) / 2;
            c  = max(-1, min(1, c));    % clamp for stability
            eR = acos(c);
            eT = norm(t_hat - t_gt);

            rot_err_list{p}(end+1)   = rad2deg(eR);   % deg
            trans_err_list{p}(end+1) = eT;            % m
        end
    end

    %% ========= Aggregate statistics & visualization =========

    mean_trans = zeros(P,1);
    std_trans  = zeros(P,1);
    mean_rot   = zeros(P,1);
    std_rot    = zeros(P,1);

    for p = 1:P
        mean_trans(p) = mean(trans_err_list{p});
        std_trans(p)  = std(trans_err_list{p});
        mean_rot(p)   = mean(rot_err_list{p});
        std_rot(p)    = std(rot_err_list{p});
    end

    is_rotation_mode = strcmp(perturb_mode, 'rotation');
    unit_str = ternary(is_rotation_mode, 'deg', 'm');

    % Console summary
    fprintf('==== Summary (%s perturbation) ====\n', perturb_mode);
    for p = 1:P
        fprintf(['Perturb = %5.3f %s --> ' ...
                 'mean trans = %.4f m (±%.4f), ' ...
                 'mean rot = %.4f° (±%.4f), N = %d\n'], ...
            perturb_values(p), unit_str, ...
            mean_trans(p), std_trans(p), ...
            mean_rot(p),   std_rot(p), ...
            numel(trans_err_list{p}));
    end

    % Figure 1: Errors vs perturbation
    figure('Name', sprintf('Perturbation - %s', perturb_mode));
    hold on; box on; grid on;

    yyaxis left;
    plot(perturb_values, mean_trans, 'b-o', 'LineWidth', 2, 'MarkerSize', 4);
    ylabel('Translation Errors (m)');

    yyaxis right;
    plot(perturb_values, mean_rot, 'r-o', 'LineWidth', 2, 'MarkerSize', 4);
    ylabel('Rotation Errors (°)');

    xlabel(sprintf('Perturbation of %s parameters (%s)', perturb_mode, unit_str));
    title(sprintf('Average Errors - %s perturbation', perturb_mode));
    legend('Translation Errors', 'Rotation Errors', 'Location', 'northwest');

    saveas(gcf, sprintf('perturb_exp_%s.png', perturb_mode));

    % Figure 2: Loss & keep ratio
    J_hist_mean     = zeros(P,1);
    J_hist_std      = zeros(P,1);
    best_curve_mean = zeros(P,1);
    keep_ratio      = zeros(P,1);

    for p = 1:P
        J_hist_mean(p)     = mean(J_hist_all{p});
        J_hist_std(p)      = std(J_hist_all{p});
        best_curve_mean(p) = mean(bestcurve_all{p});
        keep_ratio(p)      = mean(keep_all{p});
    end

    f2 = figure('Name', sprintf('Loss & Keep - %s', perturb_mode));
    hold on; box on; grid on;

    yyaxis left;
    plot(perturb_values, J_hist_mean, '-o', 'LineWidth', 2, 'MarkerSize', 4);
    plot(perturb_values, best_curve_mean, '--', 'LineWidth', 2, 'MarkerSize', 4);
    ylabel('Cost J (mean across tfrecords)');

    kept_idx = keep_ratio > 0.5;
    scatter(perturb_values(kept_idx), J_hist_mean(kept_idx), 70, ...
        'o', 'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'none', 'LineWidth', 1.6);

    yyaxis right;
    plot(perturb_values, 100 * keep_ratio, '-s', 'LineWidth', 1.5, 'MarkerSize', 5);
    ylabel('Keep ratio (%)');

    xlabel(sprintf('Perturbation of %s parameters (%s)', perturb_mode, unit_str));
    title(sprintf('Loss & Keep Visualization - %s', perturb_mode));

    legend({'J_{hist} (mean)', 'Best-J curve (mean)', ...
            'Kept points (>50%)', 'Keep ratio (%)'}, 'Location', 'best');

    saveas(f2, sprintf('perturb_exp_%s_loss.png', perturb_mode));

    % Save MAT (results for later plotting / analysis)
    out_prefix = sprintf('perturb_exp_%s', perturb_mode);

    save([out_prefix '.mat'], ...
        'perturb_mode', 'perturb_values', ...
        'trans_err_list', 'rot_err_list', ...
        'mean_trans', 'std_trans', ...
        'mean_rot', 'std_rot');

    save([out_prefix '_loss.mat'], ...
        'perturb_mode', 'perturb_values', ...
        'J_hist_all', 'bestcurve_all', 'keep_all', ...
        'J_hist_mean', 'J_hist_std', ...
        'best_curve_mean', 'keep_ratio');
end

fprintf('\nAll selected perturbation modes finished (including basic visualizations).\n');

%% ===== Helper: ternary operator for strings/numbers =====
function y = ternary(cond, a, b)
    if cond
        y = a;
    else
        y = b;
    end
end

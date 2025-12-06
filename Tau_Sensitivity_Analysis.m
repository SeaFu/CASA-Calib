% =======================================================================
% Tau_Sensitivity_Analysis.m
%
% CASA-Calib / CASA-Loss: sensitivity of the IoU threshold τ.
%
% This script visualizes:
%   (1) The relative improvement of MAE_rot / MAE_trans as τ varies
%       (tau sweep vs. baseline τ = 0).
%   (2) A distance-to-ideal score S(τ) with a 2% stability band.
%   (3) A Pareto plot of translation vs rotation error highlighting τ = 0.8.
%
% These figures correspond to the τ-selection analysis (Fig. 4) in the
% CASA-Calib paper, where τ is the IoU-based reliability threshold used
% in the CASA_Loss coupling term α(L_IoU).
%
% Author: Yuan-Ting Fu
% Project: CASA-Calib — Context-Aware Semantic Alignment for Camera–LiDAR Calibration
% =======================================================================

%% ================================================================
% Tau-sweep evaluation relative to baseline (tau = 0)
% Definition:
%   %Δ = (baseline − current) / baseline × 100
% Positive values = improvement (lower error)
% Negative values = degradation (higher error)
%
% Both curves share the same y-axis (percentage). A 0% reference
% dashed line is plotted for clarity.
% ================================================================

clear; close all; clc;

% ----- Input Data (from CASA-Calib validation sweep) -----
tau = 0:0.1:0.9;

MAE_rot_deg = [ ...
    0.114107827, 0.116952593, 0.148851881, 0.149515865, 0.155641969, ...
    0.133834580, 0.134272463, 0.120763643, 0.122512957, 0.132753268 ];

MAE_trans_cm = [ ...
    3.788937262, 3.922781357, 3.807671352, 3.553581045, 3.699753129, ...
    3.116013644, 2.955340867, 2.913516577, 2.763030405, 3.323161248 ];

% ----- Compute percentage improvements relative to baseline (tau=0) -----
idx   = 2:numel(tau);         % indices for tau = 0.1 : 0.1 : 0.9
tau_s = tau(idx);

rot_base  = MAE_rot_deg(1);
tran_base = MAE_trans_cm(1);

pct_rot  = (rot_base  - MAE_rot_deg(idx)) ./ rot_base  * 100;
pct_tran = (tran_base - MAE_trans_cm(idx)) ./ tran_base * 100;

% ----- Plotting: relative changes vs tau -----
figure('Color','w','Position',[120 120 820 520]); 
hold on; box on; grid on;

% 0% reference line
yline(0,'--','Color',[0.4 0.4 0.4],'LineWidth',1);

% Percentage curves
p1 = plot(tau_s, pct_tran, '-o', 'LineWidth', 2, 'MarkerSize', 6, ...
          'Color', [0.00 0.45 0.74]);
p2 = plot(tau_s, pct_rot,  '-s', 'LineWidth', 2, 'MarkerSize', 6, ...
          'Color', [0.85 0.33 0.10]);

xlabel('IoU threshold \tau');
ylabel('Relative change vs. \tau = 0 (%)');
title('Relative Error Change vs. IoU Threshold (Baseline = \tau = 0)');

legend([p1 p2], {'Translation MAE', 'Rotation MAE'}, 'Location', 'best');

% Set y-limits with margin
all_pct = [pct_tran(:); pct_rot(:)];
yl      = [min(all_pct) max(all_pct)];
margin  = 0.1 * max(1, range(yl));
ylim([yl(1)-margin, yl(2)+margin]);

% ----- Optional: Save figure -----
% print(gcf, 'tau_sweep_relative_to_tau0.png', '-dpng', '-r300');
% print(gcf, 'tau_sweep_relative_to_tau0.pdf', '-dpdf');


%% ================================================================
% Additional Figures:
%   - Fig X: Distance-to-ideal score S(tau) with 2% stability band
%   - Fig Y: Pareto plot (MAE_trans vs MAE_rot), highlight tau = 0.8
%
% Outputs (optional):
%   figX_tau_score.(pdf|png)
%   figY_pareto.(pdf|png)
% ================================================================

clear; clc; close all;

%% ----- Validation Sweep Data (same as above, for completeness) -----
% Columns: tau, MAE_rot_deg, MAE_trans_cm
D = [ ...
  0.0  0.114107827  3.788937262;
  0.1  0.116952593  3.922781357;
  0.2  0.148851881  3.807671352;
  0.3  0.149515865  3.553581045;
  0.4  0.155641969  3.699753129;
  0.5  0.133383458  3.116013644;
  0.6  0.134272463  2.955340867;
  0.7  0.120763643  2.913156177;
  0.8  0.122512957  2.763030405;
  0.9  0.132753268  3.323161248 ];

tau  = D(:,1);
rot  = D(:,2);
tran = D(:,3);

%% ----- Relative Improvements vs tau = 0 -----
rot0  = rot(1);
tran0 = tran(1);

imp_rot  = (rot0  - rot)./rot0  * 100;
imp_tran = (tran0 - tran)./tran0 * 100;

i08 = find(abs(tau - 0.8) < 1e-9, 1);

fprintf('\n[CASA-Calib] Tau sensitivity (relative to \\tau = 0)\n');
fprintf('At tau=0.8: ΔMAE_trans = %+0.1f%% (%.3f → %.3f cm)\n', ...
    imp_tran(i08), tran0, tran(i08));
fprintf('At tau=0.8: ΔMAE_rot   = %+0.1f%% (%.4f → %.4f deg)\n', ...
    imp_rot(i08), rot0, rot(i08));

%% ----- Min–max Normalization -----
z_rot  = (rot  - min(rot))  ./ (max(rot)  - min(rot));
z_tran = (tran - min(tran)) ./ (max(tran) - min(tran));

%% ----- Composite Scores -----
S_sum  = 0.5*z_tran + 0.5*z_rot;
S_dist = sqrt(z_tran.^2 + z_rot.^2);

[Smin, idxMin] = min(S_dist);
tau_best       = tau(idxMin);

% Stability band = within 2% above the minimum
stable_mask = S_dist <= Smin * 1.02;
stable_min  = min(tau(stable_mask));
stable_max  = max(tau(stable_mask));

fprintf('\nBest tau (min distance score) = %.1f (S = %.3f)\n', tau_best, Smin);
fprintf('Stability band (<=2%% above minimum): [%.1f, %.1f]\n', ...
    stable_min, stable_max);

%% ================= Figure X: Distance-to-Ideal Score =================
figX = figure('Color','w','Position',[100 100 920 500]);
hold on;

yl = [min(S_dist)*0.9, max(S_dist)*1.1];

% Stability band region
patch([stable_min stable_max stable_max stable_min], ...
      [yl(1) yl(1) yl(2) yl(2)], ...
      [0.9 0.9 0.9], 'EdgeColor','none', 'FaceAlpha',0.6, ...
      'DisplayName','Stability ≤2% band');

% Distance score
plot(tau, S_dist, '-o', 'LineWidth', 2, 'MarkerSize', 6, ...
     'DisplayName', 'Distance-to-ideal score S(\tau)');

% Optional: additive score
plot(tau, S_sum, '--s', 'LineWidth', 1.5, 'MarkerSize', 5, ...
     'DisplayName', 'Additive score (reference)');

% Highlight tau = 0.8
xline(0.8, ':', 'LineWidth', 1.5, 'Color',[0.2 0.2 0.2], ...
      'DisplayName','\tau = 0.8');

% Mark best tau
plot(tau_best, Smin, 'p', 'MarkerSize', 12, 'LineWidth', 2, ...
     'Color',[0.8 0.2 0.2], 'MarkerFaceColor',[0.9 0.4 0.4], ...
     'DisplayName','Best');

text(tau_best, Smin, sprintf('  \\leftarrow BEST (%.1f)', tau_best), ...
     'FontSize', 11, 'Color',[0.5 0 0]);

grid on; box on; ylim(yl);

xlabel('\tau');
ylabel('Score (lower is better)');
title('Sensitivity of \tau via Distance-to-Ideal Score');
legend('Location','northwest');

%% ==================== Figure Y: Pareto Plot ====================
figY = figure('Color','w','Position',[120 120 560 520]); 
hold on;

% Scatter all candidates
scatter(tran, rot, 50, 'filled', 'MarkerFaceColor',[0.2 0.4 0.8], ...
        'DisplayName','\tau candidates');

% Label each tau
for i = 1:numel(tau)
    text(tran(i) + 0.02*range(tran), rot(i), ...
         sprintf('\\tau=%.1f', tau(i)), 'FontSize', 9);
end

% Highlight tau = 0.8
plot(tran(i08), rot(i08), 'o', 'MarkerSize', 11, 'LineWidth', 2, ...
     'Color',[0.8 0.2 0.2], 'MarkerFaceColor',[0.9 0.4 0.4], ...
     'DisplayName','\tau = 0.8');

% Ideal corner (min of each axis)
tran_star = min(tran);
rot_star  = min(rot);

plot(tran_star, rot_star, 'k*', 'MarkerSize', 12, 'LineWidth', 1.5, ...
     'DisplayName','Ideal corner');

% Optional guide line
plot([tran(i08) tran_star], [rot(i08) rot_star], ':', 'Color',[0.3 0.3 0.3]);

grid on; box on;

xlabel('MAE_{trans} (cm)  \downarrow');
ylabel('MAE_{rot} (deg)   \downarrow');
title('Pareto Trade-off: Translation vs Rotation');

xlim([min(tran)-0.1*range(tran), max(tran)+0.1*range(tran)]);
ylim([min(rot)-0.1*range(rot),  max(rot)+0.1*range(rot)]);
legend('Location','northeast');

% Optional saving:
% exportgraphics(figX, 'figX_tau_score.pdf', 'ContentType','vector');
% exportgraphics(figX, 'figX_tau_score.png', 'Resolution', 300);
% exportgraphics(figY, 'figY_pareto.pdf', 'ContentType','vector');
% exportgraphics(figY, 'figY_pareto.png', 'Resolution', 300);

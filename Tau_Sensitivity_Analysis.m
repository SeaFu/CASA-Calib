% ================================================================
% τ-sweep 相對於 baseline(τ=0) 的百分比變化
% 定義：%Δ = (baseline − current) / baseline × 100
% 正值 = 改善（誤差下降）；負值 = 退步（誤差上升）
% 兩條曲線皆用同一個百分比 y 軸。會畫 0% 參考虛線。
% ================================================================

clear; close all; clc;

% ----- Data（表格）-----
tau = 0:0.1:0.9;

MAE_rot_deg = [ ...
    0.114107827, 0.116952593, 0.148851881, 0.149515865, 0.155641969, ...
    0.133834580, 0.134272463, 0.120763643, 0.122512957, 0.132753268];

MAE_trans_cm = [ ...
    3.788937262, 3.922781357, 3.807671352, 3.553581045, 3.699753129, ...
    3.116013644, 2.955340867, 2.913516577, 2.763030405, 3.323161248];

% ----- 以 τ=0 為 baseline 計算百分比變化（只顯示 0.1~0.9）-----
idx = 2:numel(tau);                 % 對應 tau=0.1:0.1:0.9
tau_s = tau(idx);

rot_base  = MAE_rot_deg(1);
tran_base = MAE_trans_cm(1);

pct_rot  = (rot_base  - MAE_rot_deg(idx)) ./ rot_base  * 100;  % %
pct_tran = (tran_base - MAE_trans_cm(idx)) ./ tran_base * 100;  % %

% ----- 繪圖 -----
figure('Color','w','Position',[120 120 820 520]); hold on; box on; grid on;

% 0% 參考線
yline(0,'--','Color',[0.4 0.4 0.4],'LineWidth',1);

% 兩條百分比曲線（共用同一個 y 軸）
p1 = plot(tau_s, pct_tran, '-o', 'LineWidth', 2, 'MarkerSize', 6, ...
          'Color', [0.00 0.45 0.74]);
p2 = plot(tau_s, pct_rot,  '-s', 'LineWidth', 2, 'MarkerSize', 6, ...
          'Color', [0.85 0.33 0.10]);

xlabel('IoU threshold \tau');
ylabel('Relative change vs \tau=0  (%)');
title('Relative Error Change vs \tau (baseline = \tau=0)');

legend([p1 p2], {'Translation MAE','Rotation MAE'}, 'Location','best');

% 自動留白一點，避免貼邊
all_pct = [pct_tran(:); pct_rot(:)];
yl = [min(all_pct) max(all_pct)];
margin = 0.1 * max(1, range(yl));    % 至少保留 ±0.1%
ylim([yl(1)-margin, yl(2)+margin]);

% -----（可選）輸出圖檔 -----
% print(gcf, 'tau_sweep_relative_to_tau0.png', '-dpng', '-r300');
% print(gcf, 'tau_sweep_relative_to_tau0.pdf', '-dpdf');

%% =========================
%  Tau Sensitivity Plots
%  Fig X: Distance-to-ideal score S(tau) with stability band (<=2% worse)
%  Fig Y: Pareto plot (MAE_trans vs MAE_rot), highlight tau=0.8
%  --------------------------------------------
%  If you prefer loading from CSV, see the section near the end.
%  Output files: figX_tau_score.(pdf|png), figY_pareto.(pdf|png)
%  --------------------------------------------
clear; clc; close all;

%% ====== Data (from your validation sweep) ======
% tau, MAE_rot_deg, MAE_trans_cm
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
  0.9  0.132753268  3.323161248];

tau  = D(:,1);
rot  = D(:,2);  % deg
tran = D(:,3);  % cm

% Sanity print
fprintf('tau  rot(deg)   trans(cm)\n');
disp([tau rot tran]);

%% ====== Relative improvement vs tau=0 (optional numbers for text) ======
rot0  = rot(tau==0);
tran0 = tran(tau==0);
imp_rot  = (rot0  - rot)./rot0  * 100;   % higher = better
imp_tran = (tran0 - tran)./tran0 * 100;  % higher = better

i08 = find(abs(tau-0.8)<1e-9,1);
fprintf('\nAt tau=0.8: ΔMAE_trans = %+0.1f%% (%.3f -> %.3f cm)\n', ...
    imp_tran(i08), tran0, tran(i08));
fprintf('At tau=0.8: ΔMAE_rot   = %+0.1f%% (%.4f -> %.4f deg)\n', ...
    imp_rot(i08), rot0, rot(i08));

%% ====== Min–max normalization (lower is better) ======
z_rot  = (rot - min(rot))   ./ (max(rot)  - min(rot));
z_tran = (tran - min(tran)) ./ (max(tran) - min(tran));

%% ====== Composite scores ======
S_sum   = 0.5*z_tran + 0.5*z_rot;                   % equal-weight additive
S_geo   = sqrt(max(z_tran.*z_rot, 0));              % geometric mean (not used as main)
S_dist  = sqrt(z_tran.^2 + z_rot.^2);               % distance to ideal (recommended)

[ Smin, idxMin ]   = min(S_dist);
tau_best           = tau(idxMin);

% Stability band: within <= 2% of the minimum
stable_mask = S_dist <= Smin * 1.02;
stable_min  = min(tau(stable_mask));
stable_max  = max(tau(stable_mask));

fprintf('\nBest by distance score: tau = %.1f (Smin = %.3f)\n', tau_best, Smin);
fprintf('Stability band (<=2%% worse than min): [%.1f, %.1f]\n', stable_min, stable_max);

%% ====== Fig X: Score vs tau with stability band ======
figX = figure('Color','w','Position',[100 100 920 500]);

% Shade stability band
hold on;
yl = [min(S_dist)*0.9, max(S_dist)*1.1];
patch([stable_min stable_max stable_max stable_min], [yl(1) yl(1) yl(2) yl(2)], ...
      [0.9 0.9 0.9], 'EdgeColor','none', 'FaceAlpha',0.6, 'DisplayName','Stability band (<=2%)');

% Plot distance score
p1 = plot(tau, S_dist, '-o', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName','Distance-to-ideal score S(\tau)');
% (Optional) also show additive score for reference
p2 = plot(tau, S_sum,  '--s', 'LineWidth', 1.5, 'MarkerSize', 5, 'DisplayName','Additive score (ref)');

% Vertical line at tau=0.8
xline(0.8, ':', 'LineWidth', 1.5, 'Color',[0.2 0.2 0.2], 'DisplayName','\tau=0.8');

% Annotate best point
plot(tau_best, Smin, 'p', 'MarkerSize', 12, 'LineWidth', 2, 'Color',[0.8 0.2 0.2], ...
     'MarkerFaceColor',[0.9 0.4 0.4], 'DisplayName','Best');
text(tau_best, Smin, sprintf('  \\leftarrow BEST (%.1f)', tau_best), 'FontSize', 11, 'Color',[0.5 0 0]);

grid on; box on; ylim(yl);
xlabel('\tau', 'FontSize', 12);
ylabel('Score (lower is better)', 'FontSize', 12);
title('Sensitivity of \tau via Distance-to-Ideal Score', 'FontSize', 13);
legend('Location','northwest');

% Save
% exportgraphics(figX, 'figX_tau_score.pdf', 'ContentType','vector');
% exportgraphics(figX, 'figX_tau_score.png', 'Resolution', 300);

%% ====== Fig Y: Pareto plot (MAE_trans vs MAE_rot) with ideal point ======
figY = figure('Color','w','Position',[120 120 560 520]); hold on;

% Scatter all tau points
scatter(tran, rot, 50, 'filled', 'MarkerFaceColor',[0.2 0.4 0.8], 'DisplayName','\tau candidates');

% Label each point with tau value
for i=1:numel(tau)
    text(tran(i)+0.02*range(tran), rot(i), sprintf('\\tau=%.1f', tau(i)), 'FontSize', 9, 'Color',[0.1 0.1 0.1]);
end

% Highlight tau=0.8
plot(tran(i08), rot(i08), 'o', 'MarkerSize', 11, 'LineWidth', 2, ...
     'Color',[0.8 0.2 0.2], 'MarkerFaceColor',[0.9 0.4 0.4], 'DisplayName','\tau=0.8');

% Ideal point (min of each axis)
tran_star = min(tran);
rot_star  = min(rot);
plot(tran_star, rot_star, 'k*', 'MarkerSize', 12, 'LineWidth', 1.5, 'DisplayName','Ideal corner');

% (Optional) dashed guide from tau=0.8 to ideal corner
plot([tran(i08) tran_star], [rot(i08) rot_star], ':', 'Color',[0.3 0.3 0.3]);

% Axes & labels (lower-left is better)
grid on; box on;
xlabel('MAE_{trans} (cm) \downarrow', 'FontSize', 12);
ylabel('MAE_{rot} (deg) \downarrow',  'FontSize', 12);
title('Pareto Trade-off of Rotation vs Translation', 'FontSize', 13);

% Tight limits with margins
xlim([min(tran)-0.1*range(tran), max(tran)+0.1*range(tran)]);
ylim([min(rot) -0.1*range(rot),  max(rot) +0.1*range(rot)]);

legend('Location','northeast');

% Save
% exportgraphics(figY, 'figY_pareto.pdf', 'ContentType','vector');
% exportgraphics(figY, 'figY_pareto.png', 'Resolution', 300);

%% ====== (Optional) Load from CSV instead of hard-coding ======
% CSV format expected: columns named tau, MAE_rot_deg, MAE_trans_cm
%{
T = readtable('your_tau_results.csv');
tau  = T.tau;
rot  = T.MAE_rot_deg;
tran = T.MAE_trans_cm;
% Then re-run the blocks above from "Relative improvement..." onward.
%}

% disp('Done. Files written: figX_tau_score.(pdf|png), figY_pareto.(pdf|png)');


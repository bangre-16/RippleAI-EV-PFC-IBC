% =========================================================================
% ROBUST 2000-SAMPLE DATASET EXTRACTOR FOR PFC INTERLEAVED BOOST CONVERTER
% =========================================================================
clear; clc;

model_name = 'PFC_IBC_AC_Model';

if ~bdIsLoaded(model_name)
    load_system(model_name);
end

set_param(model_name, 'ZeroCrossControl', 'DisableAll');

N_samples = 2000;
rng(42);

% 1. Define Parameter Ranges
fp_range      = [18000, 35000];       % Switching Frequency: 18 kHz to 35 kHz[cite: 1]
RL_range      = [150, 220];           % Load Resistance: 150 to 220 Ohms[cite: 1]
Vin_rms_range = [195, 245];           % Grid AC: 195V to 245V RMS[cite: 1]
D_range       = [58.0, 63.0];         % Duty Cycle: 58% to 63%[cite: 1]
L_range       = [700e-6, 800e-6];     % Boost Inductance: 700 uH to 800 uH[cite: 1]

% 2. Uniform Latin Hypercube Sampling[cite: 1]
samples = lhsdesign(N_samples, 5);

fp_arr      = fp_range(1)      + samples(:, 1) * (fp_range(2) - fp_range(1));
RL_arr      = RL_range(1)      + samples(:, 2) * (RL_range(2) - RL_range(1));
Vin_rms_arr = Vin_rms_range(1) + samples(:, 3) * (Vin_rms_range(2) - Vin_rms_range(1));
Vin_pk_arr  = Vin_rms_arr * sqrt(2);
D_arr       = D_range(1)       + samples(:, 4) * (D_range(2) - D_range(1));
L_arr       = L_range(1)       + samples(:, 5) * (L_range(2) - L_range(1));

% 3. Extraction Loop
dataset = zeros(N_samples, 7);

fprintf('Starting physical dataset extraction for %d iterations...\n', N_samples);
fprintf('Progress: ');

hWait = waitbar(0, 'Extracting simulation dataset...');
tic;

for k = 1:N_samples
    assignin('base', 'fp_val', fp_arr(k));
    assignin('base', 'RL_val', RL_arr(k));
    assignin('base', 'Vin_peak_val', Vin_pk_arr(k));
    assignin('base', 'D_val', D_arr(k));
    assignin('base', 'L_val', L_arr(k));
    
    try
        simOut = sim(model_name, 'StopTime', '0.06');
        
        io_ts = simOut.get('Io_out');
        t = io_ts.Time;
        i_data = io_ts.Data;
        
        % Extract steady-state window (last 10 ms cycle)
        idx = find(t >= 0.05);
        i_window = i_data(idx);
        
        delta_Io = max(i_window) - min(i_window);
        i_avg    = mean(i_window);
        vo_avg   = i_avg * RL_arr(k);
        
        dataset(k, :) = [fp_arr(k), RL_arr(k), Vin_rms_arr(k), D_arr(k), L_arr(k)*1e6, delta_Io, vo_avg];
    catch
        dataset(k, :) = [fp_arr(k), RL_arr(k), Vin_rms_arr(k), D_arr(k), L_arr(k)*1e6, NaN, NaN];
    end
    
    if mod(k, 50) == 0 || k == N_samples
        waitbar(k/N_samples, hWait, sprintf('Extracted %d / %d samples...', k, N_samples));
        fprintf('%d.. ', k);
    end
end

close(hWait);
elapsed_time = toc;
fprintf('\nCompleted in %.2f seconds.\n', elapsed_time);

% 4. Save to CSV
valid_idx = ~isnan(dataset(:, 6));
clean_data = dataset(valid_idx, :);

dataTable = array2table(clean_data, 'VariableNames', { ...
    'Switching_Freq_Hz', ...
    'Load_Resistance_Ohm', ...
    'Vin_RMS_V', ...
    'Duty_Cycle_Pct', ...
    'Inductance_uH', ...
    'Current_Ripple_A', ...
    'Vout_DC_V'});

writetable(dataTable, 'dataset_pfc_ibc_2000.csv');
fprintf('Saved %d clean rows to "dataset_pfc_ibc_2000.csv".\n', height(dataTable));
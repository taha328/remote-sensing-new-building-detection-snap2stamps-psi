function export_ps_points()
% Export a minimal, parseable PSI point table from a completed StaMPS run.
%
% This script runs inside a StaMPS processing directory after stamps(1,8)
% and writes plain CSV outputs without relying on MATLAB table support.

outdir = fullfile(pwd, 'export');
if exist(outdir, 'dir') ~= 7
    mkdir(outdir);
end

psver_struct = load('psver.mat');
psver = psver_struct.psver;

ps = load(sprintf('ps%d.mat', psver));
n_ps = double(ps.n_ps);

point_id = (1:n_ps)';
x_local_m = nan(n_ps, 1);
y_local_m = nan(n_ps, 1);
if isfield(ps, 'xy') && size(ps.xy, 2) >= 3
    x_local_m = double(ps.xy(:, 2));
    y_local_m = double(ps.xy(:, 3));
end

lon = double(ps.lonlat(:, 1));
lat = double(ps.lonlat(:, 2));

temporal_coherence = nan(n_ps, 1);
pmfile = sprintf('pm%d.mat', psver);
if exist(pmfile, 'file') == 2
    pm = load(pmfile);
    if isfield(pm, 'coh_ps')
        temporal_coherence = double(pm.coh_ps(:));
    end
end

scene_elevation_m = nan(n_ps, 1);
hgtfile = sprintf('hgt%d.mat', psver);
if exist(hgtfile, 'file') == 2
    hgt = load(hgtfile);
    if isfield(hgt, 'hgt')
        scene_elevation_m = double(hgt.hgt(:));
    end
end

dem_error_phase_per_m = nan(n_ps, 1);
scla_candidates = {sprintf('scla_smooth%d.mat', psver), sprintf('scla%d.mat', psver)};
for k = 1:numel(scla_candidates)
    if exist(scla_candidates{k}, 'file') == 2
        scla = load(scla_candidates{k});
        if isfield(scla, 'K_ps_uw')
            dem_error_phase_per_m = double(scla.K_ps_uw(:));
            break;
        end
        if isfield(scla, 'K_ps')
            dem_error_phase_per_m = double(scla.K_ps(:));
            break;
        end
    end
end

mean_velocity_mm_yr = nan(n_ps, 1);
try
    ps_plot('v-do', -1);
    velocity_plot = load('ps_plot_v-do.mat');
    if isfield(velocity_plot, 'ph_disp')
        velocity_values = velocity_plot.ph_disp;
        if size(velocity_values, 2) >= 1
            mean_velocity_mm_yr = double(velocity_values(:, 1));
        end
    end
catch ME
    fprintf('Warning: unable to export velocity from ps_plot(''v-do''): %s\n', ME.message);
end

master_day = repmat(double(ps.master_day), n_ps, 1);
n_ifg = repmat(double(ps.n_ifg), n_ps, 1);
n_image = repmat(double(ps.n_image), n_ps, 1);

% These fields are intentionally left empty until a project-specific
% emergence derivation is implemented from actual StaMPS time-series outputs.
pre_stability_fraction = nan(n_ps, 1);
post_stability_fraction = nan(n_ps, 1);
residual_height_m = nan(n_ps, 1);

points_csv = fullfile(outdir, 'ps_points.csv');
fid = fopen(points_csv, 'w');
if fid < 0
    error('Unable to open ps_points.csv for writing');
end
fprintf(fid, 'point_id,x_local_m,y_local_m,lon,lat,temporal_coherence,scene_elevation_m,dem_error_phase_per_m,mean_velocity_mm_yr,pre_stability_fraction,post_stability_fraction,first_stable_epoch,residual_height_m,master_day,n_ifg,n_image\n');
for i = 1:n_ps
    fprintf(
        fid,
        '%d,%.6f,%.6f,%.10f,%.10f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%s,%.6f,%.0f,%.0f,%.0f\n',
        point_id(i),
        x_local_m(i),
        y_local_m(i),
        lon(i),
        lat(i),
        temporal_coherence(i),
        scene_elevation_m(i),
        dem_error_phase_per_m(i),
        mean_velocity_mm_yr(i),
        pre_stability_fraction(i),
        post_stability_fraction(i),
        '',
        residual_height_m(i),
        master_day(i),
        n_ifg(i),
        n_image(i)
    );
end
fclose(fid);

timeseries_csv = fullfile(outdir, 'ps_timeseries.csv');
fid = fopen(timeseries_csv, 'w');
if fid < 0
    error('Unable to open ps_timeseries.csv for writing');
end
fprintf(fid, 'point_id,epoch,metric_name,value\n');
fclose(fid);
end

export_ps_points();

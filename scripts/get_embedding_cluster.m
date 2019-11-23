function get_embedding_cluster(ckpt, attr_bins, attr_centers, margin, maxDist)
if ~exist('ckpt', 'var') || isempty(ckpt)
    ckpt = '../checkpoints/elo_UTK_cnn/50_net.pth';
end
if ~exist('attr_bins', 'var') || isempty(attr_bins)
    attr_bins = [1 21 41 61 81];
end
if ~exist('attr_centers', 'var') || isempty(attr_centers)
    attr_centers = [10 30 50 70 90];
end
if ~exist('margin', 'var') || isempty(margin)
    margin = (attr_bins(2)-attr_bins(1))/2;
end
if ~exist('maxDist', 'var') || isempty(maxDist)
    maxDist = 2 * margin;
end
min_kept = 10;
max_kept = 50;

%%
[ckpt_dir, epoch, ~] = fileparts(ckpt);
f = readNPY(fullfile(ckpt_dir, sprintf('features_%s.npy', epoch(1:end-4))));
l = readNPY(fullfile(ckpt_dir, sprintf('labels_%s.npy', epoch(1:end-4))));

% RANSAC
points = [l, f];
fitLineFcn = @(points) polyfit(points(:,1), points(:,2), 1);
evalLineFcn = @(model, points) sum((points(:, 2) - polyval(model, points(:,1))).^2, 2);
[model, ix] = ransac([l, f], fitLineFcn, evalLineFcn, 4, maxDist);

% fprintf('corr: %.4f\nmean: %.4f\nstd : %.4f\n', abs(corr(l, f, 'type', 'Spearman')), mean(f), std(f));
fprintf('corr: %.4f \n--embedding_mean %.4f \\\n--embedding_std %.4f \\\n', abs(corr(l, f, 'type', 'Spearman')), mean(f), std(f));

figure;
plot(l, f, '.');
hold on;
x = linspace(attr_bins(1), attr_bins(end), 100);
y = model(1)*x+model(2);

plot(l(~ix), f(~ix), 'g.')

emb = attr_centers;
% fprintf('clusters: [')
fprintf('--embedding_bins "[')
for i = 1:length(attr_centers)
    c = attr_centers(i);
    idx = find(abs(l-c) < margin & ix);
    if length(idx) < min_kept
        idx = find(abs(l-c) < (attr_bins(2)-attr_bins(1))/2 & ix);
    end
    if length(idx) > max_kept
        idx = idx(randperm(length(idx), max_kept));
    end
    fc = f(idx);
    mf = mean(fc);
    emb(i) = mf;
    fprintf('%.4f', mf);
    if i ~= length(attr_centers)
        fprintf(', ');
    end
    plot(c, mf, 'r*')
end
% fprintf(']\n\n')
fprintf(']" \\\n\n')
% disp(emb)

fprintf('Attribute bins:\n')
x = attr_centers;
fprintf('"[')
for i = 1:numel(x)
    fprintf('%.4f', x(i));
    if i ~= length(y)
        fprintf(', ');
    end
end
fprintf(']"\n\n')

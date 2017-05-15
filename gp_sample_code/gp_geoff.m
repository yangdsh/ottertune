n = 101;
k = 1;
xs = randn(n, k);
ys = xs(:,1) + cos(3*xs(:,1)) + xs(:,1).^2/2 + randn(n,1)*.1;

%%

sigma = 1.07;
ridge = .1 * (xs(:,1) < 0) + .0001 * (xs(:,1) >= 0);

gx = sort([(-3.5:.01:3.5)'; xs(:,1)]);  % I added the sample points to the grid so that we can compare predicted v. actual y-values at the sample points
gx = [gx zeros(length(gx), k-1)];

K = exp(-pdist2(xs, xs)/sigma) + diag(ridge);
K2 = exp(-pdist2(xs, gx)/sigma);
K3 = exp(-pdist2(gx, gx)/sigma);

yhat = K2' * (K \ ys);
sigma = sqrt(diag(K3 - K2'*(K\K2)));

plot(xs(:,1), ys, 'o');
co = get(gca, 'colororder');
line(gx(:,1), yhat, 'color', co(2,:), 'linewidth', 2);
line(gx(:,1), yhat+1.96*sigma, 'color', co(2,:))
line(gx(:,1), yhat-1.96*sigma, 'color', co(2,:))

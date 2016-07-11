function [yhats,sigmas,eips] = gp(xs,ys,xt,ridge,nfeats) 
batch_size = 3000;
xs=reshape(xs,[],nfeats);
xs=cell2mat(xs);
xt=reshape(xt,[],nfeats);
xt=cell2mat(xt);
%if size(ys) == [1,1]
%disp('here')
%ys = [ys{1,1}]
%disp(ys)
%disp(class(ys))
%disp(size(ys))
%else
ys = cell2mat(ys');
%end
ridge = cell2mat(ridge');
%k = 10;
%xs = randn(n, k);
%ys = xs(:,1) + cos(3*xs(:,1)) + xs(:,1).^2/2 + randn(n,1)*.1;

%%

sigma_cl = 1.07;
y_best = min(ys);
%ridge = .1 * (xs(:,1) < 0) + .0001 * (xs(:,1) >= 0);
%ridge = .1 * (xs(:,1) < 0) + .0001 * (xs(:,1) >= 0);

%gx = sort([(-3.5:.01:3.5)'; xs(:,1)]);  % I added the sample points to the grid so that we can compare predicted v. actual y-values at the sample points
%gx = [gx zeros(length(gx), k-1)];

K = exp(-pdist2(xs, xs)/sigma_cl) + diag(ridge);

arr_offset = 1;
yhats = zeros(size(xt,1),1);
sigmas = zeros(size(xt,1),1);
eips = zeros(size(xt,1),1);
n_xt = size(xt,1);
while arr_offset <= n_xt
    if arr_offset + batch_size > n_xt
        end_offset = n_xt;
    else
        end_offset = arr_offset + batch_size ;
    end
    xt_ = xt(arr_offset:end_offset,:);
    K2 = exp(-pdist2(xs, xt_)/sigma_cl);
    K3 = exp(-pdist2(xt_, xt_)/sigma_cl);

    yhat = K2' * (K \ ys);
    sigma = sqrt(diag(K3 - K2'*(K\K2)));
    u = (y_best - yhat) ./ sigma;
    phi1 = 0.5 * erf(u/sqrt(2))+0.5;
    phi2 = (1/sqrt(2*pi))*exp((u.^2) *-0.5);
    eip = sigma .* (u .* phi1 + phi2);

    yhats(arr_offset:end_offset,:) = yhat; %[yhats ; yhat];
    sigmas(arr_offset:end_offset,:) = sigma; %[sigmas ; sigma];
    eips(arr_offset:end_offset,:) = eip;
    arr_offset = end_offset + 1;
end
%disp(size(yhats));
%disp(size(sigmas));
%disp(sigma_cl)


%plot(xs(:,1), ys, 'o');
%co = get(gca, 'colororder');
%line(gx(:,1), yhats, 'color', co(2,:), 'linewidth', 2);
%line(gx(:,1), yhats+1.96*sigmas, 'color', co(2,:))
%line(gx(:,1), yhats-1.96*sigmas, 'color', co(2,:))

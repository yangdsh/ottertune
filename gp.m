function [yhats,sigmas,eips] = gp(xs,ys,xt,ridge,nfeats) 
batch_size = 3000;
xs=reshape(xs,[],nfeats);
xs=cell2mat(xs);
xt=reshape(xt,[],nfeats);
xt=cell2mat(xt);
ys = cell2mat(ys');
ridge = cell2mat(ridge');

%%

sigma_cl = 1.07;
y_best = min(ys);

K = exp(-pdist2(xs, xs)/sigma_cl) + diag(ridge);

arr_offset = 1;
yhats = zeros(size(xt,1),1);
sigmas = zeros(size(xt,1),1);
eips = zeros(size(xt,1),1);
n_xt = size(xt,1);`
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

    yhats(arr_offset:end_offset,:) = yhat;
    sigmas(arr_offset:end_offset,:) = sigma;
    eips(arr_offset:end_offset,:) = eip;
    arr_offset = end_offset + 1;
end

%plot(xs(:,1), ys, 'o');
%co = get(gca, 'colororder');
%line(gx(:,1), yhats, 'color', co(2,:), 'linewidth', 2);
%line(gx(:,1), yhats+1.96*sigmas, 'color', co(2,:))
%line(gx(:,1), yhats-1.96*sigmas, 'color', co(2,:))

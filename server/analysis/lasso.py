'''
Created on Jul 8, 2016

@author: dvanaken
'''

from collections import Counter
import numpy as np
import os.path
#from sklearn.linear_model import enet_path
from sklearn.linear_model import lasso_path
from sklearn.preprocessing import StandardScaler

from .matrix import Matrix
from .preprocessing import PolynomialFeatures, Shuffler, dummy_encoder_helper, DummyEncoder
from .util import stdev_zero, stopwatch

def get_coef_range(X, y):
    print "starting experiment"
    with stopwatch("lasso paths"):
        #alphas, coefs, dual_gaps = enet_path(X, y, 
        #                                     l1_ratio=1.0, 
        #                                     verbose=True,
        #                                     return_models=False, 
        #                                     positive=False, 
        #                                     max_iter=1000)

        alphas, coefs, dual_gaps = lasso_path(X, y, 
                                             #l1_ratio=1.0, 
                                             verbose=True,
                                             #return_models=False, 
                                             positive=False, 
                                             max_iter=1000)

    return alphas, coefs, dual_gaps

def run_lasso(dbms, basepaths, savedir, featured_metrics, knobs_to_ignore,
              include_polynomial_features=True):
    import gc

    # Load matrices
    assert len(basepaths) > 0
    Xs = []
    ys = []
    
    with stopwatch("matrix concatenation"):
        for basepath in basepaths:
            X_path = os.path.join(basepath, "X_data_enc.npz")
            y_path = os.path.join(basepath, "y_data_enc.npz")
            
            Xs.append(Matrix.load_matrix(X_path))
            ys.append(Matrix.load_matrix(y_path).filter(featured_metrics,
                                                        "columns"))

        # Combine matrix data if more than 1 matrix
        if len(Xs) > 1:
            X = Matrix.vstack(Xs, require_equal_columnlabels=True)
            y = Matrix.vstack(ys, require_equal_columnlabels=True)
        else:
            X = Xs[0]
            y = ys[0]
        del Xs
        del ys
        gc.collect()
    
    with stopwatch("preprocessing"):
        # Filter out columns with near zero standard
        # deviation (i.e., constant columns)
        if y.shape[1] > 1:
            column_mask = ~stdev_zero(y.data, axis=0)
            filtered_columns = y.columnlabels[column_mask]
            y = y.filter(filtered_columns, 'columns')
        column_mask = ~stdev_zero(X.data, axis=0)
        removed_columns = X.columnlabels[~column_mask]
        print "removed columns = {}".format(removed_columns)
        filtered_columns = set(X.columnlabels[column_mask])
        filtered_columns -= set(knobs_to_ignore)
        filtered_columns = np.array(sorted(filtered_columns))
        X = X.filter(filtered_columns, 'columns')
        print "\ncolumnlabels:",X.columnlabels
        
        # Dummy-code categorical features
        n_values,cat_feat_indices,_  = dummy_encoder_helper(dbms, X.columnlabels)
        if len(cat_feat_indices) > 0:
            encoder = DummyEncoder(n_values, cat_feat_indices)
            encoder.fit(X.data, columnlabels=X.columnlabels)
            X = Matrix(encoder.transform(X.data),
                       X.rowlabels,
                       encoder.columnlabels)
        
        # Scale the data
        X_standardizer = StandardScaler()
        X.data = X_standardizer.fit_transform(X.data)
        y_standardizer = StandardScaler()
        y.data = y_standardizer.fit_transform(y.data)
        if include_polynomial_features:
            X_poly = PolynomialFeatures()
            X_data = X_poly.fit_transform(X.data)
            X_columnlabels = np.expand_dims(np.array(X.columnlabels, dtype=str), axis=0)
            X_columnlabels = X_poly.fit_transform(X_columnlabels).squeeze()
            X = Matrix(X_data, X.rowlabels, X_columnlabels)

        # Shuffle the data rows (experiments x metrics)
        shuffler = Shuffler(shuffle_rows=True, shuffle_columns=False)
        X = shuffler.fit_transform(X, copy=False)
        y = shuffler.transform(y, copy=False)
        assert np.array_equal(X.rowlabels, y.rowlabels)
        gc.collect()
        
    print "\nfeatured_metrics:",featured_metrics

    with stopwatch("lasso paths"):
        # Fit the model to calculate the components
        alphas, coefs, _ = get_coef_range(X.data, y.data)
    # Save model
    np.savez(os.path.join(savedir, "lasso_path.npz"),
             alphas=alphas,
             coefs=coefs,
             feats=X.columnlabels,
             metrics=y.columnlabels)
    
    with stopwatch("lasso processing"):
        nfeats = X.columnlabels.shape[0]
        lasso = Lasso(alphas, X.columnlabels, coefs)
        print lasso.get_top_summary(nfeats, "")
        top_knobs = get_features_list(lasso.get_top_features(n=nfeats))
        print "\nfeat list length: {}".format(len(top_knobs))
        print "nfeats = {}".format(nfeats)
        top_knobs = lasso.get_top_features(nfeats)
        print top_knobs
        final_ordering = []
        for knob in top_knobs:
            if '#' in knob:
                knob = knob.split('#')[0]
                if knob not in final_ordering:
                    final_ordering.append(knob)
            else:
                final_ordering.append(knob)
        final_ordering = np.append(final_ordering, removed_columns)
    with open(os.path.join(savedir, "featured_knobs.txt"), "w") as f:
        f.write("\n".join(final_ordering))

class Lasso(object):

    def __init__(self, alphas, column_labels, coefs):
        # alphas in descending order
        self._alphas = alphas
        self._column_labels = column_labels
        self._coefs = coefs
        self._fallout_idxs = []
        self._sorted_fallout_idxs = []
        sorted_idx_ctr = Counter()
        for coef in self._coefs:
            fallouts = self._get_fallouts(coef)
            self._fallout_idxs.append(fallouts)
            sorted_fallouts = self._get_sorted_fallouts(coef,fallouts)
            self._sorted_fallout_idxs.append(sorted_fallouts)
            sorted_idx_ctr[tuple(sorted_fallouts[:10])] += 1

        # Fallout order is always really close for all knobs, but may
        # differ by 1-2 columns. Use majority vote to decide which
        # ordering to use.
        self._master_sorted_idxs = np.array([])
        if len(self._sorted_fallout_idxs) == len(sorted_idx_ctr):
            self._master_sorted_idxs = self._sorted_fallout_idxs[0]
            self._master_idxs = self._fallout_idxs[0]
            self._master_coefs = self._coefs[0]
        else:
            master = sorted_idx_ctr.most_common(1)[0][0]
            for sort_idxs,idxs,coef in zip(self._sorted_fallout_idxs, \
                    self._fallout_idxs,self._coefs):
                m = tuple(sort_idxs[:10])
                if m == master:
                    self._master_sorted_idxs = sort_idxs
                    self._master_idxs = idxs
                    self._master_coefs = coef
                    break
        if self._master_sorted_idxs.shape[0] == 0:
            raise Exception()


    def _get_fallouts(self,coefs):
        i = 0
        zero_arr = np.zeros((len(self._alphas,)))
        fallout_indices = np.empty((coefs.shape[0]))
        for i,param_coefs in enumerate(coefs):
            near_zero = np.isclose(param_coefs,zero_arr)
            fallout_indices[i] = np.count_nonzero(near_zero)-1
        return fallout_indices

    def _get_sorted_fallouts(self,coefs,fallout_idxs):
        if len(fallout_idxs) == len(set(fallout_idxs)):
            # no need to break ties
            return np.argsort(fallout_idxs)
        sorted_fallouts = []
        
        coef_matrix = np.hstack([coefs,np.zeros((len(self._column_labels),1))])
        for i in range(len(self._alphas)):
            mask = fallout_idxs == i
            nz_count = np.count_nonzero(mask)
            if nz_count == 0:
                continue
            else:
                coefs = coef_matrix[:,i+1][mask]
                nz_idxs = np.nonzero(mask)
                sort_idxs = np.argsort(np.absolute(coefs))[::-1]
                sorted_fallouts.extend(nz_idxs[0][sort_idxs])
        return np.array(sorted_fallouts)
            

    def get_top_features(self,n=10):
        return self._column_labels[self._master_sorted_idxs][:n]

    def get_top_coefs(self,n=10):
        return self._master_coefs[self._master_sorted_idxs,:][:n,:]

#    def plot_path(self,title=None,savepath=None,show=False):
#        title_base = "Lasso Paths"
#        if title:
#            title = "{}: {}".format(title_base,title)
#        else:
#            title = title_base
#        top_coefs = self.get_top_coefs()
#        top_features = self.get_top_features()
#        plt.figure(1)
#        ax = plt.gca()
#        ax.set_color_cycle(2 * ['b', 'r', 'g', 'c', 'k'])
#        l1 = plt.plot(-np.log10(self._alphas), top_coefs.T)
#
#        plt.xlabel('-Log(alpha)')
#        plt.ylabel('coefficients')
#        plt.title(title)
#        plt.legend(top_features, loc='upper left', fontsize='x-small')
#        plt.axis('tight')
#        if savepath:
#            plt.savefig(savepath) 
#        if show:
#            plt.show()

    def get_top_summary(self, n=20, title=None):
        summary = ""
        top_feats = self.get_top_features(n=n)

        # Use coefs corresponding to last alpha s.t. all n top feats in regression
        alpha_idx = self._master_idxs[self._master_sorted_idxs][:n][-1]
        max_alpha_idx = len(self._alphas) - 1
        alpha_idx = alpha_idx+1 if alpha_idx < max_alpha_idx else max_alpha_idx
        top_coefs = self.get_top_coefs(n=n)
        print "top_coefs_shape={}, alpha_idx={}, n={}".format(top_coefs.shape, alpha_idx, n)
        top_coefs = top_coefs[:,int(alpha_idx)]

        # Print summary
        c1 = len("RANK") + 2
        c2 = max([len(feat) for feat in top_feats]) + 2
        c3 = len("COEF") + 1
        summary += ("{rank:{w1}}{feat:{w2}}{coef:{w3}}\n").format(
                rank="RANK",w1=c1,feat="FEATURES",w2=c2,coef="COEF",w3=c3)
        table_width = len(summary)
        sep="{n:{fil}{al}{w}}\n".format(w=table_width, al='<', fil='-', n="")
        summary = sep + summary + sep
        if title:
#             sep2="{n:{fil}{al}{w}}\n".format(w=table_width, al='<', fil='*', n="")
            summary = sep + "{0:{al}{w}}\n".format(title.upper(),al="^",w=table_width) + summary
        for i,feat,coef in zip(range(n),top_feats,top_coefs):
            summary += "{0:{al}{w1}}{1:{al}{w2}}{2:+0.3f}\n".format(str(i+1) + ".",feat,coef,w1=c1,w2=c2,w3=c3,al="<")
        return summary

def merge_top(top_feats_list,n=10):
    all_feats = set()
    for fl in top_feats_list:
        all_feats.update(fl)
    all_feats = sorted(list(all_feats))
    #num_workloads = len(top_feats_list)
    num_feats = len(all_feats)

    counts = np.zeros((num_feats,num_feats))
    for feat_list in top_feats_list:
        print "len feat_list = {}".format(len(feat_list))
        print "num_feats = {}".format(num_feats)
        assert len(feat_list) <= num_feats
        for i,feat in enumerate(all_feats):
            try:
                rank = feat_list.index(feat)
                counts[rank,i] += 1
            except ValueError:
                continue

    ranked_idxs = []
    for i,row in enumerate(counts):
        winning_index = np.argmax(row)
        ranked_idxs.append(winning_index)
        counts[:,winning_index] = 0
        if i < num_feats - 1:
            counts[i+1] += row
    assert len(ranked_idxs) == len(all_feats)
    ranked_feats = np.array(all_feats)[ranked_idxs]

    split_feats = get_features_list(ranked_feats,split=True)
    _,feats_indices = np.unique(split_feats,return_index=True)
    ranked_split_feats = np.array(split_feats)[sorted(feats_indices)]
    return ranked_feats,ranked_split_feats

def get_features_list(features, split=False):
    split_feats = []
    for feat in features:
        if split and '*' in feat:
            split_feats.extend(feat.split("*"))
        else:
            split_feats.append(feat)
    return split_feats

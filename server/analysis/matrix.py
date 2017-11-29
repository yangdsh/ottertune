'''
Created on Mar 23, 2016

@author: dvanaken
'''

import numpy as np

from .util import array_tostring

class Matrix(object):
    """
    Attributes:
    - __data: m x n matrix
    - __rowlabels: vector of length m
    - __columnlabels: vector of length n
    """
    
    def __init__(self, data, rowlabels, columnlabels, copy=False):
        if copy:
            self.__data = data.copy()
            self.__rowlabels = rowlabels.copy()
            self.__columnlabels = columnlabels.copy()
        else:
            self.__data = data
            self.__rowlabels = rowlabels
            self.__columnlabels = columnlabels
        self._check_invariants()
    
    def get_data(self):
        return self.__data
    
    def set_data(self, newdata):
        self.__data = newdata
        self._check_invariants()
    
    data = property(get_data, set_data)
        
    def get_rowlabels(self):
        return self.__rowlabels
    
    def set_rowlabels(self, newrowlabels):
        self.__rowlabels = newrowlabels
        self._check_invariants()
    
    rowlabels = property(get_rowlabels, set_rowlabels)

    def get_columnlabels(self):
        return self.__columnlabels
    
    def set_columnlabels(self, newcolumnlabels):
        self.__columnlabels = newcolumnlabels
        self._check_invariants()

    columnlabels = property(get_columnlabels, set_columnlabels)

    @property
    def shape(self):
        self._check_invariants()
        return self.__data.shape
    

    @staticmethod
    def vstack(matrices, require_equal_columnlabels=True):
        """Returns a new matrix equal to the row-wise
        concatenation of the given matrices"""
        assert len(matrices) > 0
        if len(matrices) == 1:
            return matrices[0].copy()
        
        for matrix in matrices:
            assert isinstance(matrix, Matrix)
            if matrix.columnlabels.shape != matrices[0].columnlabels.shape:
                print ""
                print matrix.columnlabels
                print ""
                print matrices[0].columnlabels
                print ""
                print set(matrices[0].columnlabels) - set(matrix.columnlabels)
            assert matrix.columnlabels.shape == matrices[0].columnlabels.shape
            assert matrix.data.shape[1] == matrices[0].data.shape[1]
            if require_equal_columnlabels:
                assert np.array_equal(matrix.columnlabels, matrices[0].columnlabels)

        data = np.vstack([mtx.data for mtx in matrices])
        rowlabels = np.hstack([mtx.rowlabels for mtx in matrices])
        return Matrix(data, rowlabels, matrices[0].columnlabels.copy())
        
    
    @staticmethod
    def hstack(matrices, require_equal_rowlabels=True):
        """Returns a new matrix equal to the column-wise
        concatenation of the given matrices"""
        assert len(matrices) > 0
        if len(matrices) == 1:
            return matrices[0].copy()
        
        for matrix in matrices:
            assert isinstance(matrix, Matrix)
            assert matrix.data.shape[0] == matrices[0].data.shape[0]
            assert matrix.rowlabels.shape == matrices[0].rowlabels.shape
            if require_equal_rowlabels:
                assert np.array_equal(matrix.rowlabels, matrices[0].rowlabels)

        data = np.hstack([mtx.data for mtx in matrices])
        columnlabels = np.hstack([mtx.columnlabels for mtx in matrices])
        return Matrix(data, matrices[0].rowlabels.copy(), columnlabels)
    
    def _unique_helper(self, data):
        cdata = np.ascontiguousarray(data).view(np.dtype(
                (np.void, data.dtype.itemsize * data.shape[1])))
        _, unique_indices = np.unique(cdata, return_index=True)
        unique_indices.sort()
        return unique_indices
    
    def unique_rows(self, return_index=False):
        """Returns a new matrix containing the unique rows
        in this matrix"""
        
        unique_indices = self._unique_helper(self.__data)
        if unique_indices.size == self.__data.shape[0]:
            # Rows are already unique
            newmatrix = self.copy()
        else:
            newmatrix = Matrix(self.__data[unique_indices],
                               self.__rowlabels[unique_indices],
                               self.__columnlabels.copy())
        if return_index:
            return newmatrix, unique_indices
        else:
            return newmatrix
    
    def unique_columns(self, return_index=False):
        """Returns a new matrix containing the unique columns
        in this matrix"""
        
        unique_indices = self._unique_helper(self.__data.T)
        if unique_indices.size == self.__data.shape[1]:
            # Columns are already unique
            newmatrix = self.copy()
        else:
            newmatrix = Matrix(self.__data[:, unique_indices],
                               self.__rowlabels.copy(),
                               self.__columnlabels[unique_indices])
        if return_index:
            return newmatrix, unique_indices
        else:
            return newmatrix
    
    def shuffle_matrix(self, rows_or_columns, shuffle_indices=None, seed=None):
        from .preprocessing import get_shuffle_indices
        
        assert rows_or_columns == 'rows' or rows_or_columns == 'columns'
        length = self.shape[0] if rows_or_columns == 'rows' else self.shape[1]
        if shuffle_indices is not None:
            assert shuffle_indices.ndim == 1 and shuffle_indices.size == length
        else:
            shuffle_indices = get_shuffle_indices(length, seed)

        if rows_or_columns == 'rows':
            newmatrix = Matrix(self.data[shuffle_indices],
                               self.rowlabels[shuffle_indices],
                               self.columnlabels.copy())
        else:
            newmatrix = Matrix(self.data[:, shuffle_indices],
                               self.rowlabels.copy(),
                               self.columnlabels[shuffle_indices])
        return newmatrix, shuffle_indices
    
    def _check_invariants(self):
        assert self.__data is not None
        assert self.__rowlabels is not None
        assert self.__columnlabels is not None
        assert self.__data.ndim == 2
        assert self.__rowlabels.ndim == 1
        assert self.__columnlabels.ndim == 1
        numrows, numcolumns = self.__data.shape
        assert self.__rowlabels.shape[0] == numrows
        assert self.__columnlabels.shape[0] == numcolumns
    
    def __eq__(self, other):
        return isinstance(other, Matrix) and \
               np.array_equal(self.__data, other.__data) and \
               np.array_equal(self.__rowlabels, other.__rowlabels) and \
               np.array_equal(self.__columnlabels, other.__columnlabels)
               
    def __str__(self):
        return ("\nDATA:\n{}\n\nROWLABELS:\n{}\n\nCOLUMNLABELS:\n{}\n"
                .format(self.__data, self.__rowlabels, self.__columnlabels))
    
    def copy(self):
        """Returns a copy of this matrix"""
        return Matrix(self.__data.copy(),
                      self.__rowlabels.copy(),
                      self.__columnlabels.copy())

    @staticmethod
    def load_matrix(path):
        with np.load(path) as mtx:
            data = mtx['data']
            rowlabels = mtx['rowlabels']
            columnlabels = mtx['columnlabels']
        return Matrix(data, rowlabels, columnlabels)
    

    def save_matrix(self, path):
        with open(path, 'w') as f:
            np.savez_compressed(f,
                                data=self.__data,
                                rowlabels=self.__rowlabels,
                                columnlabels=self.__columnlabels)
    
    def save_matrix_csv(self, path):
        import csv
#         from .util import array_tostring

        with open(path, 'w') as f:
            writer = csv.writer(f, delimiter=',')
#             writer.writerow(array_tostring(self.rowlabels))
#             writer.writerow(array_tostring(self.columnlabels))
            writer.writerow(self.rowlabels)
            writer.writerow(self.columnlabels)
            writer.writerows(self.data)
    
    @staticmethod
    def load_matrix_csv(path):
        import csv

        with open(path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            rowlabels = np.array(reader[0])
            columnlabels = np.array(reader[1])
        data = np.genfromtxt(path, delimiter=',', skip_header=2)
        return Matrix(data, rowlabels, columnlabels)
    
    def get_membership_mask(self, labels, rows_or_columns):
        assert rows_or_columns in ['rows', 'columns']
        assert isinstance(labels, np.ndarray)
        assert labels.size > 0

        if rows_or_columns == "rows":
            filter_labels = self.rowlabels
        else:
            filter_labels = self.columnlabels

        labels = array_tostring(labels)
        filter_labels = array_tostring(filter_labels)
        
        return np.in1d(filter_labels.ravel(),
                       labels).reshape(filter_labels.shape)
    
    def filter(self, labels, rows_or_columns):
        """Returns a new matrix filtered by either the rows or
           columns given in 'labels'"""

        assert rows_or_columns in ['rows', 'columns']
        logical_filter = self.get_membership_mask(labels, rows_or_columns)

        if rows_or_columns == "rows":
            return Matrix(self.__data[logical_filter],
                          self.__rowlabels[logical_filter],
                          self.__columnlabels)
        else:
            return Matrix(self.__data[:, logical_filter],
                          self.__rowlabels,
                          self.__columnlabels[logical_filter])        
        
                
def matrix_tests():
    A = np.array([[2,7,6],
                  [9,4,1],
                  [0,2,4]])
    A_rl = np.array(['Ar1', 'Ar2', 'Ar3'])
    A_cl = np.array(['Ac1', 'Ac2', 'Ac3'])
    matrix_A = Matrix(A, A_rl, A_cl)
    
    B = np.array([[2,7,6],
                  [9,5,1],
                  [0,2,4]])
    B_rl = np.array(['Br1', 'Br2', 'Br3'])
    B_cl = np.array(['Bc1', 'Bc2', 'Bc3'])
    matrix_B = Matrix(B, B_rl, B_cl)
    
    # Test hstack
    print "Testing 'hstack'..."
    AB_hstack_data_exp = np.array([[2,7,6,2,7,6],
                                   [9,4,1,9,5,1],
                                   [0,2,4,0,2,4]])
    A_cl_B_cl_exp = np.array(['Ac1', 'Ac2', 'Ac3', 'Bc1', 'Bc2', 'Bc3'])
    AB_hstack_exp = Matrix(AB_hstack_data_exp, A_rl, A_cl_B_cl_exp)
    AB_hstack = Matrix.hstack([matrix_A, matrix_B], require_equal_rowlabels=False)
    assert AB_hstack_exp == AB_hstack
    try:
        Matrix.hstack([matrix_A, matrix_B], require_equal_rowlabels=True)
        print "Failed hstack require equal columns test"
    except AssertionError:
        print "Passed hstack require equal columns test"
        
    print "Passed all tests for 'hstack'"
    print ""
    print "Testing 'vstack'..."
    AB_vstack_data_exp = np.array([[2,7,6],
                                   [9,4,1],
                                   [0,2,4],
                                   [2,7,6],
                                   [9,5,1],
                                   [0,2,4]])
    A_rl_B_rl_exp = np.array(['Ar1', 'Ar2', 'Ar3', 'Br1', 'Br2', 'Br3'])
    AB_vstack_exp = Matrix(AB_vstack_data_exp, A_rl_B_rl_exp, A_cl)
    AB_vstack = Matrix.vstack([matrix_A, matrix_B], require_equal_columnlabels=False)
    assert AB_vstack_exp == AB_vstack
    try:
        Matrix.vstack([matrix_A, matrix_B], require_equal_columnlabels=True)
        print "Failed vstack require equal columns test"
    except AssertionError:
        print "Passed vstack require equal columns test"
        
    print "Passed all tests for 'vstack'"
    print ""
    print "Testing 'unique_rows'..."
    unique_rows_data_exp = np.array([[2,7,6],
                                     [9,4,1],
                                     [0,2,4],
                                     [9,5,1]])
    unique_rl = np.array(['Ar1', 'Ar2', 'Ar3', 'Br2'])
    unique_rows_exp = Matrix(unique_rows_data_exp, unique_rl, A_cl)
    unique_rows, _ = AB_vstack_exp.unique_rows(return_index=True)
    assert unique_rows_exp == unique_rows
    unique_rows = AB_hstack_exp.unique_rows()
    assert AB_hstack_exp == unique_rows
    
    print "Passed all tests for 'unique_rows'"
    print ""
    print "Testing 'unique_columns'..."
    unique_columns_data_exp = np.array([[2,7,6,7],
                                        [9,4,1,5],
                                        [0,2,4,2]])
    unique_cl = np.array(['Ac1', 'Ac2', 'Ac3', 'Bc2'])
    unique_columns_exp = Matrix(unique_columns_data_exp, A_rl, unique_cl)
    unique_columns, _ = AB_hstack_exp.unique_columns(return_index=True)
    assert unique_columns_exp == unique_columns
    unique_columns2 = unique_columns.unique_columns()
    assert unique_columns == unique_columns2
    print "Passed all tests for 'unique_columns'"
    print ""
    
if __name__ == "__main__":
    matrix_tests()

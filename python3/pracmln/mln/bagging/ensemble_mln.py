from scipy import sparse
import numpy as np


class EnsembleMLN(object):
    def __init__(self, mln, database):
        self._mln = mln
        self._materialized_mln = None
        self._grounded_mrf = None
        self._database = database

    def _materialize_mln(self):
        self._materialized_mln = self._mln.materialize()

    def _calculate_sparse_weight_matrix(self):
        val = []
        col_ind = []
        row_ptr = [0]

        for formula in self._materialized_mln.formulas:
            sparse_vector = formula.weight
            vector_index, vector_value = sparse_vector
            val.extend(vector_value)
            col_ind.extend(vector_index)
            row_ptr.append(row_ptr[-1] + len(vector_value))

        return sparse.csr_matrix((val, col_ind, row_ptr))

    def _get_truth_vector(self, world):
        truth_vector = [x(world) for x in self._grounded_mrf.itergroundings()]

        return sparse.csr_matrix(truth_vector)

    def perform_exact_inference_with_given_world(self, world):
        if self._materialized_mln is None:
            self._materialize_mln()

        self._grounded_mrf = self._materialized_mln.ground(self._database)

        sparse_weight_matrix = self._calculate_sparse_weight_matrix()
        truth_vector = self._get_truth_vector(world)

        a = truth_vector.dot(sparse_weight_matrix).todense()
        b = np.exp(a)


from scipy import sparse
import numpy as np


class EnsembleMLN(object):
    def __init__(self, mln, database):
        self._mln = mln
        self._materialized_mln = None
        self._grounded_mrf = None
        self._database = database

    def _materialize_mln(self):
        self._materialized_mln = self._mln.materialize(*self._database)

    def _get_weight_matrix(self):
        weight_matrix = []

        for formula in self._grounded_mrf.formulas:
            weight_matrix.append(formula.weight)

        return np.array(weight_matrix)

    def _get_truth_vector(self, world):
        truth_vector = [x(world) for x in self._grounded_mrf.itergroundings()]

        return sparse.csr_matrix(truth_vector)

    def perform_exact_inference_with_given_world(self,weight_matrix, cw_predicates):
        if self._materialized_mln is None:
            self._materialize_mln()

        self._grounded_mrf = self._materialized_mln.ground(*self._database)
        self._grounded_mrf.apply_cw(*cw_predicates)

        evidence = self._grounded_mrf.evidence
        target0_evidence = evidence.copy()
        target1_evidence = evidence.copy()

        target0_evidence[-2] = 1
        target0_evidence[-1] = 0

        target1_evidence[-2] = 0
        target1_evidence[-1] = 1

        world0 = []
        world1 = []

        # global GLOBAL_MRF
        # if GLOBAL_MRF is None:
        #    GLOBAL_MRF = list(mrf.itergroundings())

        # for x in GLOBAL_MRF:
        for x in self._grounded_mrf.itergroundings():
            world0.append(x(target0_evidence))
            world1.append(x(target1_evidence))

        truth_vectors = np.array([world0, world1])
        temp_result = truth_vectors.dot(weight_matrix)

        return np.argmax(temp_result,axis=0)


class EnsembleMLNSparse(object):
    def __init__(self, mln, database):
        self._mln = mln
        self._materialized_mln = None
        self._grounded_mrf = None
        self._database = database

    def _materialize_mln(self):
        self._materialized_mln = self._mln.materialize(*self._database)

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

        return np.exp(a)


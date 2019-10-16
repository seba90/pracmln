from os import path

from pracmln.mln.base import parse_mln
from scipy import sparse


def test_inference():
    mln_file = open(path.join(path.dirname(__file__), 'bagging.mln'))
    mln = parse_mln(mln_file.read())
    materialized_mln = mln.materialize()
    mln_file.close()

    val = []
    col_ind = []
    row_ptr = [0]

    for formula in materialized_mln.formulas:
        sparse_vector = eval(formula.weight)
        vector_index, vector_value = sparse_vector
        val.extend(vector_value)
        col_ind.extend(vector_index)
        row_ptr.append(row_ptr[-1]+len(vector_value))

    sparse_weight_matrix = sparse.csr_matrix((val,col_ind,row_ptr))
    assert 0 == 0


test_inference()

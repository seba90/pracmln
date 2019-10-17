from os import path

from pracmln.mln.base import parse_mln
from scipy import sparse
from pracmln.mln.database import Database
import numpy as np

from python3.pracmln.mln.bagging.ensemble_mln import EnsembleMLN


def test_inference():
    mln_file = open(path.join(path.dirname(__file__), 'bagging.mln'))
    mln = parse_mln(mln_file.read())
    materialized_mln = mln.materialize()
    mrf = materialized_mln.ground(Database(materialized_mln))
    mln_file.close()

    ensemble = EnsembleMLN(mln, Database(materialized_mln))
    ensemble.perform_exact_inference_with_given_world(list(mrf.worlds())[1])

    assert 0 == 0


test_inference()

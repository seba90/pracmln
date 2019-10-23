from os import path

from pracmln.mln.base import parse_mln
from python3.pracmln.mln.efficient_database_parser import use_mln_to_parse_from_file_path_with_multi_cpu


def test_use_mln_to_parse_from_file_path_with_multi_cpu():

    mln_file = open(path.join(path.dirname(__file__), 'test_mlns', 'cat-in-the-data-test.mln'))
    mln = parse_mln(mln_file.read())
    mln_file.close()

    db_file_path = path.join(path.dirname(__file__), 'test_dbs', 'cat-in-the-data-small.db')
    dbs = use_mln_to_parse_from_file_path_with_multi_cpu(mln, db_file_path)

    assert len(dbs) == 1588

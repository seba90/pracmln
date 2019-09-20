from pracmln.mln.base import parse_mln
import os.path as path


def test_performance_parse_mln():
    with open(path.join(path.dirname(__file__),"performance_test.mln"), "r") as mln_file:
        parse_mln(mln_file.read())

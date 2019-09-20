from pracmln.mln import MLNParsingError
from pracmln.mln.base import parse_mln
import os.path as path
from os import listdir

TEST_MLNS_FOLDER_PATH = path.join(path.dirname(__file__),"test_mlns")


def test_performance_parse_mln():
    num_wrong_parsed_mlns = 0

    for mln_filename in listdir(TEST_MLNS_FOLDER_PATH):
        # Remove if condition if you want to measure the performance of the parser
        if mln_filename == 'performance_test.mln': continue

        mln_file_path = path.join(TEST_MLNS_FOLDER_PATH, mln_filename)
        with open(mln_file_path, "r") as mln_file:

            try:
                parse_mln(mln_file.read())
            except MLNParsingError as e:
                print("Parsing Error in: " + mln_file_path)
                print(e)
                num_wrong_parsed_mlns += 1

    assert num_wrong_parsed_mlns == 0
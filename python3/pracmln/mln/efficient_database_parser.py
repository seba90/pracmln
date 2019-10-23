import multiprocessing
from psutil._compat import xrange
from pracmln.mln.database import parse_db

CPUS = multiprocessing.cpu_count()


def use_mln_to_parse_from_file_path_with_multi_cpu(mln, database_file_path):
    print('Parsing dbs ...')
    db_file = open(database_file_path)
    dbs_content = db_file.read().split('---\n')
    splitted_dbs_content = split_list_in_equal_n_chunks(dbs_content, CPUS)
    db_file.close()

    manager = multiprocessing.Manager()
    shared_list = manager.list()
    jobs = []

    for i in range(CPUS):
        p = multiprocessing.Process(target=parse_databases_in_thread, args=(splitted_dbs_content[i], mln, shared_list))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    result_list = []

    for l in shared_list:
        result_list += l

    print('Done Parsing dbs')

    return result_list


def parse_databases_in_thread(databases_content, mln, databases_parsed_result):
    dbs = []
    for content in databases_content:
        dbs.append(parse_db(mln, content)[0])
        if (len(dbs) % 1000) == 0:
            print(len(dbs))

    databases_parsed_result.append(dbs)


def split_list_in_equal_n_chunks(l, n):
    k, m = divmod(len(l), n)

    return [l[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in xrange(n)]

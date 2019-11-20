import json
import random
import time
from os import path, listdir

import csv

from pracmln.mln.base import parse_mln
from python3.pracmln.mln.bagging.ensemble_mln import EnsembleMLN
from python3.pracmln.mln.efficient_database_parser import use_mln_to_parse_from_file_path_with_multi_cpu

import pickle
import numpy as np
from python3.pracmln.mln.learning import BPLL
from python3.pracmln.mln.util import mergedom, mergedom_on_reference
from collections import defaultdict

GLOBAL_MRF = None
PATH_TO_ENSEMBLE = path.join(path.dirname(__file__), 'ensemble')

def get_training_databases(mln):
    PICKLE_FILE_PATH = path.join(path.dirname(__file__), 'result.p')
    if path.exists(PICKLE_FILE_PATH):
        dbs = pickle.load(open(PICKLE_FILE_PATH, "rb"))
    else:
        dbs = use_mln_to_parse_from_file_path_with_multi_cpu(mln, path.join(path.dirname(__file__), 'cat-in-the-data.db'))
        pickle.dump(dbs, open(PICKLE_FILE_PATH, "wb"))

    return dbs


def get_test_databases(mln):
    PICKLE_FILE_PATH = path.join(path.dirname(__file__), 'test.p')
    if path.exists(PICKLE_FILE_PATH):
        dbs = pickle.load(open(PICKLE_FILE_PATH, "rb"))
    else:
        dbs = use_mln_to_parse_from_file_path_with_multi_cpu(mln, path.join(path.dirname(__file__), 'cat-in-the-data-test.db'))
        pickle.dump(dbs, open(PICKLE_FILE_PATH, "wb"))

    return dbs


def infer_target(mln, db, cw_predicates):
    mrf = mln.ground(db)

    mrf.apply_cw(*cw_predicates)
    weights = mrf._weights()

    evidence = mrf.evidence
    target0_evidence = evidence.copy()
    target1_evidence = evidence.copy()

    target0_evidence[-2] = 1
    target0_evidence[-1] = 0

    target1_evidence[-2] = 0
    target1_evidence[-1] = 1

    world0 = []
    world1 = []

    #global GLOBAL_MRF
    #if GLOBAL_MRF is None:
    #    GLOBAL_MRF = list(mrf.itergroundings())

    #for x in GLOBAL_MRF:
    for x in mrf.itergroundings():
        world0.append(x(target0_evidence))
        world1.append(x(target1_evidence))

    truth_vectors = np.array([world0, world1])
    return np.argmax(np.sum(np.multiply(weights,truth_vectors),axis=1))


def test_eval_inference():
    mln_file_template = open(path.join(path.dirname(__file__), 'cat-in-the-data-template.mln'))
    mln_template = parse_mln(mln_file_template.read())
    mln_file_template.close()

    infer_db = get_training_databases(mln_template)[0]
    result = []
    for mln_filename in listdir(PATH_TO_ENSEMBLE):
        if mln_filename == 'bagged_mln.mln' or mln_filename == 'bagged_mln.weights': continue
        mln_file = open(path.join(PATH_TO_ENSEMBLE, mln_filename))
        mln = parse_mln(mln_file.read())
        mln_file.close()
        cw_predicates = [x for x in mln.prednames if x != 'target']
        #mln = mln.materialize(infer_db)


        result.append(infer_target(mln, infer_db, cw_predicates))

    return result

def test_transform_mlns_into_ensemble_mln():
    formulas = defaultdict(list)
    sparse_vectors = {}

    for mln_filename in listdir(PATH_TO_ENSEMBLE):
        if mln_filename == 'bagged_mln.mln' or mln_filename == 'bagged_mln.weights': continue
        mln_file = open(path.join(PATH_TO_ENSEMBLE, mln_filename))
        mln = parse_mln(mln_file.read())

        for formula in mln.formulas:
            formulas[str(formula)].append(formula.weight)

        mln_file.close()

    weight_matrix = []

    with open(path.join(PATH_TO_ENSEMBLE, 'bagged_mln.mln'), "w") as template_file:
        mln_file = open(path.join(PATH_TO_ENSEMBLE, 'trained_1.mln'))
        mln = parse_mln(mln_file.read())
        mln_file.close()

        for predicate in mln.predicates:
            template_file.write('{}\n'.format(str(predicate)))

        for formula, weights in formulas.items():
            template_file.write('{} {}\n'.format(0., formula))
            weight_matrix.append(weights)

    with open(path.join(PATH_TO_ENSEMBLE, 'bagged_mln.weights'), "w") as template_file:
        template_file.write(str(weight_matrix).replace("'",''))



    #Right now no csr representation is required
    #formulas_with_zero = 0
    #for formula, weights in formulas.items():
    #    index = []
    #    values = []
#
#        for i, value in enumerate(weights):
#            if value != 0.:
#                index.append(i)
#                values.append(value)
#        sparse_vectors[formula] = [index, values]
#        if len(index) < 100:
#            formulas_with_zero += 1

    pass


def test_eval_inference_bagging():
    print('Inference ...')

    mln_file = open(path.join(PATH_TO_ENSEMBLE, 'bagged_mln.mln'))
    mln = parse_mln(mln_file.read())
    mln_file.close()

    weights_file = open(path.join(PATH_TO_ENSEMBLE, 'bagged_mln.weights'))
    weight_matrix = np.array(json.loads(weights_file.read()))
    weights_file.close()

    cw_predicates = [x for x in mln.prednames if x != 'target']

    infer_db = get_training_databases(mln)[0]
    ensemble = EnsembleMLN(mln, [infer_db])
    return ensemble.perform_exact_inference_with_given_world(weight_matrix, cw_predicates)


def test_inference():
    mln_file = open(path.join(path.dirname(__file__), 'trained.mln'))
    mln = parse_mln(mln_file.read())
    mln_file.close()
    cw_predicates = [x for x in mln.prednames if x != 'target']
    dbs = get_test_databases(mln)


    print('Creating samples ...')
    print(len(dbs))
    print('Done')

    target_result = []

    i = 0

    mln = mln.materialize()

    for db in dbs:
        if i % 1000 == 0:
            print(i)
        target_result.append(infer_target(mln, db, cw_predicates))
        i += 1

    PICKLE_FILE_PATH = path.join(path.dirname(__file__), 'lolol.p')
    pickle.dump(target_result, open(PICKLE_FILE_PATH, "wb"))
    print(len(target_result))


    #world = [random.randint(0, 1) for _ in range(4)]

    #ensemble = EnsembleMLN(mln, dbs)
    #print('Inference ...')
    #print(ensemble.perform_exact_inference_with_given_world(world))

    assert 0 == 0


def transform_to_csv():
    PICKLE_FILE_PATH = path.join(path.dirname(__file__), 'lolol.p')
    vector = pickle.load(open(PICKLE_FILE_PATH, "rb"))
    id = [x for x in range(300000, 500000)]
    len(id)
    with open(path.join(path.dirname(__file__), 'result.csv'), mode='w') as employee_file:
        employee_writer = csv.writer(employee_file, delimiter=',')

        employee_writer.writerow(['id', 'target'])

        for i, v in zip(id, vector):
            employee_writer.writerow([i, v])


def test_learn():

    mln_file = open(path.join(path.dirname(__file__), 'cat-in-the-data-template.mln'))
    mln = parse_mln(mln_file.read())
    mln_file.close()

    mln_domains = mln.domains
    dbs = get_training_databases(mln)
    dbs = dbs[:25]
    start = time.time()
    time.clock()
    print('Merging Domains ...')
    for db in dbs:
        mergedom_on_reference(mln_domains, db.domains)
    print('Merged Domains')

    print('Materializing MLN ...')
    materialized_mln = mln.materialize()
    print('Materialized')

    total_time = 0
    for x in range(500):
        trained_mln = None
        while True:
            try:
                print('MLN_{}'.format(x))
                print('Creating samples ...')
                sample_size = random.choices(dbs, k=int(len(dbs)))
                print('Sample Size: {}'.format(len(sample_size)))
                print('Done')

                params = {}
                params['verbose'] = True
                params['multicore'] = True
                trained_mln = materialized_mln.learn(sample_size, BPLL, **params)
            except:
                continue
            break

        trained_mln.tofile(path.join(path.dirname(__file__),'ensemble' ,'trained_{}.mln'.format(x)))
        elapsed = time.time() - start
        total_time += elapsed
        print("loop cycle time: %f, seconds count: %02d" % (time.clock(), elapsed))
    assert 0 == 0

    print("loop cycle time: %f, seconds count: %02d" % (1.0, total_time))

def test_eval():
    print('EVAL Regular MLN ...')
    start = time.time()
    time.clock()

    result = test_eval_inference()

    elapsed = time.time() - start
    print("loop cycle time: %f, seconds count: %02d" % (time.clock(), elapsed))

    print('EVAL bagged MLN ...')
    start = time.time()
    time.clock()

    bagged_result = test_eval_inference_bagging()

    elapsed = time.time() - start
    print("loop cycle time: %f, seconds count: %02d" % (time.clock(), elapsed))

    print(np.array_equal(result, bagged_result))

#test_learn()
#test_transform_mlns_into_ensemble_mln()
test_eval()
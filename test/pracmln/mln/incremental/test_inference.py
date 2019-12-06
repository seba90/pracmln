import json
import random
from sklearn.utils import shuffle
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score
import time
from os import path, listdir
import statistics

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
        dbs = use_mln_to_parse_from_file_path_with_multi_cpu(mln, path.join(path.dirname(__file__), 'nursery.db'))
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


def is_ensemble_mln(mln_filename):
    return mln_filename.startswith('bagged_mln')

def infer_target(mln, db, cw_predicates):
    mrf = mln.ground(db)

    mrf.apply_cw(*cw_predicates)
    weights = [float(x) for x in mrf._weights()]

    evidence = mrf.evidence
    class_ground_atoms = []

    for gndatoms in mrf.gndatoms:
        if gndatoms.predname == 'class':
            class_ground_atoms.append(gndatoms)


    target0_evidence = evidence.copy()
    target1_evidence = evidence.copy()
    target2_evidence = evidence.copy()
    target3_evidence = evidence.copy()

    target0_evidence[class_ground_atoms[0].idx] = 1
    target0_evidence[class_ground_atoms[1].idx] = 0
    target0_evidence[class_ground_atoms[2].idx] = 0
    target0_evidence[class_ground_atoms[3].idx] = 0

    target1_evidence[class_ground_atoms[0].idx] = 0
    target1_evidence[class_ground_atoms[1].idx] = 1
    target1_evidence[class_ground_atoms[2].idx] = 0
    target1_evidence[class_ground_atoms[3].idx] = 0

    target2_evidence[class_ground_atoms[0].idx] = 0
    target2_evidence[class_ground_atoms[1].idx] = 0
    target2_evidence[class_ground_atoms[2].idx] = 1
    target2_evidence[class_ground_atoms[3].idx] = 0

    target3_evidence[class_ground_atoms[0].idx] = 0
    target3_evidence[class_ground_atoms[1].idx] = 0
    target3_evidence[class_ground_atoms[2].idx] = 0
    target3_evidence[class_ground_atoms[3].idx] = 1



    world0 = []
    world1 = []
    world2 = []
    world3 = []

    #global GLOBAL_MRF
    #if GLOBAL_MRF is None:
    #    GLOBAL_MRF = list(mrf.itergroundings())

    #for x in GLOBAL_MRF:
    for x in mrf.itergroundings():
        world0.append(x(target0_evidence))
        world1.append(x(target1_evidence))
        world2.append(x(target2_evidence))
        world3.append(x(target3_evidence))

    truth_vectors = np.array([world0, world1,world2,world3])
    argmax = np.argmax(np.sum(np.multiply(weights, truth_vectors), axis=1))

    return str(class_ground_atoms[argmax])


def test_eval_inference(k=0):
    print('Inference with MLN_{}'.format(k))
    mln_file_template = open(path.join(path.dirname(__file__), 'ensemble', 'trained_{}.mln'.format(k)))
    mln_template = parse_mln(mln_file_template.read())
    mln_file_template.close()

    _, infer_dbs = get_training_and_test_set_in_n_chunks()
    result = []
    mln_number = 0
    start = time.time()
    time.clock()

    cw_predicates = [x for x in mln_template.prednames if x != 'class']

    total = len(infer_dbs)
    correct = 0
    wrong = 0
    xval_result = defaultdict(lambda: defaultdict(int))
    predictions = []
    truths = []

    for db in infer_dbs:
        truth = ''
        class_predicate = ''


        for key in db.evidence.keys():
            if key.startswith('class'):
                #class(vgood)
                class_predicate =  key
                truth = key.split('(')[1].replace(')','')
                break
        truths.append(class_predicate)
        db.retract(class_predicate)
        mln = mln_template.materialize(db)
        prediction = infer_target(mln, db, cw_predicates)
        predictions.append(prediction)

        if class_predicate == prediction:
            correct += 1
        else:
            wrong += 1
        # if is_ensemble_mln(mln_filename): continue
        # mln_number += 1
        # mln_file = open(path.join(PATH_TO_ENSEMBLE, mln_filename))
        # mln = parse_mln(mln_file.read())
        # mln_file.close()
        # cw_predicates = [x for x in mln.prednames if x != 'target']
        #
        # result.append(infer_target(mln, infer_db, cw_predicates))
        # if (mln_number % 100) == 0:
        #     elapsed = time.time() - start
        #     print("Run Number %f: ,loop cycle time: %f, seconds count: %02d" % (mln_number,time.clock(), elapsed))

    print('Accuracy:', accuracy_score(truths, predictions))
    print('F1 score:', f1_score(truths, predictions, average='micro'))
    print('Recall:', recall_score(truths, predictions, average='micro'))
    print('Precision:', precision_score(truths, predictions, average='micro'))
    print('\n clasification report:\n', classification_report(truths, predictions))
    print('\n confussion matrix:\n', confusion_matrix(truths, predictions))

    return predictions, truths

def test_transform_mlns_into_ensemble_mln():
    formulas = defaultdict(list)

    mln_number = 0
    for mln_filename in listdir(PATH_TO_ENSEMBLE):
        if is_ensemble_mln(mln_filename): continue
        mln_number += 1
        mln_file = open(path.join(PATH_TO_ENSEMBLE, mln_filename))
        mln = parse_mln(mln_file.read())

        for formula in mln.formulas:
            formulas[str(formula)].append(formula.weight)

        mln_file.close()

        if (mln_number % 100) == 0:
            weight_matrix = []

            with open(path.join(PATH_TO_ENSEMBLE, 'bagged_mln_{}.mln'.format(mln_number)), "w") as template_file:
                mln_file = open(path.join(PATH_TO_ENSEMBLE, 'trained_1.mln'))
                mln = parse_mln(mln_file.read())
                mln_file.close()

                for predicate in mln.predicates:
                    template_file.write('{}\n'.format(str(predicate)))

                for formula, weights in formulas.items():
                    template_file.write('{} {}\n'.format(0., formula))
                    weight_matrix.append(weights)

            with open(path.join(PATH_TO_ENSEMBLE, 'bagged_mln_{}.weights'.format(mln_number)), "w") as template_file:
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

    result = []
    infer_db = get_training_databases("lol")[0]

    for x in range(100, 10100, 100):
        start = time.time()
        time.clock()

        mln_file = open(path.join(PATH_TO_ENSEMBLE, 'bagged_mln_{}.mln'.format(x)))
        mln = parse_mln(mln_file.read())
        mln_file.close()

        weights_file = open(path.join(PATH_TO_ENSEMBLE, 'bagged_mln_{}.weights'.format(x)))
        weight_matrix = np.array(json.loads(weights_file.read()))
        weights_file.close()

        cw_predicates = [x for x in mln.prednames if x != 'target']

        ensemble = EnsembleMLN(mln, [infer_db])
        result = ensemble.perform_exact_inference_with_given_world(weight_matrix, cw_predicates)
        elapsed = time.time() - start
        print("FAST: Run Number %f: ,loop cycle time: %f, seconds count: %f" % (x, time.clock(), elapsed))
    return result


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

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def get_training_and_test_set_in_n_chunks(chunks=1):
    dbs_loaded = get_training_databases("")
    filtered_databases = defaultdict(list)
    training_set = []
    test_set = []
    shuffled_databases = {}

    for db in dbs_loaded:
        atom = list(db.gndatoms(['class']))[0]
        filtered_databases[atom[0]].append(db)

    for key, value in filtered_databases.items():
        shuffled_databases[key] = shuffle(value, random_state=42)

    for shuffled_database in shuffled_databases.values():
        training_set_size = int(len(shuffled_database)*0.8)
        training_set.extend(shuffled_database[:training_set_size])
        test_set.extend(shuffled_database[training_set_size:])

    return shuffle(training_set, random_state=42), shuffle(test_set, random_state=42)

def test_learn(k=1):
    training_set, _ = get_training_and_test_set_in_n_chunks(k)
    dbs_splitted = split(training_set,k)
    mln_file = open(path.join(path.dirname(__file__), 'nursery-template.mln'))
    mln = parse_mln(mln_file.read())
    mln_file.close()

    mln_domains = mln.domains

    #
    print('Merging Domains ...')
    for db in training_set:
        mergedom_on_reference(mln_domains, db.domains)
    print('Merged Domains')

    print('Materializing MLN ...')
    materialized_mln = mln.materialize()
    print('Materialized')

    x = 0
    for dbs in dbs_splitted:
        total_time = 0

        start = time.time()
        time.clock()
        print('MLN_{}'.format(x))

        params = {}
        params['verbose'] = True
        params['multicore'] = True
        trained_mln = materialized_mln.learn(dbs, BPLL, **params)


        trained_mln.tofile(path.join(path.dirname(__file__),'ensemble' ,'trained_{}.mln'.format(x)))
        elapsed = time.time() - start
        total_time += elapsed
        print("loop cycle time: %f, seconds count: %02d" % (time.clock(), elapsed))
        x += 1
    assert 0 == 0

    print("loop cycle time: %f, seconds count: %02d" % (1.0, total_time))


def test_eval():
    print('EVAL Regular MLN ...')
    #start = time.time()
    #time.clock()

    result = []
    truth = []
    correct = 0
    wrong = 0

    for x in range(10):
        predications, truths = test_eval_inference(x)
        result.append(predications)
        truth = truths

    total = len(truth)

    for x in range(total):
        votes = []
        for prediction in result:
            votes.append(prediction[x])
        vote = statistics.mode(votes)

        if vote == truth[x]:
            correct += 1
        else:
            wrong += 1

    print('Correct:{}  {}'.format(correct, correct / total))
    print('Wrong:{}  {}'.format(wrong, wrong / total))
    print('right/Wrong:{}'.format(correct / wrong))

    #elapsed = time.time() - start
    #print("loop cycle time: %f, seconds count: %02d" % (time.clock(), elapsed))

    print('EVAL bagged MLN ...')
    #start = time.time()
    #time.clock()



    #elapsed = time.time() - start
    #print("loop cycle time: %f, seconds count: %02d" % (time.clock(), elapsed))



test_learn(1)
#test_eval_inference()
#test_transform_mlns_into_ensemble_mln()
#test_eval()
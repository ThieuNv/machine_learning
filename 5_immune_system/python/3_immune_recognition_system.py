# Artificial Immune Recognition System - AIRS 2 (version 2 - better, stable)
# Mien 2 chieu: x thuoc [0,1] , y thuoc [0,1]
# Phan loai thanh 1 trong 2 class sau:
#  A (x thuoc [0.0, 0.4999999], y thuoc [0.0, 0.4999999]) va B (x thuoc [0.5, 1], y thuoc [0.5, 1])

from random import random
from math import sqrt



def random_vector(min_max):
    """
    :param min_max: [ [0,1], [0,1] ]
    :return:
    """
    rand_vector = []
    for i in range(len(min_max)):
        rand_vector.append(min_max[i][0] + (min_max[i][1] - min_max[i][0]) * random())
    return rand_vector


def create_cell(vector, class_label):
    cell = {
        "label": class_label,
        "vector": vector
    }
    return cell


def initialize_cells(domain):
    """
    :param domain: { "A": [[0, 0.4999999], [0, 0.4999999]], "B": [[0.5,1],[0.5,1]]}
    :return: mem_cells :
    [
        { "label": "A", "vector": [0.09, 0.40] }
        { "label": "B", "vector": [0.85, 0.64] }
    ]
    """
    mem_cells = []
    for key, value in domain.iteritems():
        mem_cells.append(create_cell(random_vector([[0, 1], [0, 1]]), key))
    return mem_cells


def generate_random_pattern(domain) :
    """
    :param domain: { "A": [[0, 0.4999999], [0, 0.4999999]], "B": [[0.5,1],[0.5,1]]}
    :return: pattern { "label": A, "vector": [0.01, 0.33] }
    """
    class_label = domain.keys()[int( random() + len(domain) - 1 )]
    pattern = {
        "label": class_label,
        "vector": random_vector(domain[class_label])
    }
    return pattern


def distance(c1, c2):
    """
    :param c1: [0.0, 0.0]
    :param c2: [1.0, 1.0]
    :return:
    """
    sum = 0.0
    for i in range(len(c1)):
        sum += (c1[i] - c2[i]) ** 2.0
    return sqrt(sum)


def stimulate(cells, pattern):
    """
    :param cells: List cac cell
    :param pattern: 1 ma~u
    :return:
    """
    max_dist = distance([0.0, 0.0], [1.0, 1.0])     # 1.4142
    for cell in cells:
        cell["affinity"] = distance(cell["vector"], pattern["vector"]) / max_dist
        cell["stimulation"] = 1.0 - cell["affinity"]

    return cells


def get_most_stimulated_cell(mem_cells, pattern):
    """
    Tim te bao kich thich thag pattern nhat.
    :param mem_cells:
    :param pattern:
    :return:
    """
    mem_cells = stimulate(mem_cells, pattern)
    sorted_cells = sorted(mem_cells, key=lambda elem: elem['stimulation'])

    # Sap xep tang dan` do kich thich --> Lay thag cuoi cung thi max(stimulation)
    return sorted_cells[-1]


def mutate_cell(cell, best_match):
    """
    :param cell: { label, vector }
    :param best_match: { label, vector, affinity, stimulation }
    :return:
    """
    range = 1.0 - best_match["stimulation"]     # > 0 do stimulation < 1.0
    for i, val in enumerate(cell["vector"]):
        min_value = max([ (val - (range / 2)), 0.0 ])
        max_value = min([ (val + (range / 2)), 1.0 ])
        cell["vector"][i] = min_value + (max_value - min_value) * random()
    return cell


def create_arb_pool(pattern, best_match, clone_rate, mutate_rate):
    """
    :param pattern:  { label: A, vector: [0.1, 0.4] }
    :param best_match: { label, vector, affinity, stimulation }
    :param clone_rate: 10
    :param mutate_rate: 2.0
    :return: pool [] (so phan tu la: best_match(cell) + dot biet (num_clones) )
     [  {label, vector}, {label, vector} ,... ]
    """
    pool = []
    pool.append(create_cell(best_match["vector"], best_match["label"]))

    num_clones = int(round((best_match["stimulation"] * clone_rate * mutate_rate), 0))  # num_clones < 20
    for i in range(num_clones):
        cell = create_cell(best_match["vector"], best_match["label"])
        pool.append(mutate_cell(cell, best_match))

    # Se nhan ban so luong cell, dong thoi phai dot bien cell do.
    return pool


def competition_for_resources(pool, clone_rate, max_resources):
    """
    :param pool: [ {label, vector, affinity, stimulation}, {}... ]
    :param clone_rate:  10
    :param max_resources:   150
    :return:
    """
    for cell in pool:
        cell["resources"] = cell["stimulation"] * clone_rate

    pool = sorted(pool, key=lambda elem: elem['resources'])

    total_resources = 0.0
    for cell in pool:
        total_resources += cell["resources"]

    while total_resources > max_resources:
        cell = pool.pop(-1)                     # Xoa de giam resources
        total_resources -= cell["resources"]


def refine_arb_pool(pool, pattern, stim_thresh, clone_rate, max_res):
    """
    :param pool: [ {label, vector}, {label, vector} ... ] --> Tat ca cung 1 label (do cung xuat phat tu best_match)
    :param pattern:
    :param stim_thresh: 0.9
    :param clone_rate: 10
    :param max_res: 150
    :return:
    """
    mean_stim, candidate = 0.0, None
    while mean_stim < stim_thresh:
        pool_with_stimulate = stimulate(pool, pattern)

        pool = sorted(pool_with_stimulate, key=lambda elem: elem['stimulation'])
        candidate = pool[-1]     # Lay thag co max(stimulation)

        sum_stimulation = 0.0
        for cell in pool:
            sum_stimulation += cell["stimulation"]
        mean_stim = sum_stimulation / len(pool)

        if mean_stim < stim_thresh:
            candidate = competition_for_resources(pool, clone_rate, max_res)     # Gan candidate = None
            for i in range(len(pool)):
                cell = create_cell(pool[i]["vector"], pool[i]["label"])
                cell_temp = mutate_cell(cell, pool[i])
                pool.append(cell_temp)

    return candidate


def add_candidate_to_memory_pool(candidate, best_match, mem_cells):
    if candidate["stimulation"] > best_match["stimulation"]:
        mem_cells.append(candidate)


def train_system(mem_cells, domain, num_patterns, clone_rate, mutate_rate, stim_thresh, max_res):
    """
    :param mem_cells: [ { "label": "A", "vector": [0.09, 0.40] }, { "label": "B", "vector": [0.85, 0.64] } ]
    :param domain: { "A": [[0, 0.4999999], [0, 0.4999999]], "B": [[0.5,1],[0.5,1]]}
    :param num_patterns: 50
    :param clone_rate: 10
    :param mutate_rate:
    :param stim_thresh:
    :param max_res:
    :return:

    best_match: { label: A, vector: [0.1, 0.4], affinity: 0.07, stimulation: 0.93 }
    """
    for i in range(num_patterns):
        pattern = generate_random_pattern(domain)
        best_match = get_most_stimulated_cell(mem_cells, pattern)

        if best_match["label"] != pattern["label"]:
            mem_cells.append(create_cell(pattern["vector"], pattern["label"]))
        else :
            if best_match["stimulation"] < 1.0:
                pool = create_arb_pool(pattern, best_match, clone_rate, mutate_rate)            # Tao tap dot bien cua best_match
                candidate = refine_arb_pool(pool,pattern, stim_thresh, clone_rate, max_res)     # Loc tap dot bien so vs pattern
                add_candidate_to_memory_pool(candidate, best_match, mem_cells)

        print " > iterator= %d, mem_cells= %d" %((i+1), len(mem_cells))


def classify_pattern(mem_cells, pattern):
    mem_cells_with_stimulation = stimulate(mem_cells, pattern)
    new_mem_cells = sorted(mem_cells_with_stimulation, key=lambda cell: cell['stimulation'])
    return new_mem_cells[-1]    # return best cell


def test_system(mem_cells, domain, num_trials = 50):
    """
    :param mem_cells:
    :param domain:
    :param num_trials:
    :return:
    """
    correct = 0
    for i in range(num_trials):
        pattern = generate_random_pattern(domain)
        best = classify_pattern(mem_cells, pattern)
        if best["label"] == pattern["label"]:
            correct += 1

    print "Finished test with a score of %d / %d" %(correct, num_trials)
    return correct


def execute(domain, num_patterns, clone_rate, mutate_rate, stim_thresh, max_res, num_trials):
    """
    :param domain:  { "A": [[0, 0.4999999], [0, 0.4999999]], "B": [[0.5,1],[0.5,1]]}
    :param num_patterns:    50
    :param clone_rate:  10
    :param mutate_rate: 2.0
    :param stim_thresh: 0.9
    :param max_res: 150
    :return:
    """
    mem_cells = initialize_cells(domain)

    # Sau num_patterns --> Tao ra duoc so luong mem_cells
    train_system(mem_cells, domain, num_patterns, clone_rate, mutate_rate, stim_thresh, max_res)

    test_system(mem_cells, domain, num_trials)
    return mem_cells


if __name__ == "__main__":
    # problem configuration
    domain = { "A": [[0, 0.4999999], [0, 0.4999999]], "B": [[0.5,1],[0.5,1]]}
    num_patterns = 50

    # algorithm configuration
    clone_rate = 10
    mutate_rate = 2.0
    stim_thresh = 0.9
    max_res = 150
    num_trials = 50

    # execute the algorithm
    execute(domain, num_patterns, clone_rate, mutate_rate, stim_thresh, max_res, num_trials)
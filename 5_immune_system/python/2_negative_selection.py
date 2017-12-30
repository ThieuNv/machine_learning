
from random import random
from math import sqrt

def random_vector(min_max):
    vector = []
    for i in range(len(min_max)):
        vector.append(min_max[i][0] + (min_max[i][1] - min_max[i][0]) * random())
    return vector


def euclidean_distance(c1, c2):
    """
    :param c1: [0.75, 0.06]
    :param c2: [0.6, 0.99]
    :return:
    """
    sum = 0.0
    for i in range(len(c1)):
        sum += (c1[i] - c2[i]) ** 2
    return sqrt(sum)



def matches(vector, dataset, min_dist):
    """
    :param vector:  [0.26, 0.19]
    :param dataset: [ {"vector": [0.28, 0.67]}, {"vector": [0.91, 0.13]}, ...]
    :param min_dist:  0.0
    :return: True / False
    """
    flag = False

    for pattern in dataset:
        dist = euclidean_distance(vector, pattern["vector"])
        if dist <= min_dist:
            flag = True
            break
    return flag


def contains(vector, space):
    """
    :param vector:  [0.26, 0.19]
    :param space:   [ [0.5, 1.0], [0.5, 1.0] ]
    :return: True / False
    """
    flag = True
    for ind, value in enumerate(vector):
        if value < space[ind][0] or value > space[ind][1]:
            flag = False
            break
    return flag




def generate_self_dataset(number_records, self_space, search_space):
    """
    :param number_records:  150
    :param self_space:      [ [0.5, 1.0], [0.5, 1.0] ]
    :param search_space:    [ [0.0, 1.0], [0.0, 1.0] ]
    :return: self_dataset [ { "vector": [0.6, 0.9] }, { "vector": [0.64, 0.71] }, ... ]
    """
    self_dataset = []
    while len(self_dataset) < number_records:
        pattern = {}
        pattern["vector"] = random_vector(search_space)

        if matches(pattern["vector"], self_dataset, 0.0):   # Da~ ton tai trong self_dataset
            continue

        if contains(pattern["vector"], self_space):   # Chua ton tai thi kiem tra xem no co thuoc mien xac dinh ko
            self_dataset.append(pattern)

    return self_dataset


def generate_detectors(max_detectors, search_space, self_dataset, min_dist):
    """
    :param max_detectors:   300
    :param search_space:    [ [0.0, 1.0], [0.0, 1.0] ]
    :param self_dataset:    [ {"vector": [0.56, 0.64]}, {"vector"}: [0.83, 0.64], ....]
    :param min_dist:    0.05
    :return: detectors[]
    """
    detectors = []
    while len(detectors) < max_detectors:
        detector = {}
        detector["vector"] = random_vector(search_space)

        # Neu khong match vs self_dataset voi min_dist = 0.05 va khong match vs detectors voi min_dist = 0.0 thi add
        if not matches(detector["vector"], self_dataset, min_dist):
            if not matches(detector["vector"], detectors, 0.0):
                detectors.append(detector)

    return detectors


def apply_detectors(detectors, bounds, self_dataset, min_dist, trials = 50):
    """
    :param detectors: [ {"vector": [0.28, 0.67]}, {"vector": [0.91, 0.13]}, ...]
    :param bounds:  [ [0.0, 1.0], [0.0, 1.0] ]
    :param self_dataset: [ {"vector": [0.65, 0.51]}, {"vector": [0.91, 0.75]}, ...]
    :param min_dist:    0.05
    :param trials:  50
    :return: number of correct result
    """
    correct = 0
    for i in range(trials):
        input = {}
        input["vector"] = random_vector(bounds)

        actual = "N" if matches(input["vector"], detectors, min_dist) else "S"      # N: Non-self, S: self
        expected = "S" if matches(input["vector"], self_dataset, min_dist) else "N"
        if actual == expected:
            correct += 1
        print "# %d/%d : predicted = %s, expected = %s" %((i+1), trials, actual, expected)

    print "Done. Result: %d / %d" %(correct, trials)

def execute(bounds, self_space, max_detectors, max_self, min_dist, trials):
    """
    :param bounds: search_space
    :param self_space:
    :param max_detectors:
    :param max_self:
    :param min_dist:
    :return:    detectors
    """
    self_dataset = generate_self_dataset(max_self, self_space, bounds)
    print "Done: prepared %d self patterns." %(len(self_dataset))

    detectors = generate_detectors(max_detectors, bounds, self_dataset, min_dist)
    print "Done: prepared %d detectors." %(len(detectors))

    apply_detectors(detectors, bounds, self_dataset, min_dist, trials)
    return detectors

if __name__ == "__main__":

    # problem configuration
    problem_size = 2
    max_self = 500
    search_space = []       # [ [0.0, 1.0], [0.0, 1.0] ]
    self_space = []         # [ [0.5, 1.0], [0.5, 1.0] ]

    for i in range(problem_size):
        search_space.append([0.0, 1.0])
        self_space.append([0.5, 1.0])

    # algorithm configuration
    max_detectors = 1000     # So luong may nhan dang
    min_dist = 0.05         # Khoang cach min

    trials = 1000             # Test

    # execute the algorithm
    execute(search_space, self_space, max_detectors, max_self, min_dist, trials)

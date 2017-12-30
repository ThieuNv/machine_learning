# min(fx) = x1 ^ 2 + x2 ^ 2 + x3 ^ 2    (-5 <= xi <= 5)
# co the thu vs cac dau vao tham so khac nhu:
#   max_gens: So the he doi` sau - 100, 1000, 5000...
#   pop_size: population: 100, 1000, 5000, 10000...
#   clone_factor : 0.1, 0.2, 0.5... chi so clone se~ dung: size(pop_after_clones) = pop_size * clone_factor
#   num_rand: 1, 2, 5. So luong child duoc random them after clone progress

from random import random
from math import exp

def objective_function(vector): # [2.9862668802929733, -4.0627908751049056]
    """ Ham muc tieu """
    fx = 0.0
    for x in vector:
        fx += x**2
    return fx

def decode(bitstring, search_space, bits_per_param):
    # "11000000100101000101010" - 32 bit
    # search_space = [[-5, 5], [-5, 5]]
    # bits_per_param = 16

    # Nhiem vu ham nay: tra ve mang 2 phan tu x1, x2.
    # Vi ban dau ta sinh ra chuoi~ 32 bit ngau nhien.
    # Gio ta can giai ma~ 16 bit dau`, va 16 bit sau.

    vector = []         # 2.9862668802929733,  -4.0627908751049056 (ket qua)
    for i in range(len(search_space)):
        offset, sum = i*bits_per_param, 0.0     # 0  va 0.0   -> 16 va 0.0 (vong lap sau)
        param = bitstring[offset : (offset + bits_per_param)]     # Lay 16 bit dau` --> Lai lay tiep 16 bit sau (cho vong lap sau)
        sum += int(param, 2)
        min, max = search_space[i]

        temp = min + ( (max-min)/ ((2.0 ** bits_per_param) - 1) ) * sum

        vector.append(temp)
    return vector


def random_bitstring(number_bits):      # 32 bit
    bitstring = ''
    for i in range(number_bits):
        x = random()                             # 0.00 -> 1.00
        bitstring += "1" if x < 0.5 else "0"    # x < 0.5 ? "1": "0"
    return bitstring


def evaluate(pop, search_space, bits_per_param):
    """
    :param pop:
    :param search_space:
    :param bits_per_param:
    :return: pop_with_evaluate

    pop[0]: {   dictionary
        bitstring: "00100100000101110010001101001111"
        vector: [-3.5902189669642173, -3.6207370107576105]
        cost: 25.99940873181957
    }
    """
    for elem in pop:
        elem["vector"] = decode(elem["bitstring"], search_space, bits_per_param)
        elem["cost"] = objective_function(elem["vector"])

    return pop


def find_best_cost(pop):
    """
    :param pop:
    :return: elem co min cost
    """
    min = pop[0]
    for elem in pop:
        if min["cost"] > elem["cost"]:
            min = elem
    return min


def get_num_clones(pop_size, clone_factor):
    """
    :param pop_size: 100
    :param clone_factor: 0.1
    :return:
    """
    return int(pop_size * clone_factor)   # 10 (Chi nhan len 10 lan`)


def calculate_affinity(pop):
    """
    :param pop:
    :return: sorted pop
        {
            bitstring: '11000...'
            vector: [x1, x2]
            cost: fx
            affinity: 0.0 --> 1.0
        }
    """

    # Sap xep lai cost tang dan trong pop
    sorted_pop = sorted(pop, key=lambda elem: elem['cost'])

    # Tinh do rong cua cost
    range_cost = sorted_pop[-1]["cost"] - sorted_pop[0]["cost"]

    # So sanh va tinh ai luc
    if range_cost == 0.0:
        for elem in sorted_pop:
            elem["affinity"] = 1.0
    else:
        for elem in sorted_pop:
            elem["affinity"] = 1.0 - elem["cost"]/range_cost

    return sorted_pop


def calculate_mutation_rate(antibody, mutate_factor = -2.5):
    """
    :param antibody: { bitstring: '11....', vector: [x1, x2], cost: fx, affinity: 0.0 -> 1.0 }
    :param mutate_factor:
    :return:
    """
    return exp( mutate_factor * antibody["affinity"])


def point_mutation(bitstring, mutation_rate):
    """
    :param bitstring: '100011...' - 32 bit
    :param mutation_rate: 0.08
    :return:
    """
    child = ""
    for bit in bitstring:
        if random() < mutation_rate:
            child += "0" if bit == "1" else "1"
        else:
            child += bit
    return child


def clone_and_hypermutate(pop, clone_factor):
    """
    :param pop: list cac dict( 'bitstring': '10101...' - 32 bit, 'vector': '[x1, x2]', 'cost': fx)
    :param clone_factor: 0.1
    :return: the he sau
    """
    clones = []
    num_clones = get_num_clones(len(pop), clone_factor)     # 10

    # Tinh ai luc va sap xep lai pop theo cost tang dan
    sorted_pop = calculate_affinity(pop)

    # Nhan ban 10 lan: --> co duoc pop la: 100 * 10 = 1000
    for antibody in sorted_pop:
        m_rate = calculate_mutation_rate(antibody)      # Gia su: 0.08

        # Nhan ban = dot bien cac bit.
        for i in range(num_clones):
            clone = {}                          # clone { bitstring: '11111110100001001000000001111001'}
            clone["bitstring"] = point_mutation(antibody["bitstring"], m_rate)
            clones.append(clone)

    return clones


def get_next_generation_pop(population, pop_size):
    sort_pop = sorted(population, key=lambda elem: elem['cost'])
    return sort_pop[0: pop_size]


def random_insertion(search_space, population, number_random, bits_per_param):
    """
    Sau khi clone ra 1000 child, ta tiep tuc random them : num_rand child nua.
    Roi sap xep lai : 1000 + num_rand (child) nay. Va lay pop moi la: len(population)

    :return: population moi, voi so luong : 1000
    """
    if number_random == 0:
        return population
    else:
        rands = []
        for i in range(number_random):
            bitstr = random_bitstring(len(search_space) * bits_per_param)
            dict = {'bitstring': bitstr}
            rands.append(dict)
        new_rands = evaluate(rands, search_space, bits_per_param)

        # Return population vs cost nho nhat
        new_population = population + new_rands
        return get_next_generation_pop( new_population, len(population))


def search(search_space, max_gens, pop_size, clone_factor, num_rand, bits_per_param=16):
    """
    :param search_space:  [[-5, 5], [-5, 5]]
    :param max_gens:    100
    :param pop_size:    100
    :param clone_factor:    0.1
    :param num_rand:    2
    :param bits_per_param: 16
    :return:

     - Khoi tao random 100 chuoi~ ngau nhien (1 chuoi~ la cac so nhi phan 32 bit) (vi len(search_space) = 2)
    """
    pop = []
    for i in range(pop_size):
        bitstr = random_bitstring(len(search_space) * bits_per_param)
        dict = {'bitstring': bitstr}
        pop.append(dict)

    # Khoi tao xong population, thi se dung ham danh gia voi tung represent (element trong population)
    pop = evaluate(pop, search_space, bits_per_param)

    # Tim best co cost min
    best = find_best_cost(pop)

    # Clone
    for i in range(0, max_gens):
        clones = clone_and_hypermutate(pop, clone_factor)       # The he sau clone: 1000 element
        pop_clones = evaluate(clones, search_space, bits_per_param)

        next_generation_pop = get_next_generation_pop(pop_clones, pop_size)    # The he sau khi danh gia: 100 element

        next_generation_pop_after_insert = random_insertion(search_space, next_generation_pop, num_rand, bits_per_param)

        new_population = next_generation_pop_after_insert + [best]
        best = find_best_cost(new_population)     # Child co cost global min

        print "Gen %d, fx = %f, s=[%f, %f]" %((i+1), best["cost"], best["vector"][0], best["vector"][1])

    return best

if __name__ == '__main__':
    # problem configuration
    problem_size = 2
    search_space = []

    for i in range(problem_size):
        search_space.append([-5, +5])

    # algorithm configuration
    max_gens = 20000
    pop_size = 100
    clone_factor = 0.1
    num_rand = 2

    # execute the algorithm
    best = search(search_space, max_gens, pop_size, clone_factor, num_rand)

    print "Done! Solution: fx = %f, s=[%f, %f]" %(best["cost"], best["vector"][0], best["vector"][1])


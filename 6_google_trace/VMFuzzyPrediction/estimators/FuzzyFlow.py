import pandas as pd
import numpy as np
import math
class FuzzyFlow(object):

    def get_midpoint(self,ptuple):
        return 0.5 * (ptuple[0] + ptuple[1])

    def get_midpoint_vector(self,tuple_vector):
        return [self.get_midpoint(x) for x in tuple_vector];

    def get_fuzzy_class(self,point, partition_size):
        return int(math.floor(point / partition_size))

    def get_fuzzy_dataset(self,data):
        u_class = []
        for item in data:
            u_class.append(self.get_fuzzy_class(item, self.partition_size))
        return u_class

    def mapping_class(self,u_class):
        unique_class = np.unique(u_class)
        index = np.arange(unique_class.shape[0])
        inverted = {}
        mapping = {}
        for idx, val in enumerate(unique_class):
            mapping[val] = idx
            inverted[idx] = val
        return mapping, inverted

    def defuzzy(self,index, inverted, midpoints):
        f_class = inverted[index]
        return midpoints[f_class]
    def fit_transform(self,dat,skip_value=0.25):
        distance = round(dat.max() / (dat.max() / skip_value + 4), 4)
        partition_size = distance
        self.partition_size = partition_size
        umin = math.floor(min(dat))
        umax = math.ceil(max(dat))

        # 2: Partition of universe
        # Method: Dividing in the half-thousands

        nIter = int((umax - umin) / partition_size)
        u_vectorized = []

        for i in range(nIter):
            u_vectorized.append((umin + i * partition_size, umin + (i + 1) * partition_size));

        u_midpoints = self.get_midpoint_vector(u_vectorized)
        u_class = np.array(self.get_fuzzy_dataset(dat), dtype=np.int32)

        u_unique_inverted, u_unique_mapping = self.mapping_class(u_class)
        u_class_transform = [u_unique_inverted[item] for item in u_class]

        self.u_unique_mapping = u_unique_mapping
        self.u_midpoints = u_midpoints

        self.u_class = u_class
        self.u_class_transform = u_class_transform
        return self
    def inverse_transform(self, ypred):
        return [self.defuzzy(item%len(self.u_unique_mapping),self.u_unique_mapping,self.u_midpoints) for item in ypred]

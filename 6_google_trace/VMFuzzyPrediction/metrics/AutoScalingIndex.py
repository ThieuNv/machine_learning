import numpy as np
class ADI(object):
    def __init__(self, lower_bound=0.5, upper_bound = 0.8, metric=None):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.metric = metric
    def util_lv(self, resource_used=None, resource_allocated=None):
        return float(resource_used) / resource_allocated
    def basic_ADI(self, util_lv):
        lv = 0
        if util_lv <= self.lower_bound :
            lv = self.lower_bound - util_lv
        elif util_lv >= self.upper_bound :
            lv = util_lv - self.upper_bound
        return lv
    def calculate_ADI(self, resource_used=None, resource_allocated=None):
        time_used = zip(resource_used,resource_allocated)
        lv_time = [self.basic_ADI(self.util_lv(resource_used=w, resource_allocated=m))for w, m in time_used]
        return lv_time
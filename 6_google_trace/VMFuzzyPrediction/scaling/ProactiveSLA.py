from __future__ import division
import numpy as np
import math
from BaseStrategy import BaseStrategy
class ProactiveSLA(BaseStrategy):
    def __init__(self,max_vms=10, sla = 3.0, past_consecutive_values = 3, capacity_VM = 0.25,metric=None):
        self.max_vms = max_vms
        self.sla = sla
        self.k = past_consecutive_values
        self.metric = metric
        self.capacity_VM = capacity_VM

    def allocate_VM(self, res_consump):
        return math.ceil(res_consump / self.capacity_VM)
    def basic_allocate_VMs(self, resource_used):
        actual_allocated = np.array([self.allocate_VM(item) for item in resource_used])
        return actual_allocated
    def allocate_VMs(self, resource_used=None, resource_predicted = None):
        self.data_used = resource_used
        self.data_pred = resource_predicted
        allocated = np.zeros(len(resource_used))
        allocated[:self.k] = resource_used[:self.k]
        for idx in range(self.k,len(resource_used)):
            allocated[idx] = self.sla*resource_predicted[idx] + \
                             (1.0/ self.k) * sum([max(0,(resource_used[i]-resource_predicted[i]))
                                                  for i in range(idx - self.k, idx)])
        self.allocated_CPU = allocated
        return [self.allocate_VM(item) for item in allocated]



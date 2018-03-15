from scaling.ProactiveSLA import ProactiveSLA
from scaling.BaseStrategy import BaseStrategy
import numpy as np
class ProactiveManager(BaseStrategy):
    def __init__(self, max_vms=10, sla=3.0, past_consecutive_values=3, capacity_VM = [0.25,0.0308], metrics=['CPU','RAM']):
        self.max_vms = max_vms
        self.sla = sla
        self.k = past_consecutive_values
        self.capacity_VM = capacity_VM
        self.metrics = metrics
        self.manager = []
        for idx, metric in enumerate(metrics):
            manager = ProactiveSLA(max_vms,sla,past_consecutive_values,capacity_VM[idx],metric=metric)
            self.manager.append(manager)

    def allocate_VMs(self, resource_used=None, resource_predicted=None):
        if (resource_used is not np.array) or (resource_predicted is not np.array) :
            resource_used = np.array(resource_used)
            resource_predicted = np.array(resource_predicted)
        number_of_VMs = []
        for idx, metric in enumerate(self.metrics):
            allocated = self.manager[idx].allocate_VMs(resource_used[:, idx], resource_predicted=resource_predicted[:, idx])
            number_of_VMs.append(allocated)
        return [max(vms) for vms in zip(*number_of_VMs)]
    def sla_violate(self, allocated_VMs = None, used_VMs=None):
        total_time = len(allocated_VMs)
        if (allocated_VMs is not np.array) or (used_VMs is not np.array):
            allocated_VMs = np.array(allocated_VMs)
            used_VMs = np.array(used_VMs)
        max_VMs_used = [max(used) for used in used_VMs]
        number_of_violate = (max_VMs_used-allocated_VMs)
        return float(len(number_of_violate[number_of_violate>=0])) / total_time






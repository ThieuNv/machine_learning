from scaling.OnDemand import OnDemand
from scaling.BaseStrategy import BaseStrategy
import numpy as np
class OnDemandManager(BaseStrategy):
    def __init__(self, capacity_VM = [0.25,0.0308], metrics=['CPU','RAM']):
        # self.max_vms = max_vms
        # self.sla = sla
        # self.k = past_consecutive_values
        self.capacity_VM = capacity_VM
        self.metrics = metrics
        self.manager = []
        for idx, metric in enumerate(capacity_VM):
            manager = OnDemand(capacity_VM=metric)
            self.manager.append(manager)
    """
        Allocate VMs based on resource metric used
    """
    def allocate_VMs(self, resource_used=None):
        if resource_used is not np.array :
            resource_used = np.array(resource_used)
        number_of_VMs = []
        for idx, metric in enumerate(self.metrics):
            allocated = self.manager[idx].allocate_VMs(resource_used[:, idx])
            number_of_VMs.append(allocated)
        # return [max(vms) for vms in zip(*number_of_VMs)]
        return number_of_VMs
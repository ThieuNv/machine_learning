from abc import abstractmethod
class BaseStrategy(object):
    @abstractmethod
    def basic_allocate_VMs(self, resource_used):
        pass
    def sla_violate(self, allocated = None, used=None):
        number_of_points = len(allocated)
        delta = allocated - used
        violate_count = len(delta[delta>0])
        return float(violate_count) / number_of_points
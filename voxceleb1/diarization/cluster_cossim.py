from .cluster import Cluster

class ClusterCossim(Cluster):

    def matches(self, embedding):
        dist = (self._average * embedding).sum().item()/self._average.norm().item()/embedding.norm().item()
        return dist > self._threshold
        

    

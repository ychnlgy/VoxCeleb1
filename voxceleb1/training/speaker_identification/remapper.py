class ReMapper:

    def __init__(self):
        self._map = {}

    def __len__(self):
        return len(self._map)

    def __getitem__(self, key):
        if key in self._map:
            return self._map[key]
        else:
            self._map[key] = val = len(self._map)
            return val

    def freeze(self):
        self.__getitem__ = self._frozen_getitem

    # === PRIVATE ===

    def _frozen_getitem(self, key):
        return self._map[key]

class ReMapper:

    def __init__(self):
        self._map = {}
        self._locked = True
        self._restore = self._locked

    def activate(self, lock):
        self._restore, self._locked = self._locked, lock
        return self
    
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._locked = self._restore

    def __len__(self):
        return len(self._map)

    def __getitem__(self, key):
        if self._locked or key in self._map:
            return self._map[key]
        else:
            self._map[key] = val = len(self._map)
            return val

    # === PRIVATE ===

    def _frozen_getitem(self, key):
        return self._map[key]

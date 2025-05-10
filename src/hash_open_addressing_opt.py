from src.base_hash import BaseHashTable
import mmh3

class OpenAddrOpt(BaseHashTable):
    EMPTY = object()
    DELETED = object()

    def __init__(self, exp=20, hash_fn=mmh3.hash):
        self.size = 1 << exp
        self.mask = self.size - 1
        self.hash_fn = hash_fn
        self.slots = [OpenAddrOpt.EMPTY] * self.size
        self.data  = [None] * self.size
        self.n = 0
        self.insert_probes = 0
        self.search_probes = 0
        self.collisions = 0

    def _h1(self, key: str) -> int:
        return self.hash_fn(key) & self.mask

    def _h2(self, key: str) -> int:
        return ((self.hash_fn(key[::-1]) >> 1) & self.mask) | 1

    def insert(self, key: str, value: int) -> None:
        if self.n / self.size > 0.5:
            self._resize()
        h1 = self._h1(key)
        h2 = self._h2(key)
        for i in range(self.size):
            idx = (h1 + i * h2) & self.mask
            self.insert_probes += 1
            slot = self.slots[idx]
            if slot not in (OpenAddrOpt.EMPTY, OpenAddrOpt.DELETED) and slot != key:
                self.collisions += 1
            if slot in (OpenAddrOpt.EMPTY, OpenAddrOpt.DELETED) or slot == key:
                if slot in (OpenAddrOpt.EMPTY, OpenAddrOpt.DELETED):
                    self.n += 1
                self.slots[idx] = key
                self.data[idx]  = value
                return
        raise RuntimeError("Hash table is full")

    def search(self, key: str):
        h1 = self._h1(key)
        h2 = self._h2(key)
        for i in range(self.size):
            idx = (h1 + i * h2) & self.mask
            self.search_probes += 1
            slot = self.slots[idx]
            if slot is OpenAddrOpt.EMPTY:
                return None
            if slot == key:
                return self.data[idx]
        return None

    def delete(self, key: str):
        h1 = self._h1(key)
        h2 = self._h2(key)
        for i in range(self.size):
            idx = (h1 + i * h2) & self.mask
            if self.slots[idx] == key:
                val = self.data[idx]
                self.slots[idx] = OpenAddrOpt.DELETED
                self.data[idx]  = None
                self.n -= 1
                return val
            if self.slots[idx] is OpenAddrOpt.EMPTY:
                break
        return None

    def _resize(self) -> None:
        old_slots, old_data = self.slots, self.data
        self.size <<= 1
        self.mask = self.size - 1
        self.slots = [OpenAddrOpt.EMPTY] * self.size
        self.data  = [None] * self.size
        self.n = 0
        for s, d in zip(old_slots, old_data):
            if s not in (OpenAddrOpt.EMPTY, OpenAddrOpt.DELETED):
                self.insert(s, d)

    def __len__(self):
        return self.n

    def __str__(self):
        # Hiển thị các cặp key-value hiện có trong bảng
        pairs = []
        for k, v in zip(self.slots, self.data):
            if k not in (OpenAddrOpt.EMPTY, OpenAddrOpt.DELETED):
                pairs.append(f"{k}: {v}")
        return '\n'.join(pairs) if pairs else '<empty>'


class HashTableLinear(BaseHashTable):
    EMPTY = object()
    DELETED = object()

    def __init__(self, exp=20, hash_fn=mmh3.hash):
        self.size = 1 << exp
        self.mask = self.size - 1
        self.hash_fn = hash_fn
        self.slots = [HashTableLinear.EMPTY] * self.size
        self.data  = [None] * self.size
        self.n = 0
        self.insert_probes = 0
        self.search_probes = 0
        self.collisions = 0

    def _h1(self, key: str) -> int:
        return self.hash_fn(key) & self.mask

    def insert(self, key: str, value: int) -> None:
        if self.n / self.size > 0.5:
            self._resize()
        idx = self._h1(key)
        start = idx
        while True:
            self.insert_probes += 1
            slot = self.slots[idx]
            if slot not in (HashTableLinear.EMPTY, HashTableLinear.DELETED) and slot != key:
                self.collisions += 1
            if slot in (HashTableLinear.EMPTY, HashTableLinear.DELETED) or slot == key:
                if slot in (HashTableLinear.EMPTY, HashTableLinear.DELETED):
                    self.n += 1
                self.slots[idx] = key
                self.data[idx]  = value
                return
            idx = (idx + 1) & self.mask
            if idx == start:
                raise RuntimeError("Hash table is full")

    def search(self, key: str):
        idx = self._h1(key)
        start = idx
        while True:
            self.search_probes += 1
            slot = self.slots[idx]
            if slot is HashTableLinear.EMPTY:
                return None
            if slot == key:
                return self.data[idx]
            idx = (idx + 1) & self.mask
            if idx == start:
                return None

    def delete(self, key: str):
        idx = self._h1(key)
        start = idx
        while True:
            slot = self.slots[idx]
            if slot == key:
                val = self.data[idx]
                self.slots[idx] = HashTableLinear.DELETED
                self.data[idx]  = None
                self.n -= 1
                return val
            if slot is HashTableLinear.EMPTY:
                return None
            idx = (idx + 1) & self.mask
            if idx == start:
                return None

    def _resize(self) -> None:
        old_slots, old_data = self.slots, self.data
        self.size <<= 1
        self.mask = self.size - 1
        self.slots = [HashTableLinear.EMPTY] * self.size
        self.data  = [None] * self.size
        self.n = 0
        for s, d in zip(old_slots, old_data):
            if s not in (HashTableLinear.EMPTY, HashTableLinear.DELETED):
                self.insert(s, d)

    def __len__(self):
        return self.n

    def __str__(self):
        pairs = []
        for k, v in zip(self.slots, self.data):
            if k not in (HashTableLinear.EMPTY, HashTableLinear.DELETED):
                pairs.append(f"{k}: {v}")
        return '\n'.join(pairs) if pairs else '<empty>'


class HashTableQuadratic(BaseHashTable):
    EMPTY = object()
    DELETED = object()

    def __init__(self, exp=20, hash_fn=mmh3.hash):
        self.size = 1 << exp
        self.mask = self.size - 1
        self.hash_fn = hash_fn
        self.slots = [HashTableQuadratic.EMPTY] * self.size
        self.data  = [None] * self.size
        self.n = 0
        self.insert_probes = 0
        self.search_probes = 0
        self.collisions = 0

    def _h1(self, key: str) -> int:
        return self.hash_fn(key) & self.mask

    def insert(self, key: str, value: int) -> None:
        if self.n / self.size > 0.5:
            self._resize()
        h1 = self._h1(key)
        i = 0
        while i < self.size:
            idx = (h1 + i*i) & self.mask
            self.insert_probes += 1
            slot = self.slots[idx]
            if slot not in (HashTableQuadratic.EMPTY, HashTableQuadratic.DELETED) and slot != key:
                self.collisions += 1
            if slot in (HashTableQuadratic.EMPTY, HashTableQuadratic.DELETED) or slot == key:
                if slot in (HashTableQuadratic.EMPTY, HashTableQuadratic.DELETED):
                    self.n += 1
                self.slots[idx] = key
                self.data[idx]  = value
                return
            i += 1
        raise RuntimeError("Hash table is full")

    def search(self, key: str):
        h1 = self._h1(key)
        i = 0
        while i < self.size:
            idx = (h1 + i*i) & self.mask
            self.search_probes += 1
            slot = self.slots[idx]
            if slot is HashTableQuadratic.EMPTY:
                return None
            if slot == key:
                return self.data[idx]
            i += 1
        return None

    def delete(self, key: str):
        h1 = self._h1(key)
        i = 0
        while i < self.size:
            idx = (h1 + i*i) & self.mask
            if self.slots[idx] == key:
                val = self.data[idx]
                self.slots[idx] = HashTableQuadratic.DELETED
                self.data[idx]  = None
                self.n -= 1
                return val
            if self.slots[idx] is HashTableQuadratic.EMPTY:
                return None
            i += 1
        return None

    def _resize(self) -> None:
        old_slots, old_data = self.slots, self.data
        self.size <<= 1
        self.mask = self.size - 1
        self.slots = [HashTableQuadratic.EMPTY] * self.size
        self.data  = [None] * self.size
        self.n = 0
        for s, d in zip(old_slots, old_data):
            if s not in (HashTableQuadratic.EMPTY, HashTableQuadratic.DELETED):
                self.insert(s, d)

    def __len__(self):
        return self.n

    def __str__(self):
        pairs = []
        for k, v in zip(self.slots, self.data):
            if k not in (HashTableQuadratic.EMPTY, HashTableQuadratic.DELETED):
                pairs.append(f"{k}: {v}")
        return '\n'.join(pairs) if pairs else '<empty>'

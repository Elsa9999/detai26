from src.base_hash import BaseHashTable

class HashTableChainingOpt(BaseHashTable):
    def __init__(self, size=11):
        self.size = size
        self.buckets = [[] for _ in range(size)]
        self.collisions = 0
        self.insert_probes = 0
        self.search_probes = 0

    def _idx(self, key: str) -> int:
        return hash(key) % self.size

    def insert(self, key: str, value: int) -> None:
        idx = self._idx(key)
        bucket = self.buckets[idx]
        self.insert_probes += 1
        if len(bucket) > 0:
            self.collisions += 1
        for i, (k, _) in enumerate(bucket):
            self.insert_probes += 1
            if k == key:
                bucket[i] = (key, value)
                return
        bucket.append((key, value))

    def search(self, key: str):
        idx = self._idx(key)
        bucket = self.buckets[idx]
        self.search_probes += 1
        for k, v in bucket:
            self.search_probes += 1
            if k == key:
                return v
        return None

    def delete(self, key: str):
        idx = self._idx(key)
        bucket = self.buckets[idx]
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket.pop(i)
                return v
        return None

    def __len__(self):
        return sum(len(b) for b in self.buckets)

    def __str__(self):
        result = ""
        for i, bucket in enumerate(self.buckets):
            if bucket:
                result += f"[{i}] " + ", ".join(f"{k}:{v}" for k,v in bucket) + "\n"
        return result

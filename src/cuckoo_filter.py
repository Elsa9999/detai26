import numpy as np
import mmh3

class CuckooFilter:
    def __init__(self, capacity=1000, bucket_size=4, max_kicks=500):
        """
        Khởi tạo Cuckoo Filter
        capacity: số lượng bucket
        bucket_size: số lượng item trong mỗi bucket
        max_kicks: số lần kick tối đa khi insert
        """
        self.capacity = capacity
        self.bucket_size = bucket_size
        self.max_kicks = max_kicks
        self.buckets = np.zeros((capacity, bucket_size), dtype=np.int64)
        self.count = 0
        self.items = set()  # Lưu các item đã thêm vào
        self.total_checks = 0
        self.false_positives = 0

    def _get_hash_values(self, item):
        """Tạo 2 hash values cho một item"""
        h1 = mmh3.hash(str(item)) % self.capacity
        h2 = (h1 ^ mmh3.hash(str(item), 1)) % self.capacity
        return h1, h2

    def _get_fingerprint(self, item):
        """Tạo fingerprint cho item"""
        return mmh3.hash(str(item), 2) % (2**32)

    def insert(self, item):
        """Thêm item vào filter"""
        if self.count >= self.capacity * self.bucket_size:
            return False

        fp = self._get_fingerprint(item)
        h1, h2 = self._get_hash_values(item)

        # Thử thêm vào bucket 1
        for i in range(self.bucket_size):
            if self.buckets[h1][i] == 0:
                self.buckets[h1][i] = fp
                self.count += 1
                self.items.add(item)
                return True

        # Thử thêm vào bucket 2
        for i in range(self.bucket_size):
            if self.buckets[h2][i] == 0:
                self.buckets[h2][i] = fp
                self.count += 1
                self.items.add(item)
                return True

        # Nếu cả 2 bucket đều đầy, thử kick
        for _ in range(self.max_kicks):
            # Chọn ngẫu nhiên 1 bucket
            bucket_idx = np.random.choice([h1, h2])
            # Chọn ngẫu nhiên 1 vị trí trong bucket
            pos = np.random.randint(0, self.bucket_size)
            # Lưu fingerprint cũ
            old_fp = self.buckets[bucket_idx][pos]
            # Thay thế bằng fingerprint mới
            self.buckets[bucket_idx][pos] = fp
            # Cập nhật fingerprint cần thêm
            fp = old_fp
            # Tính lại hash values cho fingerprint cũ
            h1, h2 = self._get_hash_values(str(fp))

            # Thử thêm fingerprint cũ vào bucket mới
            for i in range(self.bucket_size):
                if self.buckets[h1][i] == 0:
                    self.buckets[h1][i] = fp
                    self.count += 1
                    self.items.add(item)
                    return True
                if self.buckets[h2][i] == 0:
                    self.buckets[h2][i] = fp
                    self.count += 1
                    self.items.add(item)
                    return True

        return False

    def contains(self, item):
        """Kiểm tra xem item có thể tồn tại trong filter không"""
        self.total_checks += 1
        fp = self._get_fingerprint(item)
        h1, h2 = self._get_hash_values(item)

        # Kiểm tra trong cả 2 bucket
        exists = (fp in self.buckets[h1]) or (fp in self.buckets[h2])

        # Nếu item không có trong set nhưng filter báo có
        if exists and item not in self.items:
            self.false_positives += 1

        return exists

    def delete(self, item):
        """Xóa item khỏi filter"""
        if item not in self.items:
            return False

        fp = self._get_fingerprint(item)
        h1, h2 = self._get_hash_values(item)

        # Xóa fingerprint khỏi bucket 1
        for i in range(self.bucket_size):
            if self.buckets[h1][i] == fp:
                self.buckets[h1][i] = 0
                self.count -= 1
                self.items.remove(item)
                return True

        # Xóa fingerprint khỏi bucket 2
        for i in range(self.bucket_size):
            if self.buckets[h2][i] == fp:
                self.buckets[h2][i] = 0
                self.count -= 1
                self.items.remove(item)
                return True

        return False

    def get_load_factor(self):
        """Tính load factor của filter"""
        return self.count / (self.capacity * self.bucket_size)

    def get_false_positive_rate(self):
        """Tính xác suất false positive thực tế"""
        if self.total_checks == 0:
            return 0
        return self.false_positives / self.total_checks 
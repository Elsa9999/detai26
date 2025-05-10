import numpy as np
import mmh3

class BloomFilter:
    def __init__(self, m=1000, k=3):
        """
        Khởi tạo Bloom Filter
        m: kích thước bit array
        k: số lượng hash function
        """
        self.m = m
        self.k = k
        self.bit_array = np.zeros(m, dtype=bool)
        self.items = set()  # Lưu các item đã thêm vào
        self.total_checks = 0
        self.false_positives = 0

    def _get_hash_values(self, item):
        """Tạo k hash values cho một item"""
        return [mmh3.hash(str(item), i) % self.m for i in range(self.k)]

    def add(self, item):
        """Thêm item vào filter"""
        hash_values = self._get_hash_values(item)
        for h in hash_values:
            self.bit_array[h] = True
        self.items.add(item)

    def check(self, item):
        """Kiểm tra xem item có thể tồn tại trong filter không"""
        self.total_checks += 1
        hash_values = self._get_hash_values(item)
        
        # Nếu tất cả các bit đều là 1
        exists = all(self.bit_array[h] for h in hash_values)
        
        # Nếu item không có trong set nhưng filter báo có
        if exists and item not in self.items:
            self.false_positives += 1
            
        return {
            'exists': exists,
            'false_positive_rate': self.get_false_positive_rate(),
            'stats': {
                'items_added': len(self.items),
                'total_checks': self.total_checks,
                'false_positives': self.false_positives,
                'actual_false_positive_rate': self.get_actual_false_positive_rate()
            }
        }

    def get_false_positive_rate(self):
        """Tính xác suất false positive lý thuyết"""
        n = len(self.items)
        return (1 - np.exp(-self.k * n / self.m)) ** self.k

    def get_actual_false_positive_rate(self):
        """Tính xác suất false positive thực tế"""
        if self.total_checks == 0:
            return 0
        return self.false_positives / self.total_checks 
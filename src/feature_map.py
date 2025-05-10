# src/feature_map.py

from src.hash_chaining_optimized import HashTableChainingOpt
from src.hash_open_addressing_opt import (
    OpenAddrOpt,
    HashTableLinear,
    HashTableQuadratic
)

class FastFeatureMap:
    """
    Ứng dụng hash table để ánh xạ feature (string) sang id (int) nhanh chóng.
    Hỗ trợ các phương pháp chaining, double hashing, linear probing, quadratic probing.
    """

    def __init__(self, method: str = "chaining"):
        if method == "chaining":
            self.table = HashTableChainingOpt(size=1024)
        elif method == "open_double":
            self.table = OpenAddrOpt(exp=10)
        elif method == "linear":
            self.table = HashTableLinear(exp=10)
        elif method == "quadratic":
            self.table = HashTableQuadratic(exp=10)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Đếm số feature đã phát hiện
        self.feature_count = 0

    def get_feature_id(self, feature: str) -> int:
        """
        Trả về id của feature, nếu chưa có thì gán id mới và insert vào bảng.
        """
        fid = self.table.search(feature)
        if fid is None:
            fid = self.feature_count
            self.table.insert(feature, fid)
            self.feature_count += 1
        return fid

    def vectorize(self, features: list[str]) -> list[int]:
        """
        Chuyển list các feature thành vector nhị phân (1-hot) độ dài = feature_count.
        """
        print("DEBUG: features input:", features)
        ids = [self.get_feature_id(f) for f in features]
        print("DEBUG: ids:", ids)
        print("DEBUG: feature_count:", self.feature_count)
        vec = [0] * self.feature_count
        for fid in ids:
            vec[fid] = 1
        print("DEBUG: vec:", vec)
        return vec

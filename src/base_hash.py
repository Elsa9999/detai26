class BaseHashTable:
    def insert(self, key, value):
        raise NotImplementedError

    def search(self, key):
        raise NotImplementedError

    def delete(self, key):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

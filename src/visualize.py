import matplotlib.pyplot as plt
from hash_chaining_optimized import HashTableChainingOpt
from hash_open_addressing_opt import OpenAddrOpt


def plot_chain(ht: HashTableChainingOpt):
    lengths = []
    for b in ht.buckets:
        count = 0
        node = b
        while node:
            count += 1
            node = node.next
        lengths.append(count)
    plt.figure(figsize=(10,4))
    plt.bar(range(len(lengths)), lengths)
    plt.title("Chain lengths in ChainingOpt")
    plt.xlabel("Bucket Index")
    plt.ylabel("Length")
    plt.show()


def plot_occupancy(oa: OpenAddrOpt):
    occ = [1 if s not in (OpenAddrOpt.EMPTY, OpenAddrOpt.DELETED) else 0 for s in oa.slots]
    plt.figure(figsize=(10,2))
    plt.bar(range(len(occ)), occ)
    plt.title(f"Occupancy: {oa.n}/{oa.size}")
    plt.xlabel("Slot Index")
    plt.yticks([])
    plt.show()

if __name__ == "__main__":
    hc = HashTableChainingOpt(size=128)
    ho = OpenAddrOpt(exp=7)
    for key in ["apple","banana","apple","pear","grape"]:
        hc.insert(key, 1)
        ho.insert(key, 1)
    plot_chain(hc)
    plot_occupancy(ho)
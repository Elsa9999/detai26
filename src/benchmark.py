import timeit
import tracemalloc
import math
import csv

# Tất cả import đều package‑qualified
from src.hash_chaining_optimized import HashTableChainingOpt
from src.hash_open_addressing_opt import OpenAddrOpt, HashTableLinear, HashTableQuadratic

def exp_for(n, load_factor=0.5):
    """Tính exponent sao cho kích thước bảng ≈ n/load_factor và là log2."""
    return math.ceil(math.log2(n / load_factor))

def measure_table(name, ctor_fn, n):
    """Đo thời gian insert, search và memory usage của một hash table"""
    table = ctor_fn()
    
    # Insert
    start = timeit.default_timer()
    for i in range(n):
        table.insert(f"key_{i}", i)
    ti = timeit.default_timer() - start
    
    # Search
    start = timeit.default_timer()
    for i in range(n):
        table.search(f"key_{i}")
    ts = timeit.default_timer() - start
    
    # Memory
    if name == 'Chaining':
        mem = get_memory_usage_chaining(table)
    else:
        mem = get_memory_usage_open_addressing(table)
        
    # Stats
    avg_insert_probe = table.insert_probes / n if hasattr(table, 'insert_probes') else 0
    avg_search_probe = table.search_probes / n if hasattr(table, 'search_probes') else 0
    collisions = table.collisions if hasattr(table, 'collisions') else 0
    load_factor = n / table.size if hasattr(table, 'size') else 0
    
    return ti, ts, mem, avg_insert_probe, avg_search_probe, collisions, load_factor

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='CLI: Benchmark various HashTable implementations'
    )
    parser.add_argument('-s','--sizes', type=int, nargs='+',
                        default=[100_000, 200_000, 500_000, 1_000_000],
                        help='List of N values to benchmark')
    parser.add_argument('-o','--output', help='Optional CSV output file')
    args = parser.parse_args()

    methods = [
        ('Chaining',   lambda exp: HashTableChainingOpt(size=1<<exp)),
        ('OpenDouble', lambda exp: OpenAddrOpt(exp=exp)),
        ('Linear',     lambda exp: HashTableLinear(exp=exp)),
        ('Quadratic',  lambda exp: HashTableQuadratic(exp=exp)),
    ]

    # Header
    hdr = (
        f"{'N':>8} | {'Method':<10} | {'Insert(s)':>10} | "
        f"{'Search(s)':>10} | {'Mem(MB)':>8} | {'InsProbe':>8} | "
        f"{'SrchProbe':>8} | {'Coll':>6} | {'Load':>6}"
    )
    print(hdr)
    print('-' * len(hdr))

    results = []
    for n in args.sizes:
        exp = exp_for(n)
        for name, ctor_fn in methods:
            ti, ts, mem, ip, sp, coll, load = measure_table(name, lambda ctor_fn=ctor_fn: ctor_fn(exp), n)
            print(f"{n:8} | {name:<10} | {ti:10.4f} | {ts:10.4f} | {mem:8.2f} | "
                  f"{ip:8.2f} | {sp:8.2f} | {coll:6d} | {load:6.2f}")
            results.append({
                'N': n, 'Method': name,
                'Insert': ti, 'Search': ts, 'MemoryMB': mem,
                'AvgInsertProbe': ip, 'AvgSearchProbe': sp,
                'Collisions': coll, 'LoadFactor': load
            })

    if args.output:
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['N','Method','Insert','Search','MemoryMB',
                                                 'AvgInsertProbe','AvgSearchProbe','Collisions','LoadFactor'])
            writer.writeheader()
            writer.writerows(results)
        print(f"Results written to {args.output}")

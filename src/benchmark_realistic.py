import os
import timeit
import pandas as pd
import sys
from src.hash_chaining_optimized import HashTableChainingOpt
from src.hash_open_addressing_opt import OpenAddrOpt, HashTableLinear, HashTableQuadratic
import random
import tracemalloc
import psutil

def get_memory_usage_chaining(table):
    """Đo memory usage cho hash table kiểu chaining"""
    mem = sys.getsizeof(table)
    if hasattr(table, 'buckets'):
        mem += sys.getsizeof(table.buckets)
        for bucket in table.buckets:
            mem += sys.getsizeof(bucket)
            for item in bucket:
                mem += sys.getsizeof(item)
                if isinstance(item[0], str):
                    mem += len(item[0].encode('utf-8'))
                if isinstance(item[1], str):
                    mem += len(item[1].encode('utf-8'))
    return round(mem / (1024 * 1024), 2)  # MB

def get_memory_usage_open_addressing(table):
    """Đo memory usage cho hash table kiểu open addressing"""
    mem = sys.getsizeof(table)
    if hasattr(table, 'slots'):
        mem += sys.getsizeof(table.slots)
        for slot in table.slots:
            if slot not in (table.EMPTY, table.DELETED):
                mem += sys.getsizeof(slot)
                if isinstance(slot[0], str):
                    mem += len(slot[0].encode('utf-8'))
                if isinstance(slot[1], str):
                    mem += len(slot[1].encode('utf-8'))
    return round(mem / (1024 * 1024), 2)  # MB

def get_constructor(algo, exp):
    """Trả về constructor cho thuật toán hash table tương ứng"""
    if algo == 'chaining':
        return lambda: HashTableChainingOpt(size=1<<exp)
    if algo == 'open_double':
        return lambda: OpenAddrOpt(exp=exp)
    if algo == 'linear':
        return lambda: HashTableLinear(exp=exp)
    if algo == 'quadratic':
        return lambda: HashTableQuadratic(exp=exp)
    raise ValueError(f"Unknown algorithm: {algo}")

def benchmark_from_file(path, algo, exp, chunksize=1000000):
    """
    Stream CSV file in chunks, insert into hash table, measure insert+search+peak memory.
    CSV must have columns 'key','value'.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss

    constructor = get_constructor(algo, exp)
    table = constructor()

    # --- INSERT PHASE ---
    def do_insert():
        try:
            reader = pd.read_csv(path, chunksize=chunksize)
            for chunk in reader:
                if 'key' not in chunk.columns or 'value' not in chunk.columns:
                    raise ValueError("CSV must have 'key' and 'value' columns")
                for k, v in zip(chunk['key'].astype(str), chunk['value']):
                    table.insert(k, v)
        except Exception as e:
            raise RuntimeError(f"Error during insertion: {str(e)}")

    t_insert = timeit.timeit(do_insert, number=1)
    mem_after = process.memory_info().rss
    mem_mb = (mem_after - mem_before) / (1024 * 1024)

    # --- SEARCH PHASE ---
    try:
        first = pd.read_csv(path, nrows=10000)
        if 'key' not in first.columns:
            raise ValueError("CSV must have 'key' column")
        search_keys = first['key'].astype(str).tolist()
        random.shuffle(search_keys)
        reps = 10000

        def do_search():
            for k in search_keys:
                table.search(k)

        total_searches = reps * len(search_keys)
        t_search_total = timeit.timeit(do_search, number=reps)
        t_search_ms = t_search_total * 1000  # tổng thời gian search (ms)
        # Không chia nhỏ nữa, trả về tổng thời gian search (ms)

        # Tính các chỉ số thống kê
        n = sum(1 for _ in pd.read_csv(path))
        avg_insert_probe = table.insert_probes / n if hasattr(table, 'insert_probes') else 0
        avg_search_probe = table.search_probes / len(search_keys) if hasattr(table, 'search_probes') else 0
        collisions = table.collisions if hasattr(table, 'collisions') else 0
        load_factor = n / table.size if hasattr(table, 'size') else 0

        return t_insert, t_search_ms, mem_mb, avg_insert_probe, avg_search_probe, collisions, load_factor

    except Exception as e:
        raise RuntimeError(f"Error during search: {str(e)}") 
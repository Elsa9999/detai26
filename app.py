import pandas as pd
print("HELLO FROM APP.PY")
import flask
import pandas
import werkzeug
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from src.routes import routes
from src.ai_demo import FeatureHasher, LSH, ai_demo
from src.bloom_filter import BloomFilter
from src.cuckoo_filter import CuckooFilter
import numpy as np
import os
from werkzeug.utils import secure_filename
import sys
from src.gui import upload as gui_upload
from src.feature_map import FastFeatureMap
from src.hash_chaining_optimized import HashTableChainingOpt
from src.hash_open_addressing_opt import OpenAddrOpt
from src.hash_open_addressing_opt import HashTableLinear
from src.hash_open_addressing_opt import HashTableQuadratic

print("Starting application...")
app = Flask(__name__)
app.secret_key = 'hastable2024'

print("Initializing AI objects...")
# Khởi tạo các đối tượng AI
feature_hasher = FeatureHasher()
# Luôn khởi tạo LSH với vector mẫu
lsh = LSH(n_vectors=5, n_bits=4, n_tables=2)
lsh.add_vector(np.array([1,2,3,4,5]), "vec1")
lsh.add_vector(np.array([2,3,4,5,6]), "vec2")
lsh.add_vector(np.array([5,4,3,2,1]), "vec3")

print("Initializing filters...")
# Khởi tạo Bloom Filter và Cuckoo Filter
bloom_filter = BloomFilter(m=1000, k=3)
cuckoo_filter = CuckooFilter(capacity=1000, bucket_size=4, max_kicks=500)

print("Configuring upload...")
# Cấu hình upload
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'csv', 'npy'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Tạo thư mục uploads nếu chưa tồn tại
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("Registering routes...")
# In‑memory tables cho CRUD demo
tables = {}

def new_table(algo: str):
    if algo == 'chaining':
        return HashTableChainingOpt(size=16)
    if algo == 'open_double':
        return OpenAddrOpt(exp=4)
    if algo == 'linear':
        return HashTableLinear(exp=4)
    if algo == 'quadratic':
        return HashTableQuadratic(exp=4)
    raise ValueError(f"Unknown algorithm: {algo}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_text_file(file):
    return file.read().decode('utf-8')

def read_csv_file(file):
    df = pd.read_csv(file)
    return df.to_string()

def read_npy_file(file):
    return np.load(file).tolist()

def format_vector(vector, max_items=10):
    """Format vector để hiển thị đẹp hơn"""
    if isinstance(vector, np.ndarray):
        vector = vector.tolist()
    return str(vector[:max_items]) + ('...' if len(vector) > max_items else '')

def get_memory_usage_open_addressing(ht):
    mem = sys.getsizeof(ht)
    if hasattr(ht, 'slots'):
        mem += sys.getsizeof(ht.slots)
        for slot in ht.slots:
            if slot not in (ht.EMPTY, ht.DELETED):
                mem += sys.getsizeof(slot)
    if hasattr(ht, 'data'):
        mem += sys.getsizeof(ht.data)
        for data in ht.data:
            if data is not None:
                mem += sys.getsizeof(data)
    return mem / (1024 * 1024)  # MB

def get_memory_usage_chaining(ht):
    mem = sys.getsizeof(ht)
    if hasattr(ht, 'buckets'):
        mem += sys.getsizeof(ht.buckets)
        for bucket in ht.buckets:
            mem += sys.getsizeof(bucket)
            for k, v in bucket:
                mem += sys.getsizeof(k)
                mem += sys.getsizeof(v)
    return mem / (1024 * 1024)  # MB

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/benchmark', methods=['GET', 'POST'])
def benchmark():
    if request.method == 'POST':
        n_items = int(request.form.get('n_items', 100000))
        algorithms = request.form.getlist('algorithms')
        table_size_exp = int(request.form.get('table_size_exp', 20))
        table_size = 2 ** table_size_exp
        
        # Tạo dữ liệu ngẫu nhiên
        keys = [str(i) for i in range(n_items)]
        values = [str(i) for i in range(n_items)]
        
        results = {
            'n_items': n_items,
            'algorithms': algorithms,
            'table_size': table_size,
            'insert_times': {},
            'search_times': {},
            'memories': {},
            'load_factors': {},
            'avg_insert_probe': {},
            'avg_search_probe': {},
            'collisions': {},
            'chart_data': {
                'methods': [],
                'insert_times': [],
                'search_times': [],
                'memories': []
            }
        }
        
        # Thực hiện benchmark cho từng thuật toán
        for algo in algorithms:
            try:
                if algo == 'chaining':
                    from src.hash_chaining_optimized import HashTableChainingOpt
                    ht = HashTableChainingOpt(table_size)
                elif algo == 'open_double':
                    from src.hash_open_addressing_opt import OpenAddrOpt
                    ht = OpenAddrOpt(table_size_exp)
                elif algo == 'linear':
                    from src.hash_open_addressing_opt import HashTableLinear
                    ht = HashTableLinear(table_size_exp)
                elif algo == 'quadratic':
                    from src.hash_open_addressing_opt import HashTableQuadratic
                    ht = HashTableQuadratic(table_size_exp)
                else:
                    continue
                # Reset counters nếu cần
                if hasattr(ht, 'insert_probes'): ht.insert_probes = 0
                if hasattr(ht, 'search_probes'): ht.search_probes = 0
                if hasattr(ht, 'collisions'): ht.collisions = 0
                
                # Đo thời gian insert
                import time
                start_time = time.time()
                for k, v in zip(keys, values):
                    ht.insert(k, v)
                insert_time = (time.time() - start_time) * 1000  # ms
                
                # Đo thời gian search
                start_time = time.time()
                for k in keys:
                    ht.search(k)
                search_time = (time.time() - start_time) * 1000  # ms
                
                # Tính memory usage sát thực tế
                if algo == 'chaining':
                    memory = get_memory_usage_chaining(ht)
                else:
                    memory = get_memory_usage_open_addressing(ht)
                
                # Tính load factor
                if algo == 'chaining':
                    load_factor = len(ht) / table_size
                    avg_insert_probe = 1  # Chaining: mỗi insert chỉ probe 1 bucket
                    avg_search_probe = 1  # Chaining: mỗi search chỉ probe 1 bucket
                    collisions = ht.collisions
                else:
                    avg_insert_probe = ht.insert_probes / n_items if n_items else 0
                    avg_search_probe = ht.search_probes / n_items if n_items else 0
                    collisions = ht.collisions
                    load_factor = ht.n / table_size
                
                # Lưu kết quả (làm tròn 6 chữ số)
                results['insert_times'][algo] = f"{insert_time:.6f}"
                results['search_times'][algo] = f"{search_time:.6f}"
                results['memories'][algo] = f"{memory:.6f}"
                results['load_factors'][algo] = f"{load_factor:.6%}"
                results['avg_insert_probe'][algo] = f"{avg_insert_probe:.6f}"
                results['avg_search_probe'][algo] = f"{avg_search_probe:.6f}"
                results['collisions'][algo] = collisions
                
                # Thêm dữ liệu cho biểu đồ
                results['chart_data']['methods'].append(algo)
                results['chart_data']['insert_times'].append(insert_time)
                results['chart_data']['search_times'].append(search_time)
                results['chart_data']['memories'].append(memory)
            except Exception as e:
                print(f"Error benchmarking {algo}: {str(e)}")
                continue
        
        return render_template('benchmark.html', results=results)
    return render_template('benchmark.html', results=None)

@app.route('/benchmark_realistic')
def benchmark_realistic():
    return render_template('benchmark_realistic.html')

@app.route('/feature_map', methods=['GET', 'POST'])
def feature_map():
    if request.method == 'POST':
        text = request.form.get('sample_text', '') or request.form.get('text', '')
        method = request.form.get('method', 'chaining')
        tokens = text.split() if text else []
        fmap = FastFeatureMap(method=method)
        vector = fmap.vectorize(tokens)
        mapping = [(tok, fmap.get_feature_id(tok)) for tok in tokens]
        return render_template('feature_map.html', tokens=tokens, vector=vector, mapping=mapping)
    # Xử lý GET có query string
    text = request.args.get('text', '')
    method = request.args.get('method', 'chaining')
    tokens = text.split() if text else []
    vector = []
    mapping = []
    if tokens:
        fmap = FastFeatureMap(method=method)
        vector = fmap.vectorize(tokens)
        mapping = [(tok, fmap.get_feature_id(tok)) for tok in tokens]
    return render_template('feature_map.html', tokens=tokens, vector=vector, mapping=mapping)

@app.route('/crud', methods=['GET', 'POST'])
def crud():
    if request.method == 'POST':
        algo = request.form['algorithm']
        key = request.form['key']
        val = request.form.get('value', '')
        act = request.form['crud_action']

        try:
            if algo not in tables:
                tables[algo] = new_table(algo)
            tbl = tables[algo]

            if act == 'insert':
                v = int(val) if val.isdigit() else val
                # Kiểm tra key đã tồn tại chưa (nếu muốn)
                try:
                    tbl.insert(key, v)
                    flash(f"Inserted '{key}' → {v}", 'success')
                except Exception as e:
                    flash(f"Insert failed: {str(e)}", 'danger')
            elif act == 'search':
                try:
                    r = tbl.search(key)
                    if r is not None:
                        flash(f"Found: {r}", 'info')
                    else:
                        flash(f"Not found: '{key}'", 'warning')
                except Exception as e:
                    flash(f"Search failed: {str(e)}", 'danger')
            elif act == 'delete':
                try:
                    r = tbl.delete(key)
                    if r is not None:
                        flash(f"Deleted '{key}'", 'success')
                    else:
                        flash(f"Key '{key}' not found for deletion", 'warning')
                except Exception as e:
                    flash(f"Delete failed: {str(e)}", 'danger')
            else:
                flash(f"Unknown action: {act}", 'danger')
        except Exception as e:
            flash(f"Operation failed: {str(e)}", 'danger')

        return redirect(url_for('crud'))

    return render_template('crud.html', tables=tables)

@app.route('/theory')
def theory():
    return render_template('theory.html')

@app.route('/ai_demo/feature_hash', methods=['POST'])
def feature_hash():
    text = request.form.get('text', '')
    hash_algo = request.form.get('hash_algo', 'chaining')
    # Xử lý file upload
    if 'file' in request.files:
        file = request.files['file']
        if file and allowed_file(file.filename):
            if file.filename.endswith('.txt'):
                text = read_text_file(file)
            elif file.filename.endswith('.csv'):
                text = read_csv_file(file)
    # Cập nhật thuật toán hash
    feature_hasher.hash_algo = hash_algo
    # Chuyển đổi text thành vector (lấy dòng đầu tiên nếu nhiều dòng)
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return render_template('ai_demo.html', feature_hash_results=None, error='Không có dữ liệu đầu vào!')
    results = feature_hasher._process_batch([lines[0]])
    # Format kết quả để hiển thị
    hash_vector = results['hash_vectors'][0] if results['hash_vectors'] else []
    tfidf_vector = results['tfidf_vectors'][0] if results['tfidf_vectors'] else []
    feature_mapping = results['feature_mappings'][0] if results['feature_mappings'] else {}
    formatted_results = {
        'hash_vector': format_vector(hash_vector),
        'tfidf_vector': format_vector(tfidf_vector),
        'feature_mapping': {k: v for k, v in list(feature_mapping.items())[:10]},
        'token_stats': results['token_stats'],
        'input_text': lines[0][:200] + ('...' if len(lines[0]) > 200 else ''),
        'total_tokens': len(feature_mapping),
        'non_zero_elements': int(np.count_nonzero(hash_vector)) if len(hash_vector) else 0,
        'hash_algo': hash_algo
    }
    return render_template('ai_demo.html', feature_hash_results=formatted_results)

@app.route('/ai_demo/lsh', methods=['POST'])
def lsh_search():
    query = request.form.get('query_vector', '')
    k = int(request.form.get('k', 5))
    
    try:
        query_vector = None
        # Xử lý file upload
        if 'file' in request.files:
            file = request.files['file']
            if file and file.filename:
                if file.filename.endswith('.csv'):
                    df = pd.read_csv(file)
                    query_vector = df.values.flatten()
                elif file.filename.endswith('.npy'):
                    query_vector = np.load(file)
                else:
                    return render_template('ai_demo.html', error="File không đúng định dạng!")
        if query_vector is None:
            # Parse query vector từ text, hỗ trợ cả dấu cách và dấu phẩy
            try:
                query_vector = np.array([float(x) for x in query.replace(',', ' ').split()])
            except Exception:
                return render_template('ai_demo.html', error="Vui lòng nhập vector hợp lệ hoặc upload file hợp lệ!")
        if query_vector is None or len(query_vector) == 0:
            return render_template('ai_demo.html', error="Bạn chưa nhập vector truy vấn!")
        
        # Kiểm tra có vector mẫu không
        if not lsh.vectors:
            return render_template('ai_demo.html', error="Chưa có vector mẫu trong hệ thống LSH!")
        
        # Kiểm tra độ dài vector
        if len(query_vector) != lsh.n_vectors:
            return render_template('ai_demo.html', error=f"Độ dài vector truy vấn ({len(query_vector)}) không khớp với vector mẫu ({lsh.n_vectors})!")
        
        # Tìm kiếm
        results = lsh.search(query_vector, k)
        
        # Format kết quả
        formatted_results = {
            'query_vector': str(query_vector),
            'results': [{
                'id': r['id'],
                'cosine_similarity': f"{r['cosine_similarity']:.4f}",
                'vector': str(r['vector'])
            } for r in results['results']],
            'stats': results['stats'],
            'query_info': results['query_info']
        }
        
        return render_template('ai_demo.html', lsh_results=formatted_results)
            
    except Exception as e:
        return render_template('ai_demo.html', error=f"Lỗi: {str(e)}")

@app.route('/ai_demo/bloom', methods=['POST'])
def bloom_search():
    try:
        query = request.form.get('query', '').strip()
        if not query:
            return render_template('ai_demo.html', error="Vui lòng nhập từ khóa cần kiểm tra!")

        # Kiểm tra từ khóa
        results = bloom_filter.check(query)
        
        # Format kết quả
        formatted_results = {
            'query': query,
            'result': 'Có thể tồn tại (có thể là false positive)' if results['exists'] else 'Chắc chắn không tồn tại',
            'false_positive_rate': f"{results['false_positive_rate']:.2%}",
            'stats': {
                'items_count': results['stats']['items_added'],
                'total_checks': results['stats']['total_checks'],
                'false_positives': results['stats']['false_positives'],
                'actual_false_positive_rate': f"{results['stats']['actual_false_positive_rate']:.2%}"
            }
        }
        
        return render_template('ai_demo.html', bloom_results=formatted_results)
    except Exception as e:
        return render_template('ai_demo.html', error=f"Lỗi: {str(e)}")

@app.route('/ai_demo/bloom/add', methods=['POST'])
def bloom_add():
    try:
        item = request.form.get('item', '').strip()
        if not item:
            return jsonify({'error': 'Vui lòng nhập item cần thêm!'})
        
        bloom_filter.add(item)
        return jsonify({
            'success': True,
            'message': f'Đã thêm item "{item}"',
            'stats': {
                'items_count': len(bloom_filter.items),
                'false_positive_rate': f"{bloom_filter.get_false_positive_rate():.2%}"
            }
        })
    except Exception as e:
        return jsonify({'error': f'Lỗi: {str(e)}'})

@app.route('/ai_demo/bloom/delete', methods=['POST'])
def bloom_delete():
    try:
        item = request.form.get('item', '').strip()
        if not item:
            return jsonify({'error': 'Vui lòng nhập item cần xóa!'})
        
        if item in bloom_filter.items:
            bloom_filter.items.remove(item)
            return jsonify({
                'success': True,
                'message': f'Đã xóa item "{item}"',
                'stats': {
                    'items_count': len(bloom_filter.items),
                    'false_positive_rate': f"{bloom_filter.get_false_positive_rate():.2%}"
                }
            })
        else:
            return jsonify({'error': f'Item "{item}" không tồn tại trong filter!'})
    except Exception as e:
        return jsonify({'error': f'Lỗi: {str(e)}'})

@app.route('/ai_demo/bloom/list', methods=['GET'])
def bloom_list():
    try:
        return jsonify({
            'items': list(bloom_filter.items),
            'count': len(bloom_filter.items)
        })
    except Exception as e:
        return jsonify({'error': f'Lỗi: {str(e)}'})

@app.route('/ai_demo/cuckoo', methods=['POST'])
def cuckoo_search():
    try:
        query = request.form.get('query', '').strip()
        if not query:
            return render_template('ai_demo.html', error="Vui lòng nhập từ khóa cần kiểm tra!")

        # Kiểm tra từ khóa
        exists = cuckoo_filter.contains(query)
        
        # Format kết quả
        formatted_results = {
            'query': query,
            'result': 'Có thể tồn tại (có thể là false positive)' if exists else 'Chắc chắn không tồn tại',
            'load_factor': f"{cuckoo_filter.get_load_factor():.2%}",
            'stats': {
                'items_count': cuckoo_filter.count,
                'total_checks': cuckoo_filter.total_checks,
                'false_positives': cuckoo_filter.false_positives,
                'actual_false_positive_rate': f"{cuckoo_filter.get_false_positive_rate():.2%}"
            }
        }
        
        return render_template('ai_demo.html', cuckoo_results=formatted_results)
    except Exception as e:
        return render_template('ai_demo.html', error=f"Lỗi: {str(e)}")

@app.route('/ai_demo/cuckoo/add', methods=['POST'])
def cuckoo_add():
    try:
        item = request.form.get('item', '').strip()
        if not item:
            return jsonify({'error': 'Vui lòng nhập item cần thêm!'})
        
        if cuckoo_filter.insert(item):
            return jsonify({
                'success': True,
                'message': f'Đã thêm item "{item}"',
                'stats': {
                    'items_count': cuckoo_filter.count,
                    'load_factor': f"{cuckoo_filter.get_load_factor():.2%}"
                }
            })
        else:
            return jsonify({'error': 'Không thể thêm item do filter đã đầy!'})
    except Exception as e:
        return jsonify({'error': f'Lỗi: {str(e)}'})

@app.route('/ai_demo/cuckoo/delete', methods=['POST'])
def cuckoo_delete():
    try:
        item = request.form.get('item', '').strip()
        if not item:
            return jsonify({'error': 'Vui lòng nhập item cần xóa!'})
        
        if cuckoo_filter.delete(item):
            return jsonify({
                'success': True,
                'message': f'Đã xóa item "{item}"',
                'stats': {
                    'items_count': cuckoo_filter.count,
                    'load_factor': f"{cuckoo_filter.get_load_factor():.2%}"
                }
            })
        else:
            return jsonify({'error': f'Item "{item}" không tồn tại trong filter!'})
    except Exception as e:
        return jsonify({'error': f'Lỗi: {str(e)}'})

@app.route('/ai_demo/cuckoo/list', methods=['GET'])
def cuckoo_list():
    try:
        return jsonify({
            'items': list(cuckoo_filter.items),
            'count': cuckoo_filter.count
        })
    except Exception as e:
        return jsonify({'error': f'Lỗi: {str(e)}'})

@app.route('/ai_demo/bloom/reset', methods=['POST'])
def bloom_reset():
    try:
        bloom_filter = BloomFilter(m=1000, k=3)
        return jsonify({
            'success': True,
            'message': 'Đã reset Bloom Filter',
            'stats': {
                'items_count': 0,
                'false_positive_rate': '0.00%'
            }
        })
    except Exception as e:
        return jsonify({'error': f'Lỗi: {str(e)}'})

@app.route('/ai_demo/cuckoo/reset', methods=['POST'])
def cuckoo_reset():
    try:
        cuckoo_filter = CuckooFilter(capacity=1000, bucket_size=4, max_kicks=500)
        return jsonify({
            'success': True,
            'message': 'Đã reset Cuckoo Filter',
            'stats': {
                'items_count': 0,
                'load_factor': '0.00%'
            }
        })
    except Exception as e:
        return jsonify({'error': f'Lỗi: {str(e)}'})

@app.route('/upload_file', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        try:
            print("Received POST request")
            f = request.files.get('datafile')
            print(f"File received: {f.filename if f else 'None'}")
            algos = request.form.getlist('algorithms')
            print(f"Selected algorithms: {algos}")
            exp = int(request.form.get('table_size_exp', 22))
            chunks = int(request.form.get('chunksize', 1_000_000))

            # Kiểm tra file hợp lệ
            if not f or not f.filename.endswith('.csv'):
                print("Invalid file format")
                flash('Vui lòng upload file CSV có định dạng .csv!', 'danger')
                return redirect(request.url)

            # Giới hạn kích thước file (10MB)
            f.seek(0, 2)
            size = f.tell()
            f.seek(0)
            print(f"File size: {size} bytes")
            if size > 10 * 1024 * 1024:
                print("File too large")
                flash('File quá lớn! Vui lòng chọn file nhỏ hơn 10MB.', 'danger')
                return redirect(request.url)

            # Tạo thư mục uploads nếu chưa tồn tại
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            name = secure_filename(f.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], name)
            print(f"Saving file to: {path}")
            f.save(path)

            # Kiểm tra cột CSV
            try:
                df = pd.read_csv(path, nrows=1)
                print(f"CSV columns: {df.columns.tolist()}")
                if 'key' not in df.columns or 'value' not in df.columns:
                    print("Missing required columns")
                    flash("File CSV phải có cột 'key' và 'value'!", 'danger')
                    os.remove(path)
                    return redirect(request.url)
            except Exception as e:
                print(f"Error reading CSV: {str(e)}")
                flash(f'Lỗi khi đọc file CSV: {str(e)}', 'danger')
                os.remove(path)
                return redirect(request.url)

            # Chạy benchmark cho từng thuật toán
            from src.benchmark_realistic import benchmark_from_file
            results = []
            for algo in algos:
                try:
                    ti, ts, mem, ip, sp, coll, load = benchmark_from_file(path, algo, exp, chunks)
                    
                    # Chuyển đổi và làm tròn các giá trị
                    ti_us = round(ti * 1000, 2)  # ms -> μs
                    ts_us = round(ts * 1000, 2)  # ms -> μs
                    mem_mb = round(mem, 2)
                    ip_avg = round(ip, 2)
                    sp_avg = round(sp, 2)
                    load_pct = round(load * 100, 2)  # Chuyển sang phần trăm
                    
                    results.append({
                        'algo': algo,
                        'ti': ti_us,
                        'ts': ts_us,
                        'mem': mem_mb,
                        'avg_insert_probe': ip_avg,
                        'avg_search_probe': sp_avg,
                        'collisions': coll,
                        'load_factor': load_pct,
                        'stats': {
                            'insert_speed': f"{round(1000/ti_us if ti_us > 0 else 0, 2)} ops/ms",
                            'search_speed': f"{round(1000/ts_us if ts_us > 0 else 0, 2)} ops/ms",
                            'memory_efficiency': f"{round(mem_mb/load_pct if load_pct > 0 else 0, 2)} MB/%"
                        }
                    })
                except Exception as e:
                    results.append({
                        'algo': algo,
                        'ti': None,
                        'ts': None,
                        'mem': None,
                        'avg_insert_probe': None,
                        'avg_search_probe': None,
                        'collisions': None,
                        'load_factor': None,
                        'stats': None,
                        'error': str(e)
                    })

            # Xóa file sau khi benchmark xong
            try:
                os.remove(path)
            except:
                pass

            flash('Benchmark hoàn tất!', 'success')
            return render_template(
                'benchmark_compare.html',
                filename=name, exp=exp, chunksize=chunks,
                results=results
            )
        except Exception as e:
            flash(f'Lỗi không xác định: {str(e)}', 'danger')
            return redirect(request.url)

    return render_template('upload.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/feature_hashing', methods=['POST'])
def feature_hashing():
    data_source = request.form.get('data_source')
    if data_source == 'sample_social':
        data = get_sample_social_data()
        if not data:
            return render_template('feature_hashing.html', error="Không có dữ liệu mẫu mạng xã hội!")
    # ... tiếp tục xử lý

def get_sample_social_data():
    return [
        "Tôi thích dùng mạng xã hội để kết nối bạn bè.",
        "Bài viết này rất hay và bổ ích.",
        "Chia sẻ cảm xúc trên Facebook thật dễ dàng."
    ]

# Đăng ký blueprint sau khi đã định nghĩa các route
app.register_blueprint(routes)

# Đăng ký blueprint ai_demo
app.register_blueprint(ai_demo, url_prefix='/ai_demo')

if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(debug=True, port=5000)
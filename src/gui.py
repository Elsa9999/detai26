import os
import csv
import timeit
import tracemalloc
from io import StringIO
from flask import (Flask, render_template, request,
                   redirect, url_for, flash, make_response, send_file, Response)
from werkzeug.utils import secure_filename
import pandas as pd
import io
import json

from src.hash_chaining_optimized import HashTableChainingOpt
from src.hash_open_addressing_opt import OpenAddrOpt, HashTableLinear, HashTableQuadratic
from src.benchmark_realistic import benchmark_from_file
from src.benchmark import measure_table
from src.feature_map import FastFeatureMap

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

def allowed_file(fn: str) -> bool:
    return fn.lower().endswith('.csv')

# Tính đường dẫn tới root chứa /templates và /static
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)
app.secret_key = 'super-secret-key'
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static', 'uploads')


# === ROUTES ===

# Trang chủ
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        act = request.form.get('action')
        if act == 'benchmark':
            n    = int(request.form.get('n_items', 100000))
            algos = request.form.getlist('algorithms')
            exp  = int(request.form.get('table_size_exp', 20))
            # Chuyển sang route benchmark_compare
            results = []
            mapping = {
                'chaining':    ('Chaining',    lambda: HashTableChainingOpt(size=1<<exp)),
                'open_double': ('OpenDouble',  lambda: OpenAddrOpt(exp=exp)),
                'linear':      ('Linear',      lambda: HashTableLinear(exp=exp)),
                'quadratic':   ('Quadratic',   lambda: HashTableQuadratic(exp=exp)),
            }
            for algo in algos:
                if algo not in mapping:
                    continue
                name, ctor = mapping[algo]
                try:
                    ti, ts, mem, ip, sp, coll, load = measure_table(name, ctor, n)
                    results.append({
                        'algo': algo, 
                        'ti': ti, 
                        'ts': ts, 
                        'mem': mem,
                        'avg_insert_probe': ip,
                        'avg_search_probe': sp,
                        'collisions': coll,
                        'load_factor': load
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
                        'error': str(e)
                    })
            return render_template(
                'benchmark_compare.html',
                filename=f'Synthetic {n} items', exp=exp, chunksize='-',
                results=results
            )
        if act == 'feature_map':
            txt = request.form.get('sample_text', '')
            return redirect(url_for('feature_map', text=txt))
    return render_template('index.html')


# Upload & Realistic benchmark
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        try:
            f = request.files.get('datafile')
            algos = request.form.getlist('algorithms')
            exp = int(request.form.get('table_size_exp', 22))
            chunks = int(request.form.get('chunksize', 1_000_000))

            # Kiểm tra file hợp lệ
            if not f or not allowed_file(f.filename):
                flash('Vui lòng upload file CSV có định dạng .csv!', 'danger')
                return redirect(request.url)

            # Giới hạn kích thước file (10MB)
            f.seek(0, 2)
            size = f.tell()
            f.seek(0)
            if size > 10 * 1024 * 1024:
                flash('File quá lớn! Vui lòng chọn file nhỏ hơn 10MB.', 'danger')
                return redirect(request.url)

            # Tạo thư mục uploads nếu chưa tồn tại
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            name = secure_filename(f.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], name)
            f.save(path)

            # Kiểm tra cột CSV
            try:
                df = pd.read_csv(path, nrows=1)
                if 'key' not in df.columns or 'value' not in df.columns:
                    flash("File CSV phải có cột 'key' và 'value'!", 'danger')
                    os.remove(path)
                    return redirect(request.url)
            except Exception as e:
                flash(f'Lỗi khi đọc file CSV: {str(e)}', 'danger')
                os.remove(path)
                return redirect(request.url)

            # Chạy benchmark cho từng thuật toán
            results = []
            for algo in algos:
                try:
                    ti, ts, mem, ip, sp, coll, load = benchmark_from_file(path, algo, exp, chunks)
                    results.append({
                        'algo': algo,
                        'ti': ti,
                        'ts': ts,
                        'mem': mem,
                        'avg_insert_probe': ip,
                        'avg_search_probe': sp,
                        'collisions': coll,
                        'load_factor': load
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


# Export CSV synthetic
@app.route('/export_synthetic/<int:n>/<algo>/<int:exp>')
def export_synthetic(n, algo, exp):
    funcs = {
      'chaining':    ('Chaining',    lambda: HashTableChainingOpt(size=1<<exp)),
      'open_double': ('OpenDouble',  lambda: OpenAddrOpt(exp=exp)),
      'linear':      ('Linear',      lambda: HashTableLinear(exp=exp)),
      'quadratic':   ('Quadratic',   lambda: HashTableQuadratic(exp=exp)),
    }
    if algo not in funcs:
        return redirect(url_for('index'))

    name, ctor = funcs[algo]
    ti, ts, mem, ip, sp, coll, load = measure_table(name, ctor, n)

    buf = StringIO()
    w = csv.writer(buf)
    w.writerow(['algorithm','n','exp','insert','search','memory','avg_insert_probe','avg_search_probe','collisions','load_factor'])
    w.writerow([algo, n, exp, f"{ti:.6f}", f"{ts:.6f}", f"{mem:.6f}", f"{ip:.2f}", f"{sp:.2f}", coll, f"{load:.2f}"])
    resp = make_response(buf.getvalue())
    resp.headers['Content-Disposition'] = f'attachment; filename=synthetic_{algo}_{n}.csv'
    resp.headers['Content-Type'] = 'text/csv'
    return resp


# Feature map
@app.route('/feature_map', methods=['GET', 'POST'])
def feature_map():
    error = None
    tokens = []
    vector = []
    mapping = []
    if request.method == 'POST':
        text = request.form.get('sample_text', '')
        method = request.form.get('method', 'chaining')
        if not text.strip():
            error = "Vui lòng nhập văn bản!"
        else:
            try:
                tokens = text.split()
                fmap = FastFeatureMap(method=method)
                vector = fmap.vectorize(tokens)
                # Lấy feature duy nhất và id, sort theo id
                unique_features = list(dict.fromkeys(tokens))
                mapping = [(tok, fmap.get_feature_id(tok)) for tok in unique_features]
            except Exception as e:
                error = f"Lỗi xử lý: {str(e)}"
    else:
        text = request.args.get('text', '')
        method = request.args.get('method', 'chaining')
        if text.strip():
            try:
                tokens = text.split()
                fmap = FastFeatureMap(method=method)
                vector = fmap.vectorize(tokens)
                # Lấy feature duy nhất và id, sort theo id
                unique_features = list(dict.fromkeys(tokens))
                mapping = [(tok, fmap.get_feature_id(tok)) for tok in unique_features]
            except Exception as e:
                error = f"Lỗi xử lý: {str(e)}"
    return render_template('feature_map.html', tokens=tokens, vector=vector, mapping=mapping, error=error)


# CRUD demo
@app.route('/crud', methods=['GET','POST'])
def crud():
    if request.method == 'POST':
        algo = request.form['algorithm']
        key  = request.form['key']
        val  = request.form.get('value','')
        act  = request.form['crud_action']

        if algo not in tables:
            tables[algo] = new_table(algo)
        tbl = tables[algo]

        if act == 'insert':
            v = int(val) if val.isdigit() else val
            tbl.insert(key, v)
            flash(f"Inserted '{key}' → {v}", 'crud')
        elif act == 'search':
            r = tbl.search(key)
            flash(f"{'Found: '+str(r) if r is not None else 'Not found'}", 'crud')
        elif act == 'delete':
            r = tbl.delete(key)
            flash(f"{'Deleted' if r is not None else 'Key not found'}", 'crud')

        return redirect(url_for('crud'))

    return render_template('crud.html', tables=tables)


@app.route('/export_compare_csv', methods=['POST'])
def export_compare_csv():
    try:
        filename = request.form.get('filename', 'benchmark')
        exp = request.form.get('exp', '22')
        chunksize = request.form.get('chunksize', '1000000')
        results = request.form.getlist('results')

        # Tạo DataFrame từ kết quả
        data = []
        for r in results:
            try:
                result = json.loads(r)
                data.append({
                    'Algorithm': result['algo'],
                    'Insert Time (s)': f"{result['ti']:.6f}" if result['ti'] is not None else 'N/A',
                    'Search Time (s)': f"{result['ts']:.6f}" if result['ts'] is not None else 'N/A',
                    'Memory Usage (MB)': f"{result['mem']:.2f}" if result['mem'] is not None else 'N/A',
                    'Avg Insert Probe': f"{result['avg_insert_probe']:.2f}" if result.get('avg_insert_probe') is not None else 'N/A',
                    'Avg Search Probe': f"{result['avg_search_probe']:.2f}" if result.get('avg_search_probe') is not None else 'N/A',
                    'Collisions': result.get('collisions', 'N/A'),
                    'Load Factor': f"{result['load_factor']:.2f}" if result.get('load_factor') is not None else 'N/A',
                    'Error': result.get('error', '')
                })
            except:
                continue

        if not data:
            flash('Không có dữ liệu để xuất!', 'warning')
            return redirect(url_for('index'))

        # Tạo DataFrame và xuất CSV
        df = pd.DataFrame(data)
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        return Response(
            output,
            mimetype='text/csv',
            headers={
                'Content-Disposition': f'attachment; filename={filename}_benchmark_results.csv'
            }
        )

    except Exception as e:
        flash(f'Lỗi khi xuất CSV: {str(e)}', 'danger')
        return redirect(url_for('index'))


@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == "__main__":
    app.run(debug=True) 
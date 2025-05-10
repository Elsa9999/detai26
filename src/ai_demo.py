import numpy as np
from typing import List, Tuple, Dict
import mmh3
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from datetime import datetime
import os
import random
import string
import requests
from flask import Blueprint, request, jsonify, send_file, current_app, render_template
import pandas as pd
import logging
import traceback
import sys
import time

# Cấu hình logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

ai_demo = Blueprint('ai_demo', __name__)

# Thư mục lưu trữ dữ liệu mẫu
SAMPLE_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'samples')

# Tạo thư mục nếu chưa tồn tại
try:
    os.makedirs(SAMPLE_DATA_DIR, exist_ok=True)
    logger.info(f"Đã tạo thư mục dữ liệu mẫu: {SAMPLE_DATA_DIR}")
except Exception as e:
    logger.error(f"Lỗi khi tạo thư mục dữ liệu mẫu: {str(e)}")
    raise

def generate_sample_text(n_samples: int) -> List[str]:
    """Tạo dữ liệu văn bản mẫu thực tế, đa chủ đề"""
    topics = [
        "công nghệ", "thể thao", "giải trí", "kinh doanh", "giáo dục",
        "sức khỏe", "du lịch", "ẩm thực", "thời trang", "xe cộ"
    ]
    templates = [
        "Hôm nay tôi đọc một bài báo về {topic} rất thú vị.",
        "{topic_cap} đang phát triển mạnh mẽ trong những năm gần đây.",
        "Tôi thích tham gia các hoạt động liên quan đến {topic} cùng bạn bè.",
        "Bài học về {topic} giúp tôi mở mang kiến thức.",
        "Tin tức mới nhất về {topic} khiến nhiều người quan tâm.",
        "Chuyên gia nhận định {topic} sẽ thay đổi trong tương lai.",
        "Nhiều sự kiện {topic} diễn ra sôi động ở thành phố lớn.",
        "Tôi thường xuyên tìm hiểu về {topic} trên mạng xã hội.",
        "{topic_cap} là lĩnh vực tôi muốn theo đuổi.",
        "Gia đình tôi thường nói chuyện về {topic} vào cuối tuần."
    ]
    sentences = []
    for i in range(n_samples):
        topic = random.choice(topics)
        template = random.choice(templates)
        text = template.format(topic=topic, topic_cap=topic.capitalize())
        sentences.append(text)
    return sentences

def generate_sample_vectors(n_samples: int, vector_size: int) -> np.ndarray:
    """Tạo các nhóm vector gần nhau để dễ kiểm thử LSH"""
    n_clusters = max(2, min(5, n_samples // 100))
    cluster_centers = [np.random.uniform(-1, 1, vector_size) for _ in range(n_clusters)]
    vectors = []
    for i in range(n_samples):
        center = random.choice(cluster_centers)
        # Thêm nhiễu nhỏ quanh tâm cụm
        noise = np.random.normal(0, 0.1, vector_size)
        vec = center + noise
        vectors.append(vec)
    return np.array(vectors)

@ai_demo.route('/sample_data/<data_type>')
def download_sample_data(data_type):
    """Tải xuống dữ liệu mẫu"""
    try:
        logger.info(f"Bắt đầu tạo dữ liệu mẫu loại: {data_type}")
        
        if data_type == 'text':
            # Tạo dữ liệu văn bản mẫu
            texts = generate_sample_text(1000)
            df = pd.DataFrame({'text': texts})
            file_path = os.path.join(SAMPLE_DATA_DIR, 'sample_text.csv')
            df.to_csv(file_path, index=False, encoding='utf-8')
            logger.info(f"Đã lưu dữ liệu văn bản mẫu vào: {file_path}")
            return send_file(file_path, as_attachment=True)
            
        elif data_type == 'vectors':
            # Tạo dữ liệu vector mẫu
            vectors = generate_sample_vectors(1000, 100)
            file_path = os.path.join(SAMPLE_DATA_DIR, 'sample_vectors.npy')
            np.save(file_path, vectors)
            logger.info(f"Đã lưu dữ liệu vector mẫu vào: {file_path}")
            return send_file(file_path, as_attachment=True)
            
        logger.error(f"Loại dữ liệu không hợp lệ: {data_type}")
        return jsonify({'error': 'Loại dữ liệu không hợp lệ'}), 400
        
    except Exception as e:
        logger.error(f"Lỗi khi tạo dữ liệu mẫu: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@ai_demo.route('/create_data', methods=['POST'])
def create_data():
    try:
        data_type = request.form.get('data_type')
        sample_count = int(request.form.get('sample_count', 1000))
        vector_size = int(request.form.get('vector_size', 100))
        if data_type == 'text':
            texts = generate_sample_text(sample_count)
            df = pd.DataFrame({'text': texts})
            file_path = os.path.join(SAMPLE_DATA_DIR, 'generated_text.csv')
            df.to_csv(file_path, index=False, encoding='utf-8')
            message = 'Tạo dữ liệu văn bản thành công'
            return render_template('create_data_result.html', file_path=file_path, message=message, sample_count=sample_count)
        elif data_type == 'vectors':
            vectors = generate_sample_vectors(sample_count, vector_size)
            file_path = os.path.join(SAMPLE_DATA_DIR, 'generated_vectors.npy')
            np.save(file_path, vectors)
            message = 'Tạo dữ liệu vector thành công'
            return render_template('create_data_result.html', file_path=file_path, message=message, sample_count=sample_count, vector_size=vector_size)
        message = 'Loại dữ liệu không hợp lệ'
        return render_template('create_data_result.html', file_path=None, message=message, sample_count=sample_count)
    except Exception as e:
        return render_template('create_data_result.html', file_path=None, message=f'Lỗi: {str(e)}', sample_count=0)

@ai_demo.route('/test_api', methods=['POST'])
def test_api():
    """Kiểm tra kết nối API"""
    try:
        api_url = request.form.get('api_url')
        api_key = request.form.get('api_key')
        
        logger.info(f"Kiểm tra kết nối API: {api_url}")
        
        if not api_url:
            logger.error("URL API không được để trống")
            return jsonify({'error': 'URL API không được để trống'}), 400
            
        headers = {}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
            
        response = requests.get(api_url, headers=headers, timeout=5)
        logger.info(f"Kết quả kiểm tra API: status={response.status_code}")
        
        return jsonify({
            'status': response.status_code,
            'message': 'Kết nối API thành công' if response.ok else 'Kết nối API thất bại'
        })
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Lỗi kết nối API: {str(e)}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Lỗi không xác định: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# Cập nhật route feature_hash để hỗ trợ nhiều nguồn dữ liệu
@ai_demo.route('/feature_hash', methods=['POST'])
def feature_hash():
    data_source = request.form.get('data_source', 'text')
    hash_algo = request.form.get('hash_algo', 'chaining')
    batch_size = int(request.form.get('batch_size', 1000))
    
    try:
        if data_source == 'text':
            text = request.form.get('text', '')
            if not text:
                return jsonify({'error': 'Văn bản không được để trống'}), 400
            texts = [text]
            
        elif data_source == 'file':
            file = request.files.get('file')
            if not file:
                return jsonify({'error': 'Không tìm thấy file'}), 400
                
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file)
                texts = df['text'].tolist()
            else:
                texts = file.read().decode('utf-8').splitlines()
                
        elif data_source == 'sample':
            sample_type = request.form.get('sample_data')
            file_path = os.path.join(SAMPLE_DATA_DIR, f'sample_{sample_type}.csv')
            if not os.path.exists(file_path):
                return jsonify({'error': 'Không tìm thấy dữ liệu mẫu'}), 400
            df = pd.read_csv(file_path)
            texts = df['text'].tolist()
            
        elif data_source == 'api':
            api_url = request.form.get('api_url')
            api_key = request.form.get('api_key')
            
            if not api_url:
                return jsonify({'error': 'URL API không được để trống'}), 400
                
            headers = {}
            if api_key:
                headers['Authorization'] = f'Bearer {api_key}'
                
            response = requests.get(api_url, headers=headers)
            if not response.ok:
                return jsonify({'error': 'Không thể lấy dữ liệu từ API'}), 400
                
            data = response.json()
            texts = [item['text'] for item in data]
            
        else:
            return jsonify({'error': 'Nguồn dữ liệu không hợp lệ'}), 400
            
        # Xử lý dữ liệu theo batch
        hasher = FeatureHasher(hash_algo=hash_algo, batch_size=batch_size)
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = hasher._process_batch(batch)
            results.extend(batch_results['hash_vectors'])
            
        return jsonify({
            'message': 'Xử lý thành công',
            'results': results,
            'stats': hasher.stats
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Cập nhật route LSH tương tự
@ai_demo.route('/lsh', methods=['POST'])
def lsh_search():
    data_source = request.form.get('data_source', 'vector')
    k = int(request.form.get('k', 5))
    batch_size = int(request.form.get('batch_size', 1000))
    
    try:
        if data_source == 'vector':
            vector_str = request.form.get('query_vector', '')
            if not vector_str:
                return jsonify({'error': 'Vector không được để trống'}), 400
                
            # Chuyển đổi chuỗi thành vector
            vector = np.array(eval(vector_str))
            
        elif data_source == 'file':
            file = request.files.get('file')
            if not file:
                return jsonify({'error': 'Không tìm thấy file'}), 400
                
            if file.filename.endswith('.npy'):
                vector = np.load(file)
            else:
                df = pd.read_csv(file)
                vector = df.values
                
        elif data_source == 'sample':
            sample_type = request.form.get('sample_data')
            file_path = os.path.join(SAMPLE_DATA_DIR, f'sample_{sample_type}.npy')
            if not os.path.exists(file_path):
                return jsonify({'error': 'Không tìm thấy dữ liệu mẫu'}), 400
            vector = np.load(file_path)
            
        elif data_source == 'api':
            api_url = request.form.get('api_url')
            api_key = request.form.get('api_key')
            
            if not api_url:
                return jsonify({'error': 'URL API không được để trống'}), 400
                
            headers = {}
            if api_key:
                headers['Authorization'] = f'Bearer {api_key}'
                
            response = requests.get(api_url, headers=headers)
            if not response.ok:
                return jsonify({'error': 'Không thể lấy dữ liệu từ API'}), 400
                
            data = response.json()
            vector = np.array(data['vector'])
            
        else:
            return jsonify({'error': 'Nguồn dữ liệu không hợp lệ'}), 400
            
        # Thực hiện tìm kiếm LSH
        lsh = LSH(batch_size=batch_size)
        results = lsh.search(vector, k=k)
        
        return jsonify({
            'message': 'Tìm kiếm thành công',
            'results': results['results'],
            'stats': results['stats'],
            'query_info': results['query_info']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

class FeatureHasher:
    def __init__(self, n_features: int = 1000, hash_algo: str = 'chaining', batch_size: int = 1000):
        self.n_features = n_features
        self.hash_algo = hash_algo
        self.feature_map = {}
        self.token_stats = Counter()
        self.tfidf = TfidfVectorizer(max_features=n_features)
        self.batch_size = batch_size
        self.stats = {
            'total_documents': 0,
            'total_tokens': 0,
            'unique_tokens': 0,
            'avg_tokens_per_doc': 0,
            'last_processed': None,
            'processing_times': [],
            'memory_usage': []
        }
        
    def preprocess_text(self, text: str) -> List[str]:
        # Loại bỏ số và ký tự đặc biệt, chỉ giữ lại chữ cái
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        # Tách từ và loại bỏ từ trống
        tokens = [token for token in text.split() if token]
        return tokens
        
    def hash_token(self, token: str, attempt: int = 0) -> int:
        """Hash function tùy chỉnh để tạo ra các giá trị hash đồng nhất"""
        if self.hash_algo == 'chaining':
            return mmh3.hash(token) % self.n_features
        elif self.hash_algo == 'linear':
            return (mmh3.hash(token) + attempt) % self.n_features
        elif self.hash_algo == 'quadratic':
            return (mmh3.hash(token) + attempt * attempt) % self.n_features
        else:  # double hashing
            h1 = mmh3.hash(token)
            h2 = mmh3.hash(str(h1))
            return (h1 + attempt * h2) % self.n_features
        
    def process_large_file(self, file_path: str, callback=None) -> Dict:
        """Xử lý file lớn theo batch"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Không tìm thấy file: {file_path}")
            
        start_time = datetime.now()
        total_lines = sum(1 for _ in open(file_path, 'r', encoding='utf-8'))
        processed_lines = 0
        
        results = {
            'hash_vectors': [],
            'tfidf_vectors': [],
            'feature_mappings': [],
            'token_stats': Counter(),
            'processing_stats': {
                'total_documents': 0,
                'total_tokens': 0,
                'unique_tokens': 0,
                'avg_tokens_per_doc': 0,
                'processing_times': [],
                'memory_usage': []
            }
        }
        
        with open(file_path, 'r', encoding='utf-8') as f:
            batch = []
            for line in f:
                line = line.strip()
                if line:
                    batch.append(line)
                    processed_lines += 1
                    
                    if len(batch) >= self.batch_size:
                        # Xử lý batch
                        batch_results = self._process_batch(batch)
                        self._update_results(results, batch_results)
                        
                        # Cập nhật tiến trình
                        if callback:
                            progress = (processed_lines / total_lines) * 100
                            callback(progress)
                            
                        batch = []
                        
            # Xử lý batch cuối cùng
            if batch:
                batch_results = self._process_batch(batch)
                self._update_results(results, batch_results)
                
        # Tính toán thống kê cuối cùng
        processing_time = (datetime.now() - start_time).total_seconds()
        results['processing_stats']['total_processing_time'] = processing_time
        results['processing_stats']['avg_processing_time'] = float(np.mean(results['processing_stats']['processing_times']))
        results['processing_stats']['avg_memory_usage'] = float(np.mean(results['processing_stats']['memory_usage']))
        
        return results
        
    def _process_batch(self, batch: List[str]) -> Dict:
        start_time = datetime.now()
        all_tokens = []
        for text in batch:
            tokens = self.preprocess_text(text)
            all_tokens.extend(tokens)
        self.token_stats.update(all_tokens)
        tfidf_vectors = self.tfidf.fit_transform(batch).toarray()
        hash_vectors = []
        feature_mappings = []
        for text in batch:
            tokens = self.preprocess_text(text)
            hash_vector = np.zeros(self.n_features)
            feature_mapping = {}
            for token in tokens:
                attempt = 0
                while True:
                    idx = self.hash_token(token, attempt)
                    if idx not in feature_mapping.values() or attempt > 10:
                        break
                    attempt += 1
                hash_vector[idx] += 1
                feature_mapping[token] = idx
            # Nếu không có token nào, trả về vector 0
            if not tokens:
                print("DEBUG: Không có token hợp lệ, trả về vector 0")
            hash_vectors.append(hash_vector)
            feature_mappings.append(feature_mapping)
        print("DEBUG hash_vectors:", hash_vectors)
        processing_time = (datetime.now() - start_time).total_seconds()
        memory_usage = self._get_memory_usage()
        return {
            'hash_vectors': hash_vectors,
            'tfidf_vectors': tfidf_vectors.tolist(),
            'feature_mappings': feature_mappings,
            'token_stats': dict(Counter(all_tokens)),
            'processing_stats': {
                'documents_processed': len(batch),
                'tokens_processed': len(all_tokens),
                'processing_time': processing_time,
                'memory_usage': memory_usage
            }
        }
        
    def _update_results(self, results: Dict, batch_results: Dict):
        """Cập nhật kết quả tổng hợp"""
        results['hash_vectors'].extend(batch_results['hash_vectors'])
        results['tfidf_vectors'].extend(batch_results['tfidf_vectors'])
        results['feature_mappings'].extend(batch_results['feature_mappings'])
        results['token_stats'].update(batch_results['token_stats'])
        
        # Cập nhật thống kê
        stats = results['processing_stats']
        batch_stats = batch_results['processing_stats']
        
        stats['total_documents'] += batch_stats['documents_processed']
        stats['total_tokens'] += batch_stats['tokens_processed']
        stats['unique_tokens'] = len(results['token_stats'])
        stats['avg_tokens_per_doc'] = stats['total_tokens'] / stats['total_documents']
        stats['processing_times'].append(batch_stats['processing_time'])
        stats['memory_usage'].append(batch_stats['memory_usage'])
        
    def _get_memory_usage(self) -> float:
        """Lấy thông tin sử dụng bộ nhớ"""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB

    def transform(self, texts: List[str]) -> List[np.ndarray]:
        """Chuyển đổi danh sách văn bản thành vector"""
        if not texts:
            return []
            
        # Tiền xử lý văn bản
        processed_texts = []
        for text in texts:
            tokens = self.preprocess_text(text)
            processed_texts.append(' '.join(tokens))
            
        # Tạo vector TF-IDF
        tfidf_vectors = self.tfidf.fit_transform(processed_texts).toarray()
        
        # Tạo vector hash
        hash_vectors = []
        for text in processed_texts:
            tokens = self.preprocess_text(text)
            hash_vector = np.zeros(self.n_features)
            feature_mapping = {}
            
            for token in tokens:
                attempt = 0
                while True:
                    idx = self.hash_token(token, attempt)
                    if idx not in feature_mapping.values() or attempt > 10:
                        break
                    attempt += 1
                    
                hash_vector[idx] += 1
                feature_mapping[token] = idx
                
            hash_vectors.append(hash_vector)
            
        return hash_vectors

class LSH:
    def __init__(self, n_vectors: int = 1000, n_bits: int = 8, n_tables: int = 4, batch_size: int = 1000):
        self.n_vectors = n_vectors
        self.n_bits = n_bits
        self.n_tables = n_tables
        self.batch_size = batch_size
        self.vectors = []
        self.vector_ids = []
        
        # Khởi tạo nhiều bảng hash
        self.hash_tables = []
        self.projections = []
        for _ in range(n_tables):
            proj = np.random.randn(n_bits, n_vectors)
            proj = proj / np.linalg.norm(proj, axis=1, keepdims=True)
            self.projections.append(proj)
            self.hash_tables.append({})
        
        # Lưu trữ thống kê
        self.stats = {
            'total_vectors': 0,
            'total_queries': 0,
            'avg_distance': 0,
            'avg_similarity': 0,
            'last_query': None,
            'query_times': [],
            'collision_counts': [0] * n_tables,
            'memory_usage': []
        }
        
    def _compute_hash(self, vector: np.ndarray, table_idx: int) -> str:
        """Tính toán hash cho một vector trong một bảng cụ thể"""
        # Chiếu vector lên các vector ngẫu nhiên
        projections = np.dot(self.projections[table_idx], vector)
        # Chuyển đổi thành chuỗi bit
        bits = (projections > 0).astype(int)
        return ''.join(map(str, bits))
        
    def add_vectors_from_file(self, file_path: str, callback=None) -> Dict:
        """Thêm vectors từ file lớn"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Không tìm thấy file: {file_path}")
            
        start_time = datetime.now()
        total_lines = sum(1 for _ in open(file_path, 'r', encoding='utf-8'))
        processed_lines = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            batch = []
            batch_ids = []
            
            for line in f:
                line = line.strip()
                if line:
                    # Giả sử mỗi dòng có format: id,vector
                    parts = line.split(',', 1)
                    if len(parts) == 2:
                        vector_id, vector_str = parts
                        try:
                            vector = np.array(eval(vector_str))
                            if len(vector) == self.n_vectors:
                                batch.append(vector)
                                batch_ids.append(vector_id)
                                processed_lines += 1
                                
                                if len(batch) >= self.batch_size:
                                    # Xử lý batch
                                    self._process_batch(batch, batch_ids)
                                    
                                    # Cập nhật tiến trình
                                    if callback:
                                        progress = (processed_lines / total_lines) * 100
                                        callback(progress)
                                        
                                    batch = []
                                    batch_ids = []
                        except:
                            continue
                            
            # Xử lý batch cuối cùng
            if batch:
                self._process_batch(batch, batch_ids)
                
        # Cập nhật thống kê
        processing_time = (datetime.now() - start_time).total_seconds()
        self.stats['total_processing_time'] = processing_time
        self.stats['avg_processing_time'] = float(np.mean(self.stats['query_times']))
        self.stats['avg_memory_usage'] = float(np.mean(self.stats['memory_usage']))
        
        return self.stats
        
    def _process_batch(self, vectors: List[np.ndarray], vector_ids: List[str]):
        """Xử lý một batch vectors"""
        for vector, vector_id in zip(vectors, vector_ids):
            # Chuẩn hóa vector
            vector = vector / np.linalg.norm(vector)
            
            self.vectors.append(vector)
            self.vector_ids.append(vector_id)
            
            # Thêm vector vào tất cả các bảng hash
            for i in range(self.n_tables):
                hash_val = self._compute_hash(vector, i)
                if hash_val in self.hash_tables[i]:
                    self.stats['collision_counts'][i] += 1
                self.hash_tables[i][hash_val] = self.hash_tables[i].get(hash_val, []) + [len(self.vectors) - 1]
                
        # Cập nhật thống kê
        self.stats['total_vectors'] += len(vectors)
        self.stats['memory_usage'].append(self._get_memory_usage())
        
    def _get_memory_usage(self) -> float:
        """Lấy thông tin sử dụng bộ nhớ"""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB

    def add_vector(self, vector: np.ndarray, vector_id: str = None):
        """Thêm một vector vào LSH"""
        try:
            # Chuẩn hóa vector
            vector = vector / np.linalg.norm(vector)
            
            self.vectors.append(vector)
            self.vector_ids.append(vector_id or f"vec_{len(self.vectors)}")
            
            # Thêm vector vào tất cả các bảng hash
            for i in range(self.n_tables):
                hash_val = self._compute_hash(vector, i)
                if hash_val in self.hash_tables[i]:
                    self.stats['collision_counts'][i] += 1
                self.hash_tables[i][hash_val] = self.hash_tables[i].get(hash_val, []) + [len(self.vectors) - 1]
                
            # Cập nhật thống kê
            self.stats['total_vectors'] += 1
            self.stats['memory_usage'].append(self._get_memory_usage())
            logger.info(f"Đã thêm vector {vector_id} vào LSH")
            
        except Exception as e:
            logger.error(f"Lỗi khi thêm vector: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def search(self, query_vector: np.ndarray, k: int = 5):
        """Tìm k vector gần nhất dựa trên cosine similarity"""
        start_time = time.time()
        if len(self.vectors) == 0:
            return {
                'results': [],
                'stats': self.stats,
                'query_info': {
                    'processing_time': 0,
                    'candidates_found': 0,
                    'candidates_ratio': 0
                }
            }
        # Tính cosine similarity
        vectors = np.array(self.vectors)
        sims = cosine_similarity([query_vector], vectors)[0]
        topk_idx = np.argsort(sims)[::-1][:k]
        results = []
        for idx in topk_idx:
            results.append({
                'id': self.vector_ids[idx] if idx < len(self.vector_ids) else str(idx),
                'vector': vectors[idx].tolist(),
                'cosine_similarity': float(sims[idx])
            })
        processing_time = time.time() - start_time
        self.stats['total_queries'] += 1
        self.stats['avg_similarity'] = float(np.mean(sims))
        self.stats['last_query'] = query_vector.tolist()
        query_info = {
            'processing_time': processing_time,
            'candidates_found': len(results),
            'candidates_ratio': len(results) / len(self.vectors) if self.vectors else 0
        }
        return {
            'results': results,
            'stats': self.stats,
            'query_info': query_info
        }

def demo_lsh():
    print("\n=== Demo Locality Sensitive Hashing (LSH) ===")
    
    # Khởi tạo LSH với các tham số phù hợp
    lsh = LSH(n_vectors=10, n_bits=4, n_tables=2)
    
    # Tạo dữ liệu mẫu - mỗi vector đại diện cho một tài liệu
    documents = [
        "Tôi thích ăn táo và chuối",
        "Táo và chuối là trái cây ngon",
        "Tôi thích uống nước cam",
        "Cam và chanh đều là trái cây có múi",
        "Tôi thích ăn nho và dưa hấu"
    ]
    
    # Chuyển đổi văn bản thành vector
    vectorizer = TfidfVectorizer(max_features=10)
    vectors = vectorizer.fit_transform(documents).toarray()
    
    print("\n1. Thêm các tài liệu vào LSH:")
    for i, (doc, vec) in enumerate(zip(documents, vectors)):
        lsh.add_vector(vec, f"doc{i+1}")
        print(f"\nTài liệu {i+1}:")
        print(f"Văn bản: {doc}")
        print(f"Vector: {vec}")
    
    # Tạo query
    query_text = "Tôi thích ăn táo"
    query_vector = vectorizer.transform([query_text])[0]
    
    print("\n2. Tìm kiếm tài liệu tương tự:")
    print(f"Query: {query_text}")
    print(f"Query vector: {query_vector}")
    
    # Tìm kiếm
    results = lsh.search(query_vector, k=3)
    
    print("\n3. Kết quả tìm kiếm:")
    for i, result in enumerate(results['results'], 1):
        doc_idx = int(result['id'].replace('doc', '')) - 1
        print(f"\n{i}. {result['id']}:")
        print(f"   Văn bản: {documents[doc_idx]}")
        print(f"   Vector: {result['vector']}")
        print(f"   Độ tương đồng: {result['cosine_similarity']:.4f}")
    
    print("\n4. Thống kê:")
    print(f"Tổng số tài liệu: {results['stats']['total_vectors']}")
    print(f"Tổng số query: {results['stats']['total_queries']}")
    print(f"Độ tương đồng trung bình: {results['stats']['avg_similarity']:.4f}")
    print(f"Tỷ lệ ứng viên: {results['stats']['candidates_ratio']:.4f}")
    print(f"Số collision trung bình: {results['stats']['avg_collisions']:.2f}")
    
    print("\n5. Thông tin query:")
    print(f"Thời gian xử lý: {results['query_info']['processing_time']:.4f} giây")
    print(f"Số ứng viên tìm thấy: {results['query_info']['candidates_found']}")
    print(f"Tỷ lệ ứng viên: {results['query_info']['candidates_ratio']:.4f}")

if __name__ == "__main__":
    demo_lsh() 
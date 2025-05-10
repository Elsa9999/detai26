# Hash Table Benchmark Web Application

## Project Introduction
This web application, built with Flask, allows users to benchmark and compare different hash table algorithms, including Chaining, Open Addressing (Double Hashing, Linear Probing, Quadratic Probing), using both synthetic and real-world data. The app provides an interactive interface for uploading data, running benchmarks, visualizing results, and experimenting with hash table operations.

## Features
- **Synthetic Benchmark:** Generate random key-value pairs to test and compare the performance (insertion/search time, memory usage) of various hash table algorithms.
- **Realistic Benchmark:** Upload your own CSV file (with `key,value` columns) to benchmark hash tables on real data.
- **Feature Hashing (Hashing Trick):** Convert text into hashed feature vectors and view the mapping and statistics.
- **CRUD Demo:** Practice Insert, Search, and Delete operations directly on different hash table types and observe their internal state.
- **Modern UI:** Beautiful, responsive interface with real-time feedback and visualizations.

## Installation
- Python 3.8 or higher is required
- Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run
Start the application with:
```bash
python app.py
```
Then open your browser and go to: [http://localhost:5000](http://localhost:5000)

## Example CSV Format
To use the Realistic Benchmark feature, upload a CSV file with the following format (must have columns `key,value`):

```csv
key,value
a,1
b,2
c,3
```

## How to Use
1. **Synthetic Benchmark:**
   - Select the number of items and hash table algorithms.
   - Click "Run Synthetic" to view performance and memory usage charts.
2. **Realistic Benchmark:**
   - Upload a CSV file with your own data.
   - Select algorithms and parameters, then click "Upload & Run Benchmark".
   - View detailed results and download comparison data if needed.
3. **Feature Hashing:**
   - Enter a text sample and select a hashing method.
   - View the resulting hash vector, TF-IDF vector, feature mapping, and token statistics.
4. **CRUD Demo:**
   - Choose an algorithm, enter key/value, select an action (Insert, Search, Delete), and submit.
   - See real-time feedback and the current state of all hash tables.

## Result Analysis
### 1. Synthetic Benchmark
- **Insertion/Search Time:**
  - Chaining is generally fast for moderate load factors but can slow down with many collisions.
  - Open Addressing (Double Hashing, Linear, Quadratic) shows different performance depending on collision resolution and table size.
- **Memory Usage:**
  - Chaining uses more memory due to linked lists in buckets.
  - Open Addressing is more memory-efficient but can suffer from clustering.
- **Load Factor:**
  - High load factors can degrade performance, especially for open addressing.

### 2. Realistic Benchmark
- Results may vary depending on data distribution, key patterns, and table size.
- The app visualizes and compares all selected algorithms, helping users understand trade-offs in real scenarios.

### 3. Feature Hashing
- The Hashing Trick efficiently converts text into fixed-size vectors, suitable for machine learning.
- TF-IDF vectors provide additional weighting based on term frequency and document frequency.
- The app displays feature mapping and token statistics for transparency.

### 4. CRUD Demo
- Users can observe how different hash table types handle insertions, searches, and deletions in real time.
- Error messages and success notifications help users understand the behavior and limitations of each algorithm.

## Conclusion
This project provides an educational and practical tool for studying hash table algorithms, their performance, and their application in data processing and machine learning. The interactive interface and visual feedback make it easy to experiment and gain insights into hash table behavior.

## Mô tả
Đây là ứng dụng web Flask cho phép benchmark các thuật toán hash table (Chaining, Open Addressing, Linear/Quadratic Probing) với dữ liệu tổng hợp và dữ liệu thực tế (upload file CSV). Ứng dụng hỗ trợ so sánh hiệu năng, bộ nhớ, và thao tác trực quan với các bảng băm.

## Sử dụng
- **Synthetic Benchmark**: Chạy benchmark với dữ liệu ngẫu nhiên.
- **Realistic Benchmark**: Upload file CSV (cột `key,value`) để benchmark với dữ liệu thực tế.
- **Fast Feature Map**: Vector hóa văn bản.
- **CRUD Demo**: Thực hành thao tác với hash table.

## Thư mục
- `app.py`: File chính chạy Flask app
- `templates/`: Giao diện HTML
- `src/`: Chứa các thuật toán và logic xử lý
- `uploads/`: Nơi lưu file upload tạm thời

## Liên hệ
- Nếu gặp lỗi, vui lòng kiểm tra log terminal và đảm bảo file CSV đúng định dạng UTF-8, có cột `key,value`. 
from flask import Blueprint, render_template, request, jsonify
from .ai_demo import FeatureHasher, LSH
import numpy as np

routes = Blueprint('routes', __name__)

@routes.route('/')
def index():
    return render_template('index.html')

@routes.route('/ai_demo')
def ai_demo():
    return render_template('ai_demo.html')

# Khởi tạo các đối tượng AI
hasher = FeatureHasher()
lsh = LSH(n_vectors=5)

# Thêm một số vector mẫu vào LSH
lsh.add_vector(np.array([1,2,3,4,5]), "vec1")
lsh.add_vector(np.array([2,3,4,5,6]), "vec2")
lsh.add_vector(np.array([5,4,3,2,1]), "vec3") 
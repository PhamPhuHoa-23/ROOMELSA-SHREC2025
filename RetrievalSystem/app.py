# app.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
import base64

app = Flask(__name__, static_folder='static')
CORS(app)  # Cho phép cross-origin requests

# Kết nối đến Qdrant server
client = QdrantClient(host="localhost", port=6333)


@app.route('/')
def index():
    """Trả về trang chủ"""
    return send_from_directory('static', 'index.html')


@app.route('/get_text_points', methods=['GET'])
def get_text_points():
    """Lấy tất cả các điểm từ text_collection"""
    try:
        # Lấy tất cả các điểm từ text_collection
        results = client.scroll(
            collection_name="text_collection",
            limit=1000,  # Điều chỉnh giới hạn nếu cần
            with_payload=True,
            with_vectors=True
        )

        # Chuyển đổi kết quả thành dạng JSON
        points = []
        for point in results[0]:
            points.append({
                "id": point.id,
                "vector": point.vector,
                "query": point.payload.get("query", "")
            })

        return jsonify({"status": "success", "points": points})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/query_collections', methods=['POST'])
def query_collections():
    print("It's hero time")
    """Truy vấn các collections dựa trên vector đã chọn và phương thức truy vấn"""
    try:
        data = request.json
        print(data)
        vector_id = data.get("vector_id")
        query_type = data.get("query_type", "combined")  # combined, image, shape
        k = data.get("k", 100)  # Số lượng kết quả trả về (mặc định 100 cho unique UUID)
        image_weight = data.get("image_weight", 0.5)  # Trọng số cho image_score
        shape_weight = data.get("shape_weight", 0.5)  # Trọng số cho shape_score

        # Lấy vector từ text_collection dựa trên ID
        vector_result = client.retrieve(
            collection_name="text_collection",
            ids=[vector_id],
            with_vectors=True
        )

        if not vector_result:
            return jsonify({"status": "error", "message": "Vector not found"}), 404

        query_vector = vector_result[0].vector
        query_text = vector_result[0].payload.get("query", "")

        print(query_vector)
        print(query_text)

        results_list = []

        # Phương thức truy vấn: image (2D)
        if query_type == "image":
            image_results = client.query_points(
                collection_name="image_collection",
                query=query_vector,
                limit=k,
                with_payload=True
            )

            # Chuyển đổi kết quả thành format chuẩn
            for result in image_results.points:
                item = {
                    "uuid": result.payload.get("uuid"),
                    "score": result.score,
                    "iamge_path": result.payload.get("image_path"),
                    "origin_path": result.payload.get("origin_path"),
                    "image_data": None
                }

                # Load ảnh gốc nếu có
                origin_path = result.payload.get("origin_path")
                if origin_path and os.path.exists(origin_path):
                    with open(origin_path, "rb") as img_file:
                        item["image_data"] = base64.b64encode(img_file.read()).decode('utf-8')
                # Nếu không có origin_path, thử load từ path
                elif item["image_path"] and os.path.exists(item["path"]):
                    with open(item["image_path"], "rb") as img_file:
                        item["image_data"] = base64.b64encode(img_file.read()).decode('utf-8')

                results_list.append(item)

        # Phương thức truy vấn: shape (3D)
        elif query_type == "shape":
            shape_results = client.query_points(
                collection_name="shape_collection",
                query=query_vector,
                limit=k,
                with_payload=True
            )

            # Chuyển đổi kết quả thành format chuẩn
            for result in shape_results.points:
                item = {
                    "uuid": result.payload.get("uuid"),
                    "score": result.score,
                    "image_path": result.payload.get("image_path"),
                    "origin_path": result.payload.get("origin_path"),
                    "image_data": None
                }

                # Load ảnh gốc nếu có
                origin_path = result.payload.get("origin_path")
                if origin_path and os.path.exists(origin_path):
                    with open(origin_path, "rb") as img_file:
                        item["image_data"] = base64.b64encode(img_file.read()).decode('utf-8')
                # Nếu không có origin_path, thử load từ path
                elif item["image_path"] and os.path.exists(item["image_path"]):
                    with open(item["image_path"], "rb") as img_file:
                        item["image_data"] = base64.b64encode(img_file.read()).decode('utf-8')

                results_list.append(item)

        # Phương thức truy vấn: combined (kết hợp)
        else:  # query_type == "combined"
            # Truy vấn image_collection
            image_results = client.search(
                collection_name="image_collection",
                query_vector=query_vector,
                limit=k * 3,  # Lấy nhiều hơn để đảm bảo đủ k unique UUID
                with_payload=True
            )

            # Lấy danh sách UUID duy nhất từ kết quả image
            unique_uuids = {}
            for result in image_results:
                uuid = result.payload.get("uuid")
                if uuid and uuid not in unique_uuids and len(unique_uuids) < k:
                    unique_uuids[uuid] = {
                        "uuid": uuid,
                        "image_score": result.score,
                        "shape_score": None,
                        "combined_score": None,
                        "image_path": result.payload.get("image_path"),
                        "origin_path": result.payload.get("origin_path"),
                        "image_data": None
                    }

            # Truy vấn shape_collection cho từng UUID duy nhất
            for uuid in unique_uuids.keys():
                shape_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="uuid",
                            match=models.MatchValue(value=uuid)
                        )
                    ]
                )

                shape_results = client.query_points(
                    collection_name="shape_collection",
                    query=query_vector,
                    limit=1,  # Chỉ cần 1 kết quả cho mỗi UUID
                    query_filter=shape_filter,
                    with_payload=True
                )

                if shape_results:
                    unique_uuids[uuid]["shape_score"] = shape_results.points[0].score
                    # Cập nhật origin_path nếu không có trong image
                    if not unique_uuids[uuid]["origin_path"]:
                        unique_uuids[uuid]["origin_path"] = shape_results[0].payload.get("origin_path")


            # Tính toán combined_score với trọng số và load ảnh
            for uuid, item in unique_uuids.items():
                # Tính combined_score dựa trên trọng số
                print("Helo again")
                if item["image_score"] is not None and item["shape_score"] is not None:
                    # Weighted average với trọng số có thể điều chỉnh
                    item["combined_score"] = (item["image_score"] * image_weight +
                                              item["shape_score"] * shape_weight) / (image_weight + shape_weight)
                elif item["image_score"] is not None:
                    item["combined_score"] = item["image_score"]
                elif item["shape_score"] is not None:
                    item["combined_score"] = item["shape_score"]
                else:
                    item["combined_score"] = 0
                print("Helo again 2")

                # Load ảnh gốc nếu có
                try:
                    origin_path = item["origin_path"]
                    if origin_path and os.path.exists(origin_path):
                        with open(origin_path, "rb") as img_file:
                            item["image_data"] = base64.b64encode(img_file.read()).decode('utf-8')
                    # Nếu không có origin_path, thử load từ path
                    elif item["image_path"] and os.path.exists(item["image_path"]):
                        with open(item["image_path"], "rb") as img_file:
                            item["image_data"] = base64.b64encode(img_file.read()).decode('utf-8')
                except Exception as e:
                    print(e)
                    raise ValueError(e)

            # Chuyển dict thành list
            results_list = list(unique_uuids.values())

        # Sắp xếp kết quả theo score
        if query_type == "combined":
            results_list.sort(key=lambda x: x["combined_score"] if x["combined_score"] is not None else 0, reverse=True)
        else:
            results_list.sort(key=lambda x: x["score"] if x["score"] is not None else 0, reverse=True)

        return jsonify({
            "status": "success",
            "query_text": query_text,
            "query_type": query_type,
            "results": results_list
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    # Tạo thư mục static nếu chưa tồn tại
    if not os.path.exists('static'):
        os.makedirs('static')

    # Kiểm tra xem file index.html đã tồn tại chưa
    if not os.path.exists(os.path.join('static', 'index.html')):
        # Nếu chưa có, tạo file trống để tránh lỗi
        with open(os.path.join('static', 'index.html'), 'w', encoding='utf-8') as f:
            f.write(
                '<!DOCTYPE html><html><head><title>Placeholder</title></head><body><p>Please place your HTML file here.</p></body></html>')

    print("\n" + "=" * 80)
    print("Qdrant Vector Database Visualization đã sẵn sàng!")
    print("Truy cập ứng dụng tại: http://localhost:5000")
    print("=" * 80 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
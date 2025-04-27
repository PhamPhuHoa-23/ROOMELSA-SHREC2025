from flask import Flask, request, jsonify, send_from_directory, make_response
from flask_cors import CORS
import os
import json
from io import StringIO
import csv
from qdrant_client import QdrantClient
from qdrant_client.http import models
import base64
from score.calculate_score import chamfer_to_score, advanced_scoring

app = Flask(__name__, static_folder='static')
CORS(app)  # Cho phép cross-origin requests

# Kết nối đến Qdrant server
client = QdrantClient(host="localhost", port=6333)

# Đọc file distance_mapping.json
distance_mapping = {}
try:
    distance_mapping_file = os.path.join(os.path.dirname(__file__), 'distance_mapping.json')
    if os.path.exists(distance_mapping_file):
        with open(distance_mapping_file, 'r') as f:
            distance_mapping = json.load(f)
        print(f"Loaded distance mapping with {len(distance_mapping)} query entries")
    else:
        print("Warning: distance_mapping.json not found")
except Exception as e:
    print(f"Error loading distance mapping: {e}")


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
                "query": point.payload.get("query", ""),
                "uuid": point.payload.get("uuid", "")
            })

        return jsonify({"status": "success", "points": points})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/query_collections', methods=['POST'])
def query_collections():
    """Truy vấn các collections dựa trên vector đã chọn và phương thức truy vấn"""
    try:
        data = request.json
        vector_id = data.get("vector_id")
        query_type = data.get("query_type", "combined")  # combined, 2d3d, image, shape
        k = data.get("k", 100)  # Số lượng kết quả trả về (mặc định 100 cho unique UUID)
        image_weight = data.get("image_weight", 0.5)  # Trọng số cho image_score
        shape_weight = data.get("shape_weight", 0.6)  # Trọng số cho shape_score
        chamfer_weight = data.get("chamfer_weight", 0.4)  # Trọng số cho chamfer distance

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
        query_uuid = vector_result[0].payload.get("uuid", "")  # Lấy UUID của query nếu có

        # Chuẩn bị dict để lưu distances từ distance_mapping
        distances = {}
        if query_uuid and query_uuid in distance_mapping:
            distances = distance_mapping[query_uuid]

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
                uuid = result.payload.get("uuid")
                item = {
                    "uuid": uuid,
                    "score": result.score,
                    "image_path": result.payload.get("image_path", result.payload.get("path")),
                    "origin_path": result.payload.get("origin_path"),
                    "image_data": None,
                    "distance": distances.get(uuid, None)  # Thêm distance từ mapping
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

        # Phương thức truy vấn: shape (3D)
        elif query_type == "shape":
            shape_results = client.query_points(
                collection_name="shape_collection_tune2",
                query=query_vector,
                limit=k,
                with_payload=True
            )

            # Chuyển đổi kết quả thành format chuẩn
            for result in shape_results.points:
                uuid = result.payload.get("uuid")
                item = {
                    "uuid": uuid,
                    "score": result.score,
                    "image_path": result.payload.get("model_path", result.payload.get("path")),
                    "origin_path": result.payload.get("origin_path"),
                    "image_data": None,
                    "distance": distances.get(uuid, None)  # Thêm distance từ mapping
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

        # Phương thức truy vấn: 2d3d (kết hợp 2D và 3D không có Geometric)
        elif query_type == "2d3d":
            # Truy vấn image_collection
            image_results = client.search(
                collection_name="image_collection",
                query_vector=query_vector,
                limit=k * 2,  # Lấy nhiều hơn để đảm bảo đủ k unique UUID
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
                        "image_data": None,
                        "distance": distances.get(uuid, None)  # Thêm distance từ mapping
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
                    collection_name="shape_collection_tune2",
                    query=query_vector,
                    limit=1,  # Chỉ cần 1 kết quả cho mỗi UUID
                    query_filter=shape_filter,
                    with_payload=True
                )

                if shape_results.points:
                    unique_uuids[uuid]["shape_score"] = shape_results.points[0].score
                    # Cập nhật origin_path nếu không có trong image
                    if not unique_uuids[uuid]["origin_path"]:
                        unique_uuids[uuid]["origin_path"] = shape_results.points[0].payload.get("origin_path")

            # Tính toán combined_score chỉ với trọng số 2D và 3D (không có Geometric)
            for uuid, item in unique_uuids.items():
                # Tính combined_score dựa trên trọng số
                if item["image_score"] is not None and item["shape_score"] is not None:
                    # Sử dụng weighted average đơn giản cho 2D và 3D
                    total_weight = image_weight + shape_weight
                    if total_weight > 0:
                        item["combined_score"] = (
                            (item["image_score"] * image_weight) +
                            (item["shape_score"] * shape_weight)
                        ) / total_weight
                    else:
                        item["combined_score"] = (item["image_score"] + item["shape_score"]) / 2
                elif item["image_score"] is not None:
                    item["combined_score"] = item["image_score"]
                elif item["shape_score"] is not None:
                    item["combined_score"] = item["shape_score"]
                else:
                    item["combined_score"] = 0

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
                    print(f"Error loading image: {e}")
                    # Continue without the image instead of raising an error

            # Chuyển dict thành list
            results_list = list(unique_uuids.values())

        # Phương thức truy vấn: combined (kết hợp 2D + 3D + Geometric)
        else:  # query_type == "combined"
            # Truy vấn image_collection
            image_results = client.search(
                collection_name="image_collection",
                query_vector=query_vector,
                limit=k * 2,  # Lấy nhiều hơn để đảm bảo đủ k unique UUID
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
                        "image_data": None,
                        "distance": distances.get(uuid, None)  # Thêm distance từ mapping
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
                    collection_name="shape_collection_tune2",
                    query=query_vector,
                    limit=1,  # Chỉ cần 1 kết quả cho mỗi UUID
                    query_filter=shape_filter,
                    with_payload=True
                )

                if shape_results.points:
                    unique_uuids[uuid]["shape_score"] = shape_results.points[0].score
                    # Cập nhật origin_path nếu không có trong image
                    if not unique_uuids[uuid]["origin_path"]:
                        unique_uuids[uuid]["origin_path"] = shape_results.points[0].payload.get("origin_path")

            # Tính toán combined_score với trọng số và load ảnh
            for uuid, item in unique_uuids.items():
                # Tính combined_score dựa trên trọng số
                if item["image_score"] is not None and item["shape_score"] is not None:
                    # Weighted average với trọng số có thể điều chỉnh
                    item["combined_score"] = advanced_scoring(
                        img_simi=item["image_score"],
                        shape_simi=item["shape_score"],
                        chamfer_dist=item["distance"],
                        w_img=image_weight,
                        w_shape=shape_weight,
                        w_chamfer=chamfer_weight
                    )
                elif item["image_score"] is not None:
                    item["combined_score"] = item["image_score"]
                elif item["shape_score"] is not None:
                    item["combined_score"] = item["shape_score"]
                else:
                    item["combined_score"] = 0

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
                    print(f"Error loading image: {e}")
                    # Continue without the image instead of raising an error

            # Chuyển dict thành list
            results_list = list(unique_uuids.values())

        # Sắp xếp kết quả theo score
        if query_type in ["combined", "2d3d"]:
            results_list.sort(key=lambda x: x["combined_score"] if x["combined_score"] is not None else 0, reverse=True)
        else:
            results_list.sort(key=lambda x: x["score"] if x["score"] is not None else 0, reverse=True)

        # Store the last query results globally for CSV export
        global last_query_results
        last_query_results = {
            "query_text": query_text,
            "query_uuid": query_uuid,
            "query_type": query_type,
            "results": results_list
        }

        return jsonify({
            "status": "success",
            "query_text": query_text,
            "query_uuid": query_uuid,
            "query_type": query_type,
            "has_distance_mapping": bool(distances),
            "results": results_list
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/export_csv', methods=['GET'])
def export_csv():
    """Xuất kết quả truy vấn dưới dạng CSV"""
    try:
        global last_query_results
        query_uuid = last_query_results["query_uuid"]
        results = last_query_results["results"]

        if not query_uuid or not results:
            return jsonify({"status": "error", "message": "No query results available to export"}), 400

        # Tạo CSV string
        csv_output = StringIO()
        csv_writer = csv.writer(csv_output)

        # Header row
        header = ["query_uuid"]
        for i in range(min(10, len(results))):
            header.append(f"object_uuid{i + 1}")
        csv_writer.writerow(header)

        # Data row
        row = [query_uuid]
        for result in results[:10]:  # Lấy tối đa 10 kết quả
            row.append(result["uuid"])
        csv_writer.writerow(row)

        # Tạo response với file CSV
        response = make_response(csv_output.getvalue())
        response.headers["Content-Disposition"] = f"attachment; filename=query_{query_uuid}.csv"
        response.headers["Content-Type"] = "text/csv"

        return response

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/export_all_csv', methods=['GET'])
def export_all_csv():
    """Xuất toàn bộ kết quả tìm kiếm cho tất cả query UUID, sắp xếp theo combined_score"""
    try:
        # Lấy tất cả các điểm từ text_collection
        results = client.scroll(
            collection_name="text_collection",
            limit=1000,
            with_payload=True,
            with_vectors=True
        )

        if not results[0]:
            return jsonify({"status": "error", "message": "No query vectors found"}), 400

        # Tạo CSV string
        csv_output = StringIO()
        csv_writer = csv.writer(csv_output)

        # Header row
        header = ["query_uuid"]
        for i in range(10):  # Tối đa 10 kết quả cho mỗi truy vấn
            header.append(f"object_uuid{i + 1}")
        csv_writer.writerow(header)

        # Thông số truy vấn mặc định
        query_type = "combined"
        image_weight = 0.5
        shape_weight = 0.7
        chamfer_weight = 0.0
        k = 100

        # Đếm tổng số query points để hiển thị tiến trình
        total_points = len(results[0])
        print(f"Total query points to process: {total_points}")

        # Xử lý từng query vector
        for i, point in enumerate(results[0]):
            try:
                print(f"Processing query {i + 1}/{total_points}: {point.payload.get('query', 'No query text')}")

                query_vector = point.vector
                query_uuid = point.payload.get("uuid", "")

                if not query_uuid:
                    print(f"Skipping query without UUID: ID={point.id}")
                    continue

                # Chuẩn bị dict để lưu distances từ distance_mapping
                distances = {}
                if query_uuid in distance_mapping:
                    distances = distance_mapping[query_uuid]

                # Truy vấn image_collection
                image_results = client.search(
                    collection_name="image_collection",
                    query_vector=query_vector,
                    limit=k * 2,  # Lấy nhiều hơn để đảm bảo đủ k unique UUID
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
                            "distance": distances.get(uuid, None)
                        }

                # Truy vấn shape_collection cho từng UUID duy nhất
                for uuid in list(unique_uuids.keys()):
                    shape_filter = models.Filter(
                        must=[
                            models.FieldCondition(
                                key="uuid",
                                match=models.MatchValue(value=uuid)
                            )
                        ]
                    )

                    shape_results = client.query_points(
                        collection_name="shape_collection_tune2",
                        query=query_vector,
                        limit=1,  # Chỉ cần 1 kết quả cho mỗi UUID
                        query_filter=shape_filter,
                        with_payload=True
                    )

                    if shape_results.points:
                        unique_uuids[uuid]["shape_score"] = shape_results.points[0].score

                # Tính toán combined_score với trọng số
                for uuid, item in unique_uuids.items():
                    if item["image_score"] is not None and item["shape_score"] is not None:
                        item["combined_score"] = advanced_scoring(
                            img_simi=item["image_score"],
                            shape_simi=item["shape_score"],
                            chamfer_dist=item["distance"],
                            w_img=image_weight,
                            w_shape=shape_weight,
                            w_chamfer=chamfer_weight,
                        )
                    elif item["image_score"] is not None:
                        item["combined_score"] = item["image_score"]
                    elif item["shape_score"] is not None:
                        item["combined_score"] = item["shape_score"]
                    else:
                        item["combined_score"] = 0

                # Chuyển dict thành list và sắp xếp theo combined_score
                results_list = list(unique_uuids.values())
                results_list.sort(key=lambda x: x["combined_score"] if x["combined_score"] is not None else 0,
                                  reverse=True)

                # Tạo row cho CSV với query_uuid và top 10 object_uuids
                row = [query_uuid]
                for result in results_list[:10]:  # Lấy tối đa 10 kết quả
                    row.append(result["uuid"])

                # Đảm bảo đủ 11 cột (1 cho query_uuid + 10 cho object_uuids)
                while len(row) < 11:
                    row.append("")

                csv_writer.writerow(row)

                # Log tiến trình
                if (i + 1) % 10 == 0 or (i + 1) == total_points:
                    print(f"Progress: {i + 1}/{total_points} queries processed ({((i + 1) / total_points * 100):.2f}%)")

            except Exception as e:
                print(f"Error processing query {point.id}: {e}")
                # Continue with next query instead of failing the entire export

        # Tạo response với file CSV
        response = make_response(csv_output.getvalue())
        response.headers["Content-Disposition"] = "attachment; filename=all_query_results_by_combined_score.csv"
        response.headers["Content-Type"] = "text/csv"

        return response

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# Global variable to store last query results
last_query_results = {"query_uuid": "", "results": []}

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
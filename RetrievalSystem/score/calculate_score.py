import numpy as np
import open3d as o3d

def sigmoid_normalize(x, mid=0.5, steepness=10):
    """
    Chuẩn hóa giá trị x bằng cách sử dụng hàm sigmoid
    với điểm trung tâm và độ dốc có thể điều chỉnh
    """
    return 1 / (1 + np.exp(-steepness * (x - mid)))

def chamfer_to_score(chamfer_dist, target_min=0.1, target_max=0.3, alpha=10.0, center_dist=0.17):
    """
    Chuyển đổi chamfer distance thành score trong khoảng mong muốn sử dụng hàm mũ
    để tăng khả năng phân biệt giữa các khoảng cách gần nhau

    Parameters:
    -----------
    chamfer_dist : float
        Giá trị chamfer distance
    target_min : float
        Giá trị nhỏ nhất mong muốn của score (mặc định: 0.1)
    target_max : float
        Giá trị lớn nhất mong muốn của score (mặc định: 0.3)
    alpha : float
        Hệ số điều chỉnh độ dốc của hàm mũ (alpha càng lớn, độ dốc càng lớn)
    center_dist : float
        Giá trị trung tâm để so sánh (khoảng cách này sẽ cho score trung bình)

    Returns:
    --------
    float
        Score trong khoảng [target_min, target_max]
    """
    # Tính toán score bằng hàm mũ exp(-alpha * (dist - center_dist))
    # Khi dist < center_dist, score sẽ cao hơn
    # Khi dist > center_dist, score sẽ thấp hơn
    # Độ chênh lệch tăng theo hệ số alpha
    exp_score = np.exp(-alpha * (chamfer_dist - center_dist))

    # Scale score vào khoảng [0, 1] cho các dự đoán
    # Điều này dựa vào việc kiểm thử trên dữ liệu thực tế
    # Bạn có thể điều chỉnh tham số này dựa trên phân phối thực tế

    # Với center_dist = 0.17, giả sử:
    # dist = 0.16 -> exp_score cao hơn 1
    # dist = 0.18 -> exp_score thấp hơn 1

    # Hàm sigmoid để đảm bảo score nằm trong khoảng [0, 1]
    sigmoid = 1.0 / (1.0 + np.exp(-exp_score + 1.0))

    # Map vào khoảng [target_min, target_max]
    score = target_min + sigmoid * (target_max - target_min)

    return score


def advanced_scoring(img_simi, shape_simi, chamfer_dist,
                     w_img=0.45, w_shape=0.3, w_chamfer=0.25,
                     alpha=10.0, center_dist=0.18,
                     formula_type="adaptive_weight"):
    """
    Tính toán điểm số với các công thức robust có tương tác giữa các thành phần

    Parameters:
    -----------
    img_simi : float
        Độ tương đồng hình ảnh (thường trong khoảng 0.3-0.45)
    shape_simi : float
        Độ tương đồng hình dạng (thường trong khoảng 0.1-0.2)
    chamfer_dist : float
        Chamfer distance
    w_img, w_shape, w_chamfer : float
        Trọng số cơ bản cho mỗi thành phần
    alpha : float
        Hệ số điều chỉnh độ dốc của hàm mũ cho chamfer_dist
    center_dist : float
        Giá trị chamfer distance trung tâm
    formula_type : str
        Loại công thức sử dụng ("adaptive_weight", "geometric_mean",
                              "harmonic_blend", "confidence_boost")

    Returns:
    --------
    float
        Điểm số tổng hợp
    """
    # Chuyển đổi chamfer_dist thành chamfer_score cơ bản (trong khoảng ~0-1)

    # Chuẩn hóa chamfer_score về khoảng 0.1-0.3 để phù hợp với yêu cầu ban đầu
    chamfer_score = chamfer_to_score(chamfer_dist, center_dist=center_dist)

    # Các công thức khác nhau
    if formula_type == "adaptive_weight":
        # Trọng số của chamfer_score tăng khi cả img_simi và shape_simi cao
        # Ngược lại, giảm trọng số khi chúng thấp
        visual_confidence = (img_simi + shape_simi) / 2  # Độ tin cậy trung bình

        # Điều chỉnh trọng số chamfer dựa vào độ tin cậy visual
        adaptive_w_chamfer = w_chamfer * (1 + visual_confidence)
        # Tái chuẩn hóa trọng số
        sum_weights = w_img + w_shape + adaptive_w_chamfer
        norm_w_img = w_img / sum_weights
        norm_w_shape = w_shape / sum_weights
        norm_w_chamfer = adaptive_w_chamfer / sum_weights

        # Tính điểm số cuối cùng
        score = norm_w_img * img_simi + norm_w_shape * shape_simi + norm_w_chamfer * chamfer_score

    elif formula_type == "geometric_mean":
        # Sử dụng trung bình nhân có trọng số
        # Khi một thành phần bị thấp, điểm tổng thể sẽ bị ảnh hưởng nhiều hơn
        # Tính lũy thừa cho từng thành phần theo trọng số
        score = (img_simi ** w_img) * (shape_simi ** w_shape) * (chamfer_score ** w_chamfer)

    elif formula_type == "harmonic_blend":
        # Dùng hàm sigmoid để tạo điểm threshold
        sigmoid = lambda x: 1 / (1 + np.exp(-10 * (x - 0.6)))

        # Kết hợp theo kiểu harmonic mean khi visual score cao
        # Kết hợp theo kiểu arithmetic mean khi visual score thấp
        visual_score = w_img * img_simi + w_shape * shape_simi
        visual_weight = sigmoid(visual_score)

        # Blend giữa cộng tuyến tính và nhân
        score = visual_weight * (img_simi * shape_simi * chamfer_score) + \
                (1 - visual_weight) * (w_img * img_simi + w_shape * shape_simi + w_chamfer * chamfer_score)

    elif formula_type == "confidence_boost":
        # Tính điểm tin cậy cho mỗi thành phần
        # Ý tưởng: img_simi và shape_simi cao -> boosting chamfer_score
        confidence_img = sigmoid_normalize(img_simi, mid=0.35)  # 0.35 là giá trị trung bình
        confidence_shape = sigmoid_normalize(shape_simi, mid=0.15)  # 0.15 là giá trị trung bình

        # Kết hợp độ tin cậy
        confidence = (confidence_img + confidence_shape) / 2

        # Tăng cường chamfer_score khi confidence cao
        boosted_chamfer = chamfer_score * (1 + confidence)

        # Giới hạn giá trị boosted_chamfer
        boosted_chamfer = min(boosted_chamfer, 0.3)

        # Tính điểm số cuối cùng
        score = w_img * img_simi + w_shape * shape_simi + w_chamfer * boosted_chamfer

    else:
        # Công thức cộng tuyến tính đơn giản mặc định
        score = w_img * img_simi + w_shape * shape_simi + w_chamfer * chamfer_score

    return score

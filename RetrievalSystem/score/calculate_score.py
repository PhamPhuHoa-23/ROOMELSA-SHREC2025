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
                     alpha=10.0, center_dist=0.18):
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


    Returns:
    --------
    float
        Điểm số tổng hợp
    """
    chamfer_score = chamfer_to_score(chamfer_dist, alpha=alpha, center_dist=center_dist)

    # Dùng hàm sigmoid để tạo điểm threshold
    sigmoid = lambda x: 1 / (1 + np.exp(-10 * (x - 0.15)))

    # Kết hợp theo kiểu harmonic mean khi visual score cao
    # Kết hợp theo kiểu arithmetic mean khi visual score thấp
    visual_score = w_img * img_simi + w_shape * shape_simi
    visual_weight = sigmoid(visual_score)

    score = visual_weight * ((img_simi + shape_simi)) + \
            (1 - visual_weight) * (w_img * img_simi + w_shape * shape_simi + w_chamfer * chamfer_score)

    return score

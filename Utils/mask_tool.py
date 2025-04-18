import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk


class MaskingApp:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.subdirs = self.get_valid_subdirs(root_dir)
        self.current_index = 0

        if not self.subdirs:
            print("Không tìm thấy thư mục con hợp lệ!")
            return

        # Thiết lập GUI
        self.window = tk.Tk()
        self.window.title("Masking Tool")
        self.window.geometry("1200x800")

        # Frame chính
        self.main_frame = tk.Frame(self.window)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Frame hiển thị thông tin
        self.info_frame = tk.Frame(self.main_frame)
        self.info_frame.pack(fill=tk.X)

        # Hiển thị đường dẫn thư mục hiện tại
        self.path_label = tk.Label(self.info_frame, text="", font=("Arial", 10))
        self.path_label.pack(anchor=tk.W)

        # Hiển thị nội dung query.txt
        self.query_label = tk.Label(self.info_frame, text="", font=("Arial", 12, "bold"), wraplength=1150,
                                    justify=tk.LEFT)
        self.query_label.pack(anchor=tk.W, pady=(5, 10))

        # Canvas để hiển thị ảnh và vẽ mask
        self.canvas_frame = tk.Frame(self.main_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Frame chứa công cụ vẽ
        self.tool_frame = tk.Frame(self.main_frame)
        self.tool_frame.pack(fill=tk.X, pady=(10, 5))

        # Dropdown chọn công cụ vẽ
        tk.Label(self.tool_frame, text="Công cụ:").pack(side=tk.LEFT, padx=(0, 5))
        self.tool_var = tk.StringVar(value="bbox")
        tools = [
            ("Bounding Box", "bbox"),
            ("Vẽ tự do", "freehand"),
            ("Đa giác", "polygon"),
            ("Xóa vùng", "erase")
        ]
        self.tool_dropdown = ttk.Combobox(self.tool_frame, textvariable=self.tool_var, values=[t[0] for t in tools],
                                          state="readonly", width=15)
        self.tool_dropdown.pack(side=tk.LEFT, padx=5)
        self.tool_dropdown.bind("<<ComboboxSelected>>", self.on_tool_change)

        # Kích thước bút vẽ
        tk.Label(self.tool_frame, text="Kích thước bút:").pack(side=tk.LEFT, padx=(20, 5))
        self.brush_size_var = tk.IntVar(value=5)
        self.brush_size_slider = tk.Scale(self.tool_frame, from_=1, to=20, orient=tk.HORIZONTAL,
                                          variable=self.brush_size_var, length=100)
        self.brush_size_slider.pack(side=tk.LEFT)

        # Sai số màu
        tk.Label(self.tool_frame, text="Sai số màu:").pack(side=tk.LEFT, padx=(20, 5))
        self.color_tolerance_var = tk.IntVar(value=15)
        self.color_tolerance_slider = tk.Scale(self.tool_frame, from_=1, to=50, orient=tk.HORIZONTAL,
                                               variable=self.color_tolerance_var, length=100)
        self.color_tolerance_slider.pack(side=tk.LEFT)

        # Frame hiển thị danh sách các vùng đã vẽ
        self.regions_frame = tk.Frame(self.main_frame)
        self.regions_frame.pack(fill=tk.X, pady=5)

        tk.Label(self.regions_frame, text="Các vùng đã vẽ:").pack(side=tk.LEFT, padx=(0, 5))
        self.regions_listbox = tk.Listbox(self.regions_frame, height=3, width=50)
        self.regions_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        self.remove_region_btn = tk.Button(self.regions_frame, text="Xóa vùng", command=self.remove_selected_region)
        self.remove_region_btn.pack(side=tk.LEFT, padx=5)

        # Frame chứa các nút điều khiển
        self.control_frame = tk.Frame(self.main_frame)
        self.control_frame.pack(fill=tk.X, pady=(5, 10))

        # Hiển thị trạng thái
        self.status_label = tk.Label(self.control_frame, text="Trạng thái: Sẵn sàng", font=("Arial", 10))
        self.status_label.pack(side=tk.LEFT)

        # Các nút điều khiển
        self.prev_btn = tk.Button(self.control_frame, text="Trước", command=self.prev_image)
        self.prev_btn.pack(side=tk.RIGHT, padx=5)

        self.next_btn = tk.Button(self.control_frame, text="Tiếp theo", command=self.next_image)
        self.next_btn.pack(side=tk.RIGHT, padx=5)

        self.save_btn = tk.Button(self.control_frame, text="Lưu mask", command=self.save_mask)
        self.save_btn.pack(side=tk.RIGHT, padx=5)

        self.clear_btn = tk.Button(self.control_frame, text="Xóa tất cả", command=self.clear_all)
        self.clear_btn.pack(side=tk.RIGHT, padx=5)

        # Biến để lưu trữ thông tin
        self.current_image = None
        self.display_image = None
        self.image_tk = None
        self.scale_x = 1.0
        self.scale_y = 1.0

        # Các vùng đã vẽ
        self.regions = []
        self.current_region = None
        self.drawing = False

        # Biến cho công cụ polygon
        self.polygon_points = []
        self.polygon_lines = []
        self.temp_line = None

        # Liên kết các sự kiện chuột
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        # Thêm sự kiện Right-click để hoàn thành polygon
        self.canvas.bind("<ButtonPress-3>", self.on_right_click)

        # Hiển thị ảnh đầu tiên
        self.load_current_subdir()

        self.window.mainloop()

    def get_valid_subdirs(self, root_dir):
        """Tìm các thư mục con chứa cả masked.png và query.txt"""
        valid_subdirs = []
        for subdir in sorted(os.listdir(root_dir)):
            subdir_path = os.path.join(root_dir, subdir)
            if (os.path.isdir(subdir_path) and
                    os.path.exists(os.path.join(subdir_path, "masked.png")) and
                    os.path.exists(os.path.join(subdir_path, "query.txt"))):
                valid_subdirs.append(subdir_path)
        return valid_subdirs

    def prev_image(self):
        """Chuyển đến ảnh trước đó"""
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_subdir()
        else:
            messagebox.showinfo("Thông báo", "Đã ở ảnh đầu tiên!")

    def next_image(self):
        """Chuyển đến ảnh tiếp theo"""
        if self.current_index < len(self.subdirs) - 1:
            self.current_index += 1
            self.load_current_subdir()
        else:
            messagebox.showinfo("Thông báo", "Đã đến ảnh cuối cùng!")

    def load_current_subdir(self):
        """Tải ảnh và text từ thư mục hiện tại"""
        current_path = self.subdirs[self.current_index]
        img_path = os.path.join(current_path, "masked.png")
        txt_path = os.path.join(current_path, "query.txt")

        # Hiển thị đường dẫn
        self.path_label.config(text=f"Thư mục: {current_path}")

        # Đọc và hiển thị nội dung query.txt
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                query_text = f.read().strip()
            self.query_label.config(text=f"Query: {query_text}")
        except Exception as e:
            self.query_label.config(text=f"Lỗi khi đọc query.txt: {str(e)}")

        # Đọc và hiển thị ảnh
        try:
            self.current_image = cv2.imread(img_path)
            if self.current_image is None:
                raise Exception("Không thể đọc ảnh")

            # Chuyển từ BGR sang RGB để hiển thị
            img_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            self.display_image = img_rgb.copy()

            # Điều chỉnh kích thước để hiển thị
            h, w = img_rgb.shape[:2]

            # Đợi canvas được render
            self.window.update()
            canvas_w = self.canvas.winfo_width()
            canvas_h = self.canvas.winfo_height()

            # Tính toán tỷ lệ
            self.scale_x = w / canvas_w if canvas_w > 0 else 1.0
            self.scale_y = h / canvas_h if canvas_h > 0 else 1.0
            scale = max(self.scale_x, self.scale_y)

            if scale > 1.0:
                new_w, new_h = int(w / scale), int(h / scale)
                img_resized = cv2.resize(img_rgb, (new_w, new_h))
                self.scale_x = w / new_w
                self.scale_y = h / new_h
            else:
                img_resized = img_rgb
                self.scale_x = 1.0
                self.scale_y = 1.0

            # Chuyển thành ImageTk để hiển thị
            self.image_tk = ImageTk.PhotoImage(image=Image.fromarray(img_resized))

            # Xóa canvas và hiển thị ảnh mới
            self.canvas.delete("all")
            self.canvas.config(width=img_resized.shape[1], height=img_resized.shape[0])
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tk)

            # Cập nhật trạng thái
            self.clear_all()
            self.status_label.config(text="Trạng thái: Sẵn sàng")

        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể hiển thị ảnh: {str(e)}")

    def on_tool_change(self, event):
        """Xử lý khi thay đổi công cụ vẽ"""
        tool = self.tool_var.get()

        # Nếu đang vẽ polygon và chuyển công cụ khác, kết thúc polygon hiện tại
        if len(self.polygon_points) > 0 and tool != "Đa giác":
            self.finish_polygon()

        self.status_label.config(text=f"Trạng thái: Đã chọn công cụ {tool}")

    def on_mouse_down(self, event):
        """Sự kiện khi nhấn chuột để bắt đầu vẽ"""
        tool_name = self.tool_dropdown.get()

        if tool_name == "Bounding Box":
            self.drawing = True
            self.current_region = {
                'type': 'bbox',
                'points': [(event.x, event.y)],
                'canvas_items': [],
            }
            x, y = event.x, event.y
            rect_id = self.canvas.create_rectangle(x, y, x + 1, y + 1, outline="red", width=2)
            self.current_region['canvas_items'].append(rect_id)

        elif tool_name == "Vẽ tự do":
            self.drawing = True
            self.current_region = {
                'type': 'freehand',
                'points': [(event.x, event.y)],
                'canvas_items': [],
            }

        elif tool_name == "Đa giác":
            if not self.drawing:
                self.drawing = True
                self.polygon_points = [(event.x, event.y)]
                # Vẽ điểm đầu tiên
                point_id = self.canvas.create_oval(
                    event.x - 3, event.y - 3, event.x + 3, event.y + 3,
                    fill="red", outline="red")
                self.polygon_lines = [point_id]
            else:
                # Thêm điểm mới vào polygon
                self.polygon_points.append((event.x, event.y))

                # Vẽ điểm
                point_id = self.canvas.create_oval(
                    event.x - 3, event.y - 3, event.x + 3, event.y + 3,
                    fill="red", outline="red")

                # Vẽ đường nối từ điểm trước đến điểm hiện tại
                prev_x, prev_y = self.polygon_points[-2]
                line_id = self.canvas.create_line(
                    prev_x, prev_y, event.x, event.y,
                    fill="red", width=2)

                self.polygon_lines.extend([point_id, line_id])

        elif tool_name == "Xóa vùng":
            # Lấy kích thước bút vẽ
            brush_size = self.brush_size_var.get()

            self.drawing = True
            self.current_region = {
                'type': 'erase',
                'points': [(event.x, event.y)],
                'canvas_items': [],
                'size': brush_size
            }

            # Vẽ hình tròn tại vị trí con trỏ
            circle_id = self.canvas.create_oval(
                event.x - brush_size, event.y - brush_size,
                event.x + brush_size, event.y + brush_size,
                outline="white", width=2)
            self.current_region['canvas_items'].append(circle_id)

    def on_mouse_move(self, event):
        """Sự kiện khi di chuyển chuột để vẽ"""
        if not self.drawing:
            return

        tool_name = self.tool_dropdown.get()

        if tool_name == "Bounding Box":
            # Cập nhật rectangle
            start_x, start_y = self.current_region['points'][0]
            if len(self.current_region['canvas_items']) > 0:
                rect_id = self.current_region['canvas_items'][0]
                self.canvas.coords(rect_id, start_x, start_y, event.x, event.y)

        elif tool_name == "Vẽ tự do":
            # Thêm điểm mới
            last_x, last_y = self.current_region['points'][-1]
            line_id = self.canvas.create_line(
                last_x, last_y, event.x, event.y,
                fill="red", width=self.brush_size_var.get())

            self.current_region['points'].append((event.x, event.y))
            self.current_region['canvas_items'].append(line_id)

        elif tool_name == "Đa giác":
            # Cập nhật đường tạm thời
            if len(self.polygon_points) > 0:
                last_x, last_y = self.polygon_points[-1]
                if self.temp_line:
                    self.canvas.delete(self.temp_line)
                self.temp_line = self.canvas.create_line(
                    last_x, last_y, event.x, event.y,
                    fill="red", width=2, dash=(4, 4))

        elif tool_name == "Xóa vùng":
            # Vẽ hình tròn tại vị trí con trỏ
            brush_size = self.current_region['size']
            circle_id = self.canvas.create_oval(
                event.x - brush_size, event.y - brush_size,
                event.x + brush_size, event.y + brush_size,
                outline="white", width=2)

            self.current_region['points'].append((event.x, event.y))
            self.current_region['canvas_items'].append(circle_id)

    def on_mouse_up(self, event):
        """Sự kiện khi thả chuột để hoàn thành vẽ"""
        tool_name = self.tool_dropdown.get()

        if not self.drawing:
            return

        if tool_name == "Bounding Box":
            # Hoàn thành bounding box
            start_x, start_y = self.current_region['points'][0]
            self.current_region['points'] = [
                (min(start_x, event.x), min(start_y, event.y)),
                (max(start_x, event.x), max(start_y, event.y))
            ]

            # Thêm vào danh sách vùng và hiển thị
            region_name = f"Bounding Box {len(self.regions) + 1}"
            self.current_region['name'] = region_name
            self.regions.append(self.current_region)
            self.regions_listbox.insert(tk.END, region_name)

            # Kết thúc vẽ
            self.drawing = False
            self.current_region = None

        elif tool_name == "Vẽ tự do":
            # Hoàn thành vẽ tự do
            region_name = f"Vẽ tự do {len(self.regions) + 1}"
            self.current_region['name'] = region_name
            self.regions.append(self.current_region)
            self.regions_listbox.insert(tk.END, region_name)

            # Kết thúc vẽ
            self.drawing = False
            self.current_region = None

        elif tool_name == "Xóa vùng":
            # Hoàn thành vùng xóa
            region_name = f"Vùng xóa {len(self.regions) + 1}"
            self.current_region['name'] = region_name
            self.regions.append(self.current_region)
            self.regions_listbox.insert(tk.END, region_name)

            # Kết thúc vẽ
            self.drawing = False
            self.current_region = None

        # Đối với công cụ đa giác, không kết thúc khi thả chuột
        # mà phải đợi right-click hoặc nối với điểm đầu tiên

    def on_right_click(self, event):
        """Sự kiện khi nhấn chuột phải để hoàn thành đa giác"""
        tool_name = self.tool_dropdown.get()

        if tool_name == "Đa giác" and self.drawing:
            self.finish_polygon()

    def finish_polygon(self):
        """Hoàn thành vẽ đa giác"""
        if len(self.polygon_points) < 3:
            # Cần ít nhất 3 điểm để tạo một đa giác
            for item in self.polygon_lines:
                self.canvas.delete(item)
            if self.temp_line:
                self.canvas.delete(self.temp_line)
            self.polygon_points = []
            self.polygon_lines = []
            self.temp_line = None
            self.drawing = False
            return

        # Nối điểm cuối với điểm đầu
        last_x, last_y = self.polygon_points[-1]
        first_x, first_y = self.polygon_points[0]
        close_line_id = self.canvas.create_line(
            last_x, last_y, first_x, first_y,
            fill="red", width=2)

        # Xóa đường tạm thời nếu có
        if self.temp_line:
            self.canvas.delete(self.temp_line)
            self.temp_line = None

        # Tạo vùng mới
        region = {
            'type': 'polygon',
            'points': self.polygon_points.copy(),
            'canvas_items': self.polygon_lines + [close_line_id],
            'name': f"Đa giác {len(self.regions) + 1}"
        }

        # Thêm vào danh sách vùng
        self.regions.append(region)
        self.regions_listbox.insert(tk.END, region['name'])

        # Reset các biến
        self.polygon_points = []
        self.polygon_lines = []
        self.drawing = False

    def remove_selected_region(self):
        """Xóa vùng đã chọn khỏi danh sách"""
        selection = self.regions_listbox.curselection()
        if not selection:
            return

        index = selection[0]
        region = self.regions[index]

        # Xóa các item trên canvas
        for item in region['canvas_items']:
            self.canvas.delete(item)

        # Xóa khỏi danh sách
        self.regions.pop(index)
        self.regions_listbox.delete(index)

    def clear_all(self):
        """Xóa tất cả các vùng đã vẽ"""
        # Xóa các item trên canvas
        for region in self.regions:
            for item in region['canvas_items']:
                self.canvas.delete(item)

        # Xóa polygon đang vẽ nếu có
        for item in self.polygon_lines:
            self.canvas.delete(item)
        if self.temp_line:
            self.canvas.delete(self.temp_line)

        # Reset các biến
        self.regions = []
        self.current_region = None
        self.drawing = False
        self.polygon_points = []
        self.polygon_lines = []
        self.temp_line = None

        # Xóa danh sách
        self.regions_listbox.delete(0, tk.END)

    def save_mask(self):
        """Tạo và lưu mask dựa trên các vùng đã vẽ và màu"""
        if not self.regions and not self.current_image is not None:
            messagebox.showwarning("Cảnh báo", "Vui lòng vẽ ít nhất một vùng trước khi lưu mask!")
            return

        try:
            # Lấy kích thước gốc của ảnh
            h, w = self.current_image.shape[:2]

            # Tạo mask ban đầu với tất cả là 0
            mask = np.zeros((h, w), dtype=np.uint8)

            # Chuyển đổi ảnh từ BGR sang RGB để kiểm tra màu
            img_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)

            # Màu cần check (136, 206, 235) trong RGB
            target_color = np.array([136, 206, 235])

            # Lấy sai số màu
            color_tolerance = self.color_tolerance_var.get()

            # Xử lý từng vùng
            for region in self.regions:
                region_type = region['type']
                points = region['points']

                if region_type == 'bbox':
                    # Chuyển đổi tọa độ về ảnh gốc
                    (x1, y1), (x2, y2) = points
                    x1, y1 = int(x1 * self.scale_x), int(y1 * self.scale_y)
                    x2, y2 = int(x2 * self.scale_x), int(y2 * self.scale_y)

                    # Đảm bảo tọa độ nằm trong ảnh
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    # Duyệt qua vùng bbox
                    for y in range(y1, y2):
                        for x in range(x1, x2):
                            # Kiểm tra nếu pixel có màu gần với target_color
                            pixel_color = img_rgb[y, x]
                            color_diff = np.sqrt(np.sum((pixel_color - target_color) ** 2))
                            if color_diff <= color_tolerance:
                                mask[y, x] = 1

                elif region_type == 'freehand':
                    # Tạo mask cho đường vẽ tự do
                    brush_mask = np.zeros((h, w), dtype=np.uint8)

                    # Vẽ các đường vào mask
                    for i in range(len(points) - 1):
                        pt1 = (int(points[i][0] * self.scale_x), int(points[i][1] * self.scale_y))
                        pt2 = (int(points[i + 1][0] * self.scale_x), int(points[i + 1][1] * self.scale_y))
                        brush_size = self.brush_size_var.get()
                        cv2.line(brush_mask, pt1, pt2, 1, thickness=int(brush_size * max(self.scale_x, self.scale_y)))

                    # Lọc theo màu
                    for y in range(h):
                        for x in range(w):
                            if brush_mask[y, x] > 0:
                                pixel_color = img_rgb[y, x]
                                color_diff = np.sqrt(np.sum((pixel_color - target_color) ** 2))
                                if color_diff <= color_tolerance:
                                    mask[y, x] = 1

                elif region_type == 'polygon':
                    # Tạo mask cho đa giác
                    poly_mask = np.zeros((h, w), dtype=np.uint8)

                    # Chuyển đổi tọa độ về ảnh gốc
                    scaled_points = [(int(p[0] * self.scale_x), int(p[1] * self.scale_y)) for p in points]

                    # Vẽ đa giác đặc
                    cv2.fillPoly(poly_mask, [np.array(scaled_points)], 1)

                    # Lọc theo màu
                    for y in range(h):
                        for x in range(w):
                            if poly_mask[y, x] > 0:
                                pixel_color = img_rgb[y, x]
                                color_diff = np.sqrt(np.sum((pixel_color - target_color) ** 2))
                                if color_diff <= color_tolerance:
                                    mask[y, x] = 1

                elif region_type == 'erase':
                    # Tạo mask cho vùng xóa
                    erase_mask = np.zeros((h, w), dtype=np.uint8)

                    # Vẽ các hình tròn vào mask
                    for pt in points:
                        center = (int(pt[0] * self.scale_x), int(pt[1] * self.scale_y))
                        radius = int(region['size'] * max(self.scale_x, self.scale_y))
                        cv2.circle(erase_mask, center, radius, 1, -1)

                    # Xóa các pixel trong vùng xóa
                    mask[erase_mask > 0] = 0

            # Lưu mask
            mask_path = os.path.join(self.subdirs[self.current_index], "mask.npy")
            np.save(mask_path, mask)

            # Tạo preview để kiểm tra
            preview = img_rgb.copy()
            preview[mask > 0] = [0, 255, 0]  # Đánh dấu vùng mask bằng màu xanh lá

            # Chuyển preview về kích thước hiển thị
            h_display, w_display = int(h / max(self.scale_x, self.scale_y)), int(w / max(self.scale_x, self.scale_y))
            preview_display = cv2.resize(preview, (w_display, h_display))

            # Hiển thị preview
            # Hiển thị preview
            preview_pil = Image.fromarray(preview_display)
            preview_tk = ImageTk.PhotoImage(image=preview_pil)

            # Tạo cửa sổ preview
            preview_window = tk.Toplevel(self.window)
            preview_window.title("Preview Mask")

            preview_canvas = tk.Canvas(preview_window, width=w_display, height=h_display)
            preview_canvas.pack()

            # Lưu tham chiếu để tránh garbage collection
            preview_canvas.image = preview_tk
            preview_canvas.create_image(0, 0, anchor=tk.NW, image=preview_tk)

            # Thêm nút xác nhận
            confirm_frame = tk.Frame(preview_window)
            confirm_frame.pack(pady=10)

            confirm_btn = tk.Button(confirm_frame, text="Xác nhận", command=lambda: self.confirm_mask(preview_window))
            confirm_btn.pack(side=tk.LEFT, padx=5)

            cancel_btn = tk.Button(confirm_frame, text="Hủy", command=preview_window.destroy)
            cancel_btn.pack(side=tk.LEFT, padx=5)

            messagebox.showinfo("Thành công", f"Đã lưu mask vào {mask_path}")

        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể lưu mask: {str(e)}")

        def confirm_mask(self, preview_window):
            """Xác nhận mask và chuyển đến ảnh tiếp theo"""
            preview_window.destroy()
            self.next_image()

        def next_image(self):
            """Chuyển đến ảnh tiếp theo"""
            if self.current_index < len(self.subdirs) - 1:
                self.current_index += 1
                self.load_current_subdir()
            else:
                messagebox.showinfo("Thông báo", "Đã đến ảnh cuối cùng!")

        def prev_image(self):
            """Chuyển đến ảnh trước đó"""
            if self.current_index > 0:
                self.current_index -= 1
                self.load_current_subdir()
            else:
                messagebox.showinfo("Thông báo", "Đã ở ảnh đầu tiên!")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        root_directory = sys.argv[1]
    else:
        root_directory = input("Nhập đường dẫn đến thư mục chứa các thư mục con: ")

    app = MaskingApp(root_directory)
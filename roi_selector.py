import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageOps # Import ImageOps for potential future use (like auto-orient)
import os
import json
import pyperclip

class ROISelectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ROI 选择器 - 左键 (绿色框/ROI1), 右键 (红色框/ROI2)")
        self.root.geometry("1100x750")

        self.image_label = None
        self.canvas_image_id = None
        self.original_image = None
        self.tk_image = None
        self.image_path = None
        self.folder = None

        self.roi1_coords = None
        self.roi2_coords = None

        self.start_x = self.start_y = 0
        self.current_rect_id = None
        self.roi1_rect_id = None
        self.roi2_rect_id = None

        self.display_scale = 1.0
        self.img_display_width = 0
        self.img_display_height = 0
        self.img_offset_x = 0
        self.img_offset_y = 0

        left_frame = tk.Frame(root, width=250)
        left_frame.pack(side="left", fill="y", padx=5, pady=5)
        left_frame.pack_propagate(False)

        load_btn = tk.Button(left_frame, text="选择图片文件夹", command=self.load_directory)
        load_btn.pack(pady=5, fill="x")

        self.tree = ttk.Treeview(left_frame, show="tree")
        self.tree_scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=self.tree_scrollbar.set)
        self.tree.pack(side="left", fill="both", expand=True)
        self.tree_scrollbar.pack(side="right", fill="y")
        self.tree.bind('<<TreeviewSelect>>', self.on_tree_select)


        right_frame = tk.Frame(root)
        right_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)


        self.coord_frame = tk.Frame(right_frame)
        self.coord_frame.pack(fill="x", pady=(0, 5))
        self._create_coordinate_display(self.coord_frame)

        # Middle Right: Canvas
        self.canvas = tk.Canvas(right_frame, bg='#CCCCCC', relief="sunken", borderwidth=1)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self.on_mouse_down_roi1)
        self.canvas.bind("<Button-3>", self.on_mouse_down_roi2)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move_roi1)
        self.canvas.bind("<B3-Motion>", self.on_mouse_move_roi2)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up_roi1)
        self.canvas.bind("<ButtonRelease-3>", self.on_mouse_up_roi2)
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        control_frame = tk.Frame(right_frame)
        control_frame.pack(fill="x", pady=5)
        self._create_control_buttons(control_frame)

        self.status_label = tk.Label(right_frame, text="加载文件夹中的图片并选择图片", anchor="w", relief="sunken", bd=1)
        self.status_label.pack(side="bottom", fill="x")

        self.update_coord_display()
        self.update_button_states()

    def _create_coordinate_display(self, parent_frame):
        """Creates the labels and entry fields for coordinate display."""

        tk.Label(parent_frame, text="ROI 1 (绿色框):").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.roi1_entry = tk.Entry(parent_frame, width=45, state='readonly', readonlybackground='white')
        self.roi1_entry.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        self.copy_roi1_btn = tk.Button(parent_frame, text="复制", command=lambda: self.copy_to_clipboard(self.roi1_coords), width=5)
        self.copy_roi1_btn.grid(row=0, column=2, padx=(0,10))


        tk.Label(parent_frame, text="ROI 2 (红色框):").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.roi2_entry = tk.Entry(parent_frame, width=45, state='readonly', readonlybackground='white')
        self.roi2_entry.grid(row=1, column=1, padx=5, pady=2, sticky="ew")
        self.copy_roi2_btn = tk.Button(parent_frame, text="复制", command=lambda: self.copy_to_clipboard(self.roi2_coords), width=5)
        self.copy_roi2_btn.grid(row=1, column=2, padx=(0,10))

        parent_frame.grid_columnconfigure(1, weight=1) # Allow entry field to expand

    def _create_control_buttons(self, parent_frame):
        """Creates the control buttons at the bottom."""
        # ROI Cropping Buttons
        self.crop_roi1_btn = tk.Button(parent_frame, text="裁剪&保存 ROI1 图片", command=self.crop_and_save_roi1)
        self.crop_roi1_btn.pack(side="left", padx=5)
        self.crop_roi2_btn = tk.Button(parent_frame, text="裁剪&保存 ROI2 图片", command=self.crop_and_save_roi2)
        self.crop_roi2_btn.pack(side="left", padx=5)

        # Spacer (optional, for visual separation)
        ttk.Separator(parent_frame, orient='vertical').pack(side="left", fill='y', padx=15, pady=5)

        # JSON Saving Controls
        tk.Label(parent_frame, text="JSON 文件后缀:").pack(side="left", padx=(0, 5))
        self.save_name_entry = tk.Entry(parent_frame, width=15)
        self.save_name_entry.pack(side="left", padx=5)
        self.save_name_entry.insert(0, "_roi") # Default suffix

        save_json_btn = tk.Button(parent_frame, text="保存 ROI 坐标到 JSON", command=self.save_roi_json)
        save_json_btn.pack(side="left", padx=10)

    def update_status(self, text):
        self.status_label.config(text=text)

    def update_coord_display(self):
        """Updates the read-only entry fields with current ROI coordinates."""
        def format_coords(coords):
            if coords:
                # Format as tuple string for easier copying into Python code
                return f"({coords[0]:.4f}, {coords[1]:.4f}, {coords[2]:.4f}, {coords[3]:.4f})"
            else:
                return ""

        roi1_text = format_coords(self.roi1_coords)
        self.roi1_entry.config(state='normal')
        self.roi1_entry.delete(0, tk.END)
        self.roi1_entry.insert(0, roi1_text)
        self.roi1_entry.config(state='readonly')

        roi2_text = format_coords(self.roi2_coords)
        self.roi2_entry.config(state='normal')
        self.roi2_entry.delete(0, tk.END)
        self.roi2_entry.insert(0, roi2_text)
        self.roi2_entry.config(state='readonly')

        # Update button states based on whether coords exist
        self.update_button_states()

    def update_button_states(self):
        """Enable/disable buttons based on current state."""
        # Copy buttons
        self.copy_roi1_btn.config(state=tk.NORMAL if self.roi1_coords else tk.DISABLED)
        self.copy_roi2_btn.config(state=tk.NORMAL if self.roi2_coords else tk.DISABLED)
        # Crop buttons
        self.crop_roi1_btn.config(state=tk.NORMAL if self.roi1_coords and self.original_image else tk.DISABLED)
        self.crop_roi2_btn.config(state=tk.NORMAL if self.roi2_coords and self.original_image else tk.DISABLED)

    def copy_to_clipboard(self, coords):
        """Copies the formatted coordinate string to the clipboard."""
        if coords:
            coord_str = f"({coords[0]:.4f}, {coords[1]:.4f}, {coords[2]:.4f}, {coords[3]:.4f})"
            try:
                pyperclip.copy(coord_str)
                self.update_status(f" 已复制到剪切板 : {coord_str}")
            except Exception as e:
                self.update_status(f"无法复制到剪切板: {e}")
                messagebox.showerror("复制错误", f"无法复制到剪切板\n请检查 pyperclip 库是否安装\n错误详情: {e}")
        else:
            self.update_status("无可复制的内容")

    def load_directory(self):
        folder = filedialog.askdirectory(title="选择图片文件夹")
        if not folder:
            self.update_status("取消选择文件夹")
            return
        self.folder = folder
        self.update_status(f"从以下文件夹中加载图片: {os.path.basename(folder)}")

        for i in self.tree.get_children():
            self.tree.delete(i)

        try:
            files = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))])
            if not files:
                self.update_status("文件夹中没有图片文件")
                return

            for file in files:
                self.tree.insert('', 'end', text=file, values=(os.path.join(folder, file),))
            self.update_status(f"找到 {len(files)} 张图片，请在文件列表中选择")
        except Exception as e:
             messagebox.showerror("路径加载错误", f"无法加载路径:\n{e}")
             self.update_status("读取路径错误")

    def on_tree_select(self, event=None):
        selected_item = self.tree.focus()
        if not selected_item: return
        try:
            filepath = self.tree.item(selected_item, 'values')[0]
            self.load_image(filepath)
        except IndexError: self.update_status("无法读取选择的图片路径")
        except Exception as e:
             messagebox.showerror("错误", f"处理图片时出现错误:\n{e}")
             self.update_status("处理图片时出现错误")

    def load_image(self, path):
        self.update_status(f"加载图片: {os.path.basename(path)}")
        try:
            self.image_path = path
            # 使用 ImageOps.exif_transpose 处理 JPEG 可能的方向问题
            self.original_image = ImageOps.exif_transpose(Image.open(path))

            self.roi1_coords = None
            self.roi2_coords = None
            self.clear_roi_rectangles()
            self.display_image()  # 先显示图片
            self.update_coord_display()  # 然后更新坐标（清除旧的）
            self.update_status(f"图片已加载 ({self.original_image.width}x{self.original_image.height})。绘制 ROI：左键点击 (ROI1 绿色)，右键点击 (ROI2 红色)。")
        except FileNotFoundError:
            messagebox.showerror("错误", f"未找到图片文件:\n{path}")
            self.update_status("错误：未找到图片文件。")
            self.original_image = None
        except Exception as e:
            messagebox.showerror("加载图片错误", f"无法加载图片:\n{path}\n错误: {e}")
            self.update_status(f"错误：加载图片失败: {os.path.basename(path)}")
            self.original_image = None

        if not self.original_image:
            self.canvas.delete("all")
            self.tk_image = None
            self.canvas_image_id = None
            self.update_coord_display()  # 如果加载失败，清除坐标
        self.update_button_states()  # 加载完成后更新按钮状态


    def on_canvas_resize(self, event=None):
        if self.original_image:
            self.display_image()

    def display_image(self):
        # (This function remains largely the same as the previous version)
        if not self.original_image:
            self.canvas.delete("all")
            self.tk_image = None
            self.canvas_image_id = None
            return

        self.root.update_idletasks()
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            self.root.after(50, self.display_image)
            return

        img_w, img_h = self.original_image.size
        scale_w = canvas_width / img_w
        scale_h = canvas_height / img_h
        self.display_scale = min(scale_w, scale_h)

        self.img_display_width = int(img_w * self.display_scale)
        self.img_display_height = int(img_h * self.display_scale)

        resized_image = self.original_image.resize((self.img_display_width, self.img_display_height), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized_image)

        self.img_offset_x = (canvas_width - self.img_display_width) // 2
        self.img_offset_y = (canvas_height - self.img_display_height) // 2

        self.canvas.delete("all")
        self.canvas_image_id = self.canvas.create_image(self.img_offset_x, self.img_offset_y,
                                                         anchor="nw", image=self.tk_image)
        self.redraw_final_rois()

    def clear_roi_rectangles(self):
        if self.roi1_rect_id: self.canvas.delete(self.roi1_rect_id); self.roi1_rect_id = None
        if self.roi2_rect_id: self.canvas.delete(self.roi2_rect_id); self.roi2_rect_id = None
        if self.current_rect_id: self.canvas.delete(self.current_rect_id); self.current_rect_id = None

    def redraw_final_rois(self):
        if self.roi1_coords: self.draw_final_rectangle(self.roi1_coords, 1)
        if self.roi2_coords: self.draw_final_rectangle(self.roi2_coords, 2)

    def draw_final_rectangle(self, norm_coords, roi_index):
        # (This function remains the same)
        if not self.original_image or norm_coords is None: return
        x1_norm, y1_norm, x2_norm, y2_norm = norm_coords
        disp_x1 = x1_norm * self.img_display_width
        disp_y1 = y1_norm * self.img_display_height
        disp_x2 = x2_norm * self.img_display_width
        disp_y2 = y2_norm * self.img_display_height
        canvas_x1 = self.img_offset_x + disp_x1
        canvas_y1 = self.img_offset_y + disp_y1
        canvas_x2 = self.img_offset_x + disp_x2
        canvas_y2 = self.img_offset_y + disp_y2
        outline_color = "green" if roi_index == 1 else "red"
        rect_id = self.canvas.create_rectangle(canvas_x1, canvas_y1, canvas_x2, canvas_y2, outline=outline_color, width=2)
        if roi_index == 1:
            if self.roi1_rect_id: self.canvas.delete(self.roi1_rect_id)
            self.roi1_rect_id = rect_id
        else:
            if self.roi2_rect_id: self.canvas.delete(self.roi2_rect_id)
            self.roi2_rect_id = rect_id

    # --- Coordinate Conversion ---
    def canvas_to_image_coords(self, canvas_x, canvas_y):
        # (This function remains the same)
        if not self.original_image: return None, None
        img_x = canvas_x - self.img_offset_x
        img_y = canvas_y - self.img_offset_y
        img_x = max(0, min(img_x, self.img_display_width))
        img_y = max(0, min(img_y, self.img_display_height))
        if self.display_scale <= 0: return None, None # Avoid division by zero
        original_x = img_x / self.display_scale
        original_y = img_y / self.display_scale
        return original_x, original_y

    def normalize_coords(self, x1, y1, x2, y2):
        # (This function remains the same)
        if not self.original_image: return None
        orig_w, orig_h = self.original_image.size
        if orig_w <= 0 or orig_h <= 0: return None
        norm_x1 = min(x1, x2) / orig_w
        norm_y1 = min(y1, y2) / orig_h
        norm_x2 = max(x1, x2) / orig_w
        norm_y2 = max(y1, y2) / orig_h
        norm_x1 = max(0.0, min(1.0, norm_x1))
        norm_y1 = max(0.0, min(1.0, norm_y1))
        norm_x2 = max(0.0, min(1.0, norm_x2))
        norm_y2 = max(0.0, min(1.0, norm_y2))
        return norm_x1, norm_y1, norm_x2, norm_y2

    # --- Mouse Event Handlers ---
    # on_mouse_down_roi1, on_mouse_down_roi2, on_mouse_move_roi1, on_mouse_move_roi2
    # remain largely the same, just updating status and clearing old final rects
    def on_mouse_down_roi1(self, event):
        if not self.original_image: return
        self.start_x, self.start_y = event.x, event.y
        if self.current_rect_id: self.canvas.delete(self.current_rect_id)
        if self.roi1_rect_id: self.canvas.delete(self.roi1_rect_id); self.roi1_rect_id = None; self.roi1_coords = None; self.update_coord_display() # Clear stored & display
        self.current_rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='green', width=1, dash=(4, 2))
        self.update_status("正在选取 ROI 1 (绿色)...")

    def on_mouse_down_roi2(self, event):
        if not self.original_image: return
        self.start_x, self.start_y = event.x, event.y
        if self.current_rect_id: self.canvas.delete(self.current_rect_id)
        if self.roi2_rect_id: self.canvas.delete(self.roi2_rect_id); self.roi2_rect_id = None; self.roi2_coords = None; self.update_coord_display() # Clear stored & display
        self.current_rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red', width=1, dash=(4, 2))
        self.update_status("正在选取 ROI 2 (红色)...")

    def on_mouse_move_roi1(self, event):
        if self.current_rect_id:
             curr_x = max(0, min(event.x, self.canvas.winfo_width()))
             curr_y = max(0, min(event.y, self.canvas.winfo_height()))
             self.canvas.coords(self.current_rect_id, self.start_x, self.start_y, curr_x, curr_y)

    def on_mouse_move_roi2(self, event):
        if self.current_rect_id:
            curr_x = max(0, min(event.x, self.canvas.winfo_width()))
            curr_y = max(0, min(event.y, self.canvas.winfo_height()))
            self.canvas.coords(self.current_rect_id, self.start_x, self.start_y, curr_x, curr_y)

    def on_mouse_up_roi1(self, event):
        # （逻辑更新为调用 update_coord_display）
        if not self.original_image or not self.current_rect_id:
            return
        end_x, end_y = event.x, event.y
        self.canvas.delete(self.current_rect_id)
        self.current_rect_id = None
        orig_x1, orig_y1 = self.canvas_to_image_coords(self.start_x, self.start_y)
        orig_x2, orig_y2 = self.canvas_to_image_coords(end_x, end_y)
        if orig_x1 is None:
            self.update_status("无法获取 ROI 1 的坐标。")
            return
        self.roi1_coords = self.normalize_coords(orig_x1, orig_y1, orig_x2, orig_y2)
        if self.roi1_coords:
            self.draw_final_rectangle(self.roi1_coords, 1)
            self.update_status("ROI 1 已设置。")  # 状态通过坐标显示更新
        else:
            self.update_status("设置 ROI 1 失败。")
        self.update_coord_display()  # 更新文本框


    def on_mouse_up_roi2(self, event):
        # (Logic updated to call update_coord_display)
        if not self.original_image or not self.current_rect_id: return
        end_x, end_y = event.x, event.y
        self.canvas.delete(self.current_rect_id); self.current_rect_id = None
        orig_x1, orig_y1 = self.canvas_to_image_coords(self.start_x, self.start_y)
        orig_x2, orig_y2 = self.canvas_to_image_coords(end_x, end_y)
        if orig_x1 is None: self.update_status("无法获取 ROI  的坐标。"); return
        self.roi2_coords = self.normalize_coords(orig_x1, orig_y1, orig_x2, orig_y2)
        if self.roi2_coords:
            self.draw_final_rectangle(self.roi2_coords, 2)
            self.update_status(f"ROI  已设置。")
        else: self.update_status("设置 ROI 2 失败。")
        self.update_coord_display() # Update the text boxes


    # --- Saving ---
    def crop_and_save_roi(self, roi_index):
        """裁剪并保存选中的 ROI 区域"""
        coords_to_use = self.roi1_coords if roi_index == 1 else self.roi2_coords
        roi_name = f"ROI{roi_index}"

        if not self.original_image or not self.image_path:
            messagebox.showwarning("无图片", "请先加载一张图片")
            return
        if not coords_to_use:
            messagebox.showwarning("无 ROI 坐标", f"{roi_name} 区域尚未选取")
            return

        try:
            orig_w, orig_h = self.original_image.size
            nx1, ny1, nx2, ny2 = coords_to_use

            # 将归一化坐标转换为原始图片的绝对像素坐标
            abs_x1 = int(nx1 * orig_w)
            abs_y1 = int(ny1 * orig_h)
            abs_x2 = int(nx2 * orig_w)
            abs_y2 = int(ny2 * orig_h)

            # 确保坐标顺序正确 (左上角和右下角)
            left, upper = min(abs_x1, abs_x2), min(abs_y1, abs_y2)
            right, lower = max(abs_x1, abs_x2), max(abs_y1, abs_y2)

            # 确保裁剪区域的宽高不为零
            if left >= right or upper >= lower:
                messagebox.showerror("裁剪错误", "无法裁剪宽度或高度为零的区域")
                return

            # 裁剪原始图片
            cropped_image = self.original_image.crop((left, upper, right, lower))

            # 确定保存路径和文件名
            base_name = os.path.splitext(os.path.basename(self.image_path))[0]
            save_folder = os.path.dirname(self.image_path)
            crop_filename = f"{base_name}_crop{roi_index}.png"
            save_path = os.path.join(save_folder, crop_filename)

            # 提示用户确认或修改保存路径/文件名
            confirmed_path = filedialog.asksaveasfilename(
                initialdir=save_folder,
                initialfile=crop_filename,
                defaultextension=".png",
                filetypes=[("PNG 文件", "*.png"), ("JPEG 文件", "*.jpg"), ("所有文件", "*.*")]
            )

            if not confirmed_path:
                self.update_status(f"取消保存 {roi_name}")
                return

            # 保存裁剪后的图片
            cropped_image.save(confirmed_path)
            self.update_status(f"{roi_name} 已裁剪并保存至: {os.path.basename(confirmed_path)}")
            messagebox.showinfo("裁剪完成", f"{roi_name} 区域已保存为:\n{confirmed_path}")

        except Exception as e:
            messagebox.showerror("裁剪错误", f"裁剪或保存 {roi_name} 失败:\n{e}")
            self.update_status(f"裁剪/保存 {roi_name} 出错")

    def crop_and_save_roi1(self):
        self.crop_and_save_roi(1)

    def crop_and_save_roi2(self):
        self.crop_and_save_roi(2)

    def save_roi_json(self):
        """将选中的 ROI 坐标保存为 JSON 文件"""
        if not self.image_path:
            messagebox.showwarning("无图片", "请先加载并选择一张图片")
            return

        if not self.roi1_coords and not self.roi2_coords:
            result = messagebox.askquestion("未定义 ROI", "未定义任何 ROI，是否保存空 JSON 文件？", icon='warning')
            if result == 'no':
                self.update_status("取消保存 JSON")
                return

        save_suffix = self.save_name_entry.get().strip()
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        save_folder = os.path.dirname(self.image_path)
        save_filename = f"{base_name}{save_suffix}.json"
        save_path = os.path.join(save_folder, save_filename)

        data_to_save = {}
        status_parts = []
        if self.roi1_coords:
            data_to_save["roi1"] = self.roi1_coords
            status_parts.append("ROI1 已保存")
        else:
            status_parts.append("ROI1 未设置")
        if self.roi2_coords:
            data_to_save["roi2"] = self.roi2_coords
            status_parts.append("ROI2 已保存")
        else:
            status_parts.append("ROI2 未设置")

        try:
            # 提示用户确认 JSON 保存路径
            confirmed_path = filedialog.asksaveasfilename(
                initialdir=save_folder,
                initialfile=save_filename,
                defaultextension=".json",
                filetypes=[("JSON 文件", "*.json"), ("所有文件", "*.*")]
            )
            if not confirmed_path:
                self.update_status("取消保存 JSON")
                return

            with open(confirmed_path, "w") as f:
                json.dump(data_to_save, f, indent=4)
            final_status = f"ROI JSON 已保存至 {os.path.basename(confirmed_path)}. ({', '.join(status_parts)})"
            messagebox.showinfo("JSON 已保存", final_status)
            self.update_status(final_status)
        except Exception as e:
            messagebox.showerror("保存错误", f"保存 ROI JSON 数据失败:\n{e}")
            self.update_status("保存 ROI JSON 数据出错")



if __name__ == "__main__":
    root = tk.Tk()
    app = ROISelectorApp(root)
    root.mainloop()
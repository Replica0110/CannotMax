# -*- coding: utf-8 -*-
import csv
import logging
import os
import subprocess
import threading
import time
import tkinter as tk
from tkinter import messagebox
import cv2
import keyboard # 注意：keyboard库在某些系统下可能需要管理员权限
import numpy as np
import torch
import loadData
import recognize
import math
import pandas as pd
from loguru import logger

try:
    from train import UnitAwareTransformer 
    MODEL_CLASS_IMPORTED = True
except ImportError:
    logger.warning("警告：无法从 train.py 导入 UnitAwareTransformer。")
    MODEL_CLASS_IMPORTED = False


from recognize import MONSTER_COUNT, intelligent_workers_debug
from PIL import Image, ImageTk
from sklearn.metrics.pairwise import cosine_similarity # 导入cosine_similarity
from similar_history_match import HistoryMatch

class ArknightsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("明日方舟斗蛐蛐 - 争锋")
        self.history_match = HistoryMatch()

        self.main_panel = tk.Frame(self.root)
        self.main_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.root.bind_all("<MouseWheel>", self._on_mousewheel)
        self.root.bind_all("<Shift-MouseWheel>", self._on_shift_mousewheel)
        self.auto_fetch_running = False
        self.no_region = True
        self.first_recognize = True
        self.is_invest = tk.BooleanVar(value=False)
        self.game_mode = tk.StringVar(value="单人")
        self.device_serial = tk.StringVar(value=loadData.manual_serial or "")

        self.is_in_battle = False
        self.fast_poll_start_time = None

        self.left_monsters = {}
        self.right_monsters = {}
        self.images = {}
        self.progress_var = tk.StringVar()
        self.main_roi = None
        self.current_prediction = 0.5
        self.current_image = None
        self.current_image_name = ""

        self.connected_devices = []

        self.total_fill_count = 0
        self.incorrect_fill_count = 0
        self.start_time = None

        # 提前加载模型标志位，如果加载失败则不继续创建UI
        self.model_loaded_successfully = False
        self.load_model() # 先尝试加载模型

        # 只有模型加载成功才继续创建界面
        if not self.model_loaded_successfully:
             messagebox.showerror("启动失败", "模型加载失败，应用程序无法启动。请查看控制台日志。")
             # 确保窗口能正常退出
             try:
                 self.root.destroy()
             except tk.TclError:
                 pass # 可能已经被destroy了
             return # 不再继续执行 __init__

        # --- 模型加载成功后，继续初始化 ---
        self.load_images()
        self.get_connected_devices()
        self.create_widgets()  # 创建控件
        self.refresh_device_list()

        # --- 添加模式变化追踪 ---
        # 在 create_widgets 之后，确保 self.game_mode 和 self.invest_checkbox 已创建
        self.game_mode.trace_add("write", self._update_invest_option_on_mode_change)
        # 初始化时也调用一次，确保初始状态正确
        self._update_invest_option_on_mode_change()

        self.history_visible   = False
        self.history_container = tk.Frame(self.root, bd=1, relief="sunken")
        self.history_canvas  = tk.Canvas(self.history_container, bg="white")
        self.history_vscroll = tk.Scrollbar(
            self.history_container, orient="vertical",
            command=self.history_canvas.yview)
        self.history_hscroll = tk.Scrollbar(
            self.history_container, orient="horizontal",
            command=self.history_canvas.xview)
        self.history_canvas.configure(
            yscrollcommand=self.history_vscroll.set,
            xscrollcommand=self.history_hscroll.set)
        self.history_frame = tk.Frame(self.history_canvas, bg="white")
        self.history_canvas.create_window(
            (0, 0), window=self.history_frame, anchor="nw")
        self.history_frame.bind(
            "<Configure>",
            lambda e: self.history_canvas.configure(
                scrollregion=self.history_canvas.bbox("all"))
        )
        self.history_canvas.grid(row=0, column=0, sticky="nsew")
        self.history_vscroll.grid(row=0, column=1, sticky="ns")
        self.history_hscroll.grid(row=1, column=0, sticky="ew")
        self.history_container.grid_rowconfigure(0, weight=1)
        self.history_container.grid_columnconfigure(0, weight=1)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _update_invest_option_on_mode_change(self, *args):
        """当游戏模式改变时，处理投资/观望选项的逻辑"""
        current_mode = self.game_mode.get()
        if current_mode == "30人":
            # 30人模式下，强制为投资（即 is_invest 为 False），并禁用观望复选框
            self.is_invest.set(False) # 注意：根据原代码逻辑，False 代表投资
            self.invest_checkbox.config(state=tk.DISABLED)
            logger.info("切换到30人模式，强制设为投资，禁用观望选项。")
        else: # 单人模式或其他模式
            # 允许用户选择投资或观望，启用复选框
            self.invest_checkbox.config(state=tk.NORMAL)
            logger.info(f"切换到 {current_mode} 模式，允许选择投资/观望。")
    def _on_mousewheel(self, event):

        if self.history_visible: # 仅当历史面板可见时响应
             # 检查滚轮事件是否发生在历史面板区域内 (可选优化，避免全局滚动)
            widget_under_mouse = self.root.winfo_containing(event.x_root, event.y_root)
            if widget_under_mouse is None: return # 安全检查

            is_descendant = False
            curr = widget_under_mouse
            # 向上查找父控件，看是否属于 history_container
            while curr is not None:
                if curr == self.history_container:
                    is_descendant = True
                    break
                # 防止无限循环（虽然理论上不会，但以防万一）
                if curr == self.root:
                     break
                try:
                    curr = curr.master
                except AttributeError: # 如果控件没有 master 属性
                     break

            if not is_descendant:
                return # 如果鼠标不在历史面板上或其子控件上，则不滚动

            self.history_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


    def _on_shift_mousewheel(self, event):

        if self.history_visible: # 仅当历史面板可见时响应
             # 检查滚轮事件是否发生在历史面板区域内 (可选优化，避免全局滚动)
            widget_under_mouse = self.root.winfo_containing(event.x_root, event.y_root)
            if widget_under_mouse is None: return

            is_descendant = False
            curr = widget_under_mouse
            while curr is not None:
                 if curr == self.history_container:
                     is_descendant = True
                     break
                 if curr == self.root:
                     break
                 try:
                    curr = curr.master
                 except AttributeError:
                     break

            if not is_descendant:
                 return

            self.history_canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")

    def load_images(self):

        try:
            # 获取系统缩放因子，用于调整图标大小
            scaling_factor = self.root.tk.call('tk', 'scaling')
        except tk.TclError:
            scaling_factor = 1.0 # 如果获取失败，使用默认值

        base_size = 30 # 图标基础大小
        icon_size = int(base_size * scaling_factor) # 根据缩放因子调整

        logger.info(f"图标缩放因子: {scaling_factor}, 计算后图标尺寸: {icon_size}x{icon_size}")

        for i in range(1, MONSTER_COUNT + 1):
            try:
                img_path = f"images/{i}.png"
                if not os.path.exists(img_path):
                    logger.warning(f"警告: 找不到图片 {img_path}")
                    continue # 跳过不存在的图片

                img = Image.open(img_path)
                width, height = img.size

                # 计算缩放比例，保持宽高比
                ratio = min(icon_size / width, icon_size / height) if width > 0 and height > 0 else 1.0
                new_size = (max(1, int(width * ratio)), max(1, int(height * ratio))) # 确保尺寸至少为1

                # 高质量缩放
                img_resized = img.resize(new_size, Image.Resampling.LANCZOS)

                # 转换为Tkinter兼容格式
                photo_img = ImageTk.PhotoImage(img_resized)
                self.images[str(i)] = photo_img
            except Exception as e:
                logger.exception(f"加载或处理图片 {i} 时出错: {e}")

    def load_model(self):
        """加载预训练的PyTorch模型"""
        model_path = 'models/best_model_full.pth'
        self.model = None # 先置为 None
        self.model_loaded_successfully = False # 重置标志位
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 确保 device 在这里初始化

        try:
            if not MODEL_CLASS_IMPORTED:
                 # 如果类没有成功导入，直接抛出错误，避免后续尝试加载
                 raise ImportError("模型类 UnitAwareTransformer 未能导入，无法加载模型。请确保 train.py 文件存在且包含该类定义。")

            if not os.path.exists(model_path):
                alt_model_path = os.path.join('..', model_path)
                if os.path.exists(alt_model_path):
                    model_path = alt_model_path
                else:
                     raise FileNotFoundError(f"未找到模型文件 '{model_path}' 或 '{alt_model_path}'")

            logger.info(f"正在从 {model_path} 加载模型到 {self.device}...")
            # 直接加载完整模型对象，这时需要 UnitAwareTransformer 类定义可用
            try:
               model = torch.load(model_path, map_location=self.device, weights_only=False)
            except TypeError:
                model = torch.load(model_path, map_location=self.device)

            # 检查加载的是否是我们期望的类型
            # ================== 修改这里，检查是否是 UnitAwareTransformer 的实例 ==================
            if not isinstance(model, UnitAwareTransformer):
                 # 如果加载的不是预期的类实例，可能是状态字典或其他东西
                 # 尝试处理状态字典的情况（如果模型保存的是state_dict）
                 if isinstance(model, dict) and 'model_state_dict' in model:
                      logger.info("检测到加载的是状态字典，尝试创建模型实例并加载...")
                      # 这里需要能实例化 UnitAwareTransformer，需要知道它的初始化参数
                      # 例如: model_instance = UnitAwareTransformer(param1, param2, ...)
                      # 如果不知道参数，就无法完成加载
                      raise NotImplementedError("加载状态字典需要模型实例化参数，请在代码中提供。或者确保保存的是完整的模型对象。")
                      # model_instance.load_state_dict(model['model_state_dict'])
                      # self.model = model_instance.to(self.device)
                 else:
                      # 加载的对象类型未知
                      raise TypeError(f"加载的文件类型不是预期的 UnitAwareTransformer，而是 {type(model)}。请检查模型保存方式。")
            # ===================================================================================
            else:
                # 加载成功，是 UnitAwareTransformer 实例
                 self.model = model.to(self.device)


            self.model.eval() # 设置为评估模式
            logger.info("模型加载成功。")
            self.model_loaded_successfully = True # 设置成功标志

        # --- 更精细的错误处理 ---
        except FileNotFoundError as e:
            error_msg = f"模型文件未找到: {str(e)}\n请确认模型文件存在于 'models' 目录下或其父目录中，并且已经训练。"
            logger.error(error_msg) # 打印到控制台
            # 不在这里关闭窗口，让 __init__ 返回后检查标志位
        except AttributeError as e:
             # 特别处理找不到类定义的错误
             error_msg = f"模型加载失败: {str(e)}\n\n错误提示：很可能是因为找不到模型类 (例如 'UnitAwareTransformer') 的定义。\n"
             error_msg += "请确保：\n1. 定义模型类的 Python 文件 (例如 train.py) 与主程序在同一环境。\n"
             error_msg += "2. 主程序已正确导入该模型类 (例如 `from train import UnitAwareTransformer`)。\n"
             error_msg += "3. 模型保存时使用的类定义与当前导入的类定义一致。"
             print(error_msg)
             import traceback
             traceback.print_exc() # 打印详细堆栈
        except ImportError as e:
             # 处理自定义的 ImportError 或类未导入的情况
             error_msg = f"模型加载失败: {str(e)}"
             logger.error(error_msg)
        except NotImplementedError as e: # 处理状态字典加载问题
             error_msg = f"模型加载失败: {str(e)}"
             logger.error(error_msg)
        except TypeError as e: # 处理类型不匹配
             error_msg = f"模型加载失败: {str(e)}"
             logger.error(error_msg)
        except Exception as e:
            # 捕获其他可能的错误 (例如 pickle 错误, RuntimeError)
            error_msg = f"模型加载时发生未知错误: {str(e)}"
            if "size mismatch" in str(e):
                error_msg += "\n\n提示：可能是模型结构与加载的模型文件不匹配。请尝试重新训练模型。"
            elif "_pickle.UnpicklingError" in str(e) or "ModuleNotFoundError" in str(e):
                 error_msg += "\n\n提示：模型文件可能已损坏，或缺少必要的代码定义。请尝试重新训练或检查环境。"
            logger.error(error_msg)
            import traceback
            traceback.print_exc()
        # 不在此处调用 self.root.destroy()

    def create_widgets(self):

        # --- 顶部容器 (怪物显示区) ---
        self.top_container = tk.Frame(self.main_panel)
        self.top_container.pack(side=tk.TOP, fill=tk.X, pady=(0, 10)) # 顶部填充X轴，底部留间距

        # 创建一个居中容器来放置左右怪物框
        self.monster_center = tk.Frame(self.top_container)
        self.monster_center.pack(side=tk.TOP, anchor='center') # 居中放置

        # --- 左侧怪物框 ---
        self.left_frame = tk.Frame(self.monster_center, borderwidth=2, relief="groove", padx=10, pady=5) # 加边框和内边距
        self.left_frame.pack(side=tk.LEFT, padx=10, anchor='n', pady=5) # 左侧放置，增加外边距
        tk.Label(self.left_frame, text="左侧怪物", font=('Helvetica', 10, 'bold')).grid(row=0, column=0, columnspan=10, pady=(0, 5)) # 标题，增加底部间距

        # --- 右侧怪物框 ---
        self.right_frame = tk.Frame(self.monster_center, borderwidth=2, relief="groove", padx=10, pady=5) # 加边框和内边距
        self.right_frame.pack(side=tk.RIGHT, padx=10, anchor='n', pady=5) # 右侧放置，增加外边距
        tk.Label(self.right_frame, text="右侧怪物", font=('Helvetica', 10, 'bold')).grid(row=0, column=0, columnspan=10, pady=(0, 5)) # 标题，增加底部间距

        # --- 填充左右怪物图标和输入框 ---
        num_rows = 4 # 显示为4行
        monsters_per_row = math.ceil(MONSTER_COUNT / num_rows)

        for side, frame, monsters_dict in [("left", self.left_frame, self.left_monsters),
                                            ("right", self.right_frame, self.right_monsters)]:
            for r in range(num_rows):
                start_idx = r * monsters_per_row + 1
                end_idx = min((r + 1) * monsters_per_row + 1, MONSTER_COUNT + 1)
                col_offset = 0 # 当前行的列偏移量
                for i in range(start_idx, end_idx):
                    monster_id_str = str(i)
                    if monster_id_str in self.images: # 检查图片是否存在
                         # 图片标签
                        img_label = tk.Label(frame, image=self.images[monster_id_str], padx=1, pady=1)
                        img_label.grid(row=r * 2 + 1, column=col_offset, sticky='ew') # 图片放在奇数行

                        # 输入框
                        entry = tk.Entry(frame, width=5) # 保持宽度为5
                        entry.grid(row=r * 2 + 2, column=col_offset, pady=(0, 3), sticky='n') # 输入框放偶数行，减小底部间距
                        monsters_dict[monster_id_str] = entry
                    else:
                        # 如果图片不存在，可以放一个占位符或者空Label
                        # tk.Label(frame, text=f"({i})", width=5).grid(row=r * 2 + 1, column=col_offset, rowspan=2, sticky='nsew')
                        # print(f"警告：怪物 {i} 的图片未加载，不创建输入框。") # 减少干扰，注释掉
                        monsters_dict[monster_id_str] = None # 标记此怪物无输入框

                    col_offset += 1

            # 调整列权重使布局更均匀紧凑
            for col in range(monsters_per_row):
                frame.grid_columnconfigure(col, weight=1, minsize=40) # 设置列权重和最小宽度

        # --- 中部容器 (结果显示区) ---
        self.bottom_container = tk.Frame(self.main_panel)
        self.bottom_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=5) # 填充剩余空间

        self.result_frame = tk.Frame(self.bottom_container, relief="ridge", borderwidth=1) # 加边框
        self.result_frame.pack(fill=tk.X, pady=5) # 填充X轴，上下留间距

        # 预测结果标签
        self.result_label = tk.Label(self.result_frame, text="预测结果: ", font=("Helvetica", 14, "bold"), fg="black", justify=tk.LEFT, anchor='w') # 增大字体，加粗，靠左
        self.result_label.pack(pady=(5, 2), padx=10, fill=tk.X) # 增加内边距和填充

        # 统计信息标签
        self.stats_label = tk.Label(self.result_frame, text="统计: ", font=("Helvetica", 10), fg="gray", justify=tk.LEFT, anchor='w') # 稍小字体，灰色，靠左
        self.stats_label.pack(pady=(0, 5), padx=10, fill=tk.X) # 增加内边距和填充

        # --- 底部容器 (按钮区) ---
        self.button_frame = tk.Frame(self.bottom_container, relief="groove", borderwidth=2, padx=10, pady=10)
        self.button_frame.pack(fill=tk.X, pady=(10, 0))

        self.button_frame.grid_columnconfigure(0, weight=1)
        self.button_frame.grid_columnconfigure(1, weight=2)
        self.button_frame.grid_columnconfigure(2, weight=1)

        # 左侧（设置区域）
        left_frame = tk.LabelFrame(self.button_frame, text="配置", font=('Helvetica', 10, 'bold'))
        left_frame.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)

        mode_frame = tk.Frame(left_frame)
        mode_frame.pack(pady=(8, 5))
        tk.Label(mode_frame, text="模式:").pack(side=tk.LEFT)
        self.mode_menu = tk.OptionMenu(mode_frame, self.game_mode, "单人", "30人")
        self.mode_menu.pack(side=tk.LEFT, padx=5)

        self.invest_checkbox = tk.Checkbutton(left_frame, text="观望模式", variable=self.is_invest)
        self.invest_checkbox.pack(pady=(0, 8))

        duration_frame = tk.Frame(left_frame)
        duration_frame.pack(pady=(0, 8))
        tk.Label(duration_frame, text="时长(分钟, -1=无限):").pack(side=tk.LEFT)
        self.duration_entry = tk.Entry(duration_frame, width=6)
        self.duration_entry.insert(0, "-1")
        self.duration_entry.pack(side=tk.LEFT, padx=5)

        serial_frame = tk.Frame(left_frame)
        serial_frame.pack(pady=(0, 8))

        tk.Label(serial_frame, text="设备号:").pack(side=tk.LEFT)

        # 设备选择下拉菜单
        self.device_options = tk.StringVar()
        self.device_dropdown = tk.OptionMenu(serial_frame, self.device_options, "")
        self.device_dropdown.config(width=16)
        self.device_dropdown.pack(side=tk.LEFT, padx=3)

        # 刷新设备列表按钮
        self.refresh_button = tk.Button(serial_frame, text="刷新", command=self.refresh_device_list, width=6)
        self.refresh_button.pack(side=tk.LEFT, padx=3)

        # 中间（预测+标注区域）
        center_frame = tk.LabelFrame(self.button_frame, text="预测与标注", font=('Helvetica', 10, 'bold'))
        center_frame.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)

        # 自动斗蛐蛐按钮
        self.auto_fetch_button = tk.Button(center_frame, text="自动斗蛐蛐", command=self.toggle_auto_fetch,
                                           width=20, height=2, font=("Helvetica", 12, "bold"),
                                           bg="#AED581")
        self.auto_fetch_button.pack(pady=(10, 10))

        # 预测按钮
        self.predict_button = tk.Button(center_frame, text="预测", command=self.predict,
                                        width=20, height=2, bg="#FFF176", font=("Helvetica", 12, "bold"))
        self.predict_button.pack(pady=(0, 5))

        # 截图识别并预测按钮
        self.recognize_button = tk.Button(center_frame, text="截图识别并预测", command=self.recognize,
                                          width=20, height=2, bg="#4DD0E1", font=("Helvetica", 12))
        self.recognize_button.pack(pady=(0, 10))

        # 预测正确/错误按钮 (一行并排)
        manual_frame = tk.Frame(center_frame)
        manual_frame.pack(pady=(5, 10))

        self.fill_data_correct_button = tk.Button(manual_frame, text="预测正确 ✓", command=self.fill_data_correct,
                                                  width=12, bg="#A5D6A7")
        self.fill_data_correct_button.pack(side=tk.LEFT, padx=8)

        self.fill_data_incorrect_button = tk.Button(manual_frame, text="预测错误 ✕", command=self.fill_data_incorrect,
                                                    width=12, bg="#EF9A9A")
        self.fill_data_incorrect_button.pack(side=tk.LEFT, padx=8)

        # 右侧（工具区域）
        right_frame = tk.LabelFrame(self.button_frame, text="工具操作", font=('Helvetica', 10, 'bold'))
        right_frame.grid(row=0, column=2, sticky='nsew', padx=5, pady=5)

        self.reset_button = tk.Button(right_frame, text="清空数据", command=self.reset_entries, width=16)
        self.reset_button.pack(pady=(15, 5))

        self.reselect_button = tk.Button(right_frame, text="选择识别区域", command=self.reselect_roi, width=16)
        self.reselect_button.pack(pady=5)

        self.history_button = tk.Button(right_frame, text="显示错题本", command=self.toggle_history_panel, width=16)
        self.history_button.pack(pady=5)
    def refresh_device_list(self):
        """刷新当前可用设备列表"""
        self.get_connected_devices()
        if self.connected_devices:
            menu = self.device_dropdown["menu"]
            menu.delete(0, "end")
            for dev in self.connected_devices:
                menu.add_command(label=dev, command=lambda d=dev: self.device_options.set(d))
            self.device_options.set(self.connected_devices[0])  # 默认选第一个
        else:
            self.device_options.set("")
            menu = self.device_dropdown["menu"]
            menu.delete(0, "end")

    def get_connected_devices(self):
        """获取通过ADB连接的设备列表"""
        import subprocess
        try:
            output = subprocess.check_output(["adb", "devices"], encoding='utf-8')
            lines = output.strip().split("\n")[1:]  # 跳过第一行
            self.connected_devices = []
            for line in lines:
                if "device" in line:
                    parts = line.split()
                    if len(parts) >= 2 and parts[1] == "device":
                        self.connected_devices.append(parts[0])
        except Exception as e:
            logger.exception(f"获取设备列表出错: {e}")

    # --- 历史面板控制 ---
    def toggle_history_panel(self):

        if not self.history_visible:
            # 显示面板
            self.history_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(0, 10), pady=10) # 在主面板右侧显示，增加边距
            self.history_button.config(text="隐藏错题本")
            # 清空旧内容并重新渲染
            for widget in self.history_frame.winfo_children():
                widget.destroy()
            # 异步渲染，避免阻塞UI
            self.root.after(50, lambda: self.render_history(self.history_frame))
        else:
            # 隐藏面板
            self.history_container.pack_forget()
            self.history_button.config(text="显示错题本")
        self.history_visible = not self.history_visible

    def render_history(self, parent_frame):

        logger.info("开始渲染历史对局...")
        try:
            # 调用 HistoryMatch 计算相似度等
            self.history_match.render_similar_matches(self.left_monsters, self.right_monsters)

            # 获取计算结果
            left_rate = self.history_match.left_rate
            right_rate = self.history_match.right_rate
            sims = self.history_match.sims
            swap_flags = self.history_match.swap
            top20_indices = self.history_match.top20_idx

            # --- 清空现有内容 ---
            for widget in parent_frame.winfo_children():
                widget.destroy()

            # --- 渲染标题 ---
            header_frame = tk.Frame(parent_frame, bg="white")
            header_frame.pack(fill="x", pady=5, padx=5)

            tk.Label(header_frame, text="相似历史对局 Top 20", font=("Helvetica", 12, "bold"), bg="white").pack(side="left")

            stats_frame = tk.Frame(parent_frame, bg="white")
            stats_frame.pack(fill="x", pady=2, padx=5)
            fgL, fgR = ("#E23F25", "#666") if left_rate > right_rate else ("#666", "#25ace2")
            tk.Label(stats_frame, text="相似局胜率:", font=("Helvetica", 10), bg="white").pack(side="left", padx=(0, 5))
            tk.Label(stats_frame, text=f"左边 {left_rate:.1%} ", fg=fgL, font=("Helvetica", 10, "bold"), bg="white").pack(side="left")
            tk.Label(stats_frame, text=f"右边 {right_rate:.1%}", fg=fgR, font=("Helvetica", 10, "bold"), bg="white").pack(side="left")


            # --- 准备批量渲染 ---
            self.history_canvas.config(width=700)
            self.history_frame.config(width=700)

            self._history_parent = parent_frame
            self._top20_indices = top20_indices.tolist()
            self._sims = sims
            self._swap_flags = swap_flags
            self._batch_render_idx = 0

            parent_frame.after(10, lambda: self._render_batch(batch_size=5))

            logger.info("历史对局渲染准备就绪。")

        except AttributeError as e:
             if "'HistoryMatch' object has no attribute" in str(e):
                 logger.error(f"[渲染错题本失败] 缺少必要的属性: {e}. 请检查 HistoryMatch 类的实现。")
                 tk.Label(parent_frame, text="无法加载历史数据(属性缺失)。", fg="red", bg="white").pack(pady=10)
             else:
                logger.error(f"[渲染错题本失败] 属性错误: {e}")
                import traceback
                traceback.print_exc()
                tk.Label(parent_frame, text=f"渲染时发生属性错误: {e}", fg="red", bg="white").pack(pady=10)
        except Exception as e:
            logger.error(f"[渲染错题本失败] 发生未知错误: {e}")
            import traceback
            traceback.print_exc()
            tk.Label(parent_frame, text=f"渲染历史数据时出错: {e}", fg="red", bg="white").pack(pady=10)

    def _render_batch(self, batch_size=5):

        start = self._batch_render_idx
        end = min(start + batch_size, len(self._top20_indices))
        parent = self._history_parent
        history_match = self.history_match # 引用实例

        logger.info(f"渲染批次: {start} 到 {end-1}")

        if not hasattr(history_match, 'past_left') or \
           not hasattr(history_match, 'past_right') or \
           not hasattr(history_match, 'labels'):
            logger.warning("历史数据未完全加载，停止渲染。")
            tk.Label(parent, text="历史数据加载不完整。", fg="orange", bg="white").pack()
            parent.update_idletasks()
            self.history_canvas.configure(scrollregion=self.history_canvas.bbox("all"))
            return

        labels = history_match.labels

        for rank, data_idx in enumerate(self._top20_indices[start:end], start + 1):
            similarity_score = self._sims[data_idx]
            was_swapped = self._swap_flags[data_idx]

            try:
                 L_history = (history_match.past_left if not was_swapped else history_match.past_right)[data_idx]
                 R_history = (history_match.past_right if not was_swapped else history_match.past_left)[data_idx]

                 original_label = labels[data_idx]
                 actual_winner_label = original_label if not was_swapped else ('L' if original_label == 'R' else 'R')

                 is_left_win = (actual_winner_label == 'L')
                 is_right_win = (actual_winner_label == 'R')

            except IndexError:
                 logger.warning(f"警告: 索引 {data_idx} 超出历史数据范围，跳过此条记录。")
                 continue

            original_row_num = data_idx + 2
            record_frame = tk.Frame(parent, pady=8, padx=5, bg="white")
            record_frame.pack(fill="x")
            info_frame = tk.Frame(record_frame, bg="white")
            info_frame.pack(side="left", fill="y", padx=(0, 10))

            tk.Label(info_frame,text=f"#{rank}",font=("Helvetica", 14, "bold"),bg="white").pack(anchor="nw", pady=(0, 2))
            tk.Label(info_frame,text=f"相似度: {similarity_score:.3f}",font=("Helvetica", 9),bg="white").pack(anchor="nw")
            tk.Label(info_frame,text=f"(原始局: {original_row_num})",font=("Helvetica", 8, "italic"),fg="grey",bg="white").pack(anchor="nw")

            roster_frame = tk.Frame(record_frame, bg="white")
            roster_frame.pack(side="right", fill="both", expand=True)

            for side_name, monster_vector, is_winner, win_bg, win_fg, border_color in [
                    ('左', L_history, is_left_win, "#FFF3E0", "#E65100", "orange"),
                    ('右', R_history, is_right_win, "#E3F2FD", "#0D47A1", "blue"),
            ]:
                bg_color = win_bg if is_winner else "#f5f5f5"
                fg_color = win_fg if is_winner else "#555555"
                relief_style = "solid" if is_winner else "groove"
                border_thickness = 2 if is_winner else 1

                side_pane = tk.Frame(
                    roster_frame, bd=border_thickness, relief=relief_style, bg=bg_color,
                    highlightbackground=border_color if is_winner else "#cccccc",
                    highlightthickness=border_thickness)
                side_pane.pack(fill="x", pady=2)

                win_mark = "🏆" if is_winner else ""
                tk.Label(side_pane,text=f"{side_name}边 {win_mark}",fg=fg_color,bg=bg_color,font=("Helvetica", 9, "bold")).pack(anchor="nw", padx=4, pady=(2, 0))

                inner_content_frame = tk.Frame(side_pane, bg=bg_color)
                inner_content_frame.pack(fill="x", padx=4, pady=(0, 4))

                monsters_per_visual_row = 10
                current_col = 0
                row_frame = tk.Frame(inner_content_frame, bg=bg_color)
                row_frame.pack(fill="x")

                for monster_idx, count in enumerate(monster_vector):
                    if count > 0:
                        monster_id_str = str(monster_idx + 1)
                        if monster_id_str in self.images:
                            if current_col >= monsters_per_visual_row:
                                current_col = 0
                                row_frame = tk.Frame(inner_content_frame, bg=bg_color)
                                row_frame.pack(fill="x")

                            monster_item_frame = tk.Frame(row_frame, bg=bg_color)
                            monster_item_frame.pack(side="left", padx=1, pady=1)

                            tk.Label(monster_item_frame, image=self.images[monster_id_str], bg=bg_color).pack(side="left")
                            tk.Label(monster_item_frame, text=f"x{int(count)}", bg=bg_color, font=("Helvetica", 8)).pack(side="left", padx=(0, 3))
                            current_col += 1
                        # else:
                        #      print(f"警告: 历史记录中怪物ID {monster_id_str} 的图片未加载。") # 减少输出


        self._batch_render_idx = end
        parent.update_idletasks()
        scroll_region = self.history_canvas.bbox("all")
        if scroll_region:
             self.history_canvas.configure(scrollregion=scroll_region)

        if end < len(self._top20_indices):
            parent.after(75, lambda: self._render_batch(batch_size))
        else:
            logger.info("所有历史记录渲染完成。")
            done_label = tk.Label(parent, text="--- Top 20 显示完毕 ---", font=("Helvetica", 9, "italic"), fg="grey", bg="white")
            done_label.pack(pady=10)
            parent.update_idletasks()
            self.history_canvas.configure(scrollregion=self.history_canvas.bbox("all"))

    # --- 数据操作 ---
    def reset_entries(self):

        for entry in self.left_monsters.values():
             if entry: # 检查是否为 None (因为图片可能加载失败)
                entry.delete(0, tk.END)
                entry.config(bg="white") # 重置背景色为白色
        for entry in self.right_monsters.values():
             if entry:
                entry.delete(0, tk.END)
                entry.config(bg="white") # 重置背景色为白色
        self.result_label.config(text="预测结果: ", fg="black", font=("Helvetica", 14, "bold")) # 重置结果标签
        logger.info("输入框已清空。")

    def fill_data_correct(self):

        result_to_fill = 'R' if self.current_prediction > 0.5 else 'L'
        logger.info(f"填写数据 (符合预测): 预测值 {self.current_prediction:.3f}, 记录为 {result_to_fill}")
        self.fill_data(result_to_fill)
        self.total_fill_count += 1
        self.update_statistics()

    def fill_data_incorrect(self):

        result_to_fill = 'L' if self.current_prediction > 0.5 else 'R'
        logger.info(f"填写数据 (不符预测): 预测值 {self.current_prediction:.3f}, 记录为 {result_to_fill}")
        self.fill_data(result_to_fill)
        self.total_fill_count += 1
        self.incorrect_fill_count += 1
        self.update_statistics()

    def fill_data(self, result_label):

        image_data = np.zeros((1, MONSTER_COUNT * 2), dtype=np.int16)

        for monster_id, entry_widget in self.left_monsters.items():
            if entry_widget:
                 value_str = entry_widget.get()
                 if value_str.isdigit():
                     image_data[0, int(monster_id) - 1] = int(value_str)

        for monster_id, entry_widget in self.right_monsters.items():
             if entry_widget:
                 value_str = entry_widget.get()
                 if value_str.isdigit():
                     image_data[0, MONSTER_COUNT + int(monster_id) - 1] = int(value_str)

        data_row = image_data[0].tolist()
        data_row.append(result_label)

        if intelligent_workers_debug: # intelligent_workers_debug 需定义或导入
            if self.current_image_name:
                data_row.append(self.current_image_name)
                if self.current_image is not None:
                    try:
                        save_dir = 'data/images'
                        os.makedirs(save_dir, exist_ok=True)
                        image_path = os.path.join(save_dir, self.current_image_name)
                        if self.current_image.size > 0:
                             cv2.imwrite(image_path, self.current_image)
                             # print(f"调试图片已保存: {image_path}") # 减少输出
                        else:
                             logger.warning(f"警告: 尝试保存空的调试图片 {self.current_image_name}")
                        self.current_image = None
                        self.current_image_name = ""
                    except Exception as e:
                        logger.error(f"保存调试图片时出错: {e}")
            else:
                 # 保持列数一致
                 data_row.append("")


        csv_file_path = 'arknights.csv'
        try:
            with open(csv_file_path, 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(data_row)
        except IOError as e:
            logger.error(f"写入 CSV 文件时出错: {e}")
            messagebox.showerror("错误", f"无法写入数据到 {csv_file_path}\n请检查文件权限或磁盘空间。")
        except Exception as e:
            logger.error(f"处理或写入数据时发生未知错误: {e}")
            messagebox.showerror("错误", f"保存数据时发生错误: {e}")

    # --- 模型预测 ---
    def get_prediction(self):

        if self.model is None:
            messagebox.showerror("错误", "模型未加载，无法进行预测。")
            return 0.5 # 返回默认值

        try:
            left_counts = np.zeros(MONSTER_COUNT, dtype=np.int16)
            right_counts = np.zeros(MONSTER_COUNT, dtype=np.int16)

            for monster_id, entry_widget in self.left_monsters.items():
                 if entry_widget:
                    value_str = entry_widget.get().strip()
                    left_counts[int(monster_id) - 1] = int(value_str) if value_str.isdigit() else 0

            for monster_id, entry_widget in self.right_monsters.items():
                 if entry_widget:
                    value_str = entry_widget.get().strip()
                    right_counts[int(monster_id) - 1] = int(value_str) if value_str.isdigit() else 0

            left_signs_tensor = torch.sign(torch.tensor(left_counts, dtype=torch.float32)).unsqueeze(0).to(self.device)
            left_counts_tensor = torch.abs(torch.tensor(left_counts, dtype=torch.float32)).unsqueeze(0).to(self.device)
            right_signs_tensor = torch.sign(torch.tensor(right_counts, dtype=torch.float32)).unsqueeze(0).to(self.device)
            right_counts_tensor = torch.abs(torch.tensor(right_counts, dtype=torch.float32)).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(left_signs_tensor, left_counts_tensor, right_signs_tensor, right_counts_tensor)
                prediction = output.item()

            if np.isnan(prediction) or np.isinf(prediction):
                logger.warning("警告: 模型预测结果包含 NaN 或 Inf，返回默认值 0.5")
                prediction = 0.5

            prediction = max(0.0, min(1.0, prediction))

            return prediction

        except ValueError as e:
             logger.error(f"处理输入数据时出错: {e}")
             messagebox.showerror("错误", "输入无效，请输入有效的非负整数。")
             return 0.5
        except RuntimeError as e:
            error_msg = f"模型推理时发生运行时错误: {e}"
            if "size mismatch" in str(e):
                error_msg += "\n\n错误提示：输入/模型维度不匹配。\n请尝试删除旧模型并重新训练。"
            logger.error(error_msg)
            messagebox.showerror("模型错误", error_msg)
            return 0.5
        except Exception as e:
            logger.error(f"预测过程中发生未知错误: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("错误", f"预测时发生意外错误: {str(e)}")
            return 0.5

    def predictText(self, prediction):
        result = ""
        right_win_prob = prediction
        left_win_prob = 1.0 - right_win_prob
        if right_win_prob > 0.5:
            result = "右方胜"
            win_prob =  right_win_prob
        else:
            result = "左方胜"
            win_prob =  left_win_prob

        result_text = (f"预测结果 ({result}): {win_prob:.2%}\n"
                       f"左方胜率: {left_win_prob:.2%} | 右方胜率: {right_win_prob:.2%}")

        font_config = ("Helvetica", 12, "normal")
        fg_color = "black"

        if left_win_prob > 0.7:
            fg_color = "#D32F2F"; font_config = ("Helvetica", 12, "bold")
        elif left_win_prob > 0.6:
            fg_color = "#F44336"
        elif right_win_prob > 0.7:
            fg_color = "#0277BD"; font_config = ("Helvetica", 12, "bold")
        elif right_win_prob > 0.6:
            fg_color = "#2196F3"
        else:
             fg_color = "#555555"

        self.result_label.config(text=result_text, fg=fg_color, font=font_config)

    def predict(self):

        logger.info("执行预测...")
        self.current_prediction = self.get_prediction()
        logger.info(f"模型预测右方胜率: {self.current_prediction:.4f}")
        self.predictText(self.current_prediction)

        if self.history_visible:
            logger.info("历史面板可见，准备更新...")
            for widget in self.history_frame.winfo_children():
                widget.destroy()
            self.root.after(50, lambda: self.render_history(self.history_frame))

    # --- 图像识别 ---
    def recognize(self):
        # 如果正在进行自动斗蛐蛐，从adb加载截图
        if self.auto_fetch_running:
            screenshot = loadData.capture_screenshot()
            if screenshot is None:
                logger.error("错误：自动模式下获取截图失败。")
                return
        elif self.no_region:
            logger.warning("未定义识别区域，尝试获取截图...")
            if self.first_recognize:
                logger.info("首次识别，尝试连接 ADB 并设置默认 ROI...")
                try:
                    if loadData.screen_width == 0 or loadData.screen_height == 0:
                        loadData.initialize_load_data()
                        raise ValueError("无法获取屏幕尺寸，无法设置默认ROI。")

                    default_x1_ratio = 0.248; default_y1_ratio = 0.841
                    default_x2_ratio = 0.753; default_y2_ratio = 0.951
                    self.main_roi = [
                        (int(default_x1_ratio * loadData.screen_width), int(default_y1_ratio * loadData.screen_height)),
                        (int(default_x2_ratio * loadData.screen_width), int(default_y2_ratio * loadData.screen_height))
                    ]
                    logger.info(f"设置默认 ROI: {self.main_roi}")

                    adb_path = loadData.adb_path
                    device_serial = loadData.get_device_serial()
                    if adb_path and device_serial:
                         connect_command = f'"{adb_path}" connect {device_serial}'
                         result = subprocess.run(connect_command, shell=True, capture_output=True, text=True, timeout=10)
                         if result.returncode != 0:
                              logger.warning(f"警告: ADB 连接命令可能失败 (返回码 {result.returncode}): {result.stderr.strip()}")


                    self.first_recognize = False
                    self.no_region = False

                except Exception as e:
                    logger.error(f"首次识别初始化失败: {e}")
                    messagebox.showerror("初始化错误", f"首次识别设置默认区域或连接ADB时出错: {e}\n请尝试手动选择范围或检查ADB设置。")
                    return

            screenshot = loadData.capture_screenshot()
            if screenshot is None:
                 logger.error("错误：获取截图失败。")
                 return
        elif self.main_roi:
             screenshot = loadData.capture_screenshot()
             if screenshot is None:
                 logger.error("错误：获取截图失败。")
                 return
        else:
              logger.error("错误：无法确定识别方式。")
              return

        if self.main_roi is None:
             logger.error("错误：识别区域 (ROI) 未定义。")
             return

        # print(f"调用 recognize.process_regions 进行识别...")
        results = recognize.process_regions(self.main_roi, screenshot=screenshot)

        self.reset_entries()
        processed_monster_ids_for_debug = []

        if not results:
            logger.warning("识别未返回任何结果。")
        else:
            logger.info(f"识别结果: {results}")
            for res in results:
                region_id = res.get('region_id', -1)
                matched_id = res.get('matched_id')
                number = res.get('number')
                error_msg = res.get('error')

                if error_msg:
                    # print(f"区域 {region_id} 识别出错: {error_msg}")
                    if matched_id is not None:
                        entry_widget = None
                        id_str = str(matched_id)
                        if region_id < 3: entry_widget = self.left_monsters.get(id_str)
                        else: entry_widget = self.right_monsters.get(id_str)

                        if entry_widget:
                            entry_widget.delete(0, tk.END)
                            entry_widget.insert(0, "错误")
                            entry_widget.config(bg="#FFCCCC")

                elif matched_id is not None and number is not None and matched_id != 0:
                    entry_widget = None
                    monster_id_str = str(matched_id)

                    if region_id < 3: entry_widget = self.left_monsters.get(monster_id_str)
                    else: entry_widget = self.right_monsters.get(monster_id_str)

                    if entry_widget:
                         entry_widget.delete(0, tk.END)
                         entry_widget.insert(0, str(number))
                         processed_monster_ids_for_debug.append(matched_id)
                    # else:
                         # print(f"警告: 无法找到怪物ID {monster_id_str} 对应的输入框。")

        if intelligent_workers_debug and self.auto_fetch_running and screenshot is not None:
            try:
                x1, y1 = self.main_roi[0]
                x2, y2 = self.main_roi[1]
                roi_x1, roi_y1 = min(x1, x2), min(y1, y2)
                roi_x2, roi_y2 = max(x1, x2), max(y1, y2)

                roi_image = screenshot[roi_y1:roi_y2, roi_x1:roi_x2]

                if roi_image.size == 0:
                     logger.warning("警告：截取的 ROI 区域图像为空。")
                else:
                    timestamp = int(time.time())
                    ids_str = "_".join(map(str, sorted(list(set(processed_monster_ids_for_debug)))))
                    self.current_image_name = f"{timestamp}_ids_{ids_str}.png" if ids_str else f"{timestamp}_ids_none.png"

                    target_width = 300
                    resized_roi = roi_image
                    if roi_image.shape[1] > target_width:
                        scale_ratio = target_width / roi_image.shape[1]
                        new_height = int(roi_image.shape[0] * scale_ratio)
                        if new_height > 0: # 确保高度有效
                             resized_roi = cv2.resize(roi_image, (target_width, new_height), interpolation=cv2.INTER_AREA)

                    self.current_image = resized_roi
                    # print(f"调试截图已暂存: {self.current_image_name}")

            except Exception as e:
                logger.error(f"保存调试用 ROI 截图时出错: {e}")
                self.current_image = None
                self.current_image_name = ""

        # print("识别完成，自动执行预测...")
        self.predict()

    def reselect_roi(self):

        logger.info("准备让用户重新选择识别区域 (ROI)...")
        try:
             selected_roi = recognize.select_roi()
             if selected_roi:
                 self.main_roi = selected_roi
                 self.no_region = False
                 logger.info(f"用户已选择新的识别区域: {self.main_roi}")
                 messagebox.showinfo("选择范围", f"已选择新的识别区域:\n左上角: {self.main_roi[0]}\n右下角: {self.main_roi[1]}")
             else:
                 logger.warning("用户取消选择或选择失败。")
        except Exception as e:
             logger.error(f"选择 ROI 时发生错误: {e}")
             messagebox.showerror("错误", f"选择识别范围时出错: {e}")

    # --- 训练 (占位) ---
    def start_training(self):

        messagebox.showinfo("提示", "即将开始在后台运行训练脚本 (train.py)...")
        training_thread = threading.Thread(target=self.train_model, daemon=True)
        training_thread.start()

    def train_model(self):

        try:
            logger.info("正在启动 train.py 脚本...")
            result = subprocess.run(["python", "train.py"], check=True, capture_output=True, text=True, encoding='utf-8') # 指定编码
            logger.info("train.py 脚本执行完成。")
            logger.info("训练脚本输出:\n", result.stdout)
            self.root.after(0, lambda: messagebox.showinfo("训练完成", "模型训练已完成。\n请考虑重新启动程序或添加“重新加载模型”功能。"))
        except FileNotFoundError:
             logger.error("错误: 找不到 'python' 命令或 'train.py' 文件。")
             self.root.after(0, lambda: messagebox.showerror("错误", "无法启动训练：找不到 Python 或 train.py。"))
        except subprocess.CalledProcessError as e:
            logger.error(f"训练脚本 train.py 执行失败，返回码: {e.returncode}")
            logger.error("错误输出:\n", e.stderr)
            error_message = f"训练脚本执行失败。\n错误信息:\n{e.stderr[:500]}..."
            self.root.after(0, lambda: messagebox.showerror("训练失败", error_message))
        except Exception as e:
            logger.error(f"启动训练时发生未知错误: {e}")
            self.root.after(0, lambda: messagebox.showerror("错误", f"启动训练时发生意外错误: {e}"))


    # --- 自动获取相关 ---
    def save_statistics_to_log(self):

        try:
            elapsed_time = time.time() - self.start_time if self.start_time else 0
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            stats_text = (
                f"--- Log Entry: {now} ---\n"
                f"模式: {self.game_mode.get()} {'(投资)' if self.is_invest.get() else '(观望)'}, 设备: {self.device_serial.get()}\n" # 合并一行
                f"填写次数: {self.total_fill_count}, 预测错误: {self.incorrect_fill_count}\n"
                f"运行时长: {int(hours)}h {int(minutes)}m {int(seconds)}s\n"
                f"---------------------------\n\n"
            )

            with open("log.txt", "a", encoding='utf-8') as log_file:
                log_file.write(stats_text)
            logger.info("统计信息已保存到 log.txt")

        except Exception as e:
            logger.error(f"保存统计日志时出错: {e}")


    def toggle_auto_fetch(self):
        if not self.auto_fetch_running:
            # 在启动自动获取前，检查 ROI 是否已定义
            if self.main_roi is None:
                logger.info("自动获取启动检查：ROI 未定义，尝试进行首次初始化...")
                try:
                    # 尝试执行首次识别时的默认 ROI 设置逻辑
                    # 确保 loadData 已尝试获取屏幕尺寸
                    if loadData.screen_width == 0 or loadData.screen_height == 0:
                        # 如果还没有尺寸信息，尝试获取一次
                        logger.info("尝试获取屏幕尺寸...")
                        loadData.get_screen_dimensions() # 假设 loadData 有此方法或类似逻辑
                        if loadData.screen_width == 0 or loadData.screen_height == 0:
                            raise ValueError("无法获取屏幕尺寸，无法设置默认ROI。")

                    # 使用 recognize 函数中首次识别的默认比例计算 ROI
                    default_x1_ratio = 0.248; default_y1_ratio = 0.841
                    default_x2_ratio = 0.753; default_y2_ratio = 0.951
                    self.main_roi = [
                        (int(default_x1_ratio * loadData.screen_width), int(default_y1_ratio * loadData.screen_height)),
                        (int(default_x2_ratio * loadData.screen_width), int(default_y2_ratio * loadData.screen_height))
                    ]
                    self.no_region = False # 标记 ROI 已设置
                    self.first_recognize = False # 标记已进行过首次尝试
                    logger.info(f"自动获取启动时成功设置默认 ROI: {self.main_roi}")

                    # 可选：尝试连接 ADB (如果 recognize 中有此逻辑且重要)
                    adb_path = loadData.adb_path
                    device_serial = loadData.get_device_serial()
                    if adb_path and device_serial:
                         connect_command = f'"{adb_path}" connect {device_serial}'
                         subprocess.run(connect_command, shell=True, capture_output=True, text=True, timeout=5) # 缩短超时

                except Exception as e:
                    logger.error(f"自动获取启动时初始化默认 ROI 失败: {e}")
                    messagebox.showerror("启动失败", f"无法自动设置识别区域(ROI): {e}\n请先手动点击一次“识别并预测”或“选择范围”。")
                    # 初始化失败，不启动自动获取
                    return # 退出 toggle_auto_fetch 方法

            # 检查 ADB 连接性
            if not loadData.is_adb_device_connected():
                 messagebox.showerror("错误", "无法启动自动模式：ADB设备未连接或无法访问。")
                 return

            self.auto_fetch_running = True
            self.auto_fetch_button.config(text="停止自动斗蛐蛐", relief=tk.SUNKEN, bg="#FFABAB")
            self.start_time = time.time()
            self.total_fill_count = 0
            self.incorrect_fill_count = 0
            self.update_statistics()

            try:
                duration_mins = float(self.duration_entry.get())
                self.training_duration = duration_mins * 60 if duration_mins > 0 else -1
                logger.info(f"设置自动运行时长: {'无限' if self.training_duration == -1 else f'{duration_mins} 分钟'}")
            except ValueError:
                messagebox.showerror("错误", "无效的时长输入。自动运行时长将设为无限。")
                self.training_duration = -1
                self.duration_entry.delete(0, tk.END)
                self.duration_entry.insert(0, "-1")

            # --- 禁用相关按钮 ---
            logger.info("自动模式启动，禁用手动操作按钮...")
            self.predict_button.config(state=tk.DISABLED)
            self.recognize_button.config(state=tk.DISABLED)
            # --- 禁用手动填写按钮 ---
            self.fill_data_correct_button.config(state=tk.DISABLED)
            self.fill_data_incorrect_button.config(state=tk.DISABLED)
            # --- 禁用其他设置按钮 ---
            self.reselect_button.config(state=tk.DISABLED)
            self.mode_menu.config(state=tk.DISABLED)
            self.invest_checkbox.config(state=tk.DISABLED) # 自动模式下也禁用观望选择
            self.duration_entry.config(state=tk.DISABLED)
            self.reset_button.config(state=tk.DISABLED) # 清空按钮也禁用

            self.auto_fetch_thread = threading.Thread(target=self.auto_fetch_loop, daemon=True)
            self.auto_fetch_thread.start()
            logger.info("自动斗蛐蛐线程已启动。按 ESC 键可停止。")

        else:
            self.auto_fetch_running = False # 先设置标志位
            # UI 更新和日志记录移到 stop_auto_fetch_ui_update
            logger.info("请求停止自动斗蛐蛐...")
            # 不直接在这里更新 UI，由 stop_auto_fetch_ui_update 统一处理

    def auto_fetch_loop(self):
        logger.info("自动获取循环开始...")
        normal_interval = 0.5
        fast_interval = 0.08
        fast_poll_timeout = 60
        while self.auto_fetch_running:
            loop_start_time = time.time()
            current_interval = normal_interval
            if self.is_in_battle:
                current_interval = fast_interval
                if self.fast_poll_start_time is None:
                    self.fast_poll_start_time = loop_start_time
                elif time.time() - self.fast_poll_start_time > fast_poll_timeout:
                    logger.warning(f"快速轮询超过 {fast_poll_timeout} 秒未检测到结算，恢复正常轮询。")
                    self.is_in_battle = False
                    self.fast_poll_start_time = None
                    current_interval = normal_interval
            try:
                self.auto_fetch_data()

                if self.training_duration != -1:
                    elapsed_time = time.time() - self.start_time
                    if elapsed_time >= self.training_duration:
                        logger.info(f"达到预设运行时长 ({self.training_duration / 60:.1f} 分钟)，自动停止。")
                        self.auto_fetch_running = False
                        self.root.after(0, lambda: self.stop_auto_fetch_ui_update("达到时长"))
                        break

            except Exception as e:
                logger.error(f"自动斗蛐蛐循环中发生错误: {e}")
                import traceback
                traceback.print_exc()

                logger.warning("发生错误，暂停5秒后继续...")
                for _ in range(50):
                    if not self.auto_fetch_running:
                        break
                    time.sleep(0.1)
                if not self.auto_fetch_running:
                    break

            loop_duration = time.time() - loop_start_time
            sleep_time = max(0, current_interval - loop_duration)
            sleep_end_time = time.time() + sleep_time
            while time.time() < sleep_end_time:
                if not self.auto_fetch_running:
                    break
                time.sleep(0.05)

            if not self.auto_fetch_running:
                logger.info("检测到停止标志，退出循环。")
                break

        logger.info("自动获取循环结束。")
        self.is_in_battle = False
        self.fast_poll_start_time = None
        self.root.after(0, lambda: self.stop_auto_fetch_ui_update("循环结束"))

    def stop_auto_fetch_ui_update(self, reason="未知"):
        if self.auto_fetch_button['text'] == "停止自动斗蛐蛐":
            self.auto_fetch_button.config(text="自动斗蛐蛐", relief=tk.RAISED, bg="#AED581")
            logger.info(f"自动斗蛐蛐已停止 ({reason})。")
            self.update_statistics()
            self.save_statistics_to_log()
            logger.info("自动模式停止，启用手动操作按钮...")
            self.predict_button.config(state=tk.NORMAL)
            self.recognize_button.config(state=tk.NORMAL)
            self.fill_data_correct_button.config(state=tk.NORMAL)
            self.fill_data_incorrect_button.config(state=tk.NORMAL)
            self.reselect_button.config(state=tk.NORMAL)
            self.mode_menu.config(state=tk.NORMAL)
            self._update_invest_option_on_mode_change()
            self.duration_entry.config(state=tk.NORMAL)
            self.reset_button.config(state=tk.NORMAL)
            logger.info("相关控件已重新启用。")

    def auto_fetch_data(self):
        relative_points = {
            "right_all_join_start": (0.9297, 0.8833), # 右ALL、返回主页、加入赛事、开始游戏
            "left_all":             (0.0713, 0.8833), # 左ALL
            "right_gift_entertain": (0.8281, 0.8833), # 右礼物、自娱自乐模式
            "left_gift":            (0.1640, 0.8833), # 左礼物
            "watch_this_round":     (0.4979, 0.6324), # 本轮观望
        }

        screenshot = loadData.capture_screenshot()
        if screenshot is None:
            logger.error("截图失败")
            time.sleep(2)
            return

        match_results = loadData.match_images(screenshot, loadData.process_templates_info)
        match_results = sorted(match_results, key=lambda x: x[1], reverse=True)
        best_match_idx, best_match_score = match_results[0]
        match_threshold = 0.6 # 事件匹配阈值，用于判断当前处于哪个页面

        if best_match_score < match_threshold:
            logger.warning(f"事件匹配最高分数 {best_match_score:.3f} 低于阈值 {match_threshold}，未匹配到明确事件，执行跳过。")
            time.sleep(1)
            return
        if best_match_idx in [6, 61, 7, 71, 14]:
            if not self.is_in_battle:
                logger.info("检测到进入战斗状态，准备快速轮询监测结算...")
                self.is_in_battle = True
                self.fast_poll_start_time = None  # 重置超时计时器
        # 如果检测到结算 (8-11) 或者其他非战斗状态，则退出战斗状态
        elif best_match_idx not in [6, 61, 7, 71, 14]:
            if self.is_in_battle:
                logger.info("检测到非战斗状态，恢复正常轮询。")
                self.is_in_battle = False
                self.fast_poll_start_time = None
        try:
            if best_match_idx == 0:
                logger.info("状态: 加入赛事 -> 点击")
                loadData.click(relative_points["right_all_join_start"])
                time.sleep(1.5)
            elif best_match_idx == 1:
                logger.info("状态: 主界面 -> 选择模式")
                if self.game_mode.get() == "30人":
                    logger.info("模式: 30人 -> 点击")
                    loadData.click(relative_points["left_all"]) # 点击30人模式按钮
                    loadData.click(relative_points["right_all_join_start"]) # 点击开始按钮
                    time.sleep(3)
                else:
                    logger.info("模式: 单人 -> 点击")
                    loadData.click(relative_points["right_gift_entertain"])
                    time.sleep(3)
            elif best_match_idx == 2:
                logger.info("状态: 开始游戏 -> 点击")
                loadData.click(relative_points["right_all_join_start"])
                time.sleep(3)
            elif best_match_idx in [3, 4, 5, 15]:
                logger.info(f"状态: 投资/观望 ({best_match_idx})")
                time.sleep(0.5)
                self.root.after(0, self.reset_entries)
                time.sleep(0.2)
                if best_match_idx == 15:
                    logger.info("当前已淘汰，只进行预测")
                logger.info("执行识别预测...")
                self.recognize()
                time.sleep(0.5)

                if not self.is_invest.get() and best_match_idx != 15:
                    logger.info("执行投资...")
                    if self.current_prediction > 0.5:
                        logger.info("投右 -> 点击")
                        loadData.click(relative_points["right_gift_entertain"])
                    else:
                        logger.info("投左 -> 点击")
                        loadData.click(relative_points["left_gift"])
                    logger.info("等待20s...")
                    time.sleep(20)
                elif best_match_idx == 15:
                    logger.info("当前已淘汰，只进行预测")
                else:
                    logger.info("执行观望 -> 点击")
                    loadData.click(relative_points["watch_this_round"])
                    time.sleep(5)
            elif best_match_idx in [8, 9, 10, 11]:
                logger.info(f"状态: 结算 (匹配索引 {best_match_idx}, 分数 {best_match_score:.3f}) -> 进入处理流程")
                time.sleep(0.1)

                if best_match_idx in [8, 11]:
                    is_left_winner = False
                else:
                    is_left_winner = True

                result_label = 'L' if is_left_winner else 'R'
                logger.info(f"判定结果: {'左胜' if is_left_winner else '右胜'} -> 准备填写 {result_label}")

                prediction_was_wrong = False
                if (is_left_winner and self.current_prediction > 0.5) or (
                        not is_left_winner and self.current_prediction <= 0.5):
                    logger.warning(
                        f"预测错误: 实际={'左胜' if is_left_winner else '右胜'}, 预测胜率={self.current_prediction:.3f} -> 计为错误")
                    self.incorrect_fill_count += 1
                    prediction_was_wrong = True
                else:
                    logger.info(
                        f"预测正确: 实际={'左胜' if is_left_winner else '右胜'}, 预测胜率={1-self.current_prediction:.3f}")

                self.fill_data(result_label)
                self.total_fill_count += 1
                self.root.after(0, self.update_statistics)

                logger.info("点击下一轮...")
                loadData.click(relative_points["right_all_join_start"])
                wait_after_fill = 8
                logger.info(f"等待 {wait_after_fill} 秒...")
                sleep_end = time.time() + wait_after_fill
                while time.time() < sleep_end:
                    if not self.auto_fetch_running:
                        break
                    time.sleep(0.1)
                if not self.auto_fetch_running:
                    self.is_in_battle = False
                    self.fast_poll_start_time = None
                    return
                self.is_in_battle = False
                self.fast_poll_start_time = None
            elif best_match_idx in [6, 7, 61, 71, 14]:
                logger.info("状态: 战斗中 -> 等待")
                if not self.is_in_battle:
                    time.sleep(3)
            elif best_match_idx in [12, 13]:
                logger.info("状态: 返回主页 -> 点击")
                loadData.click(relative_points["right_all_join_start"])
                time.sleep(3)
            else:
                logger.warning(f"状态: 未处理模板 {best_match_idx} (分数 {best_match_score:.3f}) -> 等待")
                time.sleep(2)
        except KeyError as e:
            logger.error(f"错误: 无效的键 {e} 用于 relative_points。")
            time.sleep(2)
        except Exception as e:
            logger.error(f"处理状态 {best_match_idx} (分数 {best_match_score:.3f}) 时出错: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(2)


    # --- 界面更新 ---
    def update_statistics(self):
        if self.start_time:
            elapsed_time = time.time() - self.start_time
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s" # 简化显示
        else:
            time_str = "未开始"

        accuracy = ((self.total_fill_count - self.incorrect_fill_count) / self.total_fill_count * 100) \
                   if self.total_fill_count > 0 else 0

        stats_text = (f"累计: {self.total_fill_count} | "
                      f"错误: {self.incorrect_fill_count} | "
                      f"准确率: {accuracy:.1f}% | "
                      f"时长: {time_str}")

        try:
             if hasattr(self, 'stats_label'):
                  self.stats_label.config(text=stats_text)
        except tk.TclError:
             pass


    def update_device_serial(self):
        new_serial = self.device_serial.get().strip()
        if not new_serial:
             messagebox.showwarning("提示", "设备序列号不能为空。")
             return

        logger.info(f"请求更新设备序列号为: {new_serial}")
        try:
            loadData.set_device_serial(new_serial)
            messagebox.showinfo("提示", f"已尝试更新模拟器序列号为: {new_serial}\n请确保设备已通过ADB连接。")
        except Exception as e:
            logger.error(f"更新设备序列号时出错: {e}")
            messagebox.showerror("错误", f"更新设备序列号时发生错误: {e}")


if __name__ == "__main__":
    root = None  # 初始化为 None
    app_instance = None
    try:
        root = tk.Tk()
        root.withdraw()  # 先隐藏默认的空白窗口

        # 设置 DPI 感知
        try:
            from ctypes import windll
            try:
                windll.shcore.SetProcessDpiAwareness(1)
                logger.info("设置 DPI 感知 (模式 1)")
            except AttributeError:
                windll.user32.SetProcessDPIAware()
                logger.info("设置 DPI 感知 (旧模式)")
        except Exception as e:
            logger.error(f"设置 DPI 感知失败: {e}")

        # 创建 App 实例 (会尝试加载模型)
        app_instance = ArknightsApp(root)

        # 检查模型是否加载成功，不成功则不显示窗口
        if not app_instance.model_loaded_successfully:
            logger.error("模型加载失败，程序退出。")
        else:
            # 模型加载成功，显示窗口
            root.deiconify()  # 显示主窗口

            def on_esc_press(event=None):
                logger.info("检测到 ESC 键...")
                # 确保 app_instance 存在且 auto_fetch_running 属性存在
                if app_instance and hasattr(app_instance, 'auto_fetch_running') and app_instance.auto_fetch_running:
                    logger.info("自动模式运行中，发送停止信号...")
                    app_instance.auto_fetch_running = False  # 设置标志位
                    # UI更新将在循环结束时或 stop_auto_fetch_ui_update 中处理
                else:
                    logger.info("不在自动模式下或App未完全初始化。")

            root.bind('<Escape>', on_esc_press)
            logger.info("ESC 键绑定成功，可在自动模式下按 ESC 停止。")

            # 进入主循环
            root.mainloop()

    except tk.TclError as e:
        # 捕获 Tkinter 相关的致命错误 (例如创建 root 失败)
        logger.critical(f"Tkinter 初始化或运行期间发生致命错误: {e}")
        # 尝试显示消息，但可能 Tkinter 已失效
        try:
            messagebox.showerror("界面错误", f"Tkinter 发生错误导致程序无法运行：\n{e}")
        except Exception:
            pass  # 忽略显示错误本身的问题
    except Exception as main_e:
        # 捕获其他致命错误
        logger.critical(f"应用程序启动或运行期间发生致命错误: {main_e}")
        import traceback
        traceback.print_exc()
        try:
            # 确保 root 存在才尝试显示 messagebox
            if root and root.winfo_exists():
                messagebox.showerror("致命错误", f"应用程序发生无法处理的错误：\n{main_e}\n\n请查看控制台输出。")
        except Exception:
            pass
    finally:
        # 确保程序退出时，如自动斗蛐蛐仍在运行，尝试停止它
        if app_instance and hasattr(app_instance, 'auto_fetch_running') and app_instance.auto_fetch_running:
            logger.info("程序退出，强制停止自动斗蛐蛐...")
            app_instance.auto_fetch_running = False
            # 可能需要保存最后的日志
            app_instance.save_statistics_to_log()

        logger.info("应用程序关闭。")

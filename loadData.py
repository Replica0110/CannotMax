# -*- coding: utf-8 -*-
import subprocess

import numpy as np
from loguru import logger

# --- 配置 ---
adb_path = r".\platform-tools\adb.exe"
# 默认设备序列号，可以在main.py中通过 set_device_serial 修改
manual_serial = '127.0.0.1:16384'
# 全局变量，将在初始化时设置
device_serial = None
screen_width = 0
screen_height = 0
process_templates_info = []

TEMPLATE_ROIS = {
    # 加入赛事
    0: (0.2747, 0.7304, 0.9795, 0.9424),
    # 主界面-30人按钮
    1: (0.8747, 0.6361, 0.9723, 0.9531),
    # 开始游戏
    2: (0.8783, 0.6361, 0.9723, 0.9596),
    # 投资/观望界面按钮
    3: (0.0169, 0.8075, 0.2313, 0.9617), # 左投资
    4: (0.7578, 0.8139, 0.9819, 0.9617), # 右投资
    5: (0.4482, 0.6019, 0.5506, 0.6726), # 观望按钮
    # 战斗中
    6: (0.4084, 0.9033, 0.5855, 0.9547), # 支持了左侧
    61:(0.4193, 0.9076, 0.5410, 0.9504), # 支持了右侧
    7: (0.4590, 0.9076, 0.5735, 0.9504), # all 了左边
    71:(0.4096, 0.9039, 0.5470, 0.9553), # all 了右边
    14: (0.3651, 0.8760, 0.6289, 0.9788), # 等待其他玩家结束
    # (结算界面)
    8: (0.7506, 0.2378, 0.9892, 0.6833), # 右胜利
    9: (0.0060, 0.1971, 0.2554, 0.7197), # 左胜利
    10: (0.0060, 0.1971, 0.2554, 0.7197),
    11: (0.7506, 0.2378, 0.9892, 0.6833),
    # 返回主页按钮
    12: (0.7711, 0.8525, 0.9771, 0.9896),
    13: (0.6313, 0.8525, 0.9759, 0.9938),
    # 观望
    15: (0.3590, 0.5226, 0.6325, 0.7047), # 已被淘汰，截取中间部分判断

}
# --- 设备序列号管理 ---
def set_device_serial(serial=None):
    """设置要使用的设备序列号，如果不传则列出可用设备供选择"""
    global manual_serial, device_serial

    if not serial:
        # 没有指定，列出设备供选择
        devices = list_connected_devices()
        if not devices:
            logger.warning("未发现任何连接的设备，请确认ADB连接正常。")
            manual_serial = None
            device_serial = None
            return

        logger.info("检测到以下设备：")
        for idx, dev in enumerate(devices):
            logger.info(f"{idx+1}: {dev}")

        # 提示用户选择
        try:
            selected = int(input(f"请输入设备序号 (1-{len(devices)})：")) - 1
            if 0 <= selected < len(devices):
                manual_serial = devices[selected]
            else:
                logger.warning("输入的序号无效，取消设置。")
                manual_serial = None
                device_serial = None
                return
        except ValueError:
            logger.warning("输入无效，取消设置。")
            manual_serial = None
            device_serial = None
            return
    else:
        manual_serial = serial
        logger.info(f"手动设置设备序列号为: {manual_serial}")

    # 尝试重新连接并验证
    device_serial = get_device_serial()
    if device_serial != manual_serial:
        logger.warning(f"警告：尝试设置 {manual_serial}，但实际连接/检测到的设备是 {device_serial}")
    else:
        logger.info(f"设备 {device_serial} 已确认。")



def is_adb_device_connected():
    """检查指定的设备是否可以通过ADB连接"""
    try:
        if not manual_serial: # 如果没有指定设备号
             logger.error("错误：未指定设备序列号 (manual_serial)。")
             return False

        subprocess.run(f'"{adb_path}" connect {manual_serial}', shell=True, check=True, capture_output=True, timeout=5)

        result = subprocess.run(
            f'"{adb_path}" devices',
            shell=True, capture_output=True, text=True, timeout=5
        )
        if manual_serial in result.stdout and 'device' in result.stdout.split(manual_serial)[1]:
             logger.info(f"设备 {manual_serial} 连接成功且状态正常。")
             return True
        else:
             logger.warning(f"设备 {manual_serial} 连接成功但状态异常或未在列表中。")
             return False
    except subprocess.TimeoutExpired:
        logger.error(f"ADB 连接命令超时 (设备: {manual_serial})。")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"ADB 命令执行失败 (设备: {manual_serial}): {e}")
        return False
    except FileNotFoundError:
        logger.error(f"错误: 找不到 ADB 程序 '{adb_path}'。")
        return False
    except Exception as e:
        logger.error(f"检查设备连接时发生未知错误: {str(e)}")
        return False

def get_device_serial():
    """
    尝试连接手动指定的设备，如果失败或未指定，则自动选择第一个可用设备。
    返回最终使用的设备序列号，如果找不到设备则返回 None。
    """
    global device_serial # 声明要修改全局变量
    target_serial = manual_serial #优先使用手动设置的

    # 1. 尝试连接并验证手动指定的设备 (如果 manual_serial 有值)
    if target_serial:
        logger.info(f"尝试连接并验证手动指定设备: {target_serial}")
        try:
            subprocess.run(f'"{adb_path}" connect {target_serial}', shell=True, check=True, capture_output=True, timeout=3)
            # 检查设备列表和状态
            result = subprocess.run(
                f'"{adb_path}" devices', shell=True, capture_output=True, text=True, timeout=5
            )
            if target_serial in result.stdout and 'device' in result.stdout.split(target_serial)[1]:
                logger.info(f"手动指定设备 {target_serial} 有效。")
                device_serial = target_serial # 更新全局变量
                return target_serial # 成功找到并验证
        except Exception as e:
            logger.error(f"验证手动设备 {target_serial} 失败: {e}。尝试自动检测...")
            target_serial = None # 清除目标，进入自动检测逻辑

    # 2. 如果手动指定失败或未指定，则自动检测
    if not target_serial:
        logger.info("尝试自动检测可用设备...")
        try:
            result = subprocess.run(
                f'"{adb_path}" devices', shell=True, capture_output=True, text=True, timeout=5
            )
            devices = []
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                for line in lines[1:]:
                    if '\tdevice' in line:
                        dev = line.split('\t')[0]
                        devices.append(dev)

            if devices:
                selected_device = devices[0] # 选择第一个
                logger.info(f"自动选择设备: {selected_device}")
                device_serial = selected_device # 更新全局变量
                # 尝试连接选中的设备
                try:
                    subprocess.run(f'"{adb_path}" connect {selected_device}', shell=True, check=True, capture_output=True, timeout=3)
                except Exception as connect_e:
                    logger.warning(f"警告：连接自动选择的设备 {selected_device} 时出错: {connect_e}")
                return selected_device
            else:
                logger.error("未找到任何连接的Android设备。")
                device_serial = None
                return None
        except Exception as e:
            logger.error(f"自动检测设备时失败: {str(e)}")
            device_serial = None
            return None
    return None

# --- 列出所有连接的设备 ---
def list_connected_devices():
    """返回一个列表，列出所有通过adb连接的设备"""
    try:
        result = subprocess.run(
            f'"{adb_path}" devices', shell=True, capture_output=True, text=True, timeout=5
        )
        devices = []
        lines = result.stdout.strip().split('\n')
        if len(lines) > 1:
            for line in lines[1:]:
                if '\tdevice' in line:
                    dev = line.split('\t')[0]
                    devices.append(dev)
        return devices
    except Exception as e:
        logger.error(f"列出设备时发生错误: {e}")
        return []

# --- 屏幕分辨率 ---
def get_screen_dimensions():
    """
    获取连接设备的屏幕分辨率 (宽度, 高度).
    优先获取物理分辨率，假定为横屏（宽度 > 高度）。
    如果失败则返回 None.
    """
    global device_serial
    if not device_serial:
        logger.info("错误：无法获取分辨率，设备序列号未确定。")
        return None

    try:
        # 执行ADB命令获取分辨率
        # 增加超时设置，check=True 用于检查返回码
        result = subprocess.run(
            f'"{adb_path}" -s {device_serial} shell wm size',
            shell=True,
            capture_output=True,
            text=True,
            check=True,
            timeout=5 # 5秒超时
        )
        output = result.stdout.strip()

        # 解析分辨率输出
        res_str = None
        if 'Physical size:' in output:
            res_str = output.split('Physical size: ')[1]
        elif 'Override size:' in output: # 备选方案
            res_str = output.split('Override size: ')[1]
        else:
            logger.warning(f"警告：无法识别的 'wm size' 输出格式:\n{output}")
            # 尝试从最后一行解析（某些模拟器可能格式不同）
            last_line = output.splitlines()[-1]
            if 'x' in last_line and last_line.replace('x', '').isdigit():
                 res_str = last_line
                 logger.info(f"尝试从最后一行解析得到: {res_str}")
            else:
                 raise ValueError("无法解析分辨率输出格式")

        # 分割分辨率并转换为整数
        width, height = map(int, res_str.split('x'))

        # 假定游戏是横屏，宽度 > 高度
        calculated_width = max(width, height)
        calculated_height = min(width, height)

        logger.info(f"成功获取模拟器分辨率: {calculated_width}x{calculated_height}")
        return calculated_width, calculated_height

    # 更具体的异常处理
    except subprocess.CalledProcessError as e:
        logger.error(f"ADB 'wm size' 命令执行失败 (设备: {device_serial}): {e}")
        return None
    except ValueError as e:
        logger.error(f"解析分辨率时出错 (设备: {device_serial}): {e}")
        return None
    except subprocess.TimeoutExpired:
        logger.error(f"ADB 'wm size' 命令超时 (设备: {device_serial})。")
        return None
    except FileNotFoundError:
        logger.error(f"错误: 找不到 ADB 程序 '{adb_path}'。")
        return None
    except Exception as e:
        logger.error(f"获取分辨率时发生未知错误 (设备: {device_serial}): {e}")
        return None


import os
import cv2


def load_process_images(target_width, target_height):
    """加载模板图像，并结合预定义的ROI信息"""
    logger.info(f"尝试加载模板图像并关联ROI...")
    loaded_info = []

    if target_width <= 0 or target_height <= 0:
        logger.error("错误：无效的目标尺寸，无法加载模板图像。")
        return []

    images_dir = 'images/process/'

    try:
        if not os.path.exists(images_dir):
            logger.error(f"错误：目录 {images_dir} 不存在。")
            return []

        # 获取目录下所有以 .png 结尾的文件，并按数字顺序排序
        image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]

        # 只保留文件名是纯数字的，例如 "0.png", "1.png"
        valid_files = []
        for f in image_files:
            name_without_ext = os.path.splitext(f)[0]
            if name_without_ext.isdigit():
                valid_files.append(f)

        # 按数字排序
        valid_files.sort(key=lambda x: int(os.path.splitext(x)[0]))

        if not valid_files:
            logger.warning("警告：未找到符合要求的模板图片。")
            return []

        for file_name in valid_files:
            img_id = int(os.path.splitext(file_name)[0])
            img_path = os.path.join(images_dir, file_name)

            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"警告：无法读取模板图片 {img_path}")
                continue

            roi = TEMPLATE_ROIS.get(img_id)  # 获取模板对应的ROI
            if roi is None:
                logger.warning(f"警告：模板 {img_id} 没有在 TEMPLATE_ROIS 中定义ROI，将无法进行匹配。")
                continue

            loaded_info.append({'id': img_id, 'image': img, 'roi': roi})

        logger.info(f"成功加载并关联了 {len(loaded_info)} 个模板及其ROI信息。")
        return loaded_info

    except cv2.error as e:
        logger.info(f"OpenCV 错误（可能在读取时发生）: {e}")
        return []
    except Exception as e:
        logger.error(f"加载模板图像或关联ROI时发生未知错误: {e}")
        return []


def capture_screenshot():
    """捕获当前设备的屏幕截图"""
    global device_serial
    if not device_serial:
        logger.error("错误：无法截图，设备序列号未确定。")
        return None
    try:
        screenshot_data = subprocess.check_output(
            f'"{adb_path}" -s {device_serial} exec-out screencap -p',
            shell=True
        )

        # 将二进制数据转换为numpy数组
        img_array = np.frombuffer(screenshot_data, dtype=np.uint8)

        # 使用OpenCV解码图像
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            logger.error("错误：解码截图数据失败。")
            return None
        return img
    except subprocess.CalledProcessError as e:
        logger.error(f"截图命令执行失败 (设备: {device_serial}): {e}")
        return None
    except subprocess.TimeoutExpired:
        logger.error(f"截图命令超时 (设备: {device_serial})。")
        return None
    except Exception as e:
        logger.error(f"处理截图时发生未知错误: {e}")
        return None


def match_images(screenshot, templates_info):
    """在截图的指定ROI内匹配模板列表"""
    global screen_width, screen_height # 需要全局尺寸来计算绝对坐标

    if screenshot is None or not templates_info or screen_width <= 0 or screen_height <= 0:
        logger.error("错误: 截图无效、无模板信息或屏幕尺寸未知，无法匹配。")
        return []

    results = []
    for template_info in templates_info:
        template_id = template_info['id']
        template_img = template_info['image']
        roi_rel = template_info['roi']

        if template_img is None or roi_rel is None:
            continue # 跳过无效的模板信息

        try:
            # 1. 计算ROI的绝对像素坐标
            x1_rel, y1_rel, x2_rel, y2_rel = roi_rel
            abs_x1 = int(x1_rel * screen_width)
            abs_y1 = int(y1_rel * screen_height)
            abs_x2 = int(x2_rel * screen_width)
            abs_y2 = int(y2_rel * screen_height)

            # 确保坐标有效且顺序正确
            abs_x1, abs_x2 = min(abs_x1, abs_x2), max(abs_x1, abs_x2)
            abs_y1, abs_y2 = min(abs_y1, abs_y2), max(abs_y1, abs_y2)

            # 防止坐标超出边界
            abs_x1 = max(0, abs_x1)
            abs_y1 = max(0, abs_y1)
            abs_x2 = min(screen_width, abs_x2)
            abs_y2 = min(screen_height, abs_y2)

            # 2. 检查ROI和模板尺寸
            roi_h = abs_y2 - abs_y1
            roi_w = abs_x2 - abs_x1
            tpl_h, tpl_w = template_img.shape[:2]

            if roi_h < tpl_h or roi_w < tpl_w:
                # logger.info(f"警告: 模板 {template_id} (尺寸 {tpl_w}x{tpl_h}) 比其定义的ROI (尺寸 {roi_w}x{roi_h}) 还大，无法匹配。跳过...")
                results.append((template_id, 0.0)) # 记录一个零分匹配
                continue

            # 3. 提取截图中的ROI区域
            screenshot_roi = screenshot[abs_y1:abs_y2, abs_x1:abs_x2]

            if screenshot_roi.size == 0:
                logger.warning(f"警告: 提取模板 {template_id} 的ROI时得到空图像。跳过...")
                results.append((template_id, 0.0))
                continue

            # 4. 在ROI内进行模板匹配
            # logger.info(f"正在匹配模板 {template_id} (尺寸 {tpl_w}x{tpl_h}) 在ROI [{abs_x1},{abs_y1},{abs_x2},{abs_y2}] (ROI尺寸 {roi_w}x{roi_h})")
            res = cv2.matchTemplate(screenshot_roi, template_img, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            # logger.info(f"模板 {template_id} 匹配得分: {max_val:.4f}")
            results.append((template_id, max_val))

        except cv2.error as e:
            logger.error(f"OpenCV 模板匹配错误 (模板 {template_id}): {e}")
            results.append((template_id, 0.0))
        except Exception as e:
             logger.error(f"匹配模板 {template_id} 时发生未知错误: {e}")
             import traceback
             traceback.logger.info_exc() # 打印详细错误信息
             results.append((template_id, 0.0))

    # 按得分降序排序
    results = sorted(results, key=lambda x: x[1], reverse=True)
    # logger.info(f"所有模板匹配完成，最佳匹配: ID={results[0][0]}, Score={results[0][1]:.4f}" if results else "无匹配结果")
    return results


def click(point):
    """根据相对坐标点击屏幕"""
    global device_serial, screen_width, screen_height
    if not device_serial or screen_width <= 0 or screen_height <= 0:
        logger.info("错误：无法点击，设备序列号或屏幕尺寸无效。")
        return

    try:
        x_rel, y_rel = point
        # 确保坐标在 0 到 1 之间
        x_rel = max(0.0, min(1.0, x_rel))
        y_rel = max(0.0, min(1.0, y_rel))

        x_coord = int(x_rel * screen_width)
        y_coord = int(y_rel * screen_height)
        logger.info(f"点击相对坐标 {point} -> 绝对坐标: ({x_coord}, {y_coord})")
        # 执行点击命令
        subprocess.run(f'"{adb_path}" -s {device_serial} shell input tap {x_coord} {y_coord}',
                       shell=True, check=True, capture_output=True, timeout=3)
    except subprocess.CalledProcessError as e:
        logger.error(f"ADB 点击命令失败: {e}")
    except subprocess.TimeoutExpired:
        logger.error(f"ADB 点击命令超时。")
    except Exception as e:
        logger.error(f"点击操作时发生未知错误: {e}")

def initialize_load_data():
    """执行模块初始化，确定设备和屏幕尺寸，并加载模板信息"""
    global device_serial, screen_width, screen_height, process_templates_info # 修改了全局变量名

    logger.info("开始初始化 loadData 模块 (含ROI)")
    # 1. 确定设备序列号 (不变)
    device_serial = get_device_serial()
    if not device_serial:
        logger.error("错误：未能找到或确定有效的 ADB 设备。初始化中止。")
        return

    logger.info(f"使用设备: {device_serial}")

    # 2. 获取屏幕尺寸 (不变)
    dimensions = get_screen_dimensions()
    if dimensions:
        screen_width, screen_height = dimensions
    else:
        logger.warning("警告：获取屏幕分辨率失败，将使用默认值 1920x1080。识别和点击可能不准确。")
        screen_width = 1920
        screen_height = 1080
    logger.info(f"屏幕尺寸: {screen_width}x{screen_height}")

    # 3. 加载模板及其ROI信息
    process_templates_info = load_process_images(screen_width, screen_height) # 注意这里不再传递尺寸用于调整，因为我们在内部处理
    if not process_templates_info:
         logger.warning("警告：未能成功加载模板信息。自动操作将受影响。")

    logger.info("loadData 模块初始化完成")

# --- 自动执行初始化 ---
initialize_load_data()

relative_points = {
    "right_all_join_start": (0.9297, 0.8833),  # 右ALL、返回主页、加入赛事、开始游戏
    "left_all":             (0.0713, 0.8833),  # 左ALL
    "right_gift_entertain": (0.8281, 0.8833),  # 右礼物、自娱自乐
    "left_gift":            (0.1640, 0.8833),  # 左礼物
    "watch_this_round":     (0.4979, 0.6324),  # 本轮观望
}


def operation_simple(results):
    for idx, score in results:
        if score > 0.6:
            if idx == 0:  # 加入赛事
                click(relative_points[0])
                logger.info("加入赛事")
            elif idx == 1:  # 自娱自乐
                click(relative_points[2])
                logger.info("自娱自乐")
            elif idx == 2:  # 开始游戏
                click(relative_points[0])
                logger.info("开始游戏")
            elif idx in [3, 4, 5]:  # 本轮观望
                click(relative_points[4])
                logger.info("本轮观望")
            elif idx in [10, 11]:
                logger.info("下一轮")
            elif idx in [6, 7]:
                logger.info("等待战斗结束")
            elif idx == 12:  # 返回主页
                click(relative_points[0])
                logger.info("返回主页")
            break  # 匹配到第一个结果后退出

def operation(results):
    for idx, score in results:
        if score > 0.6:
            if idx in [3, 4, 5]:
                # 识别怪物类型数量，导入模型进行预测
                prediction = 0.6
                # 根据预测结果点击投资左/右
                if prediction > 0.5:
                    click(relative_points[1])  # 投资右
                    logger.info("投资右")
                else:
                    click(relative_points[0])  # 投资左
                    logger.info("投资左")
            elif idx in [1, 5]:
                click(relative_points[2])  # 点击省点饭钱
                logger.info("点击省点饭钱")
            elif idx == 2:
                click(relative_points[3])  # 点击敬请见证
                logger.info("点击敬请见证")
            elif idx in [3, 4]:
                # 保存数据
                click(relative_points[4])  # 点击下一轮
                logger.info("点击下一轮")
            elif idx == 6:
                logger.info("等待战斗结束")
            break  # 匹配到第一个结果后退出


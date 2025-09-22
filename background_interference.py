
import math

from typing import Tuple

import numpy as np

import cv2
import config

def _remove_background_interference(mask: np.ndarray, img_shape: Tuple[int, int], debug: bool = False, dbg_save=None) -> np.ndarray:
    """
    去除背景影响，保留球体区域。
    
    策略：
    1. 分析连通域的几何特征（面积、圆形度、紧凑度等）
    2. 根据球体的典型特征过滤连通域
    3. 去除明显的背景噪声（如边缘附着、线性结构等）
    4. 保留最符合球体特征的区域
    
    Args:
        mask: 输入的二值掩码
        img_shape: 原始图像形状 (高, 宽)
        debug: 调试模式
        dbg_save: 调试保存函数
    
    Returns:
        处理后的掩码，去除背景干扰
    """
    if mask is None or mask.size == 0:
        return mask
    
    # 从config读取参数，如果没有则使用默认值
    min_sphere_area = int(getattr(config, 'BG_MIN_SPHERE_AREA', 100) or 100)
    max_sphere_area_ratio = float(getattr(config, 'BG_MAX_SPHERE_AREA_RATIO', 0.8) or 0.8)  # 最大占图像面积比例
    min_circularity = float(getattr(config, 'BG_MIN_CIRCULARITY', 0.3) or 0.3)  # 最小圆形度
    min_compactness = float(getattr(config, 'BG_MIN_COMPACTNESS', 0.2) or 0.2)  # 最小紧凑度
    border_exclusion_ratio = float(getattr(config, 'BG_BORDER_EXCLUSION', 0.05) or 0.05)  # 边界排除比例
    
    h_img, w_img = img_shape
    max_sphere_area = h_img * w_img * max_sphere_area_ratio
    border_margin = int(min(h_img, w_img) * border_exclusion_ratio)
    
    # 寻找所有连通域
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    
    # 创建输出掩码
    output_mask = np.zeros_like(mask)
    
    # 评估每个连通域
    valid_contours = []
    contour_scores = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        # 1. 面积过滤
        if area < min_sphere_area or area > max_sphere_area:
            continue
        
        # 2. 边界检查：排除紧贴图像边界的区域
        x, y, w, h = cv2.boundingRect(contour)
        # if (x <= border_margin or y <= border_margin or 
        #     x + w >= w_img - border_margin or y + h >= h_img - border_margin):
        #     continue
        
        # 3. 几何特征分析
        # 计算圆形度 (4π*面积/周长²)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        
        # 计算紧凑度 (面积/外接矩形面积)
        rect_area = w * h
        compactness = area / max(rect_area, 1e-6)
        
        # 计算长宽比
        aspect_ratio = max(w, h) / max(min(w, h), 1e-6)
        
        # 计算凸包面积比
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        convexity = area / max(hull_area, 1e-6)
        
        # 4. 球体特征评分
        score = 0.0
        
        # 圆形度评分 (0-40分)
        if circularity >= min_circularity:
            score += min(40, circularity * 40)
        
        # 紧凑度评分 (0-30分)
        if compactness >= min_compactness:
            score += min(30, compactness * 30)
        
        # 长宽比评分 (0-20分) - 接近1的长宽比更好
        aspect_score = max(0, 20 - abs(aspect_ratio - 1.0) * 10)
        score += aspect_score
        
        # 凸性评分 (0-10分) - 凸性好的形状更像球体
        score += convexity * 10
        
        # 保存有效轮廓
        if score > 20:  # 最低分数阈值
            valid_contours.append(contour)
            contour_scores.append(score)
            
            if debug:
                print(f"球体候选: 面积={area:.1f}, 圆形度={circularity:.3f}, "
                      f"紧凑度={compactness:.3f}, 长宽比={aspect_ratio:.2f}, "
                      f"凸性={convexity:.3f}, 评分={score:.1f}")
    
    # 5. 选择最佳候选
    if valid_contours:
        if len(valid_contours) == 1:
            # 只有一个候选，直接使用
            cv2.drawContours(output_mask, valid_contours, -1, 255, -1)
        else:
            # 多个候选，选择评分最高的
            best_idx = np.argmax(contour_scores)
            best_contour = valid_contours[best_idx]
            cv2.drawContours(output_mask, [best_contour], -1, 255, -1)
            
            if debug:
                print(f"选择最佳球体候选 (评分: {contour_scores[best_idx]:.1f})")
    
    # 6. 可选的后处理：轻微形态学平滑
    if np.any(output_mask):
        # 小的闭运算来平滑轮廓
        kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        output_mask = cv2.morphologyEx(output_mask, cv2.MORPH_CLOSE, kernel_smooth, iterations=1)
    
    # 保存调试信息
    if debug and dbg_save:
        dbg_save(output_mask, 'background_removed_mask.png')
        if len(valid_contours) > 1:
            # 保存所有候选区域的对比图
            all_candidates_mask = np.zeros_like(mask)
            cv2.drawContours(all_candidates_mask, valid_contours, -1, 255, -1)
            dbg_save(all_candidates_mask, 'all_sphere_candidates.png')
    
    return output_mask
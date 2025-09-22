import math
from typing import List, Optional, Tuple, Dict
from typing import Tuple

import numpy as np

import cv2
import config

def _is_circle_acceptable(contour: np.ndarray, debug: bool = False) -> dict:
    """
    检测轮廓是否为可接受的圆形（例如检查是否有由强光照导致的缺口）。
    
    Args:
        contour: 输入的轮廓
        debug: 是否启用调试模式
    
    Returns:
        dict: 包含检测结果和详细信息的字典
              {
                  'is_acceptable': bool,
                  'convexity_ratio': float,
                  'circularity_ratio': float,
                  'min_convexity_threshold': float,
                  'min_circularity_threshold': float
              }
    """
    # 检查是否启用圆形检测功能
    if not getattr(config, 'ENABLE_CIRCLE_FIX', True):
        return {
            'is_acceptable': True,
            'check_enabled': False,
            'reason': 'circle_check_disabled'
        }
    
    try:
        # 1. 计算轮廓的凸包和圆形拟合
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        contour_area = cv2.contourArea(contour)
        
        # 计算最小外接圆
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        circle_area = math.pi * radius * radius
        
        # 2. 检测是否符合圆形要求的判据
        # - 轮廓面积与凸包面积比值（检查是否有明显凹陷）
        # - 轮廓面积与外接圆面积比值（检查圆形度）
        convexity_ratio = contour_area / max(hull_area, 1e-6)
        circularity_ratio = contour_area / max(circle_area, 1e-6)
        
        # 从config读取阈值，如果没有则使用默认值
        min_convexity = float(getattr(config, 'FIX_MIN_CONVEXITY', 0.85) or 0.85)
        min_circularity = float(getattr(config, 'FIX_MIN_CIRCULARITY', 0.75) or 0.75)
        
        is_acceptable = (convexity_ratio >= min_convexity) and (circularity_ratio >= min_circularity)
        
        result = {
            'is_acceptable': is_acceptable,
            'check_enabled': True,
            'convexity_ratio': round(convexity_ratio, 4),
            'circularity_ratio': round(circularity_ratio, 4),
            'min_convexity_threshold': min_convexity,
            'min_circularity_threshold': min_circularity,
            'contour_area': round(contour_area, 2),
            'hull_area': round(hull_area, 2),
            'circle_area': round(circle_area, 2),
            'circle_center': [round(cx, 2), round(cy, 2)],
            'circle_radius': round(radius, 2)
        }
        
        if not is_acceptable:
            reasons = []
            if convexity_ratio < min_convexity:
                reasons.append(f'low_convexity({convexity_ratio:.3f}<{min_convexity})')
            if circularity_ratio < min_circularity:
                reasons.append(f'low_circularity({circularity_ratio:.3f}<{min_circularity})')
            result['reject_reasons'] = reasons
        
        return result
        
    except Exception as e:
        if debug:
            print(f"圆形检测失败: {e}")
        return {
            'is_acceptable': True,
            'check_enabled': True,
            'error': str(e),
            'reason': 'check_failed_default_accept'
        }

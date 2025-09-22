import math
from typing import List, Optional, Tuple, Dict
from typing import Tuple

import numpy as np

import cv2
import config

def _fix_specular_gaps(contour: np.ndarray,
                       img_shape: Tuple[int, int],
                       debug: bool = False,
                       dbg_save=None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], dict]:
    """
    修复因高光导致的边缘缺口，返回 (repaired_mask, repaired_contour, info)。

    策略:
    - 计算凸性(面积/凸包面积)与最大角度缺口。
    - 若凸性偏低或存在较大角度缺口，则采用凸包填补；
      否则进行小尺度形态学闭运算以桥接微小孔洞。
    - 始终返回修复后的掩码与轮廓；若无需修复则返回原样。
    """
    try:
        if contour is None or len(contour) < 5:
            return None, None, {'applied': 'none', 'reason': 'too_few_points'}

        h_img, w_img = img_shape[:2]

        # 基本度量
        area = float(cv2.contourArea(contour))
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        convexity_ratio = (area / max(hull_area, 1e-6)) if hull_area > 0 else 1.0

        # 角度缺口评估
        M = cv2.moments(contour)
        if M.get('m00', 0) != 0:
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
        else:
            pts_tmp = contour.reshape(-1, 2).astype(np.float64)
            cx, cy = pts_tmp[:, 0].mean(), pts_tmp[:, 1].mean()

        pts = contour.reshape(-1, 2).astype(np.float64)
        rel = pts - np.array([cx, cy])
        phi = np.arctan2(rel[:, 1], rel[:, 0])  # [-π, π]
        phi = np.where(phi < 0, phi + 2 * np.pi, phi)  # [0, 2π)
        n_bins = int(getattr(config, 'SPEC_GAP_BINS', 360) or 360)
        bins = np.linspace(0.0, 2.0 * np.pi, n_bins, endpoint=False)
        occ = np.zeros(n_bins, dtype=bool)
        idx = np.floor(phi / (2.0 * np.pi) * n_bins).astype(int)
        idx = np.clip(idx, 0, n_bins - 1)
        occ[idx] = True
        if occ.all():
            max_gap_deg = 0.0
        else:
            # 处理环形的最长零段
            occ_int = occ.astype(np.int32)
            # 将数组翻倍，找连续0的最长段
            doubled = np.r_[occ_int, occ_int]
            max_zero = 0
            cur_zero = 0
            for v in doubled:
                if v == 0:
                    cur_zero += 1
                    max_zero = max(max_zero, cur_zero)
                else:
                    cur_zero = 0
            max_zero = min(max_zero, n_bins)  # 跨边界最长为 n_bins
            max_gap_deg = max_zero * (360.0 / n_bins)

        # 阈值
        gap_thr_deg = float(getattr(config, 'SPEC_GAP_DEG_THR', 15.0) or 15.0)
        min_convexity = float(getattr(config, 'SPEC_FIX_MIN_CONVEXITY', 0.90) or 0.90)

        need_hull = (convexity_ratio < min_convexity) or (max_gap_deg > gap_thr_deg)

        # 原始mask
        base_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.drawContours(base_mask, [contour], -1, 255, -1)

        # 创建边界限制蒙版，对靠近画面边缘的区域不做处理
        border_margin = int(getattr(config, 'SPEC_FIX_BORDER_MARGIN', 50) or 50)
        border_safe_mask = np.ones((h_img, w_img), dtype=np.uint8) * 255
        border_safe_mask[:border_margin, :] = 0  # 上边缘
        border_safe_mask[-border_margin:, :] = 0  # 下边缘
        border_safe_mask[:, :border_margin] = 0  # 左边缘
        border_safe_mask[:, -border_margin:] = 0  # 右边缘

        info = {
            'applied': 'none',
            'convexity_ratio': round(convexity_ratio, 4),
            'max_angle_gap_deg': round(float(max_gap_deg), 2),
            'gap_thr_deg': gap_thr_deg,
            'min_convexity_thr': min_convexity,
            'border_margin': border_margin
        }

        if need_hull:
            hull_mask = np.zeros_like(base_mask)
            cv2.fillPoly(hull_mask, [hull.astype(np.int32)], 255)
            
            # 应用边界限制：只在安全区域进行凸包修复，边界区域保持原样
            safe_hull_repair = cv2.bitwise_and(hull_mask, border_safe_mask)
            border_original = cv2.bitwise_and(base_mask, cv2.bitwise_not(border_safe_mask))
            repaired_mask = cv2.bitwise_or(safe_hull_repair, border_original)
            
            # 平滑一下边界
            ksz = int(max(3, round(min(h_img, w_img) * 0.006)))
            if ksz % 2 == 0:
                ksz += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
            try:
                # 只对安全区域进行形态学处理
                safe_region_smooth = cv2.morphologyEx(repaired_mask, cv2.MORPH_OPEN, kernel, iterations=1)
                safe_region_smooth = cv2.morphologyEx(safe_region_smooth, cv2.MORPH_CLOSE, kernel, iterations=1)
                safe_region_smooth = cv2.bitwise_and(safe_region_smooth, border_safe_mask)
                # 合并边界原始区域
                repaired_mask = cv2.bitwise_or(safe_region_smooth, border_original)
            except Exception:
                pass

            contours_new, _ = cv2.findContours(repaired_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours_new:
                repaired_contour = max(contours_new, key=cv2.contourArea)
            else:
                repaired_contour = hull.astype(np.int32)

            info['applied'] = 'convex_hull'
            info['kernel'] = ksz

            if debug and dbg_save is not None:
                dbg_save(repaired_mask, 'gap_fix_hull.png')

            return repaired_mask, repaired_contour, info

        # 否则做轻度闭运算桥接微小空洞
        x, y, w, h = cv2.boundingRect(contour)
        morph_factor = float(getattr(config, 'MORPH_CLOSE_FACTOR', 0.08) or 0.08)
        local_scale = max(3, int(round(min(w, h) * morph_factor)))
        if local_scale % 2 == 0:
            local_scale += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (local_scale, local_scale))
        
        # 应用边界限制：只在安全区域进行形态学闭运算
        safe_close_repair = cv2.morphologyEx(base_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        safe_close_repair = cv2.bitwise_and(safe_close_repair, border_safe_mask)
        border_original = cv2.bitwise_and(base_mask, cv2.bitwise_not(border_safe_mask))
        repaired_mask = cv2.bitwise_or(safe_close_repair, border_original)
        
        contours_new, _ = cv2.findContours(repaired_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_new:
            repaired_contour = max(contours_new, key=cv2.contourArea)
        else:
            # 回退
            repaired_mask = base_mask
            repaired_contour = contour

        info['applied'] = 'morph_close'
        info['kernel'] = local_scale

        if debug and dbg_save is not None:
            dbg_save(repaired_mask, 'gap_fix_close.png')

        return repaired_mask, repaired_contour, info

    except Exception as e:
        if debug:
            print(f"高光缺口修复失败: {e}")
        return None, None, {'applied': 'none', 'error': str(e)}



from __future__ import annotations

import argparse
import csv
import math
import os
from typing import List, Optional, Tuple, Dict

import numpy as np
import json
import cv2
import config

def _save_legendre_plot(legendre_info: dict, out_dir: str) -> Optional[str]:
    """在 debug 目录保存勒让德拟合的曲线图像。
    输出一张包含两幅子图的图片：
    - 左：R(φ) vs φ (0..2π) 
    - 右：极坐标下的拟合形状 (0..2π)
    返回保存的文件路径（若失败返回 None）。
    """
    try:
        import os
        import numpy as np
        from numpy.polynomial.legendre import legval
        import matplotlib.pyplot as plt

        coeffs = np.asarray(legendre_info.get('legendre_coeffs', []), dtype=float)
        if coeffs.size == 0:
            return None

        # φ∈[0,2π) 的 R(φ) 直接拟合结果
        phi = np.linspace(0.0, 2.0 * np.pi, 721, endpoint=False)
        r_phi = np.clip(legval(np.cos(phi), coeffs), 0.0, None)

        fig = plt.figure(figsize=(10, 4))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(np.degrees(phi), r_phi, color='tab:blue', lw=2)
        ax1.set_xlabel('φ (deg)')
        ax1.set_ylabel('R (px)')
        ax1.set_title('R(φ) from Legendre fit (non-folded)')
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(1, 2, 2, projection='polar')
        ax2.plot(phi, r_phi, color='tab:green', lw=2)
        ax2.set_title('Polar shape from R(φ)')

        expr = legendre_info.get('legendre_expression')
        if isinstance(expr, str) and len(expr) <= 120:
            fig.suptitle(expr, fontsize=9)

        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'legendre_fit.png')
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return out_path
    except Exception:
        return None

def _check_legendre_boundary(fitted_contour: np.ndarray, img_shape: Tuple[int, int], 
                            debug: bool = False) -> bool:
    """
    检查勒让德拟合的轮廓是否超出图像边界。
    
    Args:
        fitted_contour: 拟合得到的轮廓点 (N, 2)
        img_shape: 图像尺寸 (height, width)
        debug: 调试模式
    
    Returns:
        bool: True表示轮廓在边界内，False表示超出边界
    """
    if fitted_contour is None or len(fitted_contour) == 0:
        return False
    
    height, width = img_shape[:2]
    
    # 从配置文件获取参数
    margin = getattr(config, 'LEGENDRE_BOUNDARY_MARGIN', 5)
    boundary_threshold = getattr(config, 'LEGENDRE_BOUNDARY_THRESHOLD', 0.05)
    
    # 检查轮廓点是否超出边界
    x_coords = fitted_contour[:, 0]
    y_coords = fitted_contour[:, 1]
    
    # 检查是否有点超出边界（考虑容差）
    x_out_of_bounds = (x_coords < margin) | (x_coords >= width - margin)
    y_out_of_bounds = (y_coords < margin) | (y_coords >= height - margin)
    
    out_of_bounds_count = np.sum(x_out_of_bounds | y_out_of_bounds)
    total_points = len(fitted_contour)
    out_of_bounds_ratio = out_of_bounds_count / total_points if total_points > 0 else 1.0
    
    # 如果超过阈值比例的点超出边界，则认为拟合不合格
    is_within_bounds = out_of_bounds_ratio <= boundary_threshold
    
    if debug:
        print(f"边界检查: 总点数={total_points}, 超界点数={out_of_bounds_count}, "
              f"超界比例={out_of_bounds_ratio:.3f}, 阈值={boundary_threshold}, "
              f"容差={margin}像素, 通过={is_within_bounds}")
    
    return is_within_bounds


def _calculate_legendre_volume(contour: np.ndarray, img_shape: Optional[Tuple[int, int]] = None, 
                              debug: bool = False) -> Tuple[Optional[float], Optional[dict]]:
    """
    六阶勒让德多项式拟合轮廓的径向函数 R(φ)，并计算体积。

    修改后算法（非折叠版本）:
    - 以轮廓质心为极点，统计 r(φ), φ∈[0,2π) 的外边界半径(角度分箱取最大)。
    - 直接在整个 [0,2π) 角度空间进行拟合: R(φ) ≈ Σ_{n=0..6} c_n P_n(cos(φ))。
    - 基于拟合轮廓面积计算等效球体积，适合非对称物体。
    - 检查拟合曲线是否超出图像边界，如果超出则舍弃该拟合。

    Args:
        contour: 输入轮廓
        img_shape: 图像尺寸 (height, width)，用于边界检查
        debug: 调试模式

    返回 (volume_px3, info)。
    """
    try:
        if contour is None or len(contour) < 10:
            return None, None

        # 质心
        M = cv2.moments(contour)
        if M.get('m00', 0) == 0:
            return None, None
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']

        pts = contour.reshape(-1, 2).astype(np.float64)
        rel = pts - np.array([cx, cy])
        phi = np.arctan2(rel[:, 1], rel[:, 0])  # [-π, π]
        phi = np.where(phi < 0, phi + 2 * np.pi, phi)  # [0, 2π)
        rad = np.hypot(rel[:, 0], rel[:, 1])

        # 角度分箱，取最大半径近似外边界
        n_phi_bins = int(getattr(config, 'LEGENDRE_ANGLE_SAMPLES', 360) or 360)
        bins = np.linspace(0.0, 2.0 * np.pi, n_phi_bins + 1)
        idx = np.clip(np.digitize(phi, bins) - 1, 0, n_phi_bins - 1)
        r_per_bin = np.zeros(n_phi_bins, dtype=np.float64)
        has_val = np.zeros(n_phi_bins, dtype=bool)
        for i, r in zip(idx, rad):
            if not has_val[i] or r > r_per_bin[i]:
                r_per_bin[i] = r
                has_val[i] = True
        if not np.all(has_val):
            ii = np.arange(n_phi_bins)
            good = has_val
            if good.sum() >= 2:
                r_per_bin[~good] = np.interp(ii[~good], ii[good], r_per_bin[good])
            else:
                fill_val = float(np.nanmean(np.where(has_val, r_per_bin, np.nan))) if has_val.any() else 0.0
                r_per_bin[~good] = fill_val

        phi_centers = (bins[:-1] + bins[1:]) * 0.5  # 长度 n_phi_bins, [0,2π)

        # 直接在整个 [0,2π) 空间进行拟合，不再折叠
        phi = phi_centers  # 角度范围 [0,2π)
        r_phi = r_per_bin  # 对应的半径值

        # 构造勒让德范德蒙德矩阵并最小二乘拟合
        # 使用 cos(φ) 作为自变量，φ∈[0,2π) -> cos(φ)∈[-1,1]
        from numpy.polynomial.legendre import legvander, legval, leggauss
        x = np.cos(phi)  # φ∈[0,2π) -> x∈[-1,1]
        deg = 6
        V = legvander(x, deg)  # 形状 (n_phi_bins, deg+1)
        # 最小二乘求解
        c, *_ = np.linalg.lstsq(V, r_phi, rcond=None)  # c[0..6]

        # 拟合质量(R^2)
        r_pred = V @ c
        ss_res = float(np.sum((r_phi - r_pred) ** 2))
        ss_tot = float(np.sum((r_phi - np.mean(r_phi)) ** 2))
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # 体积计算：对于2D轮廓的3D体积估算
        # 方法1：基于面积的体积估算 V ≈ A^(3/2) / √π
        # 其中 A 是拟合轮廓的面积
        
        # 计算拟合轮廓的面积（使用梯形积分）
        # 对于极坐标：A = (1/2) ∫ r²(φ) dφ
        area_fitted = 0.5 * np.trapz(r_pred ** 2, phi)
        
        # 基于面积的体积估算（假设近似球形）
        # V = (4/3)π * r_equivalent³，其中 r_equivalent = √(A/π)
        r_equivalent = math.sqrt(area_fitted / math.pi)
        volume_px3 = (4.0 / 3.0) * math.pi * (r_equivalent ** 3)
        
        # 或者使用更保守的估算：V ≈ A^(3/2) / √π
        # volume_px3 = float(area_fitted ** 1.5 / math.sqrt(math.pi))
        
        volume_px3 = float(volume_px3)

        # 生成可视化轮廓：直接使用拟合结果
        n_draw = max(n_phi_bins, 360)
        phi_draw = np.linspace(0.0, 2.0 * np.pi, n_draw, endpoint=False)
        r_draw = np.clip(legval(np.cos(phi_draw), c), 0.0, None)
        xs = cx + r_draw * np.cos(phi_draw)
        ys = cy + r_draw * np.sin(phi_draw)
        fitted_contour = np.vstack([xs, ys]).T.astype(np.int32)

        # 边界检查：如果提供了图像尺寸，检查拟合曲线是否超出边界
        if img_shape is not None:
            is_within_bounds = _check_legendre_boundary(fitted_contour, img_shape, debug=debug)
            if not is_within_bounds:
                if debug:
                    print(f"勒让德拟合超出边界，舍弃此拟合结果")
                return None, None

        # 生成表达式字符串
        coeffs_str = [f"{v:.6g}" for v in c.tolist()]
        terms = [f"{coeffs_str[n]}*P{n}(cos(φ))" for n in range(deg + 1)]
        expr_str = "R(φ) = " + " + ".join(terms)

        info = {
            'center_x': float(cx),
            'center_y': float(cy),
            'legendre_degree': int(deg),
            'legendre_coeffs': [float(v) for v in c.tolist()],
            'legendre_expression': expr_str,
            'n_angle_samples': int(n_draw),
            'mean_radius_px': float(np.mean(r_draw)) if r_draw.size else None,
            'max_radius_px': float(np.max(r_draw)) if r_draw.size else None,
            'min_radius_px': float(np.min(r_draw)) if r_draw.size else None,
            'r_squared': float(r_squared),
            'fitted_contour': fitted_contour.tolist(),
            'volume_legendre_px3': float(volume_px3),
            'volume_legendre_full_px3': float(volume_px3),  # 完整体积与计算体积相同

            'fitting_method': 'legendre_poly_deg6_full_contour',
            'boundary_check_passed': img_shape is None or True  # 如果到这里说明通过了边界检查
        }

        if debug:
            print(f"勒让德拟合(6阶): R²={r_squared:.4f}, 体积={volume_px3:.2f}")

        return volume_px3, info

    except Exception as e:
        if debug:
            print(f"勒让德拟合失败: {e}")
        return None, None


def _draw_spherical_cap_cross_section(bgr: np.ndarray, center: Tuple[int, int], 
                                     legendre_info: dict, mask_arr: np.ndarray, 
                                     theta_samples: np.ndarray, r_actual: np.ndarray, 
                                     r_fitted: np.ndarray) -> None:
    """
    在overlay图像中绘制球冠截面。
    
    Args:
        bgr: BGR图像
        center: 轮廓中心点 (x, y)
        legendre_info: 勒让德拟合信息
        mask_arr: 缺损mask数组
        theta_samples: 角度采样数组
        r_actual: 实际半径数组
        r_fitted: 拟合半径数组
    """
    try:
        if not np.any(mask_arr):
            return
            
        # 获取缺损区域的角度和半径数据
        defect_indices = np.where(mask_arr)[0]
        if len(defect_indices) == 0:
            return
            
        # 直接使用缺损区域的实际半径作为截面位置
        defect_theta = theta_samples[mask_arr]
        defect_r_actual = r_actual[mask_arr]
        defect_r_fitted = r_fitted[mask_arr]
        
        # 获取缺损角度范围（从legendre_info中获取，确保一致性）
        thL_deg = legendre_info.get('defect_angle_left_deg')
        thR_deg = legendre_info.get('defect_angle_right_deg') 
        thC_deg = legendre_info.get('defect_angle_center_deg')
        
        if thL_deg is None or thR_deg is None or thC_deg is None:
            return
            
        # 计算截面参数
        avg_fitted_radius = float(np.mean(defect_r_fitted))
        avg_actual_radius = float(np.mean(defect_r_actual))
        avg_depth = avg_fitted_radius - avg_actual_radius
        
        if avg_depth <= 0:
            return
            
        # 方法1：绘制实际观测的截面轮廓（使用实际半径）
        截面点集 = []
        for i, theta in enumerate(defect_theta):
            x = int(center[0] + defect_r_actual[i] * math.cos(theta))
            y = int(center[1] + defect_r_actual[i] * math.sin(theta))
            截面点集.append((x, y))
        
        # 绘制截面轮廓（红色粗线）
        if len(截面点集) > 1:
            截面点数组 = np.array(截面点集, dtype=np.int32)
            cv2.polylines(bgr, [截面点数组], False, (0, 0, 255), 3)
        
        # 方法2：绘制拟合表面的轮廓（用于对比）
        拟合点集 = []
        for i, theta in enumerate(defect_theta):
            x = int(center[0] + defect_r_fitted[i] * math.cos(theta))
            y = int(center[1] + defect_r_fitted[i] * math.sin(theta))
            拟合点集.append((x, y))
            
        # 绘制拟合表面轮廓（蓝色细线）
        if len(拟合点集) > 1:
            拟合点数组 = np.array(拟合点集, dtype=np.int32)
            cv2.polylines(bgr, [拟合点数组], False, (255, 100, 0), 1)
        
        # 方法3：在几个关键角度绘制深度指示线
        关键角度 = [thL_deg, thC_deg, thR_deg]
        for angle_deg in 关键角度:
            if angle_deg is None:
                continue
                
            angle_rad = math.radians(angle_deg)
            
            # 找到最接近这个角度的数据点
            angle_diff = np.abs(defect_theta - angle_rad)
            closest_idx = np.argmin(angle_diff)
            
            actual_r = defect_r_actual[closest_idx]
            fitted_r = defect_r_fitted[closest_idx]
            
            actual_x = int(center[0] + actual_r * math.cos(angle_rad))
            actual_y = int(center[1] + actual_r * math.sin(angle_rad))
            fitted_x = int(center[0] + fitted_r * math.cos(angle_rad))
            fitted_y = int(center[1] + fitted_r * math.sin(angle_rad))
            
            # 绘制深度指示线
            cv2.arrowedLine(bgr, (actual_x, actual_y), (fitted_x, fitted_y), 
                           (0, 255, 255), 2, tipLength=0.2)
            
            # 绘制端点
            cv2.circle(bgr, (actual_x, actual_y), 3, (0, 0, 255), -1)  # 实际点（红色）
            cv2.circle(bgr, (fitted_x, fitted_y), 3, (255, 100, 0), -1)  # 拟合点（蓝色）
        
        # 添加截面信息文字
        info_x = center[0] + 20
        info_y = center[1] - 20
        depth_text = f"Cross-section: avg_depth={avg_depth:.1f}px"
        cv2.putText(bgr, depth_text, (info_x, info_y), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.4, (0, 0, 255), 1)
        
        # 添加图例
        legend_x = 10
        legend_y = bgr.shape[0] - 60
        cv2.putText(bgr, "Red line: Actual defect surface", (legend_x, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(bgr, "Blue line: Fitted ideal surface", (legend_x, legend_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 0), 1)
        cv2.putText(bgr, "Yellow arrows: Depth indicators", (legend_x, legend_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                
    except Exception as e:
        print(f"绘制球冠截面失败: {e}")


def _make_overlay_legendre(gray: np.ndarray, contour: Optional[np.ndarray], legendre_info: Optional[dict]) -> np.ndarray:
    """返回 BGR 带勒让德拟合的可视化图像（调试用）。
    contour: 原始轮廓点集
    legendre_info: 勒让德拟合参数信息
    """
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if contour is None or legendre_info is None:
        return bgr
    
    try:
        center = (int(round(legendre_info.get('center_x', 0))), int(round(legendre_info.get('center_y', 0))))
        cv2.circle(bgr, center, 3, (0, 0, 255), -1)

        mask_s = legendre_info.get('defect_mask_samples')
        r_act = legendre_info.get('r_actual_samples')  # 原始(可能缺损)
        # 优先使用最终拟合(pass2 或 pass1)；兼容无缺损时仅存在 r_ideal_samples_pass1
        r_fit = legendre_info.get('r_ideal_samples') or legendre_info.get('r_ideal_samples_pass1')
        r_fit_pass1 = legendre_info.get('r_ideal_samples_pass1')  # 第一轮拟合(含缺损影响)
        r_corr = legendre_info.get('r_corrected_samples')  # 补全后（缺损修复后半径，若无缺损可能不存在）
        th_all = legendre_info.get('theta_samples')  # 与上述数组同长度 (φ 采样, [0,2π))

        mode = getattr(config, 'LEGENDRE_OVERLAY_MODE', 'both')

        have_arrays = all([r_act, r_fit, th_all])
        if have_arrays:
            th = np.array(th_all, dtype=np.float64)
            rA = np.array(r_act, dtype=np.float64)
            rF = np.array(r_fit, dtype=np.float64)  # 最终拟合曲线（可能是 pass2 或 pass1）
            rF1 = np.array(r_fit_pass1, dtype=np.float64) if (r_fit_pass1 is not None) else rF  # 第一轮拟合
            # 若不存在补全数组（无缺损），则补全曲线与最终拟合相同
            rC = np.array(r_corr, dtype=np.float64) if (r_corr is not None) else rF
            mask_arr = np.array(mask_s, dtype=bool) if mask_s else np.zeros_like(rA, dtype=bool)

            # 构造极坐标点（原始 / 拟合 / 第一轮拟合 / 补全）
            xA = center[0] + rA * np.cos(th)
            yA = center[1] + rA * np.sin(th)
            xF = center[0] + rF * np.cos(th)
            yF = center[1] + rF * np.sin(th)
            xF1 = center[0] + rF1 * np.cos(th)  # 第一轮拟合点
            yF1 = center[1] + rF1 * np.sin(th)
            xC = center[0] + rC * np.cos(th)
            yC = center[1] + rC * np.sin(th)

            if mode == 'fit_only':
                # 只显示拟合后的平滑曲线（用绿色）
                ptsF = np.vstack([xF, yF]).T.astype(np.int32).reshape(-1, 1, 2)
                try:
                    cv2.polylines(bgr, [ptsF], True, (0, 255, 0), 2)
                except Exception:
                    pass
            else:
                # 是否显示原始
                show_original = (mode in ('both', 'original_corrected'))
                show_corrected = (mode in ('both', 'original_corrected'))
                show_fit = (mode == 'both')  # both 模式下既显示拟合理想也显示 corrected
                show_fit_pass1 = (mode == 'both') and (r_fit_pass1 is not None) and (not np.array_equal(rF1, rF))  # 显示第一轮拟合(仅当存在且与最终拟合不同时)

                if show_original:
                    ptsA = np.vstack([xA, yA]).T.astype(np.int32).reshape(-1, 1, 2)
                    try:
                        cv2.polylines(bgr, [ptsA], True, (255, 120, 0), 1)  # 橙色：原始轮廓
                    except Exception:
                        pass
                        
                if show_fit_pass1:
                    ptsF1 = np.vstack([xF1, yF1]).T.astype(np.int32).reshape(-1, 1, 2)
                    try:
                        cv2.polylines(bgr, [ptsF1], True, (255, 0, 255), 1)  # 紫色：第一轮拟合(含缺损影响)
                    except Exception:
                        pass
                        
                if show_corrected:
                    ptsC = np.vstack([xC, yC]).T.astype(np.int32).reshape(-1, 1, 2)
                    try:
                        cv2.polylines(bgr, [ptsC], True, (0, 255, 0), 1)  # 绿色：修复后轮廓
                    except Exception:
                        pass
                        
                if show_fit:
                    ptsF = np.vstack([xF, yF]).T.astype(np.int32).reshape(-1, 1, 2)
                    try:
                        cv2.polylines(bgr, [ptsF], True, (200, 200, 255), 1)  # 浅紫色：最终拟合
                    except Exception:
                        pass

                # 缺损填充只在同时有原始与补全时有意义
                if show_original and show_corrected and mask_arr.any():
                    overlay = bgr.copy()
                    idxs = np.where(mask_arr)[0]
                    segments = []
                    if idxs.size:
                        start = idxs[0]
                        prev = idxs[0]
                        for k in idxs[1:]:
                            if k == prev + 1:
                                prev = k
                            else:
                                segments.append((start, prev))
                                start = k
                                prev = k
                        segments.append((start, prev))
                    for (s, e) in segments:
                        idx_range = list(range(s, e + 1))
                        poly_x = np.concatenate([xC[idx_range], xA[idx_range][::-1]])
                        poly_y = np.concatenate([yC[idx_range], yA[idx_range][::-1]])
                        poly = np.vstack([poly_x, poly_y]).T.astype(np.int32)
                        if poly.shape[0] >= 3:
                            cv2.fillPoly(overlay, [poly], (0, 50, 200))
                    bgr = cv2.addWeighted(overlay, 0.35, bgr, 0.65, 0)

            # 缺损边界角度文字（若有中心角度）
            thL = legendre_info.get('defect_angle_left_deg')
            thC = legendre_info.get('defect_angle_center_deg')
            thR = legendre_info.get('defect_angle_right_deg')
            if thL is not None and thR is not None:
                for ang_deg, color in [(thL, (0,255,255)), (thR, (0,255,255)), (thC, (0,0,255))]:
                    if ang_deg is None: continue
                    ang = math.radians(ang_deg)
                    rr = float(np.max(rC)*0.9 if rC.size else 0)
                    px = int(round(center[0] + rr * math.cos(ang)))
                    py = int(round(center[1] + rr * math.sin(ang)))
                    cv2.circle(bgr, (px, py), 3, color, -1)

            # 绘制球冠截面（如果存在缺损）
            if mask_arr.any() and show_original and show_corrected:
                _draw_spherical_cap_cross_section(bgr, center, legendre_info, mask_arr, th, rA, rF)

        # 文本信息
        r_squared = legendre_info.get('r_squared')
        degree = legendre_info.get('legendre_degree', 0)
        dmax = legendre_info.get('max_diff_px')
        span_deg = legendre_info.get('defect_span_deg')
        chosen = legendre_info.get('chosen_volume_method')
        two_pass = legendre_info.get('two_pass', False)
        
        line1 = f"Legendre d{degree}" + (f" R²={r_squared:.3f}" if r_squared is not None else '')
        cv2.putText(bgr, line1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
        
        if dmax is not None and span_deg is not None:
            line2 = f"span={span_deg:.1f}° dmax={dmax:.1f}"
            cv2.putText(bgr, line2, (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
            
        if chosen:
            cv2.putText(bgr, chosen, (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
            
        # 添加曲线图例说明（当有多条曲线时）
        if have_arrays and mode in ('both', 'original_corrected'):
            legend_y = 85
            legend_font_size = 0.4
            legend_thickness = 1
            
            if show_original:
                cv2.putText(bgr, "Orange: Original contour", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, legend_font_size, (255,120,0), legend_thickness)
                legend_y += 15
                
            if show_fit_pass1 and two_pass:
                cv2.putText(bgr, "Purple: Pass1 fit (w/ defect)", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, legend_font_size, (255,0,255), legend_thickness)
                legend_y += 15
                
            if show_corrected:
                cv2.putText(bgr, "Green: Corrected contour", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, legend_font_size, (0,255,0), legend_thickness)
                legend_y += 15
                
            if show_fit and two_pass:
                cv2.putText(bgr, "Light Purple: Pass2 fit (clean)", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, legend_font_size, (200,200,255), legend_thickness)
    except Exception as e:
        print(f"绘制勒让德拟合失败: {e}")
        
    return bgr


def _calculate_spherical_cap_volume(r_actual: np.ndarray, r_fitted: np.ndarray, 
                                   phi_values: np.ndarray, defect_mask: np.ndarray, 
                                   debug: bool = False) -> float:
    """
    计算缺损部分的球冠体积。
    
    算法思路：
    1. 在缺损区域，将截面近似为一个平面
    2. 计算平面与拟合球面的交线，形成球冠
    3. 计算球冠体积作为缺损体积
    
    Args:
        r_actual: 实际观测半径数组
        r_fitted: 拟合理想半径数组  
        phi_values: 角度数组
        defect_mask: 缺损区域mask
        debug: 调试模式
        
    Returns:
        float: 缺损部分的球冠体积
    """
    try:
        if not np.any(defect_mask):
            return 0.0
            
        # 获取缺损区域的数据
        defect_indices = np.where(defect_mask)[0]
        if len(defect_indices) == 0:
            return 0.0
            
        # 计算缺损区域的平均半径作为球面半径
        r_fitted_defect = r_fitted[defect_mask]
        r_actual_defect = r_actual[defect_mask]
        
        # 使用拟合半径的平均值作为球面半径R
        R = float(np.mean(r_fitted_defect))
        
        # 计算缺损深度：实际半径与拟合半径的差值
        depth_values = r_fitted_defect - r_actual_defect
        positive_depths = depth_values[depth_values > 0]  # 只考虑正的缺损
        
        if len(positive_depths) == 0 or R <= 0:
            return 0.0
            
        # 球冠高度h应该是最大缺损深度（最深的缺失部分）
        max_depth = float(np.max(positive_depths))
        h = min(max_depth, R)  # 确保高度不超过半径
        
        # 球冠体积公式：V = π * h² * (3R - h) / 3
        cap_volume = math.pi * (h ** 2) * (3.0 * R - h) / 3.0
        
        # 根据缺损角度范围调整体积（考虑缺损只占部分角度）
        defect_span = len(defect_indices) / len(phi_values)  # 缺损占总角度的比例
        adjusted_volume = cap_volume
        
        if debug:
            print(f"球冠体积计算: R={R:.2f}, h={max_depth:.2f}, "
                  f"span_ratio={defect_span:.3f}, V_cap={adjusted_volume:.1f}")
            
        return float(max(0.0, adjusted_volume))
        
    except Exception as e:
        if debug:
            print(f"球冠体积计算失败: {e}")
        return 0.0


def _calculate_legendre_truncated_volume(contour: np.ndarray, img_shape: Optional[Tuple[int, int]] = None, debug: bool = False) -> Tuple[Optional[float], Optional[dict]]:

    try:
        if contour is None or len(contour) < 10:
            return None, None

        # 质心与点集
        M = cv2.moments(contour)
        if M.get('m00', 0) == 0:
            return None, None
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
        pts = contour.reshape(-1, 2).astype(np.float64)
        rel = pts - np.array([cx, cy])
        phi = np.arctan2(rel[:, 1], rel[:, 0])
        phi = np.where(phi < 0, phi + 2 * np.pi, phi)  # [0,2π)
        rad = np.hypot(rel[:, 0], rel[:, 1])

        # 原始外边界半径采样（与普通勒让德函数一致）
        n_phi_bins = int(getattr(config, 'LEGENDRE_ANGLE_SAMPLES', 360) or 360)
        n_phi_bins = max(72, min(720, n_phi_bins))
        bins = np.linspace(0.0, 2.0 * np.pi, n_phi_bins + 1)
        idx = np.clip(np.digitize(phi, bins) - 1, 0, n_phi_bins - 1)
        r_per_bin = np.zeros(n_phi_bins, dtype=np.float64)
        has_val = np.zeros(n_phi_bins, dtype=bool)
        for i, r in zip(idx, rad):
            if not has_val[i] or r > r_per_bin[i]:
                r_per_bin[i] = r
                has_val[i] = True
        if not np.all(has_val):
            ii = np.arange(n_phi_bins)
            good = has_val
            if good.sum() >= 2:
                r_per_bin[~good] = np.interp(ii[~good], ii[good], r_per_bin[good])
            else:
                fill_val = float(np.nanmean(np.where(has_val, r_per_bin, np.nan))) if has_val.any() else 0.0
                r_per_bin[~good] = fill_val
        phi_centers = (bins[:-1] + bins[1:]) * 0.5  # 代表角度

        # 直接对整个 [0,2π) 进行拟合，不再折叠
        phi_all = phi_centers  # [0,2π)
        r_all = r_per_bin      # 对应半径

        # 6阶勒让德拟合 Pass1 (含缺损) 用于缺损检测
        from numpy.polynomial.legendre import legvander, legval, leggauss
        x = np.cos(phi_all)  # φ∈[0,2π) -> x∈[-1,1]
        deg = 6
        Vmat = legvander(x, deg)
        c_pass1, *_ = np.linalg.lstsq(Vmat, r_all, rcond=None)
        r_fit_pass1 = Vmat @ c_pass1

        # 体积计算：使用统一的基于面积的方法（与普通勒让德方法一致）
        def calculate_3d_volume_from_radial(r_values, phi_values):
            """从径向函数计算3D体积（基于面积方法，与普通勒让德一致）"""
            if len(r_values) != len(phi_values):
                return 0.0
            
            # 计算拟合轮廓的面积（使用梯形积分）
            # 对于极坐标：A = (1/2) ∫ r²(φ) dφ
            area_fitted = 0.5 * np.trapz(r_values ** 2, phi_values)
            
            # 基于面积的体积估算（假设近似球形）
            # V = (4/3)π * r_equivalent³，其中 r_equivalent = √(A/π)
            r_equivalent = math.sqrt(area_fitted / math.pi)
            volume_px3 = (4.0 / 3.0) * math.pi * (r_equivalent ** 3)
            
            return float(volume_px3)
        
        V_meas = calculate_3d_volume_from_radial(r_all, phi_all)
        V_fit_full = calculate_3d_volume_from_radial(r_fit_pass1, phi_all)

        # 直接使用 φ∈[0,2π) 的拟合半径与观测半径
        r_fit_full_pass1 = np.clip(r_fit_pass1, 0.0, None)
        r_fit_full = r_fit_full_pass1.copy()
        r_meas_full = r_all.copy()

        # 差值与缺损检测 - 针对缺球冠优化的策略
        d = r_fit_full - r_meas_full  # 正值表示观测小于拟合 => 缺损
        diff_thresh = float(getattr(config, 'LEGENDRE_TRUNC_DIFF_THRESH_PX', 5.0) or 5.0)
        
        # 获取差值的统计信息
        d_mean = float(np.mean(d))
        d_std = float(np.std(d))
        i0 = int(np.argmax(d))
        d_max = float(d[i0])
        
        # 计算相对差值（百分比）
        r_base = r_meas_full[i0] if r_meas_full[i0] > 1e-6 else np.mean(r_meas_full)


        # 搜索左右边界 - 使用与椭球一致的精确边界搜索方法
        N = len(d)
        TWO_PI = 2.0 * np.pi

        # 有效性检查：只对半径有意义的角度进行边界搜索
        valid = np.isfinite(r_meas_full) & (r_meas_full > 1e-9)
        
        def _ang_step(a_from: float, a_to: float, dir_sign: int) -> float:
            """计算从 a_from 朝 dir_sign 方向前进到 a_to 的有向角度差"""
            if dir_sign >= 0:
                return (a_to - a_from) % TWO_PI
            else:
                return (a_from - a_to) % TWO_PI

        def _interp_zero_between(i_pos: int, i_neg: int, dir_sign: int, thr: float) -> Tuple[float, float]:
            """
            在两个相邻样本之间插值找到阈值交点
            i_pos: d>thr 的有效样本索引
            i_neg: d<=thr 的有效样本索引
            dir_sign: +1 表示沿角度递增方向插值；-1 表示沿角度递减方向插值
            返回 (theta_z, r_z)，r_z 为 r_fitted 与 r_meas 线性插值后取平均
            """
            d1 = float(d[i_pos])
            d2 = float(d[i_neg])
            denom = (d1 - d2)
            # 阈值插值：当 d 从 >thr 下降到 <=thr 时，求达阈值处的角度比例
            thr = float(thr)
            alpha = ((d1 - thr) / denom) if abs(denom) > 1e-12 else 0.0

            t1 = float(phi_centers[i_pos])
            t2 = float(phi_centers[i_neg])
            dt = _ang_step(t1, t2, dir_sign)
            if dir_sign >= 0:
                theta_z = (t1 + alpha * dt) % TWO_PI
            else:
                theta_z = (t1 - alpha * dt) % TWO_PI

            # 插值得到边界处的半径值
            rF1, rF2 = float(r_fit_full[i_pos]), float(r_fit_full[i_neg])
            rM1, rM2 = float(r_meas_full[i_pos]), float(r_meas_full[i_neg])
            r_fit_z = rF1 + alpha * (rF2 - rF1)
            r_meas_z = rM1 + alpha * (rM2 - rM1)
            r_z = 0.5 * (r_fit_z + r_meas_z)  # 取平均作为边界半径
            return float(theta_z), float(r_z)

        def _search_boundary(i_center: int, dir_sign: int, thr: float, strict: bool = False) -> Tuple[Optional[float], Optional[float], Optional[int], Optional[int]]:
            """
            从 i_center 出发，沿 dir_sign（+1 右/递增，-1 左/递减）寻找首个"高于阈值 -> 达到阈值"的有效过界点。
            参数:
              - thr: 阈值（第一次搜索传 0.0；重拟合后传 diff_thresh）
              - strict: True 表示严格使用 d < thr 作为到达条件；False 表示使用 d <= thr。
            返回 (theta_z, r_z, idx_pos, idx_neg)，若找不到返回 (None, None, None, None)
            """
            if not (valid[i_center] and d[i_center] > thr):
                return None, None, None, None
                
            last_pos_idx = i_center
            # 在该方向上逐步推进，跳过无效点，寻找第一个 valid 且满足到达条件的样本
            for step in range(1, N):
                j = (i_center + dir_sign * step) % N
                if not valid[j]:
                    continue
                hit = (d[j] < thr) if strict else (d[j] <= thr)
                if hit:
                    th_z, r_z = _interp_zero_between(last_pos_idx, j, dir_sign, thr)
                    return th_z, r_z, last_pos_idx, j
                else:
                    last_pos_idx = j
            # 未找到过零，返回空，外层再决定回退策略
            return None, None, None, None

        if debug:
            print(f"缺损检测: d_max={d_max:.3f} at angle {np.degrees(phi_centers[i0]):.1f}°")

        # 左右边界搜索（第一次严格过零：thr=0.0，用于后续一次性重拟合更稳健）
        thr0 = 0.0
        thL, rL, _pL, _nL = _search_boundary(i0, dir_sign=-1, thr=thr0, strict=True)
        thR, rR, _pR, _nR = _search_boundary(i0, dir_sign=+1, thr=thr0, strict=True)

        # 回退：若某侧未找到交点，则取该方向上 d 的全局最小值处近似为边界（避免"持续扩散"）
        if thL is None:
            # 沿左侧方向（递减）扫描时遇到的有效索引集合
            idxs_left = [(i0 - k) % N for k in range(1, N) if valid[(i0 - k) % N]]
            if idxs_left:
                i_minL = int(min(idxs_left, key=lambda idx: float(d[idx])))
                thL = float(phi_centers[i_minL])
                rL = float(r_meas_full[i_minL]) if np.isfinite(r_meas_full[i_minL]) else float(r_fit_full[i_minL])
            else:
                # 实在无有效点：取一个很小弧宽作为回退
                w = max(2, N // 36)
                i_tmp = (i0 - w) % N
                thL = float(phi_centers[i_tmp])
                rL = float(r_meas_full[i_tmp]) if np.isfinite(r_meas_full[i_tmp]) else float(r_fit_full[i_tmp])

        if thR is None:
            idxs_right = [(i0 + k) % N for k in range(1, N) if valid[(i0 + k) % N]]
            if idxs_right:
                i_minR = int(min(idxs_right, key=lambda idx: float(d[idx])))
                thR = float(phi_centers[i_minR])
                rR = float(r_meas_full[i_minR]) if np.isfinite(r_meas_full[i_minR]) else float(r_fit_full[i_minR])
            else:
                w = max(2, N // 36)
                i_tmp = (i0 + w) % N
                thR = float(phi_centers[i_tmp])
                rR = float(r_meas_full[i_tmp]) if np.isfinite(r_meas_full[i_tmp]) else float(r_fit_full[i_tmp])
        # 标准化弧段
        span = (thR - thL) % TWO_PI
        if span <= 0.0:
            thL, thR = thR, thL
            span = (thR - thL) % TWO_PI

        # 计算缺损中心角度
        thC = float(phi_centers[i0])
        
        # 构造缺损 mask（按 thL->thR 的正向角度区间）
        defect_mask = np.zeros(N, dtype=bool)
        for k in range(N):
            ang = float(phi_centers[k])
            if thL <= thR:
                if thL <= ang <= thR:
                    defect_mask[k] = True
            else:  # 跨越 2π
                if ang >= thL or ang <= thR:
                    defect_mask[k] = True

        # Pass2: 使用非缺损角度重新拟合，进一步减少缺损影响
        non_defect_mask = ~defect_mask
        r_all_pass2 = r_all.copy()
        # 将缺损区域的半径替换为第一轮拟合值
        r_all_pass2[defect_mask] = r_fit_full_pass1[defect_mask]
        
        c_pass2, *_ = np.linalg.lstsq(Vmat, r_all_pass2, rcond=None)
        r_fit_pass2 = Vmat @ c_pass2
        r_fit_full_pass2 = np.clip(r_fit_pass2, 0.0, None)
        r_fit_full = r_fit_full_pass2  # 用第二次结果进行修复

        # 修复：将缺损角内半径提升为第二轮拟合值
        r_corr_full = r_meas_full.copy()
        r_corr_full[defect_mask] = r_fit_full[defect_mask]

        # 计算各种体积
        V_meas = calculate_3d_volume_from_radial(r_meas_full, phi_all)  # 原始观测体积
        V_fit_full_pass2 = calculate_3d_volume_from_radial(r_fit_full_pass2, phi_all)  # 完整拟合体积  
        V_corr = calculate_3d_volume_from_radial(r_corr_full, phi_all)  # 修复后体积
        
        # 计算缺损体积：使用球冠模型
        V_loss_final = _calculate_spherical_cap_volume(r_meas_full, r_fit_full_pass2, phi_all, defect_mask, debug)
        
        # 正确的体积计算逻辑（参考椭球方法）：
        # V_full: 完整拟合体积（无缺损的理想体积）
        # V_missing: 缺损部分体积
        # V_truncated: 完整体积减去缺损体积（实际应该存在的体积）
        # V_meas: 原始观测体积（含缺损，仅供参考）
        V_full = V_fit_full_pass2  # 完整拟合体积
        V_missing = V_loss_final   # 缺损体积
        V_truncated = V_full - V_missing  # 减去缺损后的实际体积
        
        # 计算体积损失比例
        volume_loss_ratio = float(V_missing / V_full) if V_full > 0 else 0.0

        info = {
            'center_x': float(cx),
            'center_y': float(cy),
            'legendre_degree': deg,
            'legendre_coeffs_pass1': [float(v) for v in c_pass1.tolist()],
            'legendre_coeffs_pass2': [float(v) for v in c_pass2.tolist()],
            
            # 基础体积信息（与椭圆方法保持一致的结构）
            'volume_legendre_px3': float(V_meas),  # 原始观测体积（含缺损，仅供参考）
            'volume_legendre_full_px3': float(V_full),  # 完整拟合体积（对应椭圆的 volume_ellipsoid_full_px3）
            'volume_legendre_truncated_px3': float(V_truncated),  # 减去缺损后的实际体积（对应椭圆的 volume_ellipsoid_truncated_px3）
            'volume_legendre_missing_caps_px3': float(V_missing),  # 缺损体积（对应椭圆的 volume_ellipsoid_missing_caps_px3）
            'volume_legendre_missing_top_px3': float(V_missing),  # 顶部缺损体积（与椭球保持一致）
            'volume_legendre_missing_bottom_px3': 0.0,  # 底部缺损体积（勒让德通常只有一个缺损区域）
            'volume_legendre_truncated_est_px3': float(V_truncated),  # 截断估计体积（与 truncated 相同）
            
            # 选择的体积和方法（返回减去缺损后的实际体积）
            'chosen_volume_px3': float(V_truncated),
            'chosen_volume_method': 'legendre_truncated',
            
            # 缺损分析信息
            'max_diff_px': float(d_max),
            'defect_span_deg': float(np.degrees(span)),
            'defect_span_rad': float(span),
            'volume_loss_ratio': volume_loss_ratio,
            'defect_points_count': int(np.sum(defect_mask)),
            'total_points_count': int(N),
            'defect_ratio': float(np.sum(defect_mask) / N),
            
            # 采样和拟合数据
            'defect_mask_samples': defect_mask.tolist(),
            'theta_samples': phi_centers.tolist(),
            'r_actual_samples': r_meas_full.tolist(),
            'r_ideal_samples_pass1': r_fit_full_pass1.tolist(),
            'r_ideal_samples': r_fit_full.tolist(),
            'r_corrected_samples': r_corr_full.tolist(),
            
            # 缺损位置信息
            'defect_angle_left_deg': float(np.degrees(thL)),
            'defect_angle_center_deg': float(np.degrees(thC)),
            'defect_angle_right_deg': float(np.degrees(thR)),
            
            # 技术参数
            'boundary_check_passed': True,
            'diff_threshold_px': float(diff_thresh),
            'two_pass': True,
            'fitting_method': 'legendre_truncated_2pass',
        }
        if debug:
            print(f"Legendre truncated (2-pass): ")
            print(f"  - 缺损检测: d_max={d_max:.3f}, span={np.degrees(span):.1f}°, points={np.sum(defect_mask)}/{N}")
            print(f"  - 体积计算: V_meas={V_meas:.1f}, V_full={V_full:.1f}, V_missing={V_missing:.1f}")
            print(f"  - 最终体积: V_truncated={V_truncated:.1f} (完整体积减去缺损)")
            print(f"  - 损失比例: {volume_loss_ratio:.3f}, 方法: legendre_truncated")
        return V_truncated, info
    except Exception as e:
        if debug:
            print(f"勒让德截断实现失败: {e}")
        return None, None

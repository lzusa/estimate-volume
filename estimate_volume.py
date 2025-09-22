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

from edge_extend import _adaptive_edge_extend
from background_interference import _remove_background_interference
from specular_gaps import _fix_specular_gaps
from circle_acceptable import _is_circle_acceptable
from legendre import _calculate_legendre_volume, _save_legendre_plot, _make_overlay_legendre, _calculate_legendre_truncated_volume
from ellipse import _calculate_ellipsoid_volume, _calculate_ellipsoid_truncated_volume, _save_truncated_ellipse_plot, _make_overlay_ellipse

def _make_mask_fit_overlay(mask: np.ndarray,
                           fitted_contour: Optional[np.ndarray] = None,
                           ellipse_info: Optional[dict] = None,
                           contour_color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """在二值mask上叠加绘制拟合曲线/椭圆，返回BGR图像。
    - 若提供 fitted_contour，则绘制该折线;
    - 若提供 ellipse_info，则按参数绘制椭圆。
    两者都提供时优先使用 fitted_contour。
    """
    if mask is None:
        return None
    if mask.dtype != np.uint8:
        m = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        m = mask
    bgr = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)

    try:
        if fitted_contour is not None:
            cnt = np.asarray(fitted_contour, dtype=np.int32).reshape(-1, 1, 2)
            cv2.drawContours(bgr, [cnt], -1, contour_color, 1)
        elif ellipse_info is not None:
            # 兼容截断信息中的键名
            cx = ellipse_info.get('center_x', ellipse_info.get('ellipse_center_x', None))
            cy = ellipse_info.get('center_y', ellipse_info.get('ellipse_center_y', None))
            major = ellipse_info.get('major_axis_px', None)
            minor = ellipse_info.get('minor_axis_px', None)
            angle = float(ellipse_info.get('angle_deg', 0.0))
            if cx is not None and cy is not None and major is not None and minor is not None:
                center = (int(round(cx)), int(round(cy)))
                axes = (int(round(major / 2)), int(round(minor / 2)))
                cv2.ellipse(bgr, center, axes, angle, 0, 360, contour_color, 1)
    except Exception:
        pass

    return bgr

def process_image(path: str, min_area: float = config.MIN_AREA, debug: bool = False, debug_out_dir: Optional[str] = None, min_circularity: float = config.MIN_CIRCULARITY, enable_specular_fix: bool = True, low_temp_read: bool = False) -> Tuple[bool, Optional[float], Optional[float], Optional[np.ndarray], Optional[Dict[str, float]]]:

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False, None, None, None, None

    # 如果当前是低温读（来自上层调用的标志），记录meta并进入特殊处理流程
    if low_temp_read:
        # 保存基础meta（稍后会更新完整meta）
        meta = {'filename': os.path.basename(path), 'shape': None if img is None else img.shape, 'found': None, 'low_temp_read': True}
        try:
            _save_meta(meta)
        except Exception:
            pass
        # 继续常规处理，但将标记 low_temp_read 以激活后续的特殊体积计算分支

    # 基础文件名（供 debug/meta 使用），提前计算以避免作用域错误
    base = os.path.splitext(os.path.basename(path))[0]
    # 准备 debug 输出目录（每张图一个子目录）。若未提供 debug_out_dir，则在图片同目录创建 <basename>_debug
    img_dbg_dir = None
    if debug:
        try:
            if debug_out_dir:
                os.makedirs(debug_out_dir, exist_ok=True)
                img_dbg_dir = os.path.join(debug_out_dir, base)
            else:
                img_dir = os.path.dirname(path) or '.'
                img_dbg_dir = os.path.join(img_dir, base + '_debug')
            os.makedirs(img_dbg_dir, exist_ok=True)
        except Exception:
            img_dbg_dir = None

    def _save_dbg(img_to_save, name: str):
        if not debug or img_dbg_dir is None or img_to_save is None:
            return
        try:
            outp = os.path.join(img_dbg_dir, name)
            # 如果传入的是 numpy 数组但不是 uint8，则做归一化并保存
            if isinstance(img_to_save, np.ndarray):
                if img_to_save.dtype != np.uint8:
                    norm = cv2.normalize(img_to_save, None, 0, 255, cv2.NORM_MINMAX)
                    cv2.imwrite(outp, norm.astype(np.uint8))
                else:
                    cv2.imwrite(outp, img_to_save)
            else:
                try:
                    with open(outp, 'w', encoding='utf-8') as fh:
                        fh.write(str(img_to_save))
                except Exception:
                    pass
            #print(f"debug saved: {outp}")
        except Exception:
            pass

    def _save_meta(meta: dict):
        if not debug or img_dbg_dir is None:
            return
        try:
            outp = os.path.join(img_dbg_dir, f"{base}_meta.json")
            with open(outp, 'w', encoding='utf-8') as fh:
                json.dump(meta, fh, ensure_ascii=False, indent=2)
            print(f"meta saved: {outp}")
        except Exception:
            pass

    def _save_dark_mask_data(mask_data: np.ndarray, original_filename: str):
        """保存dark_mask_final的原始数据"""
        try:
            # 创建保存目录
            os.makedirs(config.DARK_MASK_DATA_DIR, exist_ok=True)
            
            # 构建文件名（与原始图像名称一致，但扩展名为.npy）
            base_name = os.path.splitext(original_filename)[0]
            output_path = os.path.join(config.DARK_MASK_DATA_DIR, f"{base_name}.npy")
            
            # 保存numpy数组
            np.save(output_path, mask_data)
            print(f"Dark mask data saved: {output_path}")
        except Exception as e:
            print(f"Failed to save dark mask data: {e}")
            pass

    #删去图像的上面1/3
    if config.NAME == 'Cu5N':
        img = img[img.shape[0]//3: , :]
    # 1. 轻度模糊减噪
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    # 2. Otsu 自动阈值（反阈值让暗球为前景=255）
    _, dark_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 3. 形态学：小开运算去噪 + 小闭运算填微孔
    km_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, km_small, iterations=1)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, km_small, iterations=1)

    

    # 5. 去除背景影响，保留球体区域
    dark_mask = _remove_background_interference(dark_mask, img.shape, debug=debug, dbg_save=_save_dbg)
    # 4. 自适应模糊边缘外延（替代简单膨胀）
    dark_mask = _adaptive_edge_extend(blur, dark_mask, debug=debug, dbg_save=_save_dbg)
    _save_dbg(dark_mask, 'dark_mask_refined.png')

    # 直接使用 refined mask 找最大连通域
    cnts, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        best_dark = max(cnts, key=cv2.contourArea)
        area_dark = cv2.contourArea(best_dark)
        if area_dark >= min_area:
            # 先进行“高光缺口修复”，避免边缘缺口影响拟合
            if enable_specular_fix:
                repaired_mask, repaired_contour, repair_info = _fix_specular_gaps(best_dark, img.shape, debug=debug, dbg_save=_save_dbg)
            else:
                repaired_mask, repaired_contour, repair_info = None, None, None
            if repaired_mask is not None and repaired_contour is not None and len(repaired_contour) >= 5:
                final_contour = repaired_contour
                mask_d = repaired_mask
                _save_dbg(mask_d, 'mask_repaired.png')
            else:
                # 回退到原始轮廓
                final_contour = best_dark
                mask_d = np.zeros(img.shape, dtype=np.uint8)
                cv2.drawContours(mask_d, [best_dark], -1, 255, -1)

            # 修复后重新计算面积
            area_dark = cv2.contourArea(final_contour)

            # 修复后再做圆形性/凸性检查；但如果为低温读则跳过此检查（低温图片可能存在顶部缺失）
            if not low_temp_read:
                circle_check_result = _is_circle_acceptable(final_contour, debug=debug)
                if not circle_check_result['is_acceptable']:
                    if debug:
                        print(f"图像 {os.path.basename(path)} 中的物体不符合圆形要求，已舍弃")
                    meta = {
                        'filename': os.path.basename(path), 
                        'shape': img.shape, 
                        'found': False, 
                        'reject_reason': 'poor_circularity',
                        'circle_check': circle_check_result,
                        'gap_fix_info': repair_info
                    }
                    _save_meta(meta)
                    overlay_reject = _make_overlay(img, None) if debug else None
                    _save_dbg(overlay_reject, 'overlay_rejected.png')
                    return False, None, None, overlay_reject, None
            else:
                # 标记为已跳过圆度检查
                circle_check_result = {'is_acceptable': True, 'skipped_for_low_temp': True}
            
            # 使用该轮廓直接计算半径并返回，减少对背景处理依赖
            (x_cd, y_cd), r_enclosing_d = cv2.minEnclosingCircle(final_contour)
            r_eq_d = math.sqrt(area_dark / math.pi)
            try:
                dist_d = cv2.distanceTransform(mask_d, cv2.DIST_L2, 5)
                _, maxVal_d, _, maxLoc_d = cv2.minMaxLoc(dist_d)
                r_dt_d = float(maxVal_d)
                cx_dt_d, cy_dt_d = maxLoc_d
            except Exception:
                dist_d = None
                r_dt_d = 0.0

            # 生成平滑包络来包含mask，去掉球形相关计算
            h_img, w_img = img.shape[:2]
            
            # 根据配置选择体积计算方法
            volume_method = getattr(config, 'VOLUME_METHOD', 'revolve')
            
            if low_temp_read:
                # 低温读：根据配置选择缺损拟合方法
                defect_method = getattr(config, 'LOW_TEMP_DEFECT_METHOD', 'ellipse')
                if defect_method == 'legendre':
                    volume_trunc, trunc_info = _calculate_legendre_truncated_volume(final_contour, img.shape, debug=debug)
                    if volume_trunc is not None:
                        chosen_volume = volume_trunc
                        chosen_method_used = 'legendre_truncated'
                        overlay_d = _make_overlay_legendre(img, final_contour, trunc_info) if debug else None
                        if debug and img_dbg_dir is not None and trunc_info is not None:
                            # 保存拟合曲线图
                            try:
                                _ = _save_legendre_plot(trunc_info, img_dbg_dir)
                            except Exception:
                                pass
                            # mask 叠加
                            fitted_contour = None
                            if trunc_info and 'fitted_contour' in trunc_info:
                                try:
                                    fitted_contour = np.array(trunc_info['fitted_contour'], dtype=np.int32)
                                except Exception:
                                    fitted_contour = None
                            mask_fit = _make_mask_fit_overlay(mask_d, fitted_contour, None)
                            _save_dbg(mask_fit, 'mask_fit_truncated_lowtemp_legendre.png')
                else:
                    # 默认使用截断椭圆经验估算
                    volume_trunc, trunc_info = _calculate_ellipsoid_truncated_volume(final_contour, debug=debug)
                    if volume_trunc is not None:
                        chosen_volume = volume_trunc
                        chosen_method_used = 'ellipsoid_truncated_lowtemp'
                        overlay_d = _make_overlay_ellipse(img, final_contour, trunc_info) if debug else None
                        if debug and img_dbg_dir is not None and trunc_info is not None:
                            mask_fit = _make_mask_fit_overlay(mask_d, None, trunc_info)
                            _save_dbg(mask_fit, 'mask_fit_truncated_lowtemp.png')
                            # 保存拟合与轮廓叠图
                            try:
                                _ = _save_truncated_ellipse_plot(trunc_info, final_contour, img, img_dbg_dir, base)
                            except Exception:
                                pass
            elif volume_method == 'ellipsoid':

                # 截断计算失败 -> 尝试普通椭圆体
                volume_ellipsoid, ellipse_info = _calculate_ellipsoid_volume(final_contour, debug=debug)
                if volume_ellipsoid is not None:
                    chosen_volume = volume_ellipsoid
                    chosen_method_used = 'ellipsoid'
                    overlay_d = _make_overlay_ellipse(img, final_contour, ellipse_info) if debug else None
                    if debug and img_dbg_dir is not None and ellipse_info is not None:
                        mask_fit = _make_mask_fit_overlay(mask_d, None, ellipse_info)
                        _save_dbg(mask_fit, 'mask_fit_overlay.png')
            elif volume_method == 'legendre':
                # 使用勒让德多项式拟合方法
                volume_legendre, legendre_info = _calculate_legendre_volume(final_contour, img.shape, debug=debug)
                if volume_legendre is not None:
                    chosen_volume = volume_legendre
                    chosen_method_used = 'legendre'
                    # 生成勒让德拟合可视化
                    overlay_d = _make_overlay_legendre(img, final_contour, legendre_info) if debug else None
                    # 额外：保存 R(θ) 曲线图到 debug 目录
                    if debug and img_dbg_dir is not None and legendre_info is not None:
                        _ = _save_legendre_plot(legendre_info, img_dbg_dir)
                        # 叠加到mask上的拟合形状
                        fitted_contour = None
                        if 'fitted_contour' in legendre_info:
                            fitted_contour = np.array(legendre_info['fitted_contour'], dtype=np.int32)
                        mask_fit = _make_mask_fit_overlay(mask_d, fitted_contour, None)
                        _save_dbg(mask_fit, 'mask_fit_overlay.png')
                else:
                    # 勒让德拟合失败或超出边界，直接舍弃当前图像
                    if debug:
                        print(f"图像 {os.path.basename(path)} 中的勒让德拟合超出边界，已舍弃")
                    meta = {
                        'filename': os.path.basename(path), 
                        'shape': img.shape, 
                        'found': False, 
                        'reject_reason': 'legendre_boundary_exceeded',
                        'circle_check': circle_check_result,
                        'gap_fix_info': repair_info
                    }
                    _save_meta(meta)
                    overlay_reject = _make_overlay(img, None) if debug else None
                    _save_dbg(overlay_reject, 'overlay_rejected.png')
                    return False, None, None, overlay_reject, None

            # 使用等面积半径作为兼容性参数（仅用于日志）
            r_px = math.sqrt(area_dark / math.pi)
            
            _save_dbg(mask_d, 'dark_mask_final.png')
            # 保存dark_mask_final的原始数据
            #_save_dark_mask_data(mask_d, os.path.basename(path))
            
            # 在overlay_d上绘制dark_mask_final的边界
            if overlay_d is not None and mask_d is not None:
                # 提取mask_d的轮廓
                contours_mask, _ = cv2.findContours(mask_d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours_mask:
                    # 绘制边界轮廓（红色，线宽2）
                    cv2.drawContours(overlay_d, contours_mask, -1, (0, 0, 255), 1)

            # 如果有拟合椭圆信息（普通椭圆或截断估计），在 overlay_d 上以线宽1 绘制椭圆
            try:
                ellipse_to_draw = None
                # 优先使用 trunc_info（低温分支），如不存在则尝试 ellipse_info
                if 'trunc_info' in locals() and trunc_info is not None:
                    info = trunc_info
                    cx = int(round(info.get('ellipse_center_x', info.get('center_x', 0))))
                    cy = int(round(info.get('ellipse_center_y', info.get('center_y', 0))))
                    major = int(round(info.get('major_axis_px', 0)))
                    minor = int(round(info.get('minor_axis_px', 0)))
                    angle = float(info.get('angle_deg', 0.0))
                    ellipse_to_draw = ((cx, cy), (major, minor), angle)
                elif 'ellipse_info' in locals() and ellipse_info is not None:
                    info = ellipse_info
                    cx = int(round(info.get('center_x', 0)))
                    cy = int(round(info.get('center_y', 0)))
                    major = int(round(info.get('major_axis_px', 0)))
                    minor = int(round(info.get('minor_axis_px', 0)))
                    angle = float(info.get('angle_deg', 0.0))
                    ellipse_to_draw = ((cx, cy), (major, minor), angle)

                if overlay_d is not None and ellipse_to_draw is not None:
                    # 以绿色线宽1绘制椭圆轮廓
                    try:
                        cv2.ellipse(overlay_d, ellipse_to_draw[0], (int(ellipse_to_draw[1][0]//2), int(ellipse_to_draw[1][1]//2)), ellipse_to_draw[2], 0, 360, (0, 255, 0), 1)
                    except Exception:
                        # 备用：如果上面参数形式不合，尝试直接使用 ellipse_to_draw
                        try:
                            cv2.ellipse(overlay_d, ellipse_to_draw, (0, 255, 0), 1)
                        except Exception:
                            pass
            except Exception:
                pass
            
            _save_dbg(overlay_d, 'dark_overlay.png')
            
            # 记录形状度量
            x, y, w, h = cv2.boundingRect(final_contour)
            bbox_w, bbox_h = float(w), float(h)
            ellipse_major = None
            ellipse_minor = None
            if len(final_contour) >= 5:
                try:
                    el = cv2.fitEllipse(final_contour)
                    (cx_e, cy_e), (MA, ma), angle = el
                    ellipse_major = float(max(MA, ma))
                    ellipse_minor = float(min(MA, ma))
                except Exception:
                    pass

            # Distance Transform 半径 (内接圆)
            dt_radius = r_dt_d if r_dt_d else None

            # 构建meta信息
            meta = {
                'filename': os.path.basename(path), 
                'shape': img.shape, 
                'found': True, 
                'low_temp_read': low_temp_read,
                'area_px': float(area_dark), 
                'radius_px': float(r_px), 
                'chosen_volume_method': chosen_method_used, 
                'chosen_volume_px3': chosen_volume,
                'bbox_w': bbox_w, 
                'bbox_h': bbox_h, 
                'ellipse_major_diam_px': ellipse_major, 
                'ellipse_minor_diam_px': ellipse_minor, 
                'dt_radius_px': dt_radius,
                'circle_check': circle_check_result,
                'gap_fix_info': repair_info
            }
            
            # 根据方法添加特定信息
            if volume_method == 'ellipsoid' or low_temp_read:
                if 'trunc_info' in locals() and trunc_info is not None:
                    meta.update(trunc_info)
                elif 'ellipse_info' in locals() and ellipse_info is not None:
                    meta.update(ellipse_info)
            elif volume_method == 'legendre' and 'legendre_info' in locals():
                meta.update(legendre_info)

            
            _save_meta(meta)
            
            # 构建extra信息
            extra = {
                'chosen_volume_px3': chosen_volume,
                'chosen_volume_method': chosen_method_used,
                'low_temp_read': low_temp_read,
                'bbox_w_px': bbox_w,
                'bbox_h_px': bbox_h,
                'ellipse_major_diam_px': ellipse_major,
                'ellipse_minor_diam_px': ellipse_minor,
                'dt_radius_px': dt_radius,
            }
            
            # 根据方法添加特定信息到extra
            if volume_method == 'ellipsoid' or low_temp_read:
                if 'trunc_info' in locals() and trunc_info is not None:
                    extra.update(trunc_info)
                elif 'ellipse_info' in locals() and ellipse_info is not None:
                    extra.update(ellipse_info)
            elif volume_method == 'legendre' and 'legendre_info' in locals():
                extra.update(legendre_info)

            # include circularity / circle_check info in extra so callers can plot/analyze
            try:
                if 'circle_check' in locals() and isinstance(circle_check_result, dict):
                    extra['circle_check'] = circle_check_result
                    extra['circularity_ratio'] = circle_check_result.get('circularity_ratio')
                else:
                    # fallback: try to compute convexity/circularity here if possible
                    if 'final_contour' in locals() and final_contour is not None:
                        try:
                            hull = cv2.convexHull(final_contour)
                            hull_area = cv2.contourArea(hull)
                            contour_area = cv2.contourArea(final_contour)
                            (cx_c, cy_c), radius_c = cv2.minEnclosingCircle(final_contour)
                            circle_area = math.pi * radius_c * radius_c
                            circularity_ratio = contour_area / max(circle_area, 1e-6)
                            extra['circularity_ratio'] = float(round(circularity_ratio, 6))
                        except Exception:
                            pass
            except Exception:
                pass

            return True, float(area_dark), float(r_px), overlay_d, extra

    # 未检测到合格的极暗区域 -> 直接返回未找到（删除完整流程）
    meta = {'filename': os.path.basename(path), 'shape': img.shape, 'found': False}
    _save_meta(meta)
    overlay_none = _make_overlay(img, None) if debug else None
    _save_dbg(overlay_none, 'overlay_none.png')
    return False, None, None, overlay_none, None


def _make_overlay(gray: np.ndarray, circle: Optional[Tuple[int, int, int]]) -> np.ndarray:
    """返回 BGR 图像（调试用，不再绘制球形）。"""
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # 不再绘制球形，只返回基础BGR图像
    return bgr




def process_folder(path: str, scale: Optional[float] = None, out_csv: Optional[str] = None, limit: Optional[int] = None, debug: bool = False, debug_out_dir: Optional[str] = None, min_circularity: float = 0.0) -> List[dict]:
    """处理文件夹或单个文件，返回每张图片的结果字典。"""
    if os.path.isdir(path):
        files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith('.bmp') or f.lower().endswith('.png') or f.lower().endswith('.jpg')])
    elif os.path.isfile(path):
        files = [path]
    else:
        raise FileNotFoundError(path)

    if limit:
        files = files[:limit]

    results = []
    for fp in files:
        found, area_px, radius_px, overlay, extra = process_image(fp, debug=debug, debug_out_dir=debug_out_dir, min_circularity=min_circularity)
        if found and extra:
            # 使用选定方法的体积
            volume_px3 = extra.get('chosen_volume_px3')
        else:
            volume_px3 = None

        radius_mm = None
        volume_mm3 = None
        if scale is not None and radius_px is not None:
            radius_mm = radius_px * float(scale)
            volume_mm3 = (4.0 / 3.0) * math.pi * (radius_mm ** 3)

        row = {
            'filename': os.path.basename(fp),
            'path': fp,
            'found': bool(found),
            'area_px': None if area_px is None else float(area_px),
            'radius_px': None if radius_px is None else float(radius_px),
            'radius_mm': None if radius_mm is None else float(radius_mm),
            'volume_px3': None if volume_px3 is None else float(volume_px3),
            'volume_mm3': None if volume_mm3 is None else float(volume_mm3),
        }
        if extra:
            row.update(extra)
        results.append(row)

        # 若 debug 且 overlay 存在，保存带标记图像到指定目录（若提供），否则保存到原图同目录
        if debug and overlay is not None:
            if debug_out_dir:
                try:
                    os.makedirs(debug_out_dir, exist_ok=True)
                    base = os.path.basename(fp)
                    name = os.path.splitext(base)[0] + '_detected.png'
                    out_img = os.path.join(debug_out_dir, name)
                except Exception:
                    out_img = os.path.splitext(fp)[0] + '_detected.png'
            else:
                out_img = os.path.splitext(fp)[0] + '_detected.png'
            try:
                cv2.imwrite(out_img, overlay)
            except Exception:
                pass

    if out_csv:
        _write_csv(out_csv, results, scale is not None)

    return results


def _write_csv(path: str, rows: List[dict], has_mm: bool = False) -> None:
    # 动态列：基础列 + 可能出现的高级体积列
    base_cols = ['filename', 'found', 'area_px', 'radius_px', 'volume_px3']
    if has_mm:
        base_cols += ['radius_mm', 'volume_mm3']
    extra_cols = [
        'volume_revolve_px3', 'volume_ellipsoid_px3', 'volume_legendre_px3', 'chosen_volume_px3', 'chosen_volume_method', 
        'bbox_w_px', 'bbox_h_px', 'ellipse_major_diam_px', 'ellipse_minor_diam_px', 'dt_radius_px',
        'center_x', 'center_y', 'major_axis_px', 'minor_axis_px', 'angle_deg',
        'semi_major_a_px', 'semi_minor_b_px', 'semi_depth_c_px', 'envelope_area_px', 'envelope_method',
        'legendre_degree', 'r_squared', 'mean_radius_px', 'max_radius_px', 'min_radius_px', 'n_angle_samples',
    # 额外：截断椭球诊断
    'ellipse_center_x', 'ellipse_center_y', 'semi_a_px', 'semi_b_px', 'semi_c_px',
    'area_ellipse_px2', 'area_contour_px2', 'area_fraction',
    'trunc_top_h_px', 'trunc_bottom_h_px', 'trunc_top_raw_px', 'trunc_bottom_raw_px', 'trunc_tolerance_px',
    'ellipse_local_y_max_px', 'ellipse_local_y_min_px', 'depth_aspect',
    'volume_ellipsoid_full_px3', 'volume_ellipsoid_truncated_px3', 'volume_ellipsoid_truncated_est_px3', 'volume_ellipsoid_scaled_px3',
    # 勒让德截断体积诊断
    'volume_legendre_full_px3', 'volume_legendre_truncated_px3', 'volume_legendre_missing_caps_px3', 'volume_legendre_truncated_est_px3',
    'volume_loss_ratio', 'max_diff_px', 'defect_span_deg', 'defect_span_rad',
    'defect_angle_left_deg', 'defect_angle_center_deg', 'defect_angle_right_deg', 'fitting_method'

    ]
    # 仅保留出现过的列
    present_extras = [c for c in extra_cols if any(c in r for r in rows)]
    fieldnames = base_cols + present_extras
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            out = {k: r.get(k) for k in fieldnames}
            writer.writerow(out)


def main() -> int:
    p = argparse.ArgumentParser(description='旋转积分法估计暗色物体体积（像素及物理单位可选）')
    p.add_argument('-i', '--input', required=True, help='单张图像或图像目录路径')
    p.add_argument('-o', '--out', help='可选: 输出 CSV 路径')
    p.add_argument('--scale', type=float, help='像素到物理单位的比例 (mm per pixel)')
    p.add_argument('--limit', type=int, help='仅处理前 N 张图片')
    p.add_argument('--debug', action='store_true', help='保存带检测标记的图像以便调试')
    p.add_argument('--debug-out', help='调试图像输出目录（可选）')
    args = p.parse_args()

    results = process_folder(args.input, scale=args.scale, out_csv=args.out, limit=args.limit, debug=args.debug, debug_out_dir=args.debug_out)

    # 简短打印
    for r in results:
        method = r.get('chosen_volume_method', 'unknown')
        if method == 'ellipsoid':
            print(f"{r['filename']} | found={r['found']} | radius_like_px={r['radius_px']} | volume_ellipsoid_px3={r.get('volume_ellipsoid_px3')} | volume_px3={r['volume_px3']} | volume_mm3={r.get('volume_mm3')} | method={method}")
        elif method == 'legendre':
            r_squared = r.get('r_squared', 0)
            print(f"{r['filename']} | found={r['found']} | radius_like_px={r['radius_px']} | volume_legendre_px3={r.get('volume_legendre_px3')} | volume_px3={r['volume_px3']} | volume_mm3={r.get('volume_mm3')} | R²={r_squared:.4f} | method={method}")
        else:
            print(f"{r['filename']} | found={r['found']} | radius_like_px={r['radius_px']} | volume_revolve_px3={r.get('volume_revolve_px3')} | volume_px3={r['volume_px3']} | volume_mm3={r.get('volume_mm3')} | method={method}")

    if args.out:
        print(f"已写入: {args.out}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
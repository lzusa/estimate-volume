import math
from typing import List, Optional, Tuple, Dict
from typing import Tuple

import numpy as np

import cv2
import config

def _fit_circle_kasa(points: np.ndarray, max_iter: int = 3, outlier_k: float = 2.5) -> Optional[Tuple[Tuple[float, float], float, np.ndarray]]:
    """最小二乘圆拟合（Kåsa 线性法）+ 简单鲁棒剔除。

    Args:
        points: (N,2) 图像坐标点
        max_iter: 迭代重拟合次数
        outlier_k: 基于 MAD 的离群阈值倍数

    Returns:
        ((cx, cy), R, inlier_mask) 或 None
    """
    try:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        N = pts.shape[0]
        if N < 3:
            return None

        inlier = np.ones(N, dtype=bool)
        for _ in range(max_iter):
            P = pts[inlier]
            if P.shape[0] < 3:
                break
            x, y = P[:, 0], P[:, 1]
            A = np.stack([x, y, np.ones_like(x)], axis=1)  # [x y 1]
            b = -(x*x + y*y)
            try:
                sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            except Exception:
                return None

            A1, B1, C1 = sol  # x^2 + y^2 + A1 x + B1 y + C1 = 0
            cx = -0.5 * A1
            cy = -0.5 * B1
            R2 = cx*cx + cy*cy - C1
            if R2 <= 1e-9:
                return None
            R = float(math.sqrt(R2))

            # 残差：|dist - R|
            dist = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
            res = np.abs(dist - R)
            # MAD -> sigma 估计
            med = np.median(res[inlier]) if np.any(inlier) else np.median(res)
            mad = np.median(np.abs(res[inlier] - med)) if np.any(inlier) else np.median(np.abs(res - med))
            sigma = 1.4826 * mad if mad > 1e-12 else np.std(res[inlier])
            if not np.isfinite(sigma) or sigma < 1e-9:
                # 收敛
                return (float(cx), float(cy)), float(R), inlier
            new_inlier = res <= (med + outlier_k * sigma)
            if new_inlier.sum() == inlier.sum():
                # 不再变化
                return (float(cx), float(cy)), float(R), inlier
            inlier = new_inlier

        # 最终解
        P = pts[inlier]
        if P.shape[0] < 3:
            return None
        x, y = P[:, 0], P[:, 1]
        A = np.stack([x, y, np.ones_like(x)], axis=1)
        b = -(x*x + y*y)
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        A1, B1, C1 = sol
        cx = -0.5 * A1
        cy = -0.5 * B1
        R2 = cx*cx + cy*cy - C1
        if R2 <= 1e-9:
            return None
        R = float(math.sqrt(R2))
        return (float(cx), float(cy)), float(R), inlier
    except Exception:
        return None

def _ray_circle_intersection(center_xy: Tuple[float, float], dir_img: Tuple[float, float], circle_center: Tuple[float, float], R: float) -> Optional[float]:
    """求从 center 沿 dir_img 的射线与圆的交点距离 t（t>=0 的最小根）。
    若无交返回 None。dir_img 不要求单位长度，内部会归一化。
    """
    try:
        cx, cy = center_xy
        dx, dy = dir_img
        norm = math.hypot(dx, dy)
        if norm < 1e-12:
            return None
        dx /= norm; dy /= norm
        x0, y0 = cx, cy
        xc, yc = circle_center
        px, py = x0 - xc, y0 - yc
        b = dx*px + dy*py
        c = px*px + py*py - R*R
        disc = b*b - c
        if disc < 0:
            return None
        s = math.sqrt(max(0.0, disc))
        t1 = -b - s
        t2 = -b + s
        # 取最小非负
        cand = [t for t in (t1, t2) if t >= 0.0]
        if not cand:
            return None
        return float(min(cand))
    except Exception:
        return None

def _calculate_ellipsoid_volume(contour: np.ndarray, debug: bool = False) -> Tuple[Optional[float], Optional[dict]]:
    """
    使用椭圆拟合计算椭圆体体积。
    
    Args:
        contour: 输入轮廓
        debug: 调试模式
    
    Returns:
        (volume_ellipsoid_px3, ellipse_info)
        volume_ellipsoid_px3: 椭圆体体积 (像素^3)
        ellipse_info: 椭圆参数信息字典
    """
    try:
        if len(contour) < 5:
            return None, None
        
        # 简单椭圆拟合（OpenCV最小二乘）
        ellipse = cv2.fitEllipse(contour)
        if ellipse is None:
            return None, None
        (cx, cy), (major_axis, minor_axis), angle = ellipse
        
        # 计算椭圆的长短半轴（像素单位）
        a = major_axis / 2.0  # 长半轴
        b = minor_axis / 2.0  # 短半轴
        
        # 假设椭圆体是旋转椭球，第三轴等于短半轴
        c = b  # 第三轴（深度方向）
        
        # 计算椭圆体体积: V = (4/3) * π * a * b * c
        volume_ellipsoid = (4.0 / 3.0) * math.pi * a * b * c
        
        ellipse_info = {
            'center_x': float(cx),
            'center_y': float(cy),
            'major_axis_px': float(major_axis),
            'minor_axis_px': float(minor_axis),
            'angle_deg': float(angle),
            'semi_major_a_px': float(a),
            'semi_minor_b_px': float(b),
            'semi_depth_c_px': float(c),
            'volume_ellipsoid_px3': float(volume_ellipsoid)
        }
        
        if debug:
            print(f"椭圆拟合: 中心=({cx:.2f}, {cy:.2f}), "
                  f"长轴={major_axis:.2f}, 短轴={minor_axis:.2f}, "
                  f"角度={angle:.2f}°, 体积={volume_ellipsoid:.2f}")
        
        return float(volume_ellipsoid), ellipse_info
        
    except Exception as e:
        if debug:
            print(f"椭圆拟合失败: {e}")
        return None, None





def _make_overlay_ellipse(gray: np.ndarray, contour: Optional[np.ndarray], ellipse_info: Optional[dict]) -> np.ndarray:
    """返回 BGR 叠加图：
    - 绿色椭圆轮廓 + 红色中心
    - 深色半透明填充弧形缺损区域（支持多段）
    - 方向箭头与少量调试文字
    """
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if contour is None or ellipse_info is None:
        return bgr

    try:
        # 通用参数读取
        cx = ellipse_info.get('center_x', ellipse_info.get('ellipse_center_x'))
        cy = ellipse_info.get('center_y', ellipse_info.get('ellipse_center_y'))
        major = ellipse_info.get('major_axis_px')
        minor = ellipse_info.get('minor_axis_px')
        angle = float(ellipse_info.get('angle_deg', 0.0))
        if cx is None or cy is None or major is None or minor is None:
            return bgr

        center = (int(round(cx)), int(round(cy)))
        axes = (int(round(major / 2)), int(round(minor / 2)))
        cv2.ellipse(bgr, center, axes, angle, 0, 360, (0, 255, 0), 1)
        cv2.circle(bgr, center, 2, (0, 0, 255), -1)

        # 先尝试按多段多边形直接填充（优先，最可靠）
        overlay = None
        try:
            segs = ellipse_info.get('defect_segments')
            if isinstance(segs, list) and segs:
                overlay = bgr.copy()
                for seg in segs:
                    poly = seg.get('poly_img')
                    if poly is None:
                        continue
                    poly_np = np.asarray(poly, dtype=np.int32)
                    if poly_np.ndim == 2:
                        poly_np = poly_np.reshape(-1, 1, 2)
                    if poly_np.shape[0] >= 3:
                        # 缺损区域改为红色填充
                        cv2.fillPoly(overlay, [poly_np], (0, 0, 255))
        except Exception:
            overlay = None

        # 若没有段多边形，则回退到基于采样数组构造多边形
        if overlay is None:
            try:
                th = ellipse_info.get('theta_samples')
                rI = ellipse_info.get('r_ideal_samples')
                rA = ellipse_info.get('r_actual_samples')
                dM = ellipse_info.get('defect_mask_samples')
                if th is not None and rI is not None and rA is not None and dM is not None:
                    th = np.asarray(th, dtype=np.float64)
                    rI = np.asarray(rI, dtype=np.float64)
                    rA = np.asarray(rA, dtype=np.float64)
                    dM = np.asarray(dM, dtype=bool)
                    n = len(th)
                    if n >= 3 and rI.shape[0] == n and rA.shape[0] == n and dM.shape[0] == n:
                        overlay = bgr.copy()
                        ang = math.radians(angle)
                        c, s = math.cos(ang), math.sin(ang)
                        def _true_segments(mask: np.ndarray) -> List[Tuple[int, int]]:
                            if mask.size == 0 or not mask.any():
                                return []
                            segs = []
                            in_seg = False
                            st = 0
                            for i in range(n):
                                if mask[i] and not in_seg:
                                    in_seg = True
                                    st = i
                                elif not mask[i] and in_seg:
                                    segs.append((st, i - 1))
                                    in_seg = False
                            if in_seg:
                                if mask[0] and segs:
                                    s0, e0 = segs[0]
                                    segs[0] = (st, e0)
                                else:
                                    segs.append((st, n - 1))
                            return segs
                        for s0, e0 in _true_segments(dM):
                            idxs = np.arange(s0, e0 + 1)
                            # 外弧：thL -> thR
                            xo = rI[idxs] * np.cos(th[idxs]); yo = rI[idxs] * np.sin(th[idxs])
                            x_out = cx + xo * c - yo * s; y_out = cy + xo * s + yo * c
                            # 内弧：反向 thR -> thL，半径与角度需同步反转
                            th_rev = th[idxs][::-1]
                            rA_rev = rA[idxs][::-1]
                            xi = rA_rev * np.cos(th_rev); yi = rA_rev * np.sin(th_rev)
                            x_in = cx + xi * c - yi * s; y_in = cy + xi * s + yi * c
                            poly_x = np.concatenate([x_out, x_in]); poly_y = np.concatenate([y_out, y_in])
                            poly = np.vstack([poly_x, poly_y]).T.astype(np.int32)
                            if poly.shape[0] >= 3:
                                # 缺损区域改为红色填充
                                cv2.fillPoly(overlay, [poly], (0, 0, 255))
            except Exception:
                overlay = None

        if overlay is not None:
            bgr = cv2.addWeighted(bgr, 0.7, overlay, 0.3, 0)

        # ===== 新增：绘制原始点（OpenCV 叠加图） =====
        try:
            draw_pts = bool(getattr(config, 'OVERLAY_DRAW_RAW_POINTS', True))
            if draw_pts and contour is not None and len(contour) > 0:
                step = int(getattr(config, 'OVERLAY_RAW_POINTS_STEP', 1) or 1)
                # BGR 颜色
                color = getattr(config, 'OVERLAY_RAW_POINTS_COLOR', (255, 255, 0))  # 青色
                if isinstance(color, (list, tuple)) and len(color) == 3:
                    color = tuple(int(v) for v in color)
                else:
                    color = (255, 255, 0)
                radius = int(getattr(config, 'OVERLAY_RAW_POINTS_RADIUS', 1) or 1)
                thickness = int(getattr(config, 'OVERLAY_RAW_POINTS_THICKNESS', -1) or -1)  # -1 实心点
                pts = np.asarray(contour).reshape(-1, 2)
                for i in range(0, len(pts), step):
                    x, y = int(round(pts[i, 0])), int(round(pts[i, 1]))
                    cv2.circle(bgr, (x, y), radius, color, thickness, lineType=cv2.LINE_AA)
        except Exception:
            pass
        # ===== 新增部分结束 =====

        # ===== 新增：在暗色 overlay 上可视化所有收缩因子 =====
        try:
            th = ellipse_info.get('theta_samples')
            rI = ellipse_info.get('r_ideal_samples')
            rA = ellipse_info.get('r_actual_samples')
            shrink = ellipse_info.get('shrink_factors')
            if th is not None and rI is not None and shrink is not None:
                th = np.asarray(th, dtype=np.float64)
                rI = np.asarray(rI, dtype=np.float64)
                shrink = np.asarray(shrink, dtype=np.float64)
                n = len(th)
                if rA is not None and len(rA) == n:
                    rA = np.asarray(rA, dtype=np.float64)
                else:
                    rA = shrink * rI  # 回退：由 shrink 重建实际半径

                if n >= 8 and rI.shape[0] == n and rA.shape[0] == n:
                    # 整体压暗背景，突出可视化
                    h, w = bgr.shape[:2]
                    dark = bgr.copy()
                    cv2.rectangle(dark, (0, 0), (w, h), (0, 0, 0), -1)
                    bgr = cv2.addWeighted(bgr, 0.6, dark, 0.4, 0)

                    ang = math.radians(angle)
                    cA, sA = math.cos(ang), math.sin(ang)

                    def color_from_shrink(v: float) -> Tuple[int, int, int]:
                        vv = float(np.clip(v, 0.0, 1.0))
                        # 0 -> 红(0,0,255), 1 -> 绿(0,255,0)
                        g = int(round(255 * vv))
                        r = int(round(255 * (1.0 - vv)))
                        return (0, g, r)  # BGR

                    # 画每个 θ 的径向线（实际半径 -> 理想半径）
                    for i in range(n):
                        ct, st = math.cos(th[i]), math.sin(th[i])
                        xo, yo = rI[i] * ct, rI[i] * st
                        xi, yi = rA[i] * ct, rA[i] * st
                        x_out = cx + xo * cA - yo * sA
                        y_out = cy + xo * sA + yo * cA
                        x_in = cx + xi * cA - yi * sA
                        y_in = cy + xi * sA + yi * cA
                        p_in = (int(round(x_in)), int(round(y_in)))
                        p_out = (int(round(x_out)), int(round(y_out)))
                        color = color_from_shrink(float(shrink[i]))
                        cv2.line(bgr, p_in, p_out, color, 1)

                    # 每隔固定角度标注 shrink 数值
                    step = max(12, n // 24)  # 大约每 15°
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    for i in range(0, n, step):
                        ct, st = math.cos(th[i]), math.sin(th[i])
                        ri = rA[i]
                        xi, yi = ri * ct, ri * st
                        x_txt = cx + xi * cA - yi * sA
                        y_txt = cy + xi * sA + yi * cA
                        txt = f"{float(shrink[i]):.2f}"
                        cv2.putText(bgr, txt, (int(round(x_txt)), int(round(y_txt))), font, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

                    # 左上角画一个渐变图例（红->绿，代表 0->1）
                    legend_w, legend_h = 120, 10
                    x0, y0 = 10, 10
                    for k in range(legend_w):
                        v = k / max(legend_w - 1, 1)
                        cv2.line(bgr, (x0 + k, y0), (x0 + k, y0 + legend_h), color_from_shrink(v), 1)
                    cv2.rectangle(bgr, (x0, y0), (x0 + legend_w, y0 + legend_h), (255, 255, 255), 1)
                    cv2.putText(bgr, "shrink 0->1", (x0, y0 + legend_h + 14), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        except Exception:
            pass
        # ===== 新增部分结束 =====

        # 轴线与方向箭头
        try:
            a = float(ellipse_info.get('semi_a_px', major / 2.0))
            b = float(ellipse_info.get('semi_b_px', minor / 2.0))
            th = math.radians(angle)
            ux, uy = math.cos(th), math.sin(th)
            vx, vy = -math.sin(th), math.cos(th)
            cx_f, cy_f = float(cx), float(cy)
            p1 = (int(round(cx_f - a * ux)), int(round(cy_f - a * uy)))
            p2 = (int(round(cx_f + a * ux)), int(round(cy_f + a * uy)))
            q1 = (int(round(cx_f - b * vx)), int(round(cy_f - b * vy)))
            q2 = (int(round(cx_f + b * vx)), int(round(cy_f + b * vy)))
            cv2.line(bgr, p1, p2, (255, 0, 0), 1)
            cv2.line(bgr, q1, q2, (255, 255, 0), 1)

            # 优先：指向“椭圆在外”的最大差值点
            mx = ellipse_info.get('max_diff_outer_point_img_x', ellipse_info.get('max_diff_point_img_x'))
            my = ellipse_info.get('max_diff_outer_point_img_y', ellipse_info.get('max_diff_point_img_y'))
            if mx is not None and my is not None:
                tip = (int(round(float(mx))), int(round(float(my))))
                cv2.arrowedLine(bgr, (int(round(cx_f)), int(round(cy_f))), tip, (255, 200, 0), 1, tipLength=0.08)
            else:
                # 次优：用索引重建（取椭圆半径 r_ell）
                idx = ellipse_info.get('max_diff_idx')
                th_samples = ellipse_info.get('theta_samples')
                rI = ellipse_info.get('r_ideal_samples')
                if idx is not None and th_samples is not None and rI is not None:
                    th_samples = np.asarray(th_samples, dtype=np.float64)
                    rI = np.asarray(rI, dtype=np.float64)
                    ii = int(idx) % len(th_samples)
                    th_loc = float(th_samples[ii])
                    ct, st = math.cos(th_loc), math.sin(th_loc)
                    # 局部->图像旋转
                    cA, sA = math.cos(math.radians(float(angle))), math.sin(math.radians(float(angle)))
                    r_tip = float(rI[ii])  # 椭圆半径
                    x_tip = cx_f + (r_tip * ct) * cA - (r_tip * st) * sA
                    y_tip = cy_f + (r_tip * ct) * sA + (r_tip * st) * cA
                    tip = (int(round(x_tip)), int(round(y_tip)))
                    cv2.arrowedLine(bgr, (int(round(cx_f)), int(round(cy_f))), tip, (255, 200, 0), 1, tipLength=0.08)
                else:
                    # 最后回退：用方向角，长度按该方向的椭圆极半径
                    dir_deg = ellipse_info.get('trunc_direction_img_deg')
                    if dir_deg is not None:
                        ang_r = math.radians(float(dir_deg))
                        th_loc = ang_r - math.radians(float(angle))
                        r_dir = (a * b) / math.sqrt((b * math.cos(th_loc)) ** 2 + (a * math.sin(th_loc)) ** 2 + 1e-12)
                        end_pt = (int(round(cx_f + r_dir * math.cos(ang_r))), int(round(cy_f + r_dir * math.sin(ang_r))))
                        cv2.arrowedLine(bgr, (int(round(cx_f)), int(round(cy_f))), end_pt, (255, 200, 0), 1, tipLength=0.08)
        except Exception:
            pass

        # ===== 新增：绘制缺损圆弧（若存在）=====
        try:
            arc = ellipse_info.get('defect_arc')
            if isinstance(arc, dict):
                ccx = float(arc.get('center_x'))
                ccy = float(arc.get('center_y'))
                R = float(arc.get('radius_px'))
                thL = float(arc.get('theta_left_rad'))
                thR = float(arc.get('theta_right_rad'))
                span = (thR - thL) % (2.0 * math.pi)
                m = max(24, int(span / (2.0 * math.pi) * 256))
                thetas = thL + np.linspace(0.0, span, m)
                ang_deg = float(ellipse_info.get('angle_deg', angle))
                cx_f = float(ellipse_info.get('ellipse_center_x', cx))
                cy_f = float(ellipse_info.get('ellipse_center_y', cy))
                xs, ys = [], []
                for th in thetas:
                    phi = th + math.radians(ang_deg)
                    dx, dy = math.cos(phi), math.sin(phi)
                    t = _ray_circle_intersection((cx_f, cy_f), (dx, dy), (ccx, ccy), R)
                    if t is None:
                        continue
                    xs.append(cx_f + t * dx)
                    ys.append(cy_f + t * dy)
                if len(xs) >= 2:
                    pts = np.vstack([xs, ys]).T.astype(np.int32).reshape(-1, 1, 2)
                    cv2.polylines(bgr, [pts], False, (0, 255, 255), 1, cv2.LINE_AA)
        except Exception:
            pass

        # ===== 新增：绘制截平面（来自球冠参数）=====
        try:
            # 若有圆弧，默认不画平面，可通过开关强制
            has_arc = isinstance(ellipse_info.get('defect_arc'), dict)
            draw_plane_default = False if has_arc else True
            draw_plane = bool(getattr(config, 'OVERLAY_DRAW_CAP_PLANE', draw_plane_default))
            t = ellipse_info.get('cap_plane_offset_t_px')
            n_local = ellipse_info.get('cap_normal_local_xyz')
            ang_deg = float(ellipse_info.get('angle_deg', angle))
            if draw_plane and t is not None and n_local is not None and len(n_local) >= 2:
                t = float(t)
                nx, ny = float(n_local[0]), float(n_local[1])

                # 椭圆局部基向量 -> 图像坐标
                ang = math.radians(ang_deg)
                ux, uy = math.cos(ang), math.sin(ang)          # 长轴方向
                vx, vy = -math.sin(ang), math.cos(ang)         # 短轴方向

                # 法向（图像坐标）
                n_img_x = nx * ux + ny * vx
                n_img_y = nx * uy + ny * vy
                n_norm = math.hypot(n_img_x, n_img_y)
                if n_norm < 1e-6:
                    raise ValueError("cap normal too small")
                n_img_x /= n_norm
                n_img_y /= n_norm

                # 通过点 P0：从中心沿法向位移 t
                cx_f, cy_f = float(ellipse_info.get('ellipse_center_x', cx)), float(ellipse_info.get('ellipse_center_y', cy))
                p0x = cx_f + t * n_img_x
                p0y = cy_f + t * n_img_y

                # 直线方向向量（与法向垂直）
                dx, dy = -n_img_y, n_img_x
                L = 2 * max(bgr.shape[0], bgr.shape[1])
                x1, y1 = int(round(p0x - L * dx)), int(round(p0y - L * dy))
                x2, y2 = int(round(p0x + L * dx)), int(round(p0y + L * dy))

                # 画线与法向箭头
                color = (255, 0, 255)  # 洋红
                cv2.line(bgr, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
                tip_len = max(12, int(0.02 * max(bgr.shape[:2])))
                tipx = int(round(p0x + tip_len * n_img_x))
                tipy = int(round(p0y + tip_len * n_img_y))
                cv2.arrowedLine(bgr, (int(round(p0x)), int(round(p0y))), (tipx, tipy), color, 1, tipLength=0.25)

                # 标注
                cv2.putText(bgr, "cap plane", (int(round(p0x)) + 4, int(round(p0y)) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        except Exception:
            pass
        # ===== 新增结束 =====
        # 轻量调试文字
        try:
            lines = []
            method_used = ellipse_info.get('volume_method_used')
            if method_used:
                lines.append(f"method: {method_used}")
            v_full = ellipse_info.get('volume_ellipsoid_full_px3')
            v_fin = ellipse_info.get('volume_ellipsoid_truncated_px3') or ellipse_info.get('volume_ellipsoid_truncated_est_px3')
            loss = ellipse_info.get('volume_loss_ratio')
            span = ellipse_info.get('defect_span_deg')
            shrink = ellipse_info.get('main_shrink_factor')
            if shrink is not None:
                lines.append(f"shrink: {float(shrink):.3f}")
            if span is not None:
                lines.append(f"span: {float(span):.1f}°")
            if v_full is not None:
                lines.append(f"Vfull: {float(v_full):.0f}")
            if v_fin is not None:
                lines.append(f"V: {float(v_fin):.0f}")
            if loss is not None:
                lines.append(f"loss: {float(loss)*100:.1f}%")
            if lines:
                font = cv2.FONT_HERSHEY_SIMPLEX
                fs, thk, pad = 0.45, 1, 4
                widths, heights = [], []
                for t in lines:
                    (w, h), base = cv2.getTextSize(t, font, fs, thk)
                    widths.append(w)
                    heights.append(h + base)
                box_w = max(widths) + pad * 2
                box_h = sum(heights) + pad * 2 + (len(lines) - 1) * 2
                x0, y0 = 10, 10 + 26  # 避免盖住图例
                overlay2 = bgr.copy()
                cv2.rectangle(overlay2, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
                bgr = cv2.addWeighted(overlay2, 0.5, bgr, 0.5, 0)
                y = y0 + pad
                for i, t in enumerate(lines):
                    (w, h), base = cv2.getTextSize(t, font, fs, thk)
                    y += h + (2 if i > 0 else 0)
                    cv2.putText(bgr, t, (x0 + pad, y), font, fs, (255, 255, 255), thk, cv2.LINE_AA)
        except Exception:
            pass
    except Exception as e:
        print(f"绘制椭圆失败: {e}")

    return bgr


def _save_truncated_ellipse_plot(ellipse_info: dict, contour: np.ndarray, gray: np.ndarray, out_dir: str, base_name: Optional[str] = None) -> Optional[str]:
    """在 debug 目录保存一张包含原始灰度图、轮廓和拟合椭圆的图像。

    文件名: <base_name>_truncated_ellipse_fit.png（若提供 base_name），否则 trun_ellipse_fit.png
    返回保存路径或 None。
    """
    try:
        import os
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
        import numpy as np

        if ellipse_info is None:
            return None

        os.makedirs(out_dir, exist_ok=True)
        fname = f"{base_name + '_' if base_name else ''}truncated_ellipse_fit.png"
        out_path = os.path.join(out_dir, fname)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(gray, cmap='gray')

        # 绘制原始轮廓（红色线）
        try:
            if contour is not None:
                pts = np.asarray(contour).reshape(-1, 2)
                ax.plot(pts[:, 0], pts[:, 1], '-', color='red', linewidth=1)
        except Exception:
            pass

        # 绘制拟合椭圆（绿色）
        try:
            cx = float(ellipse_info.get('ellipse_center_x', ellipse_info.get('center_x')))
            cy = float(ellipse_info.get('ellipse_center_y', ellipse_info.get('center_y')))
            major = float(ellipse_info.get('major_axis_px', 0))
            minor = float(ellipse_info.get('minor_axis_px', 0))
            angle = float(ellipse_info.get('angle_deg', 0))
            ell = Ellipse((cx, cy), width=major, height=minor, angle=angle, edgecolor='lime', facecolor='none', linewidth=2)
            ax.add_patch(ell)
            ax.plot(cx, cy, 'yo', markersize=4)
        except Exception:
            pass

        # 弧线绘制（若存在）
        try:
            arc = ellipse_info.get('defect_arc')
            if isinstance(arc, dict):
                ccx = float(arc.get('center_x'))
                ccy = float(arc.get('center_y'))
                R = float(arc.get('radius_px'))
                thL = float(arc.get('theta_left_rad'))
                thR = float(arc.get('theta_right_rad'))
                ang_deg = float(ellipse_info.get('angle_deg', 0.0))
                th0 = np.linspace(0, 1, 128)
                span = (thR - thL) % (2*np.pi)
                thetas = (thL + th0 * span)
                # 用射线-圆交得到弧线点
                ca, sa = math.cos(math.radians(ang_deg)), math.sin(math.radians(ang_deg))
                xs, ys = [], []
                cx = float(ellipse_info.get('ellipse_center_x', ellipse_info.get('center_x')))
                cy = float(ellipse_info.get('ellipse_center_y', ellipse_info.get('center_y')))
                for th in thetas:
                    phi = th + math.radians(ang_deg)
                    dx, dy = math.cos(phi), math.sin(phi)
                    # 射线与圆交
                    t = _ray_circle_intersection((cx, cy), (dx, dy), (ccx, ccy), R)
                    if t is None:
                        continue
                    xs.append(cx + t * dx)
                    ys.append(cy + t * dy)
                if len(xs) >= 2:
                    ax.plot(xs, ys, color='yellow', linewidth=1.5, alpha=0.9)
        except Exception:
            pass

        # 用暗色半透明填充缺损区域（优先使用预构建的多边形段）
        try:
            segs = ellipse_info.get('defect_segments')
            if isinstance(segs, list) and segs:
                for seg in segs:
                    poly = seg.get('poly_img')
                    if poly is None:
                        continue
                    poly_np = np.asarray(poly, dtype=float)
                    if poly_np.shape[0] >= 3:
                        ax.fill(poly_np[:, 0], poly_np[:, 1], color='red', alpha=0.3, linewidth=0)
            else:
                # 回退：根据采样数组构造
                th = ellipse_info.get('theta_samples')
                rI = ellipse_info.get('r_ideal_samples')
                rA = ellipse_info.get('r_actual_samples')
                dM = ellipse_info.get('defect_mask_samples')
                shrink = ellipse_info.get('shrink_factors')
                if th is not None and rI is not None and dM is not None:
                    th = np.asarray(th, dtype=np.float64)
                    rI = np.asarray(rI, dtype=np.float64)
                    if rA is not None:
                        rA = np.asarray(rA, dtype=np.float64)
                    dM = np.asarray(dM, dtype=bool)
                    n = len(th)
                    if n > 2 and rI.shape[0] == n and dM.shape[0] == n and (rA is None or rA.shape[0] == n):
                        theta_rad = math.radians(angle)
                        cos_t, sin_t = math.cos(theta_rad), math.sin(theta_rad)
                        segs2 = []
                        in_seg = False
                        st = 0
                        for i in range(n):
                            if dM[i] and not in_seg:
                                in_seg = True
                                st = i
                            elif not dM[i] and in_seg:
                                segs2.append((st, i - 1))
                                in_seg = False
                        if in_seg:
                            if dM[0] and segs2:
                                s0, e0 = segs2[0]
                                segs2[0] = (st, e0)
                            else:
                                segs2.append((st, n - 1))

                        for s2, e2 in segs2:
                            idxs = np.arange(s2, e2 + 1)
                            # 外弧：thL -> thR
                            x_out_loc = rI[idxs] * np.cos(th[idxs])
                            y_out_loc = rI[idxs] * np.sin(th[idxs])
                            x_out_img = cx + x_out_loc * cos_t - y_out_loc * sin_t
                            y_out_img = cy + x_out_loc * sin_t + y_out_loc * cos_t
                            # 内弧：反向 thR -> thL
                            if shrink is not None and len(shrink) == n:
                                sA = np.clip(np.asarray(shrink, dtype=np.float64)[idxs], 0.0, 1.0)
                                r_inner = sA * rI[idxs]
                            elif rA is not None:
                                r_inner = rA[idxs]
                            else:
                                continue
                            th_rev = th[idxs][::-1]
                            r_rev = r_inner[::-1]
                            x_in_loc = r_rev * np.cos(th_rev)
                            y_in_loc = r_rev * np.sin(th_rev)
                            x_in_img = cx + x_in_loc * cos_t - y_in_loc * sin_t
                            y_in_img = cy + x_in_loc * sin_t + y_in_loc * cos_t
                            poly_x = np.concatenate([x_out_img, x_in_img])
                            poly_y = np.concatenate([y_out_img, y_in_img])
                            poly = np.vstack([poly_x, poly_y]).T
                            if poly.shape[0] >= 3:
                                ax.fill(poly[:, 0], poly[:, 1], color='red', alpha=0.3, linewidth=0)
        except Exception:
            pass

        # 注释信息
        try:
            af = ellipse_info.get('area_fraction')
            v_trunc = ellipse_info.get('volume_ellipsoid_truncated_px3') or ellipse_info.get('volume_ellipsoid_truncated_est_px3') or ellipse_info.get('volume_ellipsoid_px3')
            v_full = ellipse_info.get('volume_ellipsoid_full_px3')
            loss_ratio = ellipse_info.get('volume_loss_ratio')
            method_used = ellipse_info.get('volume_method_used')
            main_shrink = ellipse_info.get('main_shrink_factor')
            defect_span_deg = ellipse_info.get('defect_span_deg')
            lines = []
            if method_used is not None:
                lines.append(f"method: {method_used}")
            if main_shrink is not None:
                lines.append(f"shrink: {float(main_shrink):.3f}")
            if defect_span_deg is not None:
                lines.append(f"defect_span: {float(defect_span_deg):.1f}°")
            if af is not None:
                lines.append(f"area_frac: {float(af):.3f}")
            if v_full is not None:
                lines.append(f"V_full: {float(v_full):.0f}")
            if v_trunc is not None:
                lines.append(f"V_final: {float(v_trunc):.0f}")
            if loss_ratio is not None:
                lines.append(f"loss: {float(loss_ratio)*100:.1f}%")
            if lines:
                ax.text(0.02, 0.98, "\n".join(lines), color='white', fontsize=8, transform=ax.transAxes, va='top', ha='left', bbox=dict(facecolor='black', alpha=0.5))
        except Exception:
            pass

        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return out_path
    except Exception:
        return None
def _calculate_ellipsoid_truncated_volume(contour: np.ndarray, debug: bool = False) -> Tuple[Optional[float], Optional[dict]]:
    """基于“椭圆包络 vs 原始包络”差值的截断椭球估计（单段主缺损）。

    步骤：
    1) 直接用输入轮廓拟合椭圆（不对轮廓做预处理）；
    2) 计算轮廓的原始包络：在拟合椭圆的局部坐标系中，用原始轮廓点对每个角度的径向投影最大值 r_env(θ)；
    3) 与椭圆半径 r_ell(θ) 比较，d(θ)=r_ell-r_env，取 d 最大处为缺损起点；
    4) 从起点向左右扩展至 d(θ)=0 的两处交点（线性内插角度），两交点之间记为缺损；
    5) 用缺损弧段构造“外弧(椭圆)+内弧(包络)”闭合多边形，体积用径向缩放积分。
    """
    try:
        if contour is None or len(contour) < 5:
            return None, None

        # 1) 椭圆拟合（直接使用原始轮廓）
        try:
            ellipse = cv2.fitEllipse(contour)
        except Exception:
            return None, None

        (cx, cy), (major_axis, minor_axis), angle_deg = ellipse
        a = float(major_axis) / 2.0
        b = float(minor_axis) / 2.0
        if a <= 1e-6 or b <= 1e-6:
            return None, None

        # 深度轴 c = aspect * b
        depth_aspect = float(getattr(config, 'ELLIPSOID_DEPTH_ASPECT', 1.0) or 1.0)
        c = depth_aspect * b

        # 原始点 -> 局部坐标（长轴=x'，短轴=y'）
        pts_img = contour.reshape(-1, 2).astype(np.float64)
        theta_rot = math.radians(angle_deg)
        cos_m, sin_m = math.cos(-theta_rot), math.sin(-theta_rot)
        rel = pts_img - np.array([cx, cy], dtype=np.float64)
        x_loc = rel[:, 0] * cos_m - rel[:, 1] * sin_m
        y_loc = rel[:, 0] * sin_m + rel[:, 1] * cos_m

        # 3) 角度采样，椭圆半径 r_ell 与 原始包络半径 r_env（直接用原始点）
        N = int(getattr(config, 'ELLIPSE_ANGULAR_SAMPLES', 360) or 360)
        N = int(np.clip(N, 72, 720))
        theta = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)

        # 椭圆极半径 r_ell(θ) = ab / sqrt((b cosθ)^2 + (a sinθ)^2)
        r_ell = (a * b) / np.sqrt((b * np.cos(theta)) ** 2 + (a * np.sin(theta)) ** 2)

        # 原始包络 r_env(θ)：用“射线-多边形交点”的最近正向半径（避免投影最大值导致凸包/平滑）
        def _ray_envelope(theta_arr: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
            """
            对每个 θ，从原点(椭圆中心)沿方向 θ 发射射线，计算与轮廓折线的交点；
            取最近的正向交点半径作为 r_env(θ)。
            若与某些线段共线，则取该线段端点在射线方向上的最近正向半径。
            无交则记为 NaN（后续会忽略这些角度）。
            """
            M = x.shape[0]
            r_out = np.full_like(theta_arr, np.nan, dtype=np.float64)
            if M < 2:
                return r_out

            # 连成闭合折线
            dx = np.roll(x, -1) - x
            dy = np.roll(y, -1) - y

            for j, th in enumerate(theta_arr):
                ct, st = math.cos(th), math.sin(th)
                # 与方向向量 (ct,st) 垂直的法向 n = (-st, ct)
                nx, ny = -st, ct

                # 线段参数式 P(t)=P0 + t*(P1-P0)，与过原点、方向 (ct,st) 的直线相交条件是 P(t)·n = 0
                denom = dx * nx + dy * ny                        # (P1-P0)·n
                num = -(x * nx + y * ny)                         # -P0·n
                with np.errstate(divide='ignore', invalid='ignore'):
                    t = num / denom                               # 交点参数 t

                # 真正相交 t∈[0,1]
                mask_t = (~np.isclose(denom, 0.0)) & (t >= 0.0) & (t <= 1.0)

                # 候选交点沿射线方向的半径 r
                r_list = []
                if np.any(mask_t):
                    r_cand = (x + t * dx) * ct + (y + t * dy) * st
                    r_pos = r_cand[mask_t]
                    r_pos = r_pos[r_pos >= 0.0]
                    if r_pos.size:
                        r_list.append(np.min(r_pos))

                # 处理与射线共线（重合）的线段：denom≈0 且 num≈0
                mask_col = np.isclose(denom, 0.0) & np.isclose(num, 0.0)
                if np.any(mask_col):
                    r0 = x[mask_col] * ct + y[mask_col] * st
                    r1 = (x[mask_col] + dx[mask_col]) * ct + (y[mask_col] + dy[mask_col]) * st
                    r_end = np.concatenate([r0, r1])
                    r_end = r_end[r_end >= 0.0]
                    if r_end.size:
                        r_list.append(np.min(r_end))

                if r_list:
                    r_out[j] = float(np.min(r_list))  # 最近的正向半径
                else:
                    r_out[j] = np.nan                  # 无交：标记为 NaN

            return r_out

        r_env = _ray_envelope(theta, x_loc, y_loc)

        # 差值 d(θ) 与收缩 s(θ) —— 严格使用“拟合椭圆 - 原始轮廓”，不取绝对值
        valid = np.isfinite(r_env) & (r_env > 1e-9)  # 仅有有效交点的角度
        d = r_ell - r_env
        d_masked = np.where(valid, d, -np.inf)       # 仅在有效角度上找最大差值
        shrink = np.divide(r_env, r_ell, out=np.ones_like(r_env), where=r_ell > 1e-9)
        shrink = np.clip(shrink, 0.0, 1.0)
        shrink[~valid] = 1.0                         # 无交角度不收缩

        # 4) 缺损起点：在有效角度上取 d 最大处；若最大值不超过阈值，则认为无缺损
        i0 = int(np.argmax(d_masked))
        d_max = float(d_masked[i0])
        # 最大差值角度
        th_i0 = float(theta[i0])
        cA_vis, sA_vis = math.cos(theta_rot), math.sin(theta_rot)
        ct_i0, st_i0 = math.cos(th_i0), math.sin(th_i0)
        # 椭圆外侧端点（你需要的“椭圆在外”的点）
        r_tip_outer_i0 = float(r_ell[i0])
        x_tip_outer_i0 = float(cx + (r_tip_outer_i0 * ct_i0) * cA_vis - (r_tip_outer_i0 * st_i0) * sA_vis)
        y_tip_outer_i0 = float(cy + (r_tip_outer_i0 * ct_i0) * sA_vis + (r_tip_outer_i0) * st_i0 * cA_vis)
        # 同时计算原始包络内侧端点（可做参考/可视化）
        r_env_i0 = float(r_env[i0]) if np.isfinite(r_env[i0]) else np.nan
        if np.isfinite(r_env_i0):
            x_tip_inner_i0 = float(cx + (r_env_i0 * ct_i0) * cA_vis - (r_env_i0 * st_i0) * sA_vis)
            y_tip_inner_i0 = float(cy + (r_env_i0 * ct_i0) * sA_vis + (r_env_i0 * st_i0) * cA_vis)
        else:
            x_tip_inner_i0 = None
            y_tip_inner_i0 = None
        dir_img_deg_i0 = float((math.degrees(theta_rot + th_i0) + 360.0) % 360.0)

        # 边界判定阈值（像素）：d <= diff_thresh 视为达到边界
        diff_thresh = float(getattr(config, 'ELLIPSE_BOUNDARY_DIFF_THRESH_PX', 5) or 5)

        if not np.isfinite(d_max) or d_max <= diff_thresh:
            # 无缺损
            V_full = float((4.0 / 3.0) * math.pi * a * b * c)
            info = {
                'ellipse_center_x': float(cx),
                'ellipse_center_y': float(cy),
                'major_axis_px': float(major_axis),
                'minor_axis_px': float(minor_axis),
                'angle_deg': float(angle_deg),
                'semi_a_px': float(a),
                'semi_b_px': float(b),
                'semi_c_px': float(c),
                'volume_ellipsoid_full_px3': float(V_full),
                'volume_ellipsoid_truncated_px3': float(V_full),
                'volume_ellipsoid_missing_caps_px3': 0.0,
                'volume_ellipsoid_truncated_est_px3': float(V_full),
                'depth_aspect': float(depth_aspect),
                'trunc_fit_method': 'ellipse_envelope_diff',
                'trunc_direction_img_deg': dir_img_deg_i0,
                'trunc_direction_method': 'max_diff',
                'ang_bins': int(N),
                'theta_samples': theta.tolist(),
                'r_ideal_samples': r_ell.tolist(),
                'r_actual_samples': np.where(np.isfinite(r_env), r_env, 0.0).tolist(),
                'shrink_factors': shrink.tolist(),
                'defect_mask_samples': [False] * N,
                'defect_segments': [],
                'volume_method_used': 'ellipse_envelope_diff',
                'volume_loss_ratio': 0.0,
                'main_shrink_factor': 1.0,
                'defect_span_rad': 0.0,
                'defect_span_deg': 0.0,
                'envelope_type': 'contour_radial',
                'boundary_diff_threshold_px': float(diff_thresh),
                # 最大差值点（以椭圆外侧为准）
                'max_diff_idx': int(i0),
                'max_diff_theta_local_deg': float((math.degrees(th_i0) + 360.0) % 360.0),
                'max_diff_direction_img_deg': dir_img_deg_i0,
                'max_diff_point_img_x': x_tip_outer_i0,   # 改为外侧椭圆点
                'max_diff_point_img_y': y_tip_outer_i0,
                'max_diff_outer_point_img_x': x_tip_outer_i0,
                'max_diff_outer_point_img_y': y_tip_outer_i0,
                'max_diff_inner_point_img_x': x_tip_inner_i0,
                'max_diff_inner_point_img_y': y_tip_inner_i0,
            }
            # 附带面积参考
            try:
                area_contour = float(cv2.contourArea(contour))
                area_ellipse = float(math.pi * a * b)
                info['area_contour_px2'] = area_contour
                info['area_ellipse_px2'] = area_ellipse
                info['area_fraction'] = float(np.clip(area_contour / max(area_ellipse, 1e-6), 0.0, 1.0))
            except Exception:
                pass
            if debug:
                print("未检测到缺损（d_max<=0）。")
            return V_full, info

        # 5) 向左右扩散找到 d 下降至阈值的“边界”（线性内插角度）
        TWO_PI = 2.0 * np.pi
        tol_zero = float(getattr(config, 'ELLIPSE_ZERO_CROSS_TOL', 1e-6) or 1e-6)

        def _ang_step(a_from: float, a_to: float, dir_sign: int) -> float:
            # 计算从 a_from 朝 dir_sign 方向前进到 a_to 的有向角度差
            if dir_sign >= 0:
                return (a_to - a_from) % TWO_PI
            else:
                return (a_from - a_to) % TWO_PI

        def _interp_zero_between(i_pos: int, i_neg: int, dir_sign: int, thr: float) -> Tuple[float, float]:
            """
            i_pos: d>thr 的有效样本索引
            i_neg: d<=thr 的有效样本索引
            dir_sign: +1 表示沿角度递增方向插值；-1 表示沿角度递减方向插值
            返回 (theta_z, r_z)，r_z 为 r_ell 与 r_env 线性插值后取平均
            """
            d1 = float(d[i_pos]); d2 = float(d[i_neg])
            denom = (d1 - d2)
            # 阈值插值：当 d 从 >thr 下降到 <=thr 时，求达阈值处的角度比例
            thr = float(thr)
            alpha = ((d1 - thr) / denom) if abs(denom) > 1e-12 else 0.0

            t1 = float(theta[i_pos]); t2 = float(theta[i_neg])
            dt = _ang_step(t1, t2, dir_sign)
            if dir_sign >= 0:
                theta_z = (t1 + alpha * dt) % TWO_PI
            else:
                theta_z = (t1 - alpha * dt) % TWO_PI

            rE1, rE2 = float(r_ell[i_pos]), float(r_ell[i_neg])
            rR1, rR2 = float(r_env[i_pos]), float(r_env[i_neg])
            r_ell_z = rE1 + alpha * (rE2 - rE1)
            r_env_z = rR1 + alpha * (rR2 - rR1)
            r_z = 0.5 * (r_ell_z + r_env_z)
            return float(theta_z), float(r_z)

        def _search_boundary(i_center: int, dir_sign: int, thr: float, strict: bool = False) -> Tuple[Optional[float], Optional[float], Optional[int], Optional[int]]:
                """
                从 i_center 出发，沿 dir_sign（+1 右/递增，-1 左/递减）寻找首个“高于阈值 -> 达到阈值”的有效过界点。
                参数:
                  - thr: 阈值（第一次搜索传 0.0；重拟合后传 diff_thresh）
                  - strict: True 表示严格使用 d < thr 作为到达条件；False 表示使用 d <= thr。
                返回 (theta_z, r_z, idx_pos, idx_neg)，若找不到返回 (None, None, None, None)
                """
                assert valid[i_center] and d[i_center] > thr
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

        # 左右边界搜索（第一次严格过零：thr=0.0，用于后续一次性重拟合更稳健）
        thr0 = 0.0
        thL, rL, _pL, _nL = _search_boundary(i0, dir_sign=-1, thr=thr0, strict=True)
        thR, rR, _pR, _nR = _search_boundary(i0, dir_sign=+1, thr=thr0, strict=True)

        # 回退：若某侧未找到交点，则取该方向上 d 的全局最小值处近似为边界（避免“持续扩散”）
        if thL is None:
            # 沿左侧方向（递减）扫描时遇到的有效索引集合
            idxs_left = [(i0 - k) % N for k in range(1, N) if valid[(i0 - k) % N]]
            if idxs_left:
                i_minL = int(min(idxs_left, key=lambda idx: float(d[idx])))
                thL = float(theta[i_minL])
                rL = float(r_env[i_minL]) if np.isfinite(r_env[i_minL]) else float(r_ell[i_minL])
            else:
                # 实在无有效点：取一个很小弧宽作为回退
                w = max(2, N // 36)
                i_tmp = (i0 - w) % N
                thL = float(theta[i_tmp])
                rL = float(r_env[i_tmp]) if np.isfinite(r_env[i_tmp]) else float(r_ell[i_tmp])

        if thR is None:
            idxs_right = [(i0 + k) % N for k in range(1, N) if valid[(i0 + k) % N]]
            if idxs_right:
                i_minR = int(min(idxs_right, key=lambda idx: float(d[idx])))
                thR = float(theta[i_minR])
                rR = float(r_env[i_minR]) if np.isfinite(r_env[i_minR]) else float(r_ell[i_minR])
            else:
                w = max(2, N // 36)
                i_tmp = (i0 + w) % N
                thR = float(theta[i_tmp])
                rR = float(r_env[i_tmp]) if np.isfinite(r_env[i_tmp]) else float(r_ell[i_tmp])

        # 标准化弧段
        span = (thR - thL) % TWO_PI
        if span <= 0.0:
            thL, thR = thR, thL
            span = (thR - thL) % TWO_PI

        # ========= 一次性重拟合：剔除缺损角度区间内的“内侧点”，再拟合椭圆并重复边界搜索 =========
        try:
            do_refit = bool(getattr(config, 'ELLIPSE_REFIT_AFTER_DEFECT', True))
        except Exception:
            do_refit = True

        refit_used = False
        refit_removed_pts = 0
        refit_kept_pts = int(pts_img.shape[0])

        if do_refit and span > 0.0:
            # 以当前椭圆的局部坐标/角度来判断哪些点位于缺损角度区间内
            # 点的角度与半径（相对于当前椭圆中心/朝向）
            ang_pts = (np.arctan2(y_loc, x_loc) + TWO_PI) % TWO_PI
            r_pts = np.hypot(x_loc, y_loc)

            # 当前椭圆在各点角度下的极半径
            r_ell_at_pts = (a * b) / np.sqrt((b * np.cos(ang_pts)) ** 2 + (a * np.sin(ang_pts)) ** 2 + 1e-18)

            # 判断角度是否在弧段 thL->thR 内（与上文 mask 的定义保持一致）
            in_arc = (((ang_pts - thL) % TWO_PI) <= span) & (((thR - ang_pts) % TWO_PI) < (TWO_PI - 1e-9))

            # 只剔除“在弧段内且位于椭圆内侧”的点
            margin_px = float(getattr(config, 'ELLIPSE_REFIT_INNER_MARGIN_PX', 1.5) or 1.5)
            remove_mask = in_arc & (r_pts <= (r_ell_at_pts + margin_px))

            total_pts = int(pts_img.shape[0])
            keep_mask = ~remove_mask
            kept = int(np.count_nonzero(keep_mask))
            refit_removed_pts = int(np.count_nonzero(remove_mask))
            refit_kept_pts = kept

            min_keep_pts = int(getattr(config, 'ELLIPSE_REFIT_MIN_KEEP_POINTS', 20) or 20)
            min_keep_ratio = float(getattr(config, 'ELLIPSE_REFIT_MIN_KEEP_RATIO', 0.3) or 0.3)

            if kept >= max(5, min_keep_pts) and (kept / max(total_pts, 1)) >= min_keep_ratio:
                try:
                    # 对“保留点”重新拟合椭圆（注意：用原始 contour 的对应索引）
                    keep_idx = np.nonzero(keep_mask)[0]
                    contour_refit = contour[keep_idx]
                    ellipse2 = cv2.fitEllipse(contour_refit)

                    if ellipse2 is not None:
                        (cx2, cy2), (major2, minor2), angle_deg2 = ellipse2
                        a2 = float(major2) / 2.0
                        b2 = float(minor2) / 2.0
                        if a2 > 1e-6 and b2 > 1e-6:
                            # 用新椭圆替换参数，并对“原始轮廓点”在新局部坐标系下重新计算一次
                            cx, cy = float(cx2), float(cy2)
                            major_axis, minor_axis = float(major2), float(minor2)
                            angle_deg = float(angle_deg2)
                            a, b = a2, b2
                            c = depth_aspect * b

                            theta_rot = math.radians(angle_deg)
                            cos_m, sin_m = math.cos(-theta_rot), math.sin(-theta_rot)
                            rel = pts_img - np.array([cx, cy], dtype=np.float64)
                            x_loc = rel[:, 0] * cos_m - rel[:, 1] * sin_m
                            y_loc = rel[:, 0] * sin_m + rel[:, 1] * cos_m

                            # 以相同 θ 采样，重算 r_ell 与 r_env（r_env 仍基于“原始轮廓折线”的射线交点）
                            r_ell = (a * b) / np.sqrt((b * np.cos(theta)) ** 2 + (a * np.sin(theta)) ** 2)
                            r_env = _ray_envelope(theta, x_loc, y_loc)

                            valid = np.isfinite(r_env) & (r_env > 1e-9)
                            d = r_ell - r_env
                            d_masked = np.where(valid, d, -np.inf)
                            shrink = np.divide(r_env, r_ell, out=np.ones_like(r_env), where=r_ell > 1e-9)
                            shrink = np.clip(shrink, 0.0, 1.0)
                            shrink[~valid] = 1.0

                            # 重新确定最大差值中心
                            i0 = int(np.argmax(d_masked))
                            d_max = float(d_masked[i0])

                            if np.isfinite(d_max) and d_max > diff_thresh:
                                # 重新左右搜索边界（第二次使用阈值搜索：thr=diff_thresh）
                                thL, rL, _pL, _nL = _search_boundary(i0, dir_sign=-1, thr=diff_thresh, strict=False)
                                thR, rR, _pR, _nR = _search_boundary(i0, dir_sign=+1, thr=diff_thresh, strict=False)

                                # 回退策略与首次一致
                                if thL is None:
                                    idxs_left = [(i0 - k) % N for k in range(1, N) if valid[(i0 - k) % N]]
                                    if idxs_left:
                                        i_minL = int(min(idxs_left, key=lambda idx: float(d[idx])))
                                        thL = float(theta[i_minL])
                                        rL = float(r_env[i_minL]) if np.isfinite(r_env[i_minL]) else float(r_ell[i_minL])
                                    else:
                                        w = max(2, N // 36)
                                        i_tmp = (i0 - w) % N
                                        thL = float(theta[i_tmp])
                                        rL = float(r_env[i_tmp]) if np.isfinite(r_env[i_tmp]) else float(r_ell[i_tmp])

                                if thR is None:
                                    idxs_right = [(i0 + k) % N for k in range(1, N) if valid[(i0 + k) % N]]
                                    if idxs_right:
                                        i_minR = int(min(idxs_right, key=lambda idx: float(d[idx])))
                                        thR = float(theta[i_minR])
                                        rR = float(r_env[i_minR]) if np.isfinite(r_env[i_minR]) else float(r_ell[i_minR])
                                    else:
                                        w = max(2, N // 36)
                                        i_tmp = (i0 + w) % N
                                        thR = float(theta[i_tmp])
                                        rR = float(r_env[i_tmp]) if np.isfinite(r_env[i_tmp]) else float(r_ell[i_tmp])

                                # 标准化
                                span = (thR - thL) % TWO_PI
                                if span <= 0.0:
                                    thL, thR = thR, thL
                                    span = (thR - thL) % TWO_PI

                                refit_used = True
                            else:
                                # 重拟合后不存在正差值，视为无缺损（回退不采用重拟合结果）
                                pass
                except Exception:
                    pass
        # ========= 重拟合结束 =========

        # 缺损 mask（按 thL->thR 的正向角度区间）
        mask = np.zeros(N, dtype=bool)
        for i, th in enumerate(theta):
            if ((th - thL) % TWO_PI) <= span and ((thR - th) % TWO_PI) < (TWO_PI - 1e-9):
                mask[i] = True
        if not mask.any():
            # 极端回退
            iL = int(np.argmin(np.abs(((theta - thL + np.pi) % TWO_PI) - np.pi)))
            iR = int(np.argmin(np.abs(((theta - thR + np.pi) % TWO_PI) - np.pi)))
            if iL <= iR:
                mask[iL:iR + 1] = True
            else:
                mask[iL:] = True
                mask[:iR + 1] = True

        # 主方向（图像坐标）
        th_center = (thL + span / 2.0) % TWO_PI
        trunc_dir_img_deg = float((math.degrees(theta_rot + th_center) + 360.0) % 360.0)

        # 6) 缺损多边形（外弧椭圆，内弧包络）
        idxs = []
        for i, th in enumerate(theta):
            if ((th - thL) % TWO_PI) <= span and ((thR - th) % TWO_PI) < (TWO_PI - 1e-9):
                idxs.append(i)
        idxs = np.array(sorted(idxs, key=lambda k: ((theta[k] - thL) % TWO_PI)), dtype=int)

        th_arc_out = [thL]
        r_arc_out = [(a * b) / math.sqrt((b * math.cos(thL))**2 + (a * math.sin(thL))**2)]
        th_arc_in  = [thL]
        r_arc_in   = [rL]

        th_arc_out.extend(theta[idxs].tolist())
        r_arc_out.extend(r_ell[idxs].tolist())

        # 内弧加入时对 NaN 回退到椭圆半径，避免生成 NaN 顶点
        r_env_safe = np.where(np.isfinite(r_env[idxs]), r_env[idxs], r_ell[idxs])
        th_arc_in.extend(theta[idxs].tolist())
        r_arc_in.extend(r_env_safe.tolist())

        th_arc_out.append(thR)
        r_arc_out.append((a * b) / math.sqrt((b * math.cos(thR))**2 + (a * math.sin(thR))**2))
        th_arc_in.append(thR)
        r_arc_in.append(rR)

        cA, sA = math.cos(theta_rot), math.sin(theta_rot)
        xo = np.array(r_arc_out) * np.cos(np.array(th_arc_out))
        yo = np.array(r_arc_out) * np.sin(np.array(th_arc_out))
        x_out = cx + xo * cA - yo * sA
        y_out = cy + xo * sA + yo * cA

        xi = np.array(r_arc_in)[::-1] * np.cos(np.array(th_arc_in)[::-1])
        yi = np.array(r_arc_in)[::-1] * np.sin(np.array(th_arc_in)[::-1])
        x_in = cx + xi * cA - yi * sA
        y_in = cy + xi * sA + yi * cA

        poly_x = np.concatenate([x_out, x_in])
        poly_y = np.concatenate([y_out, y_in])
        poly = np.vstack([poly_x, poly_y]).T

        defect_segments = []
        if poly.shape[0] >= 3:
            defect_segments.append({
                'idx_start': int(idxs[0]) if idxs.size else int(i0),
                'idx_end': int(idxs[-1]) if idxs.size else int(i0),
                'poly_img': poly.tolist()
            })

        # 7) 平面截断体积计算
        V_full = float((4.0 / 3.0) * math.pi * a * b * c)

        # 在缺损角内收集“内边界点”的图像坐标，用圆拟合





        use_arc = False

        # 使用平面截断模型计算体积
        # 椭圆局部坐标中的截平面法向（缺损弧段中心方向）
        nx, ny = float(math.cos(th_center)), float(math.sin(th_center))

        # 用缺损角内的 r_env 做 t 的稳健估计：t ≈ median(r_env(θ) * cos(θ - th_center))
        cos_min = float(getattr(config, 'CAP_ESTIMATE_COS_MIN', 0.2) or 0.2)
        idx_cap = np.where(mask & np.isfinite(r_env))[0]
        proj_vals = []
        for k in idx_cap:
            dth = float(((theta[k] - th_center + np.pi) % (2.0 * np.pi)) - np.pi)
            cdh = math.cos(dth)
            if cdh >= cos_min:
                proj_vals.append(float(r_env[k] * cdh))
        if proj_vals:
            t_est = float(np.median(proj_vals))
        else:
            t_est = float(r_env[i0]) if np.isfinite(r_env[i0]) else float(r_ell[i0])

        N_norm = math.hypot(a * nx, b * ny)
        t_est = float(np.clip(t_est, 0.0, N_norm))

        d_unit = t_est / max(N_norm, 1e-9)
        H = float(np.clip(1.0 - d_unit, 0.0, 2.0))

        V_cap = float(a * b * c * math.pi * (H ** 2) * (1.0 - H / 3.0))
        V_final = float(np.clip(V_full - V_cap, 0.0, V_full))

        # 参考量
        try:
            area_contour = float(cv2.contourArea(contour))
        except Exception:
            area_contour = 0.0
        area_ellipse = float(math.pi * a * b)
        area_fraction = float(np.clip(area_contour / max(area_ellipse, 1e-6), 0.0, 1.0))

        # 统计
        main_shrink_factor = float(np.min(shrink[mask])) if mask.any() else 1.0
        defect_span = float(span)
        volume_loss_ratio = (V_full - V_final) / V_full if V_full > 1e-9 else 0.0

        # 与球冠方向一致的“高度”（沿法向的物理长度）
        cap_height_px = float(N_norm - t_est)

        info = {
            'ellipse_center_x': float(cx),
            'ellipse_center_y': float(cy),
            'major_axis_px': float(major_axis),
            'minor_axis_px': float(minor_axis),
            'angle_deg': float(angle_deg),
            'semi_a_px': float(a),
            'semi_b_px': float(b),
            'semi_c_px': float(c),
            'area_ellipse_px2': float(area_ellipse),
            'area_contour_px2': float(area_contour),
            'area_fraction': float(area_fraction),

            # 平面截断模型输出的截去高度（沿截面法向）
            'trunc_top_h_px': float(cap_height_px),
            'trunc_bottom_h_px': 0.0,
            'trunc_top_raw_px': float(cap_height_px),
            'trunc_bottom_raw_px': 0.0,
            'trunc_tolerance_px': 0.0,

            'volume_ellipsoid_full_px3': float(V_full),
            'volume_ellipsoid_truncated_px3': float(V_final),
            'volume_ellipsoid_missing_caps_px3': float(V_cap),
            'volume_ellipsoid_missing_top_px3': float(V_cap),
            'volume_ellipsoid_missing_bottom_px3': 0.0,
            'volume_ellipsoid_truncated_est_px3': float(V_final),

            'depth_aspect': float(depth_aspect),
            'trunc_fit_method': 'plane_truncation',
            'trunc_phi_local_deg': float((math.degrees(th_center) + 360.0) % 360.0),
            'trunc_direction_img_deg': float(trunc_dir_img_deg),
            'trunc_direction_method': 'max_diff',
            'ang_bins': int(N),

            'shrink_threshold': 1.0,
            'main_shrink_factor': float(main_shrink_factor),
            'defect_span_rad': float(defect_span),
            'defect_span_deg': float(np.degrees(defect_span)),
            'volume_method_used': 'plane_truncation',
            'volume_loss_ratio': float(volume_loss_ratio),
            'mean_shrink_factor': float(np.mean(shrink)),
            'min_shrink_factor': float(np.min(shrink)),
            'defect_pixels_ratio': float(np.mean(mask)),
            'defect_segments': defect_segments,
            'boundary_diff_threshold_px': float(diff_thresh),

            # 采样数组（供可视化）
            'theta_samples': theta.tolist(),
            'r_ideal_samples': r_ell.tolist(),
            'r_actual_samples': r_env.tolist(),
            'defect_mask_samples': mask.tolist(),
            'shrink_factors': shrink.tolist(),

            # 平面截断几何参数（局部坐标）
            'cap_normal_local_xyz': [nx, ny, 0.0],
            'cap_plane_offset_t_px': float(t_est),
            'cap_support_norm_px': float(N_norm),
            'cap_height_px': float(cap_height_px),
            'cap_height_unit_sphere': float(H),

            # 最大差值点（以椭圆外侧为准）
            'max_diff_idx': int(i0),
            'max_diff_theta_local_deg': float((math.degrees(th_i0) + 360.0) % 360.0),
            'max_diff_direction_img_deg': dir_img_deg_i0,
            'max_diff_point_img_x': x_tip_outer_i0,
            'max_diff_point_img_y': y_tip_outer_i0,
            'max_diff_outer_point_img_x': x_tip_outer_i0,
            'max_diff_outer_point_img_y': y_tip_outer_i0,
            'max_diff_inner_point_img_x': x_tip_inner_i0,
            'max_diff_inner_point_img_y': y_tip_inner_i0,
        }
        # 标注重拟合信息
        info['refit_used'] = bool(refit_used)
        info['refit_removed_points'] = int(refit_removed_pts)
        info['refit_kept_points'] = int(refit_kept_pts)

        if debug:
            print(f"平面截断缺损: H={cap_height_px:.2f}px, V={V_final:.0f}/{V_full:.0f} ({volume_loss_ratio*100:.1f}%)，dir={trunc_dir_img_deg:.1f}°")

        return V_final, info
    except Exception as e:
        if debug:
            print(f"椭圆截断体积估算失败: {e}")
        return None, None

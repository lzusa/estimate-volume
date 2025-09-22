import numpy as np

import cv2
import config

def _adaptive_edge_extend(gray_blur: np.ndarray,
                          init_mask: np.ndarray,
                          debug: bool = False,
                          dbg_save=None) -> np.ndarray:
    """
    基于动态阈值的自适应边缘扩展:
    1. 首先进行最大扩展，获取背景灰度值
    2. 然后逐步收缩，直到灰度值发生显著变化时认为达到了边界
    
    策略：
    - 先扩展到最大范围，估算背景灰度基准值
    - 从最大扩展开始收缩，监测灰度变化
    - 当环带灰度与背景基准值差异超过阈值时，停止收缩
    
    返回扩展后的 mask。
    """
    # 读取（或默认）参数
    max_expand = int(getattr(config, 'EDGE_MAX_EXPAND', 15) or 15)  # 最大扩展步数
    min_expand = int(getattr(config, 'EDGE_MIN_EXPAND', 0) or 0)    # 最小扩展步数
    bg_sample_steps = int(getattr(config, 'EDGE_BG_SAMPLE_STEPS', 3) or 3)  # 背景采样步数
    gray_threshold = float(getattr(config, 'EDGE_GRAY_THRESHOLD', 20) or 20)  # 灰度差异阈值
    ksz = int(getattr(config, 'EDGE_KERNEL_SIZE', 3) or 3)
    if ksz < 3: ksz = 3
    if ksz % 2 == 0: ksz += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))

    # 预先准备底图（仅 debug 时使用）
    base_bgr = None
    if debug and dbg_save is not None:
        try:
            if gray_blur.dtype != np.uint8:
                gb = cv2.normalize(gray_blur, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            else:
                gb = gray_blur
            base_bgr = cv2.cvtColor(gb, cv2.COLOR_GRAY2BGR)
        except Exception:
            base_bgr = None

    # 第一阶段：扩展到最大范围，获取背景灰度基准
    expanded_mask = init_mask.copy()
    ring_grays = []  # 记录每步扩展的环带灰度
    
    # 执行最大扩展
    for step in range(1, max_expand + 1):
        prev_mask = expanded_mask.copy()
        expanded_mask = cv2.dilate(expanded_mask, kernel, iterations=1)
        
        # 计算当前步骤的环带
        ring = cv2.subtract(expanded_mask, prev_mask)
        ring_pixels = gray_blur[ring > 0]
        
        if ring_pixels.size == 0:
            break
            
        # 使用中位数作为环带代表灰度值（更稳定）
        ring_gray = float(np.median(ring_pixels))
        ring_grays.append(ring_gray)
        
        # 调试信息
        if debug and dbg_save is not None:
            try:
                vis = base_bgr.copy() if base_bgr is not None else np.zeros((gray_blur.shape[0], gray_blur.shape[1], 3), dtype=np.uint8)
                # 绘制当前状态
                def _draw_cnt_from_mask(msk, color, thickness=1):
                    try:
                        cnts, _ = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if cnts:
                            cv2.drawContours(vis, cnts, -1, color, thickness)
                    except Exception:
                        pass
                
                _draw_cnt_from_mask(init_mask, (255, 255, 0), 1)     # 初始: 青色
                _draw_cnt_from_mask(expanded_mask, (0, 255, 0), 1)   # 当前扩展: 绿色
                
                # 环带高亮
                overlay = vis.copy()
                overlay[ring > 0] = (255, 0, 255)  # 品红
                vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
                
                # 文本信息
                infos = [
                    f"Expand Step: {step}/{max_expand}",
                    f"Ring Gray: {ring_gray:.1f}",
                    f"Ring Pixels: {ring_pixels.size}",
                    f"Kernel Size: {ksz}"
                ]
                y0 = 22
                for i, t in enumerate(infos):
                    cv2.putText(vis, t, (10, y0 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                dbg_save(vis, f'expand_step_{step:02d}.png')
            except Exception:
                pass
    
    # 计算背景灰度基准值（使用最后几步的平均值）
    if len(ring_grays) >= bg_sample_steps:
        bg_gray_baseline = float(np.mean(ring_grays[-bg_sample_steps:]))
    elif len(ring_grays) > 0:
        bg_gray_baseline = float(np.mean(ring_grays))
    else:
        # 如果没有扩展成功，直接返回原mask
        return init_mask
    
    if debug:
        print(f"背景灰度基准值: {bg_gray_baseline:.1f} (基于最后{min(bg_sample_steps, len(ring_grays))}步)")
    
    # 第二阶段：从最大扩展开始收缩，寻找边界
    current_mask = expanded_mask.copy()
    shrink_steps = 0
    final_step = 0
    
    # 执行收缩，直到找到边界
    while shrink_steps < max_expand - min_expand:
        # 收缩一步
        prev_mask = current_mask.copy()
        current_mask = cv2.erode(current_mask, kernel, iterations=1)
        shrink_steps += 1
        
        # 计算被移除的环带（从prev_mask到current_mask）
        removed_ring = cv2.subtract(prev_mask, current_mask)
        removed_pixels = gray_blur[removed_ring > 0]
        
        if removed_pixels.size == 0:
            break
        
        # 计算移除环带的灰度
        removed_gray = float(np.median(removed_pixels))
        
        # 判断是否达到边界（灰度值与背景基准差异超过阈值）
        gray_diff = abs(removed_gray - bg_gray_baseline)
        
        # 调试信息
        if debug and dbg_save is not None:
            try:
                vis = base_bgr.copy() if base_bgr is not None else np.zeros((gray_blur.shape[0], gray_blur.shape[1], 3), dtype=np.uint8)
                
                def _draw_cnt_from_mask(msk, color, thickness=1):
                    try:
                        cnts, _ = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if cnts:
                            cv2.drawContours(vis, cnts, -1, color, thickness)
                    except Exception:
                        pass
                
                _draw_cnt_from_mask(init_mask, (255, 255, 0), 1)     # 初始: 青色
                _draw_cnt_from_mask(current_mask, (0, 255, 0), 1)    # 当前: 绿色
                _draw_cnt_from_mask(expanded_mask, (128, 128, 128), 1)  # 最大扩展: 灰色
                
                # 移除的环带高亮
                overlay = vis.copy()
                overlay[removed_ring > 0] = (0, 0, 255)  # 红色
                vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
                
                # 文本信息
                edge_detected = gray_diff > gray_threshold
                infos = [
                    f"Shrink Step: {shrink_steps}",
                    f"Removed Gray: {removed_gray:.1f}",
                    f"BG Baseline: {bg_gray_baseline:.1f}",
                    f"Gray Diff: {gray_diff:.1f}",
                    f"Threshold: {gray_threshold:.1f}",
                    f"Edge Detected: {edge_detected}",
                    f"Removed Pixels: {removed_pixels.size}"
                ]
                y0 = 22
                for i, t in enumerate(infos):
                    color = (0, 255, 0) if not edge_detected else (0, 0, 255)
                    cv2.putText(vis, t, (10, y0 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                
                dbg_save(vis, f'shrink_step_{shrink_steps:02d}.png')
            except Exception:
                pass
        
        # 如果灰度差异超过阈值，认为达到边界，停止收缩
        if gray_diff > gray_threshold:
            final_step = shrink_steps - 1  # 回退到上一步
            current_mask = prev_mask  # 使用上一步的mask
            if debug:
                print(f"检测到边界: 灰度差异 {gray_diff:.1f} > 阈值 {gray_threshold:.1f}, 停止在收缩步骤 {final_step}")
            break
    
    # 确保最小扩展
    total_expand_steps = max_expand - shrink_steps
    if total_expand_steps < min_expand:
        # 需要重新从初始mask扩展到min_expand步
        current_mask = init_mask.copy()
        for _ in range(min_expand):
            current_mask = cv2.dilate(current_mask, kernel, iterations=1)
        final_step = min_expand
        if debug:
            print(f"应用最小扩展约束: 扩展到 {min_expand} 步")
    
    # 保存调试信息
    if debug and dbg_save:
        try:
            dbg_save(current_mask, 'dark_mask_edge_extended.png')
            
            # 保存元数据
            info = {
                'method': 'dynamic_threshold',
                'max_expand': max_expand,
                'min_expand': min_expand,
                'bg_sample_steps': bg_sample_steps,
                'bg_gray_baseline': float(bg_gray_baseline),
                'gray_threshold': float(gray_threshold),
                'final_expand_steps': int(total_expand_steps),
                'shrink_steps': int(shrink_steps),
                'ring_grays_during_expand': [float(x) for x in ring_grays],
                'kernel_size': ksz
            }
            import json, os
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   '_last_edge_extend_debug.json'), 'w', encoding='utf-8') as fh:
                json.dump(info, fh, ensure_ascii=False, indent=2)
            
            # 保存最终对比图
            if base_bgr is not None:
                try:
                    vis = base_bgr.copy()
                    def _draw_cnt_from_mask(msk, color, thickness=2):
                        try:
                            cnts, _ = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            if cnts:
                                cv2.drawContours(vis, cnts, -1, color, thickness)
                        except Exception:
                            pass
                    
                    _draw_cnt_from_mask(init_mask, (255, 255, 0), 1)      # 初始: 青色
                    _draw_cnt_from_mask(current_mask, (0, 255, 0), 1)     # 最终: 绿色
                    
                    # 文本概要
                    summary = [
                        f"Method: Dynamic Threshold",
                        f"BG Baseline: {bg_gray_baseline:.1f}",
                        f"Gray Threshold: {gray_threshold:.1f}",
                        f"Final Expand: {total_expand_steps} steps",
                        f"Shrink Steps: {shrink_steps}",
                        f"Kernel Size: {ksz}"
                    ]
                    y0 = 22
                    for i, t in enumerate(summary):
                        cv2.putText(vis, t, (10, y0 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    
                    dbg_save(vis, 'edge_final.png')
                except Exception:
                    pass
        except Exception:
            pass
    
    return current_mask

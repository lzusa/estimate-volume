from __future__ import annotations

import argparse
import csv
import os
import math
import estimate_volume

import re
from datetime import datetime
from typing import Iterable, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import statistics
import config
import io


def read_pixel_scale_and_mass(txt_path: str) -> tuple[float, float] | tuple[None, None]:
	"""从给定的文本文件读取像素比 (mm^3 or mm per pixel?) 和质量 (g)。

	期望文件格式类似：
		像素比\t0.003402221
		质量\t0.12368

	返回 (pixel_ratio, mass_g) 或 (None, None) 如果解析失败。
	pixel_ratio 的含义: 一个像素对应的体积/长度换算，原文件使用者提供的示例是像素比（单位取决于上下文）。
	在这里我们假定像素比为 mm 每像素 (mm/pixel) 用于把半径(px) 转为 mm；若该文件实际上记录的是每像素体积(mm^3/pixel)，也能容错使用。
	"""
	if not txt_path:
		return (None, None)
	try:
		p = os.path.join(os.path.dirname(__file__), txt_path)
		with open(p, 'r', encoding='utf-8') as fh:
			text = fh.read()
	except Exception:
		return (None, None)

	pixel_ratio = None
	mass_g = None
	for line in text.splitlines():
		if not line.strip():
			continue
		parts = re.split(r"\s+|\t", line.strip())
		if len(parts) < 2:
			continue
		key = parts[0]
		val = parts[-1]
		try:
			f = float(val)
		except Exception:
			continue
		if '像素' in key or '像素比' in key or 'pixel' in key.lower():
			pixel_ratio = f
		if '质量' in key or 'mass' in key.lower():
			mass_g = f
	print('像素比 ', pixel_ratio)
	print('质量 ', mass_g)
	return (pixel_ratio, mass_g)


def parse_timestamp_from_filename(filename: str) -> str | None:
	"""从类似 Image_20250425163046637.bmp 的文件名中解析出可读时间戳。

	规则:
	- 找到最后一段连续的数字（不含扩展名），前 8 位为 YYYYMMDD，接下来的 6 位为 HHMMSS，后面的若干位为毫秒（可选，视具体文件名而定）。
	- 如果不能解析返回 None。
	"""
	m = re.search(r"(\d+)\.(?i:bmp)$", filename)
	if not m:
		# 兼容没有明确扩展名位置的情况：直接取文件名中最后一段数字
		m_all = re.findall(r"(\d+)", filename)
		if not m_all:
			return None
		digits = m_all[-1]
	else:
		digits = m.group(1)

	# 最少需要 8(date) + 6(time) = 14 位数字
	if len(digits) < 14:
		return None

	date_part = digits[0:8]
	time_h = digits[8:10]
	time_m = digits[10:12]
	time_s = digits[12:14]
	ms_part = digits[14:]

	try:
		year = int(date_part[0:4])
		month = int(date_part[4:6])
		day = int(date_part[6:8])
		hour = int(time_h)
		minute = int(time_m)
		second = int(time_s)
	except ValueError:
		return None

	if ms_part:
		# 保证毫秒为 3 位，短的右侧补零，超过 3 位截断为前三位（视为毫秒）
		ms = (ms_part + "000")[0:3]
	else:
		ms = "000"

	try:
		dt = datetime(year, month, day, hour, minute, second, int(ms) * 1000)
	except Exception:
		return None

	# 返回格式: YYYY-MM-DD HH:MM:SS.mmm
	return dt.strftime("%Y-%m-%d %H:%M:%S.") + f"{int(ms):03d}"


def parse_datetime_from_filename(filename: str) -> datetime | None:
	"""同 parse_timestamp_from_filename，但返回 datetime 对象以便匹配 CSV 时间。"""
	m = re.search(r"(\d+)\.(?i:bmp)$", filename)
	if not m:
		m_all = re.findall(r"(\d+)", filename)
		if not m_all:
			return None
		digits = m_all[-1]
	else:
		digits = m.group(1)

	if len(digits) < 14:
		return None

	date_part = digits[0:8]
	time_h = digits[8:10]
	time_m = digits[10:12]
	time_s = digits[12:14]
	ms_part = digits[14:]

	try:
		year = int(date_part[0:4])
		month = int(date_part[4:6])
		day = int(date_part[6:8])
		hour = int(time_h)
		minute = int(time_m)
		second = int(time_s)
	except ValueError:
		return None

	if ms_part:
		ms = (ms_part + "000")[0:3]
	else:
		ms = "000"

	try:
		return datetime(year, month, day, hour, minute, second, int(ms) * 1000)
	except Exception:
		return None


def list_bmp_files(directory: str) -> List[str]:
	try:
		names = os.listdir(directory)
	except OSError:
		return []
	return [n for n in names if n.lower().endswith('.bmp')]


def convert_folder(directory: str, limit: int | None = None) -> List[Tuple[str, str | None]]:
	files = sorted(list_bmp_files(directory))
	if limit:
		files = files[:limit]
	out: List[Tuple[str, str | None]] = []
	for f in files:
		readable = parse_timestamp_from_filename(f)
		out.append((f, readable))
	return out


def load_csv_times(csv_path: str) -> List[Tuple[datetime, float]]:
	"""读取 CSV，返回有序的 (datetime, temperature) 列表。"""
	rows: List[Tuple[datetime, float]] = []
	try:
		with open(csv_path, newline='', encoding='utf-8') as fh:
			reader = csv.reader(fh)
			header = next(reader, None)
			for r in reader:
				if len(r) < 4:
					continue
				date_s = r[0].strip()
				time_s = r[1].strip()
				temp_s = r[3].strip()
				dt = None
				for fmt in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S"):
					try:
						dt = datetime.strptime(f"{date_s} {time_s}", fmt)
						break
					except Exception:
						continue
				if dt is None:
					continue
				try:
					temp = float(temp_s)
				except Exception:
					continue
				rows.append((dt, temp))
	except OSError:
		return []

	rows.sort(key=lambda x: x[0])
	return rows


def find_nearest_temperature(dt: datetime, csv_rows: List[Tuple[datetime, float]]) -> Tuple[datetime | None, float | None]:

	if not csv_rows:
		return None, None
	import bisect

	times = [r[0] for r in csv_rows]
	idx = bisect.bisect_left(times, dt)
	candidates = []
	if idx > 0:
		candidates.append(csv_rows[idx - 1])
	if idx < len(csv_rows):
		candidates.append(csv_rows[idx])

	best = None
	best_diff = None
	for cand_dt, temp in candidates:
		diff = abs((cand_dt - dt).total_seconds())
		if best is None or diff < best_diff:
			best = (cand_dt, temp)
			best_diff = diff

	return best if best is not None else (None, None)


def _worker(fullpath: str, debug_flag: bool, dbg_out: str, low_temp_read: bool = False):
	"""Top-level worker for ProcessPoolExecutor (must be picklable)."""
	try:
		import estimate_volume as ev
		return ev.process_image(fullpath, debug=debug_flag, debug_out_dir=dbg_out, low_temp_read=low_temp_read)
	except Exception:
		# 与新版 process_image 返回长度保持一致 (found, area, radius, overlay, extra)
		return (False, None, None, None, None)


def main() -> int:
	# 路径准备
	default_csv = os.path.join(os.path.dirname(__file__), config.DEFAULT_CSV)
	# 诊断 LOW_TEMP 配置
	low_temp_cfg = getattr(config, 'LOW_TEMP', None)
	low_temp_dir_cfg = getattr(config, 'LOW_TEMP_DIR', None)
	print(f"Config LOW_TEMP={low_temp_cfg!r}, LOW_TEMP_DIR={low_temp_dir_cfg!r}")

	# 以 ROOM_TEMP_DIR 为基准遍历文件（LOW_TEMP_DIR 中为低温覆盖）
	room_dir = os.path.join(os.path.dirname(__file__), config.ROOM_TEMP_DIR)
	low_dir = None
	if low_temp_cfg is not None and low_temp_dir_cfg:
		candidate = os.path.join(os.path.dirname(__file__), low_temp_dir_cfg)
		if os.path.isdir(candidate):
			low_dir = candidate
			print(f"LOW_TEMP_DIR 可用，低温图片将从此目录加载: {low_dir}")
		else:
			print(f"WARNING: LOW_TEMP_DIR 路径不存在或不是目录: {candidate}. 将忽略 LOW_TEMP_DIR")

	files = sorted(list_bmp_files(room_dir))
	print(f"找到 {len(files)} 个 BMP 文件在 ROOM_TEMP_DIR: {room_dir}")

	# 根据模式决定是否加载温度 CSV
	if not config.ROOM_TEMP_TIME_MODE:
		csv_rows = load_csv_times(default_csv)
		if not csv_rows:
			print(f"未能加载或解析 CSV: {default_csv}")
	else:
		csv_rows = []  # 时间模式下不需要匹配温度

	# 元信息: (文件名, 可读时间戳, 匹配到的CSV时间字符串, 温度(或None), fullpath, 原始datetime, low_temp_read_flag, process_flag)
	meta_list: List[Tuple[str, str | None, str | None, float | None, str, datetime | None, bool, bool]] = []
	for f in files:
		readable = parse_timestamp_from_filename(f)
		dt = parse_datetime_from_filename(f)
		matched_dt_s = None
		temp: float | None = None
		if not config.ROOM_TEMP_TIME_MODE and dt is not None and csv_rows:
			matched_dt, temp = find_nearest_temperature(dt, csv_rows)
			matched_dt_s = matched_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] if matched_dt else None
		# 默认使用 room_dir 下的文件
		fullpath_room = os.path.join(room_dir, f)
		fullpath = fullpath_room
		low_temp_read = False
		process_flag = True
		# 如果有 LOW_TEMP 配置且温度可读，并且 temp < LOW_TEMP，则优先从 low_dir 中加载同名文件；
		# 如果 low_dir 中不存在该文件，则跳过处理（不做体积计算）
		if low_temp_cfg is not None and temp is not None:
			try:
				if float(temp) < float(low_temp_cfg):
					# 处于低温区间
					if low_dir is not None:
						candidate_low = os.path.join(low_dir, f)
						if os.path.isfile(candidate_low):
							# 低温目录存在对应文件 -> 使用低温图并走低温处理（截断逻辑）
							fullpath = candidate_low
							low_temp_read = True
							process_flag = True
						else:
							# 低温目录没有该文件
							if getattr(config, 'LOW_TEMP_FALLBACK_USE_ROOM', False):
								# 开启回退：使用常温目录同名文件，按普通方式处理（不视为 low_temp_read）
								fullpath = fullpath_room
								low_temp_read = False
								process_flag = True
							else:
								# 保持旧行为：跳过
								process_flag = False
					else:
						# 没有 low_dir 但温度在低温区间
						if getattr(config, 'LOW_TEMP_FALLBACK_USE_ROOM', False):
							# 允许回退 -> 使用常温图
							fullpath = fullpath_room
							low_temp_read = False
							process_flag = True
						else:
							process_flag = False
				else:
					# 非低温阶段，正常使用常温图片
					fullpath = fullpath_room
					low_temp_read = False
					process_flag = True
			except Exception:
				# 解析异常：默认处理
				process_flag = True
		# 非低温阶段（或未配置 LOW_TEMP）：使用 room_dir 中的 file
		meta_list.append((f, readable, matched_dt_s, temp, fullpath, dt, low_temp_read, process_flag))

	# 并行处理图片以估计体积（使用 ProcessPoolExecutor）

	# debug 输出目录
	debug_out_dir = os.path.join(os.path.dirname(__file__), config.DEBUG_OUT_DIR)
	#如果有这个目录，删掉目录
	if os.path.exists(debug_out_dir):
		import shutil
		shutil.rmtree(debug_out_dir)
	os.makedirs(debug_out_dir, exist_ok=True)

	results = []
	# 如果 estimate_volume 不可用，则跳过并保持旧行为（所有体积为 None）
	if estimate_volume is None:
		for (f, readable, matched_dt_s, temp, fullpath, dt, low_flag, proc) in meta_list:
			# append circularity as None when estimate_volume not available
			results.append((f, readable, matched_dt_s, temp, None, dt, None))
			print(f"{f} -> {readable if readable else '<无法解析>'} | matched: {matched_dt_s if matched_dt_s else '<无>'} | Temp: {temp if (temp is not None) else ('-' if config.ROOM_TEMP_TIME_MODE else '<无>')} | radius_px: <无> | volume_px3: <无> | circularity: <无>")
	else:
		workers = min(config.MAX_WORKERS, (os.cpu_count() or 1))
		# 为了保持结果顺序，初始化与 meta_list 等长的占位列表
		vol_results = [None] * len(meta_list)
		# 构建实际需要处理的任务列表 (meta_idx, fullpath, low_flag)
		process_entries: List[Tuple[int, str, bool]] = []
		for idx, (f, readable, matched_dt_s, temp, fullpath, dt, low_flag, proc) in enumerate(meta_list):
			if proc:
				process_entries.append((idx, fullpath, low_flag))
		debug_flag = config.DEBUG
		with ProcessPoolExecutor(max_workers=workers) as ex:
			futures = {ex.submit(_worker, fullpath, debug_flag, debug_out_dir, low_flag): idx for (idx, fullpath, low_flag) in process_entries}
			for fut in as_completed(futures):
				idx = futures[fut]
				try:
					found, area_px, radius_px, _overlay, extra = fut.result()
				except Exception:
					found, radius_px, extra = False, None, None
				# collect circularity (if provided in extra)
				if found and radius_px is not None:
					circ = None
					missing_caps = None
					major_axis_px = None
					minor_axis_px = None
					if extra and isinstance(extra, dict):
						circ = extra.get('circularity_ratio')
						# 缺失球冠体积（像素^3），来自截断椭球或勒让德信息
						missing_caps = extra.get('volume_ellipsoid_missing_caps_px3') or extra.get('volume_legendre_missing_caps_px3')
						# 椭圆拟合得到的长短轴（如果进行了椭圆拟合，ellipsoid 或 截断椭球）
						major_axis_px = extra.get('major_axis_px')
						minor_axis_px = extra.get('minor_axis_px')
					if extra and extra.get('chosen_volume_px3') is not None:
						vol_results[idx] = (radius_px, float(extra['chosen_volume_px3']), circ, missing_caps, major_axis_px, minor_axis_px)
					else:
						vol_results[idx] = (radius_px, (4.0 / 3.0) * math.pi * (radius_px ** 3), circ, missing_caps, major_axis_px, minor_axis_px)
				else:
					# 未找到，填充占位 (radius, volume, circularity, missing_caps, major_axis, minor_axis)
					vol_results[idx] = (None, None, None, None, None, None)

			# 组合最终结果并打印
			for i, (f, readable, matched_dt_s, temp, fullpath, dt, low_flag, proc) in enumerate(meta_list):
				vr = vol_results[i]
				if vr is None:
					# 未被处理的条目（例如低温但 low_dir 中无对应文件）
					radius_px, volume_px3, circularity, missing_caps, major_axis_px, minor_axis_px = (None, None, None, None, None, None)
				else:
					radius_px, volume_px3, circularity, missing_caps, major_axis_px, minor_axis_px = vr
				# 结果元组追加：长轴/短轴在末尾，保持原有索引兼容 (missing_caps 仍为 index 7)
				results.append((f, readable, matched_dt_s, temp, volume_px3, dt, circularity, missing_caps, major_axis_px, minor_axis_px))
				rad_s = f"{radius_px:.2f}" if radius_px is not None else '<无>'
				vol_s = f"{volume_px3:.1f}" if volume_px3 is not None else '<无>'
				circ_s = f"{circularity:.3f}" if circularity is not None else '<无>'
				# 缺失体积与比值
				miss_s = f"{missing_caps:.1f}" if (missing_caps is not None) else '<无>'
				ratio_s = '<无>'
				try:
					if (missing_caps is not None) and (volume_px3 is not None) and (volume_px3 > 0):
						ratio_s = f"{(missing_caps/volume_px3):.4f}"
				except Exception:
					pass
				temp_disp = '-' if (config.ROOM_TEMP_TIME_MODE) else (temp if temp is not None else '<无>')
				print(f"{f} -> {readable if readable else '<无法解析>'} | matched: {matched_dt_s if matched_dt_s else '<无>'} | Temp: {temp_disp} | radius_px: {rad_s} | volume_px3: {vol_s} | circularity: {circ_s} | missing_caps_px3: {miss_s} | missing/final: {ratio_s}")

			


	# 绘制: 根据模式
	try:
		import matplotlib.pyplot as plt
		import matplotlib
		

		# 读取像素比与质量 (两种模式都会用)
		pixel_ratio, mass_g = read_pixel_scale_and_mass(config.PIXEL_SCALE_FILE)

		# 保存统一结果 CSV（保证无论是否启用 estimate_volume 都会输出）
		try:
			csv_out_path = os.path.join(os.path.dirname(__file__), f"results_{config.NAME}_{config.VOLUME_METHOD}.csv")
			with open(csv_out_path, 'w', newline='', encoding='utf-8') as csvfile:
				writer = csv.writer(csvfile)
				# 新增列：椭圆长轴/短轴（像素）; 仍保留缺失球冠体积与比值
				header = ['Filename', 'Matched_Temperature(°C)', 'Pixel_Volume(px³)', 'Ellipsoid_Major_Axis(px)', 'Ellipsoid_Minor_Axis(px)', 'Missing_Cap(px³)', 'Missing/Final_Ratio', 'Volume(cm³)', 'Density(g/cm³)']
				writer.writerow(header)
				for r in results:
					filename = r[0]
					matched_temp = r[3]
					vol_px = r[4]
					miss_px = r[7] if len(r) > 7 else None
					major_axis_px = r[8] if len(r) > 8 else None
					minor_axis_px = r[9] if len(r) > 9 else None
					# 如果像素体积为空则跳过（不加入 CSV）
					if vol_px is None:
						continue
					vol_cm3 = ''
					density = ''
					ratio = ''
					try:
						if (miss_px is not None) and (vol_px is not None) and (float(vol_px) > 0):
							ratio = float(miss_px) / float(vol_px)
					except Exception:
						ratio = ''
					if pixel_ratio is not None:
						try:
							if config.PIXEL_RATIO_IS_VOLUME:
								vol_mm3 = float(vol_px) * float(pixel_ratio)
							else:
								vol_mm3 = float(vol_px) * (float(pixel_ratio) ** 3)
							vol_cm3 = vol_mm3 / 1000.0
							vol_cm3 = float(vol_cm3)
							if mass_g is not None and vol_cm3 > 0:
								density = float(mass_g) / vol_cm3
						except Exception:
							vol_cm3 = ''
							density = ''
					writer.writerow([
						filename,
						'' if matched_temp is None else matched_temp,
						(f"{vol_px:.3f}" if isinstance(vol_px, float) else vol_px),
						'' if major_axis_px is None else (f"{major_axis_px:.3f}" if isinstance(major_axis_px, float) else major_axis_px),
						'' if minor_axis_px is None else (f"{minor_axis_px:.3f}" if isinstance(minor_axis_px, float) else minor_axis_px),
						'' if miss_px is None else (f"{miss_px:.3f}" if isinstance(miss_px, float) else miss_px),
						'' if ratio == '' else (f"{ratio:.6f}" if isinstance(ratio, float) else ratio),
						'' if vol_cm3 == '' else (f"{vol_cm3:.6f}" if isinstance(vol_cm3, float) else vol_cm3),
						'' if density == '' else (f"{density:.6f}" if isinstance(density, float) else density)
					])
			print(f"汇总结果已保存: {csv_out_path}")
		except Exception as e:
			print(f"保存汇总CSV失败: {e}")
		if config.ROOM_TEMP_TIME_MODE:
			# 时间模式: 需要 dt 和 volume
			# time_pts: list of (datetime, volume_px3, circularity)
			time_pts = [(r[5], r[4], r[6] if len(r) > 6 else None) for r in results if r[5] is not None and r[4] is not None]
			if time_pts:
				time_pts.sort(key=lambda x: x[0])
				dts = [p[0] for p in time_pts]
				vols = [p[1] for p in time_pts]
				circularities = [p[2] for p in time_pts]
				# 相对时间(分钟)
				if config.ROOM_TEMP_TIME_USE_ELAPSED_MINUTES:
					t0 = dts[0]
					x_vals = [ (dt - t0).total_seconds()/60.0 for dt in dts ]
					x_label = 'Elapsed Time (min)'
				else:
					x_vals = dts
					x_label = 'Time'
				# 计算密度
				densities = []
				if pixel_ratio is not None and mass_g is not None:
					for v_px in vols:
						if config.PIXEL_RATIO_IS_VOLUME:
							vol_mm3 = v_px * float(pixel_ratio)
						else:
							vol_mm3 = v_px * (float(pixel_ratio) ** 3)
						vol_cm3 = vol_mm3 / 1000.0
						dens = float(mass_g) / vol_cm3 if vol_cm3 > 0 else None
						densities.append(dens)
				else:
					densities = [None]*len(vols)
				if any(d is not None for d in densities):
					y_vals = densities
					y_label = 'Density (g/cm³)'
					title_main = 'Time vs Density (' + config.NAME + ', Room Temperature)'
				else:
					y_vals = vols
					y_label = 'Volume (px³)'
					title_main = 'Time vs Volume'
				if config.ROOM_TEMP_CONSTANT_TEMP_C is not None:
					title_main += f" (Constant {config.ROOM_TEMP_CONSTANT_TEMP_C}°C)"
				plt.figure(figsize=config.PLOT_FIGSIZE)
				
				plt.plot(x_vals, y_vals, 'o', color='tab:blue', alpha=0.6, label='Data Points')
				
				plt.xlabel(x_label)
				plt.ylabel(y_label)
				plt.title(title_main)
				plt.grid(True, linestyle='--', alpha=0.5)
				plt.legend()
				
				# 设置X轴范围
				if config.X_AXIS_RANGE is not None:
					plt.xlim(config.X_AXIS_RANGE)
				
				# 设置Y轴范围
				if config.Y_AXIS_RANGE is not None:
					plt.ylim(config.Y_AXIS_RANGE)
				
				title = 'time_vs_density_' + config.NAME + '.png'
    
				out_img = os.path.join(os.path.dirname(__file__), title if any(d is not None for d in densities) else 'time_vs_volume.png')
				plt.tight_layout()
				plt.savefig(out_img, dpi=config.PLOT_DPI)
				print(f"时间-密度/体积图已保存: {out_img}")
				plt.show()  # 显示图像
				# 另外保存 时间 - Circularity 图（仅当存在 circularity 数据点时）
				try:
					# 过滤出有 circularity 的点
					circ_indices = [i for i, c in enumerate(circularities) if c is not None]
					if circ_indices:
						if config.ROOM_TEMP_TIME_USE_ELAPSED_MINUTES:
							x_circ = [x_vals[i] for i in circ_indices]
							# x_vals already in minutes when using elapsed mode
							x_label_circ = 'Elapsed Time (min)'
						else:
							x_circ = [dts[i] for i in circ_indices]
							x_label_circ = 'Time'
						y_circ = [circularities[i] for i in circ_indices]
						plt.figure(figsize=config.PLOT_FIGSIZE)
						plt.plot(x_circ, y_circ, 'o-', color='tab:purple', alpha=0.8)
						plt.xlabel(x_label_circ)
						plt.ylabel('Circularity')
						plt.title('Time vs Circularity (' + config.NAME + ')')
						plt.grid(True, linestyle='--', alpha=0.5)
						if config.X_AXIS_RANGE is not None:
							plt.xlim(config.X_AXIS_RANGE)
						plt.tight_layout()
						circ_out = os.path.join(os.path.dirname(__file__), f"time_vs_circularity_{config.NAME}.png")
						plt.savefig(circ_out, dpi=config.PLOT_DPI)
						print(f"时间-圆度(circularity)图已保存: {circ_out}")
				except Exception as _e:
					print(f"保存时间-圆度图失败: {_e}")
				
				# 保存时间和密度/体积数据到CSV文件
				try:
					csv_output_path = os.path.join(os.path.dirname(__file__), f"time_density_data_{config.NAME}.csv")
					with open(csv_output_path, 'w', newline='', encoding='utf-8') as csvfile:
						writer = csv.writer(csvfile)
						# 写入表头
						if any(d is not None for d in densities):
							if config.ROOM_TEMP_TIME_USE_ELAPSED_MINUTES:
								writer.writerow(['Elapsed_Time(min)', 'Density(g/cm³)', 'Volume(px³)', 'Circularity', 'Timestamp', 'Filename'])
							else:
								writer.writerow(['Timestamp', 'Density(g/cm³)', 'Volume(px³)', 'Circularity', 'Elapsed_Time(min)', 'Filename'])
							# 写入数据
							for i, (dt, vol) in enumerate(time_pts):
								density = densities[i] if i < len(densities) else None
								circ = circularities[i] if i < len(circularities) else None
								elapsed_min = x_vals[i] if config.ROOM_TEMP_TIME_USE_ELAPSED_MINUTES else (dt - time_pts[0][0]).total_seconds()/60.0
								timestamp = dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
								# 找到对应的文件名
								filename = None
								for r in results:
									if r[5] == dt and r[4] == vol:
										filename = r[0]
										break
								if config.ROOM_TEMP_TIME_USE_ELAPSED_MINUTES:
									writer.writerow([elapsed_min, density, vol, circ, timestamp, filename])
								else:
									writer.writerow([timestamp, density, vol, circ, elapsed_min, filename])
						else:
							if config.ROOM_TEMP_TIME_USE_ELAPSED_MINUTES:
								writer.writerow(['Elapsed_Time(min)', 'Volume(px³)', 'Circularity', 'Timestamp', 'Filename'])
							else:
								writer.writerow(['Timestamp', 'Volume(px³)', 'Circularity', 'Elapsed_Time(min)', 'Filename'])
							# 写入数据
							for i, (dt, vol) in enumerate(time_pts):
								elapsed_min = x_vals[i] if config.ROOM_TEMP_TIME_USE_ELAPSED_MINUTES else (dt - time_pts[0][0]).total_seconds()/60.0
								timestamp = dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
								# 找到对应的文件名
								filename = None
								circ = circularities[i] if i < len(circularities) else None
								for r in results:
									if r[5] == dt and r[4] == vol:
										filename = r[0]
										break
								if config.ROOM_TEMP_TIME_USE_ELAPSED_MINUTES:
									writer.writerow([elapsed_min, vol, circ, timestamp, filename])
								else:
									writer.writerow([timestamp, vol, circ, elapsed_min, filename])
					print(f"时间-密度/体积数据已保存到: {csv_output_path}")
				except Exception as e:
					print(f"保存CSV文件失败: {e}")
			else:
				print('没有足够的数据点绘制时间-体积/密度图。')
		else:
			# 温度模式 (原逻辑保留，稍作封装)
			pts = [(r[3], r[4]) for r in results if r[3] is not None and r[4] is not None]
			# pts: list of (temperature, volume_px3)
			if pts:
				pts.sort(key=lambda x: x[0])
				temps = [p[0] for p in pts]
				vols = [p[1] for p in pts]

				# 试图计算密度 (g/cm^3) 如果 pixel_ratio 与 mass_g 可用
				densities = []
				if pixel_ratio is not None and mass_g is not None:
					for v_px in vols:
						# 将体积 (px^3) 转为 mm^3
						if config.PIXEL_RATIO_IS_VOLUME:
							vol_mm3 = v_px * float(pixel_ratio)
						else:
							# pixel_ratio 代表 mm / pixel
							vol_mm3 = v_px * (float(pixel_ratio) ** 3)
						# mm^3 -> cm^3
						vol_cm3 = vol_mm3 / 1000.0
						if vol_cm3 > 0:
							dens = float(mass_g) / vol_cm3
						else:
							dens = None
						densities.append(dens)
				else:
					densities = [None] * len(vols)

			# 若成功计算密度则绘制 温度 vs 密度，否则回退到 温度 vs 体积
			if any(d is not None for d in densities):
				y_vals = densities
				y_label = 'Density (g/cm³)'
				title_main = 'Temperature vs Density (' + config.NAME + ',' + config.VOLUME_METHOD + ')'
			else:
				y_vals = vols
				y_label = 'Volume (px³)'
				title_main = 'Temperature vs Volume'

			plt.figure(figsize=config.PLOT_FIGSIZE)
			
			plt.plot(temps, y_vals, 'o', color='tab:blue', alpha=0.6, label='Original Data')
			
			# 检查是否启用拟合功能
			if config.ENABLE_FITTING:
				# 导入numpy用于勒让德多项式拟合
				import numpy as np
				from numpy.polynomial import legendre as leg
				
				# 第一次拟合
				fit_range_1 = config.TEMP_FIT_RANGE_1 if config.ENABLE_DUAL_FIT else config.TEMP_FIT_RANGE
				if fit_range_1 is not None:
					min_temp, max_temp = fit_range_1
					filtered_indices = [i for i, temp in enumerate(temps) 
										if min_temp <= temp <= max_temp]
					if filtered_indices:
						temps_fit_1 = [temps[i] for i in filtered_indices]
						y_vals_fit_1 = [y_vals[i] for i in filtered_indices]
						print(f"第一次拟合：使用温度范围 {min_temp}-{max_temp}°C 内的 {len(temps_fit_1)} 个数据点")
					else:
						temps_fit_1, y_vals_fit_1 = temps, y_vals
						print("第一次拟合：指定的温度范围内没有数据点，使用所有数据点")
				else:
					temps_fit_1, y_vals_fit_1 = temps, y_vals
				
				# 第一次拟合的阶数
				fit_degree_1 = config.FIT_DEGREE_1 if config.ENABLE_DUAL_FIT else config.FIT_DEGREE_1
				
				# 执行第一次拟合
				if len(temps_fit_1) >= fit_degree_1 + 1:
					# 将温度值标准化到[-1, 1]区间（勒让德多项式的标准定义域）
					temp_min_1, temp_max_1 = min(temps_fit_1), max(temps_fit_1)
					temp_norm_1 = [2 * (t - temp_min_1) / (temp_max_1 - temp_min_1) - 1 for t in temps_fit_1]
					
					# 进行勒让德多项式拟合
					leg_coeffs_1 = leg.legfit(temp_norm_1, y_vals_fit_1, deg=fit_degree_1)
					
					# 计算斜率（仅对一次拟合有效）
					slope_1 = None
					if fit_degree_1 == 1:
						# 对于一次勒让德多项式：P(x) = c0 + c1*x
						# 其中 x 是标准化温度，需要转换为实际温度的斜率
						# dy/dx_normalized = c1
						# dx_normalized/dx_real = 2/(temp_max_1 - temp_min_1)
						# 因此 dy/dx_real = c1 * 2/(temp_max_1 - temp_min_1)
						slope_1 = leg_coeffs_1[1] * 2 / (temp_max_1 - temp_min_1)
						print(f"第一次拟合斜率: {slope_1:.6f} (密度单位)/(°C)")
					
					# 确定第一次拟合曲线的显示范围
					display_range_1 = config.TEMP_FIT_DISPLAY_RANGE_1 if config.ENABLE_DUAL_FIT else None
					if display_range_1 is not None:
						temp_display_min_1, temp_display_max_1 = display_range_1
					elif config.X_AXIS_RANGE is not None:
						temp_display_min_1, temp_display_max_1 = config.X_AXIS_RANGE
					else:
						temp_display_min_1, temp_display_max_1 = min(temps_fit_1), max(temps_fit_1)
					
					# 生成平滑的拟合曲线（在指定范围内）
					temp_fit_1 = np.linspace(temp_display_min_1, temp_display_max_1, 200)
					temp_fit_norm_1 = 2 * (temp_fit_1 - temp_min_1) / (temp_max_1 - temp_min_1) - 1
					y_fit_1 = leg.legval(temp_fit_norm_1, leg_coeffs_1)
					
					# 在标签中包含斜率信息
					if slope_1 is not None:
						fit_label_1 = f'Fit 1 (slope: {slope_1:.6f})'
					else:
						fit_label_1 = f'Fit 1'
					
					plt.plot(temp_fit_1, y_fit_1, '-', color='tab:red', linewidth=2, label=fit_label_1)
				
					# 标注第一条拟合曲线上的特定温度点
					if (config.ENABLE_TEMP_POINT_ANNOTATION and 
						hasattr(config, 'TEMP_POINT_1') and 
						config.TEMP_POINT_1 is not None):
						temp_point_1 = config.TEMP_POINT_1
						# 计算该温度点对应的标准化值
						temp_point_norm_1 = 2 * (temp_point_1 - temp_min_1) / (temp_max_1 - temp_min_1) - 1
						# 计算拟合曲线在该点的密度值
						y_point_1 = leg.legval(temp_point_norm_1, leg_coeffs_1)
						
						# 在图上标注该点
						marker_color_1 = getattr(config, 'ANNOTATION_MARKER_COLOR_1', 'red')
						marker_size_1 = getattr(config, 'ANNOTATION_MARKER_SIZE', 8)
						plt.plot(temp_point_1, y_point_1, 'o', color=marker_color_1, 
								markersize=marker_size_1, markeredgecolor='black', markeredgewidth=1, 
								zorder=5, label=f'Point 1: {temp_point_1}°C')
						
						# 添加文字标注
						text_offset_x = getattr(config, 'ANNOTATION_TEXT_OFFSET_X', 20)
						text_offset_y = getattr(config, 'ANNOTATION_TEXT_OFFSET_Y', 10)
						text_size = getattr(config, 'ANNOTATION_TEXT_SIZE', 10)
						decimal_places = getattr(config, 'ANNOTATION_DECIMAL_PLACES', 3)
						
						annotation_text = f'({temp_point_1}°C, {y_point_1:.{decimal_places}f})'
						plt.annotate(annotation_text, 
									xy=(temp_point_1, y_point_1),
									xytext=(text_offset_x, text_offset_y), 
									textcoords='offset points',
									fontsize=text_size,
									bbox=dict(boxstyle='round,pad=0.3', facecolor=marker_color_1, alpha=0.3),
									arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
						print(f"第一条拟合曲线在 {temp_point_1}°C 处的密度值: {y_point_1:.{decimal_places}f}")
						# 提示是否为外推值
						if not (temp_min_1 <= temp_point_1 <= temp_max_1):
							print(f"  注意：{temp_point_1}°C 超出拟合数据范围 ({temp_min_1:.1f}°C - {temp_max_1:.1f}°C)，该值为外推结果")
				else:
					print(f"第一次拟合：数据点不足，无法进行{fit_degree_1}阶拟合（需要至少{fit_degree_1 + 1}个点）")
				
				# 第二次拟合（仅在启用双重拟合时）
				if config.ENABLE_DUAL_FIT:
					fit_range_2 = config.TEMP_FIT_RANGE_2
					if fit_range_2 is not None:
						min_temp_2, max_temp_2 = fit_range_2
						filtered_indices_2 = [i for i, temp in enumerate(temps) 
											  if min_temp_2 <= temp <= max_temp_2]
						if filtered_indices_2:
							temps_fit_2 = [temps[i] for i in filtered_indices_2]
							y_vals_fit_2 = [y_vals[i] for i in filtered_indices_2]
							print(f"第二次拟合：使用温度范围 {min_temp_2}-{max_temp_2}°C 内的 {len(temps_fit_2)} 个数据点")
						else:
							temps_fit_2, y_vals_fit_2 = temps, y_vals
							print("第二次拟合：指定的温度范围内没有数据点，使用所有数据点")
					else:
						temps_fit_2, y_vals_fit_2 = temps, y_vals
					
					# 执行第二次拟合
					if len(temps_fit_2) >= config.FIT_DEGREE_2 + 1:
						# 将温度值标准化到[-1, 1]区间
						temp_min_2, temp_max_2 = min(temps_fit_2), max(temps_fit_2)
						temp_norm_2 = [2 * (t - temp_min_2) / (temp_max_2 - temp_min_2) - 1 for t in temps_fit_2]
						
						# 进行勒让德多项式拟合
						leg_coeffs_2 = leg.legfit(temp_norm_2, y_vals_fit_2, deg=config.FIT_DEGREE_2)
						
						# 计算斜率（仅对一次拟合有效）
						slope_2 = None
						if config.FIT_DEGREE_2 == 1:
							# 对于一次勒让德多项式：P(x) = c0 + c1*x
							# 其中 x 是标准化温度，需要转换为实际温度的斜率
							slope_2 = leg_coeffs_2[1] * 2 / (temp_max_2 - temp_min_2)
							print(f"第二次拟合斜率: {slope_2:.6f} (密度单位)/(°C)")
						
						# 确定第二次拟合曲线的显示范围
						if config.TEMP_FIT_DISPLAY_RANGE_2 is not None:
							temp_display_min_2, temp_display_max_2 = config.TEMP_FIT_DISPLAY_RANGE_2
						elif config.X_AXIS_RANGE is not None:
							temp_display_min_2, temp_display_max_2 = config.X_AXIS_RANGE
						else:
							temp_display_min_2, temp_display_max_2 = min(temps_fit_2), max(temps_fit_2)
						
						# 生成平滑的拟合曲线（在指定范围内）
						temp_fit_2 = np.linspace(temp_display_min_2, temp_display_max_2, 200)
						temp_fit_norm_2 = 2 * (temp_fit_2 - temp_min_2) / (temp_max_2 - temp_min_2) - 1
						y_fit_2 = leg.legval(temp_fit_norm_2, leg_coeffs_2)
						
						# 在标签中包含斜率信息
						if slope_2 is not None:
							fit_label_2 = f'Fit 2 (slope: {slope_2:.6f})'
						else:
							fit_label_2 = f'Fit 2'

						
						plt.plot(temp_fit_2, y_fit_2, '--', color='tab:green', linewidth=2, label=fit_label_2)
						
						# 标注第二条拟合曲线上的特定温度点
						if (config.ENABLE_TEMP_POINT_ANNOTATION and 
							hasattr(config, 'TEMP_POINT_2') and 
							config.TEMP_POINT_2 is not None):
							temp_point_2 = config.TEMP_POINT_2
							# 计算该温度点对应的标准化值
							temp_point_norm_2 = 2 * (temp_point_2 - temp_min_2) / (temp_max_2 - temp_min_2) - 1
							# 计算拟合曲线在该点的密度值
							y_point_2 = leg.legval(temp_point_norm_2, leg_coeffs_2)
							
							# 在图上标注该点
							marker_color_2 = getattr(config, 'ANNOTATION_MARKER_COLOR_2', 'green')
							marker_size_2 = getattr(config, 'ANNOTATION_MARKER_SIZE', 8)
							plt.plot(temp_point_2, y_point_2, 's', color=marker_color_2, 
									markersize=marker_size_2, markeredgecolor='black', markeredgewidth=1, 
									zorder=5, label=f'Point 2: {temp_point_2}°C')
							
							# 添加文字标注
							text_offset_x = getattr(config, 'ANNOTATION_TEXT_OFFSET_X', 20)
							text_offset_y = -getattr(config, 'ANNOTATION_TEXT_OFFSET_Y', 10)  # 第二个点的标注向下偏移
							text_size = getattr(config, 'ANNOTATION_TEXT_SIZE', 10)
							decimal_places = getattr(config, 'ANNOTATION_DECIMAL_PLACES', 3)
							
							annotation_text = f'({temp_point_2}°C, {y_point_2:.{decimal_places}f})'
							plt.annotate(annotation_text, 
										xy=(temp_point_2, y_point_2),
										xytext=(text_offset_x, text_offset_y), 
										textcoords='offset points',
										fontsize=text_size,
										bbox=dict(boxstyle='round,pad=0.3', facecolor=marker_color_2, alpha=0.3),
										arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
							print(f"第二条拟合曲线在 {temp_point_2}°C 处的密度值: {y_point_2:.{decimal_places}f}")
							# 提示是否为外推值
							if not (temp_min_2 <= temp_point_2 <= temp_max_2):
								print(f"  注意：{temp_point_2}°C 超出拟合数据范围 ({temp_min_2:.1f}°C - {temp_max_2:.1f}°C)，该值为外推结果")
					else:
						print(f"第二次拟合：数据点不足，无法进行{config.FIT_DEGREE_2}阶拟合（需要至少{config.FIT_DEGREE_2 + 1}个点）")
			else:
				print("拟合功能已禁用，仅显示原始数据点")

			plt.xlabel('Temperature (°C)')
			plt.ylabel(y_label)
			plt.title(title_main)
			plt.grid(True, linestyle='--', alpha=0.5)
			plt.legend()
			
			# 设置X轴范围
			if config.X_AXIS_RANGE is not None:
				plt.xlim(config.X_AXIS_RANGE)
			
			# 设置Y轴范围
			if config.Y_AXIS_RANGE is not None:
				plt.ylim(config.Y_AXIS_RANGE)
			save_name = f"temp_vs_density_{config.NAME}_{config.VOLUME_METHOD}.png"
			out_img = os.path.join(os.path.dirname(__file__), save_name)
			plt.tight_layout()
			plt.savefig(out_img, dpi=config.PLOT_DPI)
			print(f"温度-密度/体积图已保存: {out_img}")
			plt.show()  # 显示图像
			
			# 保存温度和密度数据到CSV文件
			try:
				csv_output_path = os.path.join(os.path.dirname(__file__), f"temp_density_data_{config.NAME}_{config.VOLUME_METHOD}.csv")
				with open(csv_output_path, 'w', newline='', encoding='utf-8') as csvfile:
					writer = csv.writer(csvfile)
					# 写入表头
					if any(d is not None for d in densities):
						writer.writerow(['Temperature(°C)', 'Density(g/cm³)', 'Volume(px³)', 'Circularity', 'Filename', 'Timestamp', 'Matched_CSV_Time'])
						# 写入数据
						for i, (temp, vol) in enumerate(zip(temps, vols)):
							density = densities[i] if i < len(densities) else None
							# 找到对应的文件名、时间信息和 circularity
							filename = None
							timestamp = None
							matched_csv_time = None
							circ = None
							for r in results:
								if r[3] == temp and r[4] == vol:
									filename = r[0]
									timestamp = r[1]  # readable timestamp
									matched_csv_time = r[2]  # matched CSV time
									circ = r[6] if len(r) > 6 else None
									break
							writer.writerow([temp, density, vol, circ, filename, timestamp, matched_csv_time])
					else:
						writer.writerow(['Temperature(°C)', 'Volume(px³)', 'Circularity', 'Filename', 'Timestamp', 'Matched_CSV_Time'])
						# 写入数据
						for i, (temp, vol) in enumerate(zip(temps, vols)):
							# 找到对应的文件名、时间信息和 circularity
							filename = None
							timestamp = None
							matched_csv_time = None
							circ = None
							for r in results:
								if r[3] == temp and r[4] == vol:
									filename = r[0]
									timestamp = r[1]  # readable timestamp
									matched_csv_time = r[2]  # matched CSV time
									circ = r[6] if len(r) > 6 else None
									break
							writer.writerow([temp, vol, circ, filename, timestamp, matched_csv_time])
				print(f"温度-密度/体积数据已保存到: {csv_output_path}")
			except Exception as e:
				print(f"保存CSV文件失败: {e}")
			# 额外保存 时间 - Circularity 图（基于每张图片的时间戳）
			try:
				# 收集有时间和 circularity 的点
				time_circ_pts = [(r[5], r[6]) for r in results if r[5] is not None and r[6] is not None and r[4] is not None]
				if time_circ_pts:
					time_circ_pts.sort(key=lambda x: x[0])
					dts_c = [p[0] for p in time_circ_pts]
					circs = [p[1] for p in time_circ_pts]
					if config.ROOM_TEMP_TIME_USE_ELAPSED_MINUTES:
						base = dts_c[0]
						xc = [(dt - base).total_seconds() / 60.0 for dt in dts_c]
						xlabel = 'Elapsed Time (min)'
					else:
						xc = dts_c
						xlabel = 'Time'
					plt.figure(figsize=config.PLOT_FIGSIZE)
					plt.plot(xc, circs, 'o-', color='tab:purple', alpha=0.8)
					plt.xlabel(xlabel)
					plt.ylabel('Circularity')
					plt.title('Time vs Circularity (' + config.NAME + ')')
					plt.grid(True, linestyle='--', alpha=0.5)
					plt.tight_layout()
					circ_out2 = os.path.join(os.path.dirname(__file__), f"time_vs_circularity_{config.NAME}_{config.VOLUME_METHOD}.png")
					plt.savefig(circ_out2, dpi=config.PLOT_DPI)
					print(f"时间-圆度(circularity)图已保存: {circ_out2}")
			except Exception as _e:
				print(f"保存温度模式下的时间-圆度图失败: {_e}")
			else:
				print('没有足够的数据点绘制温度-体积/密度图。')
	except Exception as e:
		print(f'绘图失败: {e}')

	return 0


if __name__ == '__main__':
	raise SystemExit(main())

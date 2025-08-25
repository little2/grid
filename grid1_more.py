import os
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy import VideoFileClip
import imagehash
from insightface.app import FaceAnalysis
from datetime import datetime

import time, sys

# === 工具：把图按原比例缩放塞进固定格（letterbox，不裁切，居中） ===
def _fit_into_cell(img: Image.Image, cell_w: int, cell_h: int, bg=(0, 0, 0)):
    iw, ih = img.size
    if iw == 0 or ih == 0:
        return Image.new("RGB", (cell_w, cell_h), bg)

    scale = min(cell_w / iw, cell_h / ih)
    new_w = max(1, int(round(iw * scale)))
    new_h = max(1, int(round(ih * scale)))
    resized = img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", (cell_w, cell_h), bg)
    ox = (cell_w - new_w) // 2
    oy = (cell_h - new_h) // 2
    canvas.paste(resized, (ox, oy))
    return canvas

# === 工具：按“主图 2×2 左上 + 余格从左到右、从上到下”生成坐标顺序 ===
def _grid_coords_with_hero_first(cols: int, rows: int):
    coords = []
    for y in range(rows):
        for x in range(cols):
            # 跳过主图占位 (0,0),(1,0),(0,1),(1,1)
            if x < 2 and y < 2:
                continue
            coords.append((x, y))
    return coords



# === 新增：根据视频时长自动决定网格与采样 ===
def _decide_layout_by_duration(duration_sec: float) -> dict:
    """
    根据视频时长返回布局与采样参数：
    - <  5 min: 3x4,  主图4 + 辅助8  = 12格, sample_count=100
    - 5-10 min: 4x4,  主图4 + 辅助12 = 16格, sample_count=150
    - 10-30 min:5x5,  主图4 + 辅助21 = 25格, sample_count=200
    - > 30 min: 6x6,  主图4 + 辅助32 = 36格, sample_count=300
    """
    m = duration_sec / 60.0
    if m < 5:
        return {"cols": 3, "rows": 4, "num_aux": 8,  "sample_count": 100}
    elif m < 10:
        return {"cols": 4, "rows": 4, "num_aux": 12, "sample_count": 150}
    elif m < 30:
        return {"cols": 5, "rows": 5, "num_aux": 21, "sample_count": 200}
    else:
        return {"cols": 6, "rows": 6, "num_aux": 32, "sample_count": 300}


def _fmt_eta(elapsed, done, total):
    if done == 0: 
        return "--:--"
    rate = elapsed / done
    remain = rate * (total - done)
    m, s = divmod(int(remain), 60)
    return f"{m:02d}:{s:02d}"

def _stage(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def _progress(prefix: str, i: int, total: int, start_ts: float, every: int = 10):
    """每 every 次打印一次进度 + ETA。"""
    if i % every != 0 and i != total:
        return
    elapsed = time.time() - start_ts
    pct = 100.0 * i / total if total else 100.0
    eta = _fmt_eta(elapsed, i, total)
    print(f"{prefix}: {i}/{total} ({pct:5.1f}%)  ETA {eta}", flush=True)



# === 新增：通用时间解析 ===
def _parse_time_to_seconds(t) -> float:
    """
    支持: float / int（秒）、"mm:ss"、"hh:mm:ss"、"ss"（可含小数）
    """
    if isinstance(t, (int, float)):
        return float(t)
    s = str(t).strip()
    parts = s.split(":")
    try:
        if len(parts) == 1:
            return float(parts[0])
        elif len(parts) == 2:
            m = float(parts[0]); sec = float(parts[1])
            return m * 60 + sec
        elif len(parts) == 3:
            h = float(parts[0]); m = float(parts[1]); sec = float(parts[2])
            return h * 3600 + m * 60 + sec
    except ValueError:
        pass
    raise ValueError(f"无法解析时间格式: {t!r}")

def _safe_get_frame(clip, t: float) -> Image.Image | None:
    try:
        return Image.fromarray(clip.get_frame(float(t)))
    except Exception:
        return None


# === 新增：清晰度（锐度）评分：拉普拉斯方差 ===
def _lap_var(pil_img: Image.Image) -> float:
    arr = np.array(pil_img.convert("L"), dtype=np.float32)
    # 3x3 拉普拉斯核
    k = np.array([[0, 1, 0],
                  [1,-4, 1],
                  [0, 1, 0]], dtype=np.float32)
    # 简易卷积（零填充）
    h, w = arr.shape
    pad = 1
    padded = np.pad(arr, pad, mode="edge")
    out = np.zeros_like(arr)
    for y in range(h):
        for x in range(w):
            region = padded[y:y+3, x:x+3]
            out[y, x] = (region * k).sum()
    return float(out.var())

# === 新增：在主图附近二次搜索更优主图 ===
def _refine_hero_nearby(
    app,
    clip,
    hero_time: float,
    window: float = 1,       # 在主图前后各 1 秒
    step: float = 0.1,       # 以 0.1 秒步进抽帧
    min_face_area: float = 1e-4,  # 最小单脸面积阈值(相对画面)，太小视为噪声
) -> tuple[Image.Image | None, dict | None]:
    """
    返回(可能替换后的主图, 元数据)
    评分 = 人脸总面积(归一化) * 清晰度(拉普拉斯方差)
    增强：更细的进度反馈（每秒节流一次）与“新最佳”即时提示。
    """
    duration = max(clip.duration, 0.01)
    t0 = max(0.0, hero_time - window)
    t1 = min(duration - 1e-3, hero_time + window)

    times = np.arange(t0, t1 + 1e-9, step, dtype=np.float64)
    total = len(times)
    if total <= 0:
        _stage("精修区间为空，跳过。")
        return None, None

    _stage(f"开始精修主图：窗口 {t0:.2f}s~{t1:.2f}s，步长 {step}s，共 {total} 帧")
    start_ts = time.time()
    last_tick = start_ts

    best = None
    cnt_scanned = 0
    cnt_has_face = 0
    cnt_filtered_small = 0  # 有脸但全都低于面积阈值而被过滤
    last_best_score = -1.0

    # 先取一次尺寸，避免循环反复取
    frame0 = Image.fromarray(clip.get_frame(float(times[0])))
    H0, W0 = np.array(frame0).shape[:2]

    for i, t in enumerate(times, start=1):
        _progress("精修进度", i, total, start_ts, every=10)
        cnt_scanned += 1

        try:
            img = Image.fromarray(clip.get_frame(float(t)))
        except Exception:
            # 读帧失败照样进入下一帧，并做节流状态汇报
            now = time.time()
            if now - last_tick >= 1.0:
                _stage(f"扫描中… {i}/{total} | 有脸 {cnt_has_face} | 过滤(小脸) {cnt_filtered_small} | 当前最佳 {max(0.0,last_best_score):.2f}")
                last_tick = now
            continue

        arr = np.array(img)
        faces = app.get(arr)
        if not faces:
            # 无脸
            now = time.time()
            if now - last_tick >= 1.0:
                _stage(f"扫描中… {i}/{total} | 有脸 {cnt_has_face} | 过滤(小脸) {cnt_filtered_small} | 当前最佳 {max(0.0,last_best_score):.2f}")
                last_tick = now
            continue

        # 统计可用脸面积
        H, W = arr.shape[:2]
        area_sum = 0.0
        faces_meta = []
        for f in faces:
            x1, y1, x2, y2 = map(int, f.bbox)
            x1, y1, x2, y2 = max(0,x1), max(0,y1), min(W,x2), min(H,y2)
            a = max(0, x2-x1) * max(0, y2-y1) / (W*H)
            if a >= min_face_area:
                area_sum += a
                faces_meta.append({"bbox":[x1,y1,x2,y2], "area_norm": a})

        if area_sum > 0:
            cnt_has_face += 1
            sharp = _lap_var(img)
            score = area_sum * sharp
            candidate = (score, -abs(t-hero_time), img, {"time": float(t), "score_face_area": area_sum, "sharp": sharp, "score": float(score)})

            if best is None or candidate > best:
                best = candidate
                last_best_score = best[0]
                # 立刻报告新最佳
                meta = best[3]
                _stage(f"↑ 新最佳：t={meta['time']:.3f}s  面积={meta['score_face_area']:.5f}  清晰={meta['sharp']:.2f}  分数={meta['score']:.2f}")
        else:
            # 有脸但都太小
            cnt_filtered_small += 1

        # 每秒节流一次状态输出
        now = time.time()
        if now - last_tick >= 1.0:
            _stage(f"扫描中… {i}/{total} | 有脸 {cnt_has_face} | 过滤(小脸) {cnt_filtered_small} | 当前最佳 {max(0.0,last_best_score):.2f}")
            last_tick = now

    if best is None:
        _stage("精修未找到更优帧，沿用粗选主图")
        return None, None

    meta = best[3]
    _stage(f"精修完成：新主图时间 {meta['time']:.3f}s，综合分 {meta['score']:.2f} | 总扫 {cnt_scanned}，有脸 {cnt_has_face}，小脸过滤 {cnt_filtered_small}")
    return best[2], meta


def _safe_load_font(font_path: str | None, size: int):
    if font_path and os.path.exists(font_path):
        try:
            from PIL import ImageFont
            return ImageFont.truetype(font_path, size=size)
        except Exception:
            pass
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        from PIL import ImageFont
        return ImageFont.load_default()


def _draw_text_with_outline(draw, xy, text, font, fill=(255, 255, 255, 220)):
    x, y = xy
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(-1,1),(1,1)]:
        draw.text((x+dx, y+dy), text, font=font, fill=(0, 0, 0, 200))
    draw.text((x, y), text, font=font, fill=fill)


def _choose_hero_frame(app: FaceAnalysis, frames: List[Tuple[float, Image.Image]]):
    _stage(f"开始粗选主图：共 {len(frames)} 帧做人脸检测")
    start_ts = time.time()
    best = None
    for idx, (t, img) in enumerate(frames, start=1):
        arr = np.array(img)
        faces = app.get(arr)
        if not faces:
            score = 0.0
        else:
            h, w = arr.shape[:2]
            areas = []
            for f in faces:
                x1, y1, x2, y2 = map(int, f.bbox)
                x1, y1, x2, y2 = max(0,x1), max(0,y1), min(w,x2), min(h,y2)
                area = max(0, x2-x1) * max(0, y2-y1)
                areas.append(area / (w*h))
            score = float(np.sum(areas))
        candidate = (score, len(faces), -t, img, {"time": float(t), "score": score})
        if best is None or candidate > best:
            best = candidate

        _progress("粗选进度", idx, len(frames), start_ts, every=5)

    _stage(f"粗选完成：主图时间 {best[4]['time']:.3f}s，得分 {best[4]['score']:.6f}")
    return best[3], best[4]



def _extract_diverse_frames(
    clip: VideoFileClip,
    num_frames: int,
    extra: int = 6,
    exclude_hashes: List[imagehash.ImageHash] | None = None,
    exclude_thr: int = 6,
) -> List[Tuple[float, Image.Image]]:
    """
    从视频中均匀抽样（num_frames + extra），用感知哈希贪心选取多样帧。
    - exclude_hashes / exclude_thr：用于排除与主图（或其它）过相近的帧。
    - 自适应阈值：根据候选帧间哈希距离的分位数调节多样性筛选强度。
    """
    _stage(f"开始抽辅助帧：候选 {num_frames + extra}，目标 {num_frames}")

    duration = max(float(clip.duration or 0.01), 0.01)
    # 均匀时间点（不取到结尾，避免越界取帧）
    times = np.linspace(0.0, duration, num_frames + extra, endpoint=False, dtype=np.float64)

    raw_frames: List[Tuple[float, Image.Image]] = []
    start_ts = time.time()

    # 先“取帧”，再“算 hash”
    for k, t in enumerate(times, start=1):
        try:
            img = Image.fromarray(clip.get_frame(float(t)))
            raw_frames.append((float(t), img))
        except Exception:
            # 取帧失败跳过该点
            pass
        _progress("多样性筛选", k, len(times), start_ts, every=10)

    if not raw_frames:
        # 兜底：至少返回第一帧
        try:
            return [(0.0, Image.fromarray(clip.get_frame(0.0)))]
        except Exception:
            return []

    # 计算候选帧的 dhash
    hashes: List[imagehash.ImageHash] = [imagehash.dhash(img) for _, img in raw_frames]

    # 自适应阈值：候选帧两两距离的 25% 分位数，最低 5
    dists = []
    for i in range(len(hashes)):
        hi = hashes[i]
        for j in range(i + 1, len(hashes)):
            dists.append(abs(hi - hashes[j]))
    adaptive_thr = max(5, int(np.percentile(dists, 25))) if dists else 6

    selected: List[Tuple[float, Image.Image]] = []
    selected_hashes: List[imagehash.ImageHash] = []

    # 先进行主循环：排除与主图过近，再做多样性筛选
    for (t, img), h in zip(raw_frames, hashes):
        # 与需排除的哈希过近则跳过（例如主图）
        if exclude_hashes and any(abs(h - eh) < exclude_thr for eh in exclude_hashes):
            continue
        # 与已选中过近（多样性不足）则跳过
        if selected_hashes and any(abs(h - sh) < adaptive_thr for sh in selected_hashes):
            continue

        selected.append((t, img))
        selected_hashes.append(h)
        if len(selected) >= num_frames:
            break

    # 若还不足，做一次“宽松补齐”（仍避开主图相近，但放松与已选距离）
    if len(selected) < num_frames:
        for idx, ((t, img), h) in enumerate(zip(raw_frames, hashes)):
            if len(selected) >= num_frames:
                break
            if (t, img) in selected:
                continue
            if exclude_hashes and any(abs(h - eh) < exclude_thr for eh in exclude_hashes):
                continue
            # 放松条件：只要求“不是特别近”，阈值降一档
            if not selected_hashes or all(abs(h - sh) >= max(4, adaptive_thr - 2) for sh in selected_hashes):
                selected.append((t, img))
                selected_hashes.append(h)

    # 仍不足就按时间顺序兜底（最后兜底仍避开主图过近）
    if len(selected) < num_frames:
        for (t, img), h in zip(raw_frames, hashes):
            if len(selected) >= num_frames:
                break
            if (t, img) in selected:
                continue
            if exclude_hashes and any(abs(h - eh) < exclude_thr for eh in exclude_hashes):
                continue
            selected.append((t, img))

    # 排序 & 截断
    selected.sort(key=lambda x: x[0])
    selected = selected[:num_frames]

    _stage(f"辅助帧完成：实际选取 {len(selected)} 张")
    return selected

# === 黑边侦测：判断整行/整列是否近似黑色 ===
def _is_near_black_line(arr_row_or_col: np.ndarray, thr: int = 16, ratio: float = 0.98) -> bool:
    """
    arr_row_or_col: [W,3] 或 [H,3] 的一行/一列RGB
    thr: 像素被视为“黑”的阈值 (0-255)
    ratio: 该行/列中 >=ratio 的像素都很黑，才当作黑边行/列
    """
    if arr_row_or_col.ndim == 1:
        arr = arr_row_or_col
    else:
        arr = arr_row_or_col
    # 转灰度近似：max通道更保守（避免暗色内容被误判）
    if arr.ndim == 2 and arr.shape[1] == 3:
        v = arr.max(axis=1)
    else:
        v = arr
    return (v < thr).mean() >= ratio


def _detect_letterbox_bbox(img: Image.Image,
                           thr: int = 16,
                           ratio: float = 0.98,
                           max_frac: float = 0.20) -> tuple[int, int, int, int] | None:
    """
    侦测单张图的黑边裁切框 (left, top, right, bottom)；若无明显黑边则返回 None。
    max_frac: 每边最多允许裁掉比例（过大则认为不是简单的 letterbox）
    """
    arr = np.array(img.convert("RGB"))
    H, W = arr.shape[:2]

    # 扫描上边
    top = 0
    while top < H and _is_near_black_line(arr[top, :, :], thr=thr, ratio=ratio):
        top += 1
    # 扫描下边
    bottom = H - 1
    while bottom > top and _is_near_black_line(arr[bottom, :, :], thr=thr, ratio=ratio):
        bottom -= 1
    # 扫描左边
    left = 0
    while left < W and _is_near_black_line(arr[:, left, :], thr=thr, ratio=ratio):
        left += 1
    # 扫描右边
    right = W - 1
    while right > left and _is_near_black_line(arr[:, right, :], thr=thr, ratio=ratio):
        right -= 1

    # 合法性检查
    crop_w = right - left + 1
    crop_h = bottom - top + 1
    if crop_w <= 0 or crop_h <= 0:
        return None

    # 过度裁切保护：不允许单边超过 max_frac
    if (top / H > max_frac) or ((H - 1 - bottom) / H > max_frac) or \
       (left / W > max_frac) or ((W - 1 - right) / W > max_frac):
        return None

    # 若几乎没裁：视为无黑边
    if top == 0 and left == 0 and bottom == H - 1 and right == W - 1:
        return None

    return (left, top, right, bottom)


def _auto_detect_uniform_letterbox(frames: list[Image.Image],
                                   thr: int = 16,
                                   ratio: float = 0.98,
                                   max_frac: float = 0.20,
                                   tolerance_px: int = 4,
                                   min_consensus: float = 0.8) -> tuple[float, float, float, float] | None:
    """
    在多张图上侦测黑边；若多数图（>=min_consensus）得到“相近”的裁切框，则认为黑边一致。
    返回 “比例形式的裁切框”：(l_frac, t_frac, r_frac, b_frac)，后续可对任意同分辨率帧复用。
    tolerance_px: 各边允许的像素误差（一致性判定容差）
    """
    if not frames:
        return None

    # 采样最多 20 张以加速（包含主图+前若干辅助）
    sample = frames[:min(20, len(frames))]

    # 先以第一张的 W,H 作为参考
    W0, H0 = sample[0].size
    boxes = []
    for img in sample:
        if img.size != (W0, H0):
            # 分辨率不一致时，先统一到参考大小（避免像素维度不同导致一致性判断困难）
            tmp = img.resize((W0, H0), Image.LANCZOS)
            box = _detect_letterbox_bbox(tmp, thr=thr, ratio=ratio, max_frac=max_frac)
        else:
            box = _detect_letterbox_bbox(img, thr=thr, ratio=ratio, max_frac=max_frac)
        if box:
            boxes.append(box)

    if not boxes:
        return None

    # 用“众数区间”方式聚类：以第一候选为中心，找落在 pixel 容差内的最多集合
    def group_with_ref(ref_box):
        l0, t0, r0, b0 = ref_box
        group = [ref_box]
        for bx in boxes:
            if bx is ref_box:
                continue
            l, t, r, b = bx
            if abs(l - l0) <= tolerance_px and abs(t - t0) <= tolerance_px and \
               abs(r - r0) <= tolerance_px and abs(b - b0) <= tolerance_px:
                group.append(bx)
        return group

    best_group = []
    for ref in boxes:
        g = group_with_ref(ref)
        if len(g) > len(best_group):
            best_group = g

    if len(best_group) / len(sample) < min_consensus:
        # 共识不足，不认为是统一黑边
        return None

    # 取该组的中位数框，并转为“比例形式”
    ls, ts, rs, bs = zip(*best_group)
    l_med, t_med, r_med, b_med = int(np.median(ls)), int(np.median(ts)), int(np.median(rs)), int(np.median(bs))
    l_frac = l_med / W0
    t_frac = t_med / H0
    r_frac = r_med / W0
    b_frac = b_med / H0
    print(f"[letterbox] use frac box = {l_frac:.4f},{t_frac:.4f},{r_frac:.4f},{b_frac:.4f}", flush=True)

    return (l_frac, t_frac, r_frac, b_frac)


def make_hero_grid_adaptive(
    video_path: str,
    preview_basename: str,
    font_path: str | None = "fonts/Roboto_Condensed-Regular.ttf",
    sample_count: int | None = None,          # 允许外部覆盖；不传则按时长自动
    num_aux: int | None = None,               # 允许外部覆盖；不传则按时长自动
    manual_times: List[str | float | int] | None = None,
) -> Dict[str, Any]:
    """
    自适应网格图：
      - 根据视频时长决定总格数与 sample_count（最少 12 格）
      - 主图占 2x2，其余辅助格自动填充
      - 辅助图避免与主图/彼此过近（dhash）
      - 输出文件名加“年月日时分秒_”前缀
    """
    ts_prefix = datetime.now().strftime("%Y%m%d%H%M%S_")
    out_dir = Path(preview_basename).parent
    base_name = Path(preview_basename).name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / f"{ts_prefix}{base_name}.jpg")

    _stage("初始化 InsightFace 模型中 …")
    app = FaceAnalysis(providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    _stage("模型初始化完成。")

    with VideoFileClip(video_path, audio=False) as clip:
        duration = max(float(clip.duration or 0.01), 0.01)

        # 自动布局与采样配置
        auto_cfg = _decide_layout_by_duration(duration)
        cols = auto_cfg["cols"]
        rows = auto_cfg["rows"]
        auto_num_aux = auto_cfg["num_aux"]
        auto_sample_count = auto_cfg["sample_count"]

        # 允许外部覆盖
        if num_aux is None:
            num_aux = auto_num_aux
        if sample_count is None:
            sample_count = auto_sample_count

        # === 主图粗选候选帧 ===
        # 均匀取 sample_count 个时刻（不取到最尾帧，避免越界）
        hero_times = np.linspace(0, duration - (duration/(sample_count+1)), sample_count, endpoint=True)
        hero_frames: List[Tuple[float, Image.Image]] = []
        for t in hero_times:
            try:
                hero_frames.append((float(t), Image.fromarray(clip.get_frame(float(t)))))
            except Exception:
                pass
        if not hero_frames:
            hero_frames = [(0.0, Image.fromarray(clip.get_frame(0.0)))]

        # 选主图（粗选 + 精修）
        hero_img, hero_meta = _choose_hero_frame(app, hero_frames)
        refined_img, refined_meta = _refine_hero_nearby(
            app, clip,
            hero_time=hero_meta.get("time", 0.0),
            window=0.1, step=0.1, min_face_area=1e-4
        )
        if refined_img is not None:
            hero_img = refined_img
            hero_meta.update(refined_meta)

        hero_hash = imagehash.dhash(hero_img)

        # === 手动时间点优先 ===
        manual_frames: List[Tuple[float, Image.Image]] = []
        manual_hashes: List[imagehash.ImageHash] = []
        manual_used_times: List[float] = []

        if manual_times:
            duration_eps = max(duration - 1e-3, 0.0)
            parsed = []
            for t in manual_times:
                try:
                    sec = _parse_time_to_seconds(t)
                    parsed.append(min(max(0.0, float(sec)), duration_eps))
                except Exception:
                    continue
            parsed = sorted(set(round(x, 3) for x in parsed))
            for t in parsed:
                img = _safe_get_frame(clip, t)
                if img is None:
                    continue
                h = imagehash.dhash(img)
                if abs(h - hero_hash) < 6:
                    continue
                if manual_hashes and any(abs(h - mh) < 6 for mh in manual_hashes):
                    continue
                manual_frames.append((float(t), img))
                manual_hashes.append(h)
                manual_used_times.append(float(t))
                if len(manual_frames) >= num_aux:
                    break

        # === 自动补齐辅助帧（避开主图与已选手动帧） ===
        need_auto = max(0, num_aux - len(manual_frames))
        auto_frames: List[Tuple[float, Image.Image]] = []
        if need_auto > 0:
            auto_frames = _extract_diverse_frames(
                clip,
                num_frames=need_auto,
                extra=6,
                exclude_hashes=[hero_hash] + manual_hashes,
                exclude_thr=6,
            )

        aux_frames = (manual_frames + auto_frames)[:num_aux]

        # === 若黑边一致：统一裁切到固定长宽比（去黑边），再进入排版 ===
        # 收集一批用于判定的帧（主图 + 辅图前若干）
        judge_imgs = [hero_img] + [img for _, img in aux_frames[:15]]

        lt_rb_frac = _auto_detect_uniform_letterbox(
            judge_imgs,
            thr=32,            # 16 -> 32：容忍“灰黑”边
            ratio=0.95,        # 0.98 -> 0.95：容忍少量噪点
            max_frac=0.35,     # 0.20 -> 0.35：允许较厚上下黑边（如影院宽屏）
            tolerance_px=8,    # 4 -> 8：不同帧厚度差几像素也算一致
            min_consensus=0.6  # 0.8 -> 0.6：60% 样本一致就启用
        )


        if lt_rb_frac is not None:
            _stage("检测到一致黑边，将统一去除并保持固定长宽比。")
            # 用“比例裁切”应用到全部帧（确保任何同分辨率帧都一致）
            def _apply_frac_crop(img: Image.Image, frac_box: tuple[float, float, float, float]) -> Image.Image:
                l_frac, t_frac, r_frac, b_frac = frac_box
                W, H = img.size
                left   = int(round(W * l_frac))
                top    = int(round(H * t_frac))
                right  = int(round(W * r_frac))
                bottom = int(round(H * b_frac))
                left   = max(0, min(left, W - 1))
                top    = max(0, min(top, H - 1))
                right  = max(left + 1, min(right, W - 1))
                bottom = max(top + 1, min(bottom, H - 1))
                return img.crop((left, top, right + 1, bottom + 1))

            # 裁主图
            hero_img = _apply_frac_crop(hero_img, lt_rb_frac)
            # 裁辅图
            aux_frames = [(t, _apply_frac_crop(img, lt_rb_frac)) for (t, img) in aux_frames]
        else:
            _stage("未检测到一致黑边（可能是灰边/共识不足/厚度差异大），保持原图比例。")


    
    if aux_frames:
        base_w, base_h = aux_frames[0][1].width, aux_frames[0][1].height
    else:
        base_w, base_h = hero_img.size

    cell_w, cell_h = base_w, base_h
    hero_w, hero_h = cell_w * 2, cell_h * 2

    grid_w, grid_h = cell_w * cols, cell_h * rows
    grid = Image.new("RGB", (grid_w, grid_h), (0, 0, 0))

    # 主图填 2×2 区域（letterbox，不裁切）
    hero_canvas = _fit_into_cell(hero_img, hero_w, hero_h, bg=(0, 0, 0))
    grid.paste(hero_canvas, (0, 0))

    # 余格坐标顺序：满足 4×4 的 ①②③ / ①④⑤ / ⑥⑦⑧⑨ / ⑩⑪⑫⑬
    coords = _grid_coords_with_hero_first(cols, rows)

    # 逐格贴图（每格 letterbox）
    for (gx, gy), (_, img) in zip(coords, aux_frames):
        cell_canvas = _fit_into_cell(img, cell_w, cell_h, bg=(0, 0, 0))
        grid.paste(cell_canvas, (gx * cell_w, gy * cell_h))

    # 水印文字（留足边距，避免“下缘被拆掉”的观感）
    _stage("拼接网格与水印 …")
    draw = ImageDraw.Draw(grid)
    text = base_name[8:] if base_name.startswith("preview_") else base_name
    font = _safe_load_font(font_path, size=max(14, int(cell_h * 0.20)))
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    margin = 64
    x = int(grid.width  - text_w - margin)
    y = int(grid.height - text_h - margin)
    _draw_text_with_outline(draw, (x, y), text, font)

    grid.save(out_path, quality=90, optimize=True)
    _stage(f"已保存：{out_path}")


    return {
        "output_path": out_path,
        "grid_cols": cols, "grid_rows": rows,
        "hero_time": round(hero_meta.get("time", 0.0), 3),
        "hero_score": round(hero_meta.get("score", 0.0), 6),
        "aux_times": [round(t, 3) for t, _ in aux_frames],
        "manual_used_times": [round(t, 3) for t in manual_used_times] if manual_times else [],
        "sample_count_used": sample_count,
    }


if __name__ == "__main__":

    try:
        
        meta = make_hero_grid_adaptive(
            video_path="video/s6614244fe4b06d7f37acee3b.mp4",
            preview_basename="previews/370854",
            manual_times=["01:34","04:08","04:37","05:11","08:33"],        # 可选
            # 也可以手动覆盖：
            # sample_count=180,
            # num_aux=12,
        )





        print("✅ 网格已生成：", meta["output_path"])
        print("   主角帧时间(s)：", meta["hero_time"], " 评分：", meta["hero_score"])
        print("   辅助帧时间(s)：", meta["aux_times"])
    except Exception as e:
        import traceback
        print(f"❌ 错误：{e}")
        traceback.print_exc()

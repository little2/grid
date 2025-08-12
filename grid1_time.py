import os
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip
import imagehash
from insightface.app import FaceAnalysis
from datetime import datetime

import time, sys

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


def make_hero_grid_3x4(
    video_path: str,
    preview_basename: str,
    font_path: str | None = "fonts/Roboto_Condensed-Regular.ttf",
    sample_count: int = 150,
    num_aux: int = 8,
    manual_times: List[str | float | int] | None = None,  # ⬅️ 新增
) -> Dict[str, Any]:
    """
    生成 3x4 网格图：主图(2x2) + 8 张辅助图。
    满足：
      - 辅助图不会与主图重复（以感知哈希近似度过滤）
      - 输出文件名以 年月日时分秒_ 作为前缀
    """
    # === 时间戳前缀（Asia/Singapore 本地时间环境下一般可直接用系统时间）===
    ts_prefix = datetime.now().strftime("%Y%m%d%H%M%S_")

    # 输出路径：在原目录下，给文件名前缀加时间戳
    out_dir = Path(preview_basename).parent
    base_name = Path(preview_basename).name  # 不含扩展名
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / f"{ts_prefix}{base_name}.jpg")

    # 初始化 InsightFace（CPU 示例; 有 GPU 可改 ["CUDAExecutionProvider"]）
    _stage("初始化 InsightFace 模型中 …")
    app = FaceAnalysis(providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    _stage("模型初始化完成。")

    with VideoFileClip(video_path, audio=False) as clip:
        duration = max(clip.duration, 0.01)

        # 主图候选帧
        hero_times = np.linspace(0, duration - (duration/(sample_count+1)), sample_count, endpoint=True)
        hero_frames = []
        for t in hero_times:
            try:
                hero_frames.append((float(t), Image.fromarray(clip.get_frame(float(t)))))
            except Exception:
                continue
        if not hero_frames:
            hero_frames = [(0.0, Image.fromarray(clip.get_frame(0.0)))]

        # 选主图
        # 1) 先用你现有的 _choose_hero_frame 选出主图
        hero_img, hero_meta = _choose_hero_frame(app, hero_frames)

        # 2) 在主图附近“精修”一轮：若找到更优则替换
        refined_img, refined_meta = _refine_hero_nearby(
            app,
            clip,
            hero_time=hero_meta.get("time", 0.0),
            window=1,     # 可调：0.4~1.0 秒
            step=0.1,      # 可调：越小越细，越慢
            min_face_area=1e-4,
        )

        if refined_img is not None:
            hero_img = refined_img
            hero_meta.update(refined_meta)  # time/score 同步为新主图

        hero_hash = imagehash.dhash(hero_img)  # ⬅️ 用于后续排除重复

        # === 新增：手动指定时间点优先截帧 ===
        manual_frames: List[Tuple[float, Image.Image]] = []
        manual_hashes: List[imagehash.ImageHash] = []
        manual_used_times: List[float] = []

        if manual_times:
            # 解析 & 限定范围
            duration_eps = max(duration - 1e-3, 0.0)
            parsed_times = []
            for t in manual_times:
                try:
                    sec = _parse_time_to_seconds(t)
                    # 限制在 [0, duration) 区间
                    sec = min(max(0.0, float(sec)), duration_eps)
                    parsed_times.append(sec)
                except Exception:
                    # 无法解析就跳过
                    continue

            # 去重复时间点（按秒数近似）
            parsed_times = sorted(set(round(x, 3) for x in parsed_times))
            for t in parsed_times:
                img = _safe_get_frame(clip, t)
                if img is None:
                    continue
                h = imagehash.dhash(img)

                # 排除与主图过近
                if abs(h - hero_hash) < 6:
                    continue
                # 排除与已选手动帧过近（自我多样性）
                if manual_hashes and any(abs(h - mh) < 6 for mh in manual_hashes):
                    continue

                manual_frames.append((float(t), img))
                manual_hashes.append(h)
                manual_used_times.append(float(t))

                if len(manual_frames) >= num_aux:
                    break

        # === 原逻辑补齐：排除主图 + 已有手动帧 ===
        exclude_hashes = [hero_hash] + manual_hashes
        need_auto = max(0, num_aux - len(manual_frames))

        auto_frames: List[Tuple[float, Image.Image]] = []
        if need_auto > 0:
            auto_frames = _extract_diverse_frames(
                clip,
                num_frames=need_auto,
                extra=6,
                exclude_hashes=exclude_hashes,
                exclude_thr=6,
            )

        # 最终辅助帧 = 手动优先 + 自动补齐（保持手动在前的顺序）
        aux_frames = manual_frames + auto_frames

    # 统一尺寸（以首张辅助帧为单位；如无辅助帧则用主图尺寸）
    base_w, base_h = (aux_frames[0][1].width, aux_frames[0][1].height) if aux_frames else hero_img.size
    unit_w, unit_h = base_w, base_h
    hero_w, hero_h = unit_w * 2, unit_h * 2
    grid_w, grid_h = unit_w * 3, unit_h * 4

    grid = Image.new("RGBA", (grid_w, grid_h))
    grid.paste(hero_img.resize((hero_w, hero_h), Image.LANCZOS), (0, 0))

    # 辅助格位置
    positions = [(2, 0), (2, 1), (0, 2), (1, 2), (2, 2), (0, 3), (1, 3), (2, 3)]
    for (gx, gy), (_, img) in zip(positions, aux_frames):
        grid.paste(img.resize((unit_w, unit_h), Image.LANCZOS), (gx * unit_w, gy * unit_h))

    # 水印（仍使用原来的预览名，但显示时不含 "preview_" 前缀）
    _stage("拼接网格与水印 …")
    draw = ImageDraw.Draw(grid)
    text = base_name
    if text.startswith("preview_"):
        text = text[len("preview_"):]
    font = _safe_load_font(font_path, size=max(14, int(unit_h * 0.20)))
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # 安全边距（避免四边看起来被切）
    margin = 64

    x = int(grid.width  - text_w - margin)
    y = int(grid.height - text_h - margin)
    _draw_text_with_outline(draw, (x, y), text, font)

    grid.convert("RGB").save(out_path, quality=90, optimize=True)
    _stage(f"已保存：{out_path}")
    return {
        "output_path": out_path,
        "hero_time": round(hero_meta.get("time", 0.0), 3),
        "hero_score": round(hero_meta.get("score", 0.0), 6),
        "aux_times": [round(t, 3) for t, _ in aux_frames],
        "manual_used_times": [round(t, 3) for t in manual_used_times] if manual_times else [],
    }

if __name__ == "__main__":

    try:
        

        meta = make_hero_grid_3x4(
            video_path="video/【91】偷拍超极品正太洗澡第二弹洁白无瑕.mp4",
            preview_basename="previews/387831",
            # manual_times=["0:11", 12]  # 0分12.5秒、1分23秒、95.2秒
            manual_times=["1:15"]  # 0分12.5秒、1分23秒、95.2秒
        )

        print("✅ 网格已生成：", meta["output_path"])
        print("   主角帧时间(s)：", meta["hero_time"], " 评分：", meta["hero_score"])
        print("   辅助帧时间(s)：", meta["aux_times"])
    except Exception as e:
        import traceback
        print(f"❌ 错误：{e}")
        traceback.print_exc()

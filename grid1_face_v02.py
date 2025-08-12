import os
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip
import imagehash
from insightface.app import FaceAnalysis
from datetime import datetime


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
    """以人脸总面积为评分选择主图；并返回其时间与评分。"""
    best = None
    for t, img in frames:
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
    return best[3], best[4]


def _extract_diverse_frames(
    clip: VideoFileClip,
    num_frames: int,
    extra: int = 6,
    exclude_hashes: List[imagehash.ImageHash] | None = None,
    exclude_thr: int = 6,
) -> List[Tuple[float, Image.Image]]:
    """
    从视频中均匀抽样（num_frames+extra），用感知哈希贪心选取多样帧。
    新增: exclude_hashes/ exclude_thr 用于排除“与主图相近”的帧。
    """
    duration = max(clip.duration, 0.01)
    times = np.linspace(0, duration - (duration/(num_frames+extra+1)), num_frames + extra, endpoint=True)

    raw_frames: List[Tuple[float, Image.Image]] = []
    for t in times:
        try:
            raw_frames.append((float(t), Image.fromarray(clip.get_frame(float(t)))))
        except Exception:
            continue
    if not raw_frames:
        return [(0.0, Image.fromarray(clip.get_frame(0.0)))]

    hashes = [imagehash.dhash(fr) for _, fr in raw_frames]

    # 自适应阈值，避免过严或过松
    dists = []
    for i in range(len(hashes)):
        for j in range(i+1, len(hashes)):
            dists.append(abs(hashes[i] - hashes[j]))
    adaptive_thr = max(5, int(np.percentile(dists, 25))) if dists else 6

    selected: List[Tuple[float, Image.Image]] = []
    selected_hashes: List[imagehash.ImageHash] = []

    for (t, img), h in zip(raw_frames, hashes):
        # 先排除与主图过近的帧 —— 关键改动 👇
        if exclude_hashes and any(abs(h - eh) < exclude_thr for eh in exclude_hashes):
            continue
        # 再做多样性筛选
        if not selected or all(abs(h - sh) >= adaptive_thr for sh in selected_hashes):
            selected.append((t, img))
            selected_hashes.append(h)
        if len(selected) >= num_frames:
            break

    # 不足则补齐（仍然避开主图相近帧）
    idx = 0
    while len(selected) < num_frames and idx < len(raw_frames):
        (t, img) = raw_frames[idx]
        h = hashes[idx]
        if (t, img) not in selected:
            if not exclude_hashes or all(abs(h - eh) >= exclude_thr for eh in exclude_hashes):
                selected.append((t, img))
        idx += 1

    selected.sort(key=lambda x: x[0])
    return selected[:num_frames]


def make_hero_grid_3x4(
    video_path: str,
    preview_basename: str,
    font_path: str | None = "fonts/Roboto_Condensed-Regular.ttf",
    sample_count: int = 100,
    num_aux: int = 8,
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
    app = FaceAnalysis(providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

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
        hero_img, hero_meta = _choose_hero_frame(app, hero_frames)
        hero_hash = imagehash.dhash(hero_img)  # ⬅️ 用于后续排除重复

        # 抽辅助帧，并明确排除与主图相近的帧 —— 关键改动 👇
        aux_frames = _extract_diverse_frames(
            clip,
            num_frames=num_aux,
            extra=6,
            exclude_hashes=[hero_hash],
            exclude_thr=6,  # 主图排除阈值，越大越严格
        )

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
    draw = ImageDraw.Draw(grid)
    text = base_name
    if text.startswith("preview_"):
        text = text[len("preview_"):]
    font = _safe_load_font(font_path, size=max(16, int(unit_h * 0.28)))
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # 安全边距（避免四边看起来被切）
    margin = 64

    x = int(grid.width  - text_w - margin)
    y = int(grid.height - text_h - margin)
    _draw_text_with_outline(draw, (x, y), text, font)

    grid.convert("RGB").save(out_path, quality=90, optimize=True)

    return {
        "output_path": out_path,
        "hero_time": round(hero_meta.get("time", 0.0), 3),
        "hero_score": round(hero_meta.get("score", 0.0), 6),
        "aux_times": [round(t, 3) for t, _ in aux_frames],
    }


if __name__ == "__main__":
    video_path = "video/【91】最新偷拍超极品正太洗澡、很调皮被发现了还撸硬给你看，又大又翘.mp4"
    preview_basename = "previews/preview_209551"
    try:
        meta = make_hero_grid_3x4(video_path, preview_basename)
        print("✅ 网格已生成：", meta["output_path"])
        print("   主角帧时间(s)：", meta["hero_time"], " 评分：", meta["hero_score"])
        print("   辅助帧时间(s)：", meta["aux_times"])
    except Exception as e:
        import traceback
        print(f"❌ 错误：{e}")
        traceback.print_exc()

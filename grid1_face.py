import os
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip
import imagehash
from insightface.app import FaceAnalysis


def get_face_rich_frame(video_path: str, sample_count: int = 100):
    """
    从视频中抽样 sample_count 帧，返回人脸最多的那一帧图像
    """
    clip = VideoFileClip(video_path, audio=False)
    duration = clip.duration
    times = np.linspace(0, duration, sample_count, endpoint=False)
    app = FaceAnalysis(providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0)

    max_faces = 0
    best_frame = None
    for t in times:
        frame = Image.fromarray(clip.get_frame(t))
        faces = app.get(np.array(frame))
        if len(faces) > max_faces:
            max_faces = len(faces)
            best_frame = frame

    return best_frame if best_frame else Image.fromarray(clip.get_frame(0))


def extract_diverse_frames(video_path: str, num_frames: int = 8):
    """
    从视频中抽 num_frames 张图，确保时间均匀且图像差异大
    """
    clip = VideoFileClip(video_path, audio=False)
    duration = clip.duration
    times = np.linspace(0, duration, num_frames + 5, endpoint=False)
    frames = [(t, Image.fromarray(clip.get_frame(t))) for t in times]

    # 计算图像哈希
    frame_hashes = [imagehash.average_hash(img) for _, img in frames]
    selected = [frames[0][1]]
    selected_hashes = [frame_hashes[0]]

    for i in range(1, len(frames)):
        h = frame_hashes[i]
        if all(abs(h - sh) > 5 for sh in selected_hashes):  # 5 是 hash 差异阈值
            selected.append(frames[i][1])
            selected_hashes.append(h)
        if len(selected) >= num_frames:
            break

    return selected[:num_frames]


def make_hero_grid_3x4(video_path: str, preview_basename: str, font_path: str = "fonts/Roboto_Condensed-Regular.ttf") -> str:
    """
    主图为人脸最多的帧，其余 8 张图采用 hash 差异大的抽帧方式生成 3x4 网格
    """
    # 获取主图
    print("🔍 正在挑选主角图...", flush=True)
    hero_frame = get_face_rich_frame(video_path)

    # 获取副图
    print("📸 正在抽取辅助帧...", flush=True)
    other_frames = extract_diverse_frames(video_path, num_frames=8)

    # 网格图准备
    unit_w, unit_h = other_frames[0].size
    hero_w, hero_h = unit_w * 2, unit_h * 2
    grid_w, grid_h = unit_w * 3, unit_h * 4

    grid_img = Image.new("RGBA", (grid_w, grid_h))
    grid_img.paste(hero_frame.resize((hero_w, hero_h)), (0, 0))

    # 副图位置
    grid_positions = [
        (2, 0), (2, 1), (0, 2), (1, 2),
        (2, 2), (0, 3), (1, 3), (2, 3)
    ]
    for idx, (gx, gy) in enumerate(grid_positions):
        if idx >= len(other_frames):
            break
        img = other_frames[idx].resize((unit_w, unit_h))
        grid_img.paste(img, (gx * unit_w, gy * unit_h))

    # 加文字水印
    draw = ImageDraw.Draw(grid_img)
    font_size = int(unit_h * 0.3)
    font = ImageFont.truetype(font_path, size=font_size)
    text = Path(preview_basename).name
    if text.startswith("preview_"):
        text = text[len("preview_"):]
    bbox = draw.textbbox((0, 0), text, font=font)
    x = grid_img.width - bbox[2] - 12
    y = grid_img.height - bbox[3] - 12
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 200))

    # 保存为 JPG
    output_path = f"{preview_basename}.jpg"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    grid_img.convert("RGB").save(output_path, quality=90)
    print(f"✅ 网格已生成：{output_path}", flush=True)
    return output_path


if __name__ == "__main__":
    video_path = "video/video_2025-08-09_15-15-33.mp4"
    preview_basename = "previews/preview_209551"

    try:
        result = make_hero_grid_3x4(video_path, preview_basename)
        print(f"完成 ✅ 输出：{result}")
    except Exception as e:
        import traceback
        print(f"❌ 错误：{e}")
        traceback.print_exc()

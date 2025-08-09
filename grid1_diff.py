import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip
import imagehash

def extract_diverse_frames(video_path: str, num_frames: int = 30, sample_frames: int = 100):
    clip = VideoFileClip(video_path, audio=False)
    duration = clip.duration
    times = np.linspace(0, duration, sample_frames, endpoint=False)
    frames = [(t, Image.fromarray(clip.get_frame(t))) for t in times]

    # 使用 perceptual hash（平均哈希）计算图像差异
    frame_hashes = [imagehash.average_hash(img) for _, img in frames]

    # 选第一帧作为参考主图，其余帧计算差异度
    base_hash = frame_hashes[0]
    diff_scores = [abs(base_hash - h) for h in frame_hashes]
    frame_infos = list(zip(diff_scores, frames))

    # 选出前 num_frames 个差异最大的帧
    frame_infos.sort(reverse=True, key=lambda x: x[0])
    selected_frames = [frames[0][1]] + [img for _, (_, img) in frame_infos[1:num_frames]]

    return selected_frames[:num_frames]


def make_hero_grid_3x4(video_path: str, preview_basename: str, font_path: str = "fonts/Roboto_Condensed-Regular.ttf") -> str:
    """
    主图+辅图九宫格（3x4）：主图占左上2x2，其余填入剩下8格
    """
    frames = extract_diverse_frames(video_path, num_frames=9)

    unit_w, unit_h = frames[1].size
    hero_w, hero_h = unit_w * 2, unit_h * 2
    grid_w, grid_h = unit_w * 3, unit_h * 4  # 3列×4行

    grid_img = Image.new("RGBA", (grid_w, grid_h))

    # 主图：左上 2x2
    grid_img.paste(frames[0].resize((hero_w, hero_h)), (0, 0))

    # 其余 8 张图的网格坐标
    grid_positions = [
        (2, 0), (2, 1), (0, 2), (1, 2),
        (2, 2), (0, 3), (1, 3), (2, 3)
    ]

    for idx, (gx, gy) in enumerate(grid_positions):
        if idx + 1 >= len(frames):
            break
        img = frames[idx + 1].resize((unit_w, unit_h))
        x, y = gx * unit_w, gy * unit_h
        grid_img.paste(img, (x, y))

    # 加水印
    draw = ImageDraw.Draw(grid_img)
    font_size = int(unit_h * 0.3)
    font = ImageFont.truetype(font_path, size=font_size)
    text = Path(preview_basename).name
    if text.startswith("preview_"):
        text = text[len("preview_"):]
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = grid_img.width - text_width - 12
    y = grid_img.height - text_height - 12
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 200))

    # 保存为 JPG
    output_path = f"{preview_basename}.jpg"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    grid_img.convert("RGB").save(output_path, quality=90)
    print(f"✅ 网格已生成：{output_path}", flush=True)
    return output_path


if __name__ == "__main__":
    video_path = "video/Luba, Sanya & Sasha Blowjob vid.mkv"
    preview_basename = "previews/preview_sample"

    try:
        result = make_hero_grid_3x4(video_path, preview_basename)
        print(f"完成 ✅ 输出：{result}")
    except Exception as e:
        print(f"错误：{e}")

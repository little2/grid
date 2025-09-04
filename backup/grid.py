



import os
import time
import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from sklearn.cluster import DBSCAN
import imagehash

def smart_extract_hero_frames(video_path: str) -> list[Image.Image]:
    """
    智能分析视频帧，选出主角图为主图，其余帧按时间分布挑选为副图，确保多样性
    """
    clip = VideoFileClip(video_path, audio=False)
    duration = clip.duration
    if duration <= 0:
        raise RuntimeError("视频时长为 0")

    print(f"📹 正在分析视频：{video_path}", flush=True)
    # ✅ 动态抽帧数
    if duration < 10:
        total = 10
    elif duration < 30:
        total = 30
    elif duration < 60:
        total = 40
    elif duration < 180:
        total = 60
    elif duration < 600:
        total = 90
    else:
        total = 120


    print(f"⏱ 视频总时长：{duration:.2f}s，准备抽取 {total} 帧", flush=True)


    # 初始化 InsightFace
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))

    times = [(i + 1) * duration / (total + 1) for i in range(total)]
    all_frames = []         # 每帧: {'index': i, 'time': t, 'image': img}
    frame_data = []         # 有人脸的帧
    embeddings = []
    face_frame_map = []

    for i, t in enumerate(times):
        try:
            frame = clip.get_frame(t)
            img = Image.fromarray(frame).convert("RGB")
            all_frames.append({'index': i, 'time': t, 'image': img})

            faces = app.get(frame)
            if faces:
                print(f"✅ 检测到 {len(faces)} 张人脸", flush=True)
                frame_record = {
                    "index": i,
                    "image": img,
                    "faces": faces,
                }
                frame_data.append(frame_record)

                for face in faces:

                    emb = face.normed_embedding
                    if emb is not None:
                        embeddings.append(emb)
                        face_frame_map.append(len(frame_data) - 1)
        except Exception as e:
            print(f"⚠️ 第 {t:.2f}s 抽帧失败：{e}")

    if not frame_data:
        print("⚠️ 视频中无检测到人脸，主图将使用第 0 帧")
        return [all_frames[0]['image']] + [fr['image'] for fr in all_frames[1:9]]

    # 主图逻辑（聚类 + 最大脸面积）
    if len(embeddings) >= 2:
        embeddings = np.array(embeddings)
        clustering = DBSCAN(eps=0.4, min_samples=2, metric='cosine').fit(embeddings)
        labels = clustering.labels_

        label_counts = {l: labels.tolist().count(l) for l in set(labels) if l != -1}
        if label_counts:
            main_label = max(label_counts, key=label_counts.get)
            main_face_indices = [i for i, label in enumerate(labels) if label == main_label]
            candidate_frame_indices = list({face_frame_map[i] for i in main_face_indices})

            def face_area(face): return (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
            main_frame = max(
                (frame_data[i] for i in candidate_frame_indices),
                key=lambda fr: sum(face_area(f) for f in fr["faces"])
            )
        else:
            main_frame = frame_data[0]
    else:
        main_frame = frame_data[0]

    main_img = main_frame["image"]
    main_frame_idx = main_frame["index"]
    print(f"👑 主图采用第 {main_frame_idx + 1} 张帧")

    # ✅ 挑选副图：从 all_frames 中按 index 均匀分布挑 8 张，跳过主图帧
    print("🧠 正在挑选差异性最大的副图...", flush=True)



    # 计算 hash
    for f in all_frames:
        f['hash'] = imagehash.average_hash(f['image'])

    # 排除主图帧
    used_indices = {main_frame_idx}
    candidates = [f for f in all_frames if f['index'] != main_frame_idx]

    # 选第一张副图：取时间约 1/8 的位置作为起始点
    first_idx = len(candidates) // 8
    first_candidate = candidates[first_idx]
    selected = [first_candidate]
    selected_hashes = [first_candidate['hash']]
    used_indices.add(first_candidate['index'])

    # 选其余 7 张副图（贪心挑 hash 差异最大）
    for _ in range(7):
        best_score = -1
        best_frame = None
        for f in candidates:
            if f['index'] in used_indices:
                continue
            min_distance = min(f['hash'] - h for h in selected_hashes)
            if min_distance > best_score:
                best_score = min_distance
                best_frame = f
        if best_frame:
            selected.append(best_frame)
            selected_hashes.append(best_frame['hash'])
            used_indices.add(best_frame['index'])

    # 最终按时间排序
    selected_sorted = sorted(selected, key=lambda f: f['time'])
    other_imgs = [f['image'] for f in selected_sorted]

    print("✅ 副图选择完成：已确保内容差异性 + 时间顺序", flush=True)

    return [main_img] + other_imgs



def extract_n_frames(video_path: str, n: int = 9) -> list[Image.Image]:
    
    
    
    """
    从视频等间距抽取 n 张帧图（返回 PIL.Image 列表）
    """
    clip = VideoFileClip(video_path)
    duration = clip.duration
    if duration <= 0:
        raise RuntimeError("❌ 视频时长为 0，无法处理。")

    times = [(i + 1) * duration / (n + 1) for i in range(n)]
    frames = []
    for t in times:
        try:
            frame = clip.get_frame(t)
            frames.append(Image.fromarray(frame).convert("RGB"))
        except Exception as e:
            print(f"⚠️ 抽帧失败 @ {t:.2f}s：{e}")

    if len(frames) < n:
        raise RuntimeError(f"❌ 仅成功抽取 {len(frames)} 张图，少于期望 {n} 张。")

    return frames


def compose_hero_grid(
    images: list[Image.Image],
    output_path: str,
    font_path: str = "fonts/Roboto_Condensed-Regular.ttf",
    watermark_text: str = None
) -> str:
    """
    接收 9 张图（1 主图 + 8 副图），生成主图+辅图九宫格（3×4）
    """

    if len(images) != 9:
        raise ValueError("必须传入 9 张图像：1 主图 + 8 副图")

    # 确保统一格式
    images = [img.convert("RGB") for img in images]

    # 尺寸以副图为基准
    unit_w, unit_h = images[1].size
    hero_w, hero_h = unit_w * 2, unit_h * 2
    grid_w, grid_h = unit_w * 3, unit_h * 4
    grid_img = Image.new("RGBA", (grid_w, grid_h))

    # 主图（左上 2×2）
    grid_img.paste(images[0].resize((hero_w, hero_h)), (0, 0))

    # 副图布局
    grid_positions = [
        (2, 0), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3),
    ]

    for idx, (gx, gy) in enumerate(grid_positions):
        img = images[idx + 1].resize((unit_w, unit_h))
        grid_img.paste(img, (gx * unit_w, gy * unit_h))

    # 添加浮水印
    draw = ImageDraw.Draw(grid_img)
    font_size = int(unit_h * 0.3)
    try:
        font = ImageFont.truetype(font_path, size=font_size)
    except Exception:
        print(f"⚠️ 无法加载字体：{font_path}，使用默认字体")
        font = ImageFont.load_default()

    text = watermark_text or Path(output_path).stem
    try:
        text_width, text_height = font.getsize(text)
    except AttributeError:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

    draw.text(
        (grid_img.width - text_width - 10, grid_img.height - text_height - 10),
        text, font=font, fill=(255, 255, 255, 200)
    )

    # 输出保存
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    grid_img.convert("RGB").save(output_path, quality=90)
    print(f"✅ 已生成九宫格预览图：{output_path}")
    return output_path


# === 示例调用 ===
if __name__ == "__main__":
    video_path = "video5.mp4"
    output_path = "preview_demo.jpg"
    font_path = "fonts/Roboto_Condensed-Regular.ttf"

    frames = smart_extract_hero_frames(video_path)
    compose_hero_grid(frames, output_path, font_path=font_path, watermark_text="DEMO")


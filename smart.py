import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip
import face_recognition
from sklearn.cluster import DBSCAN
from pathlib import Path
import concurrent.futures
import json
from hashlib import md5


def is_black_frame(pil_img, threshold=10, ratio=0.95):
    gray = pil_img.convert('L')
    pixels = np.array(gray)
    dark_pixels = (pixels < threshold).sum()
    return (dark_pixels / pixels.size) > ratio


def smart_sample_timestamps(duration, max_frames=60, min_gap=1.0):
    times = []
    gap = duration / (max_frames + 1)
    for i in range(max_frames):
        t = gap * (i + 1)
        if not times or (t - times[-1]) >= min_gap:
            times.append(t)
    return times


def extract_valid_frames(video_path, max_frames=60):
    clip = VideoFileClip(video_path)
    timestamps = smart_sample_timestamps(clip.duration, max_frames)

    def process_time(t):
        try:
            frame = Image.fromarray(clip.get_frame(t))
            if not is_black_frame(frame):
                return (t, frame)
        except Exception:
            return None

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_time, timestamps))

    return [r for r in results if r is not None]


def get_face_data(frames, max_face_frames=25):
    encodings, locations, indices = [], [], []
    for i, (t, img) in enumerate(frames[:max_face_frames]):
        img_np = np.array(img)[:, :, ::-1]  # RGB to BGR
        locs = face_recognition.face_locations(img_np)
        encs = face_recognition.face_encodings(img_np, locs)
        for loc, enc in zip(locs, encs):
            encodings.append(enc)
            locations.append(loc)
            indices.append(i)
    return encodings, locations, indices


def cluster_faces(encodings):
    if not encodings:
        return []
    clustering = DBSCAN(metric='euclidean', eps=0.5, min_samples=1)
    return clustering.fit_predict(encodings)


def select_main_face_frame(frames, encodings, locations, indices, labels):
    if not labels.any():
        return None
    counts = np.bincount(labels)
    main_label = counts.argmax()
    max_area = -1
    best_idx = None
    for i, (enc, loc, idx, label) in enumerate(zip(encodings, locations, indices, labels)):
        if label == main_label:
            top, right, bottom, left = loc
            area = (right - left) * (bottom - top)
            if area > max_area:
                max_area = area
                best_idx = idx
    return best_idx


def hash_image(pil_img):
    return md5(pil_img.tobytes()).hexdigest()


def make_smart_keyframe_grid(video_path: str, preview_basename: str, max_frames=60):
    valid_frames = extract_valid_frames(video_path, max_frames)
    if len(valid_frames) < 1:
        raise RuntimeError("❌ 没有可用画面")

    encodings, locations, indices = get_face_data(valid_frames)
    labels = cluster_faces(encodings)
    main_idx = select_main_face_frame(valid_frames, encodings, locations, indices, labels)
    if main_idx is None:
        main_idx = 0

    selected_frames = [valid_frames[main_idx]]
    seen_hashes = {hash_image(valid_frames[main_idx][1])}
    rest = [i for i in range(len(valid_frames)) if i != main_idx]
    rest_sorted = sorted(rest, key=lambda i: valid_frames[i][0])

    for i in rest_sorted:
        h = hash_image(valid_frames[i][1])
        if h not in seen_hashes:
            selected_frames.append(valid_frames[i])
            seen_hashes.add(h)
        if len(selected_frames) == 9:
            break

    # 构建九宫格
    cell = 320
    grid_img = Image.new('RGB', (cell * 3, cell * 3), 'black')
    positions = {
        0: (0, 0, 2, 2),
        1: (2, 0, 1, 1),
        2: (2, 1, 1, 1),
        3: (0, 2, 1, 1),
        4: (1, 2, 1, 1),
        5: (2, 2, 1, 1),
        6: (0, 3, 1, 1),
        7: (1, 3, 1, 1),
        8: (2, 3, 1, 1),
    }

    meta = []
    for idx, (t, img) in enumerate(selected_frames):
        pos = positions[idx]
        w, h = pos[2]*cell, pos[3]*cell
        resized = img.resize((w, h))
        x, y = pos[0]*cell, pos[1]*cell
        grid_img.paste(resized, (x, y))

        # 人脸识别
        img_np = np.array(img)[:, :, ::-1]
        face_data = []
        face_locs = face_recognition.face_locations(img_np)
        face_encs = face_recognition.face_encodings(img_np, face_locs)
        for loc, enc in zip(face_locs, face_encs):
            face_data.append({
                "location": loc,
                "embedding": enc.tolist()
            })

        meta.append({
            "index": idx,
            "timestamp": t,
            "faces": face_data
        })

    # 加文字浮水印
    draw = ImageDraw.Draw(grid_img)
    font_path = "fonts/Roboto_Condensed-Regular.ttf"
    font_size = 24
    font = ImageFont.truetype(font_path, size=font_size) if os.path.exists(font_path) else ImageFont.load_default()
    text = Path(preview_basename).name
    if text.startswith("preview_"):
        text = text[len("preview_"):]
    draw.text((grid_img.width - 10 - len(text)*font_size//2, grid_img.height - font_size - 10), text, fill=(255, 255, 255), font=font)

    out_path = f"{preview_basename}.jpg"
    grid_img.save(out_path)
    print(f"✔️ Smart keyframe grid saved: {out_path}")

    json_path = f"{preview_basename}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"✔️ Metadata JSON saved: {json_path}")

    return out_path

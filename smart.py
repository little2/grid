
import os
import sys
import time
import json
import traceback
import numpy as np
from PIL import Image, ImageDraw
from moviepy.editor import VideoFileClip
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from sklearn.cluster import DBSCAN
from multiprocessing import Process, Queue
from pathlib import Path

def extract_frame_at(video_path, timestamp):
    try:
        clip = VideoFileClip(video_path)
        frame = clip.get_frame(timestamp)
        clip.reader.close()
        if clip.audio:
            clip.audio.reader.close_proc()
        return Image.fromarray(frame)
    except Exception as e:
        print(f"⚠️ 抽帧异常 @ {timestamp:.2f}s: {e}", flush=True)
        return None

def detect_faces(image, app):
    img_np = np.array(image)
    faces = app.get(img_np)
    return faces

def extract_valid_frames_worker(video_path, timestamps, queue):
    try:
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))

        results = []
        for t in timestamps:
            img = extract_frame_at(video_path, t)
            if img is None:
                continue
            faces = detect_faces(img, app)
            if faces:
                results.append((t, img, faces))
                print(f"✅ 有效画面 @ {t:.2f}s", flush=True)
            else:
                print(f"⚠️ Frame @ {t:.2f}s failed: 无人脸", flush=True)
        queue.put(results)
    except Exception as e:
        print(f"❌ 抽帧子进程异常: {e}")
        traceback.print_exc()
        queue.put([])

def safe_extract_valid_frames(video_path, max_frames=60, timeout=60):
    clip = VideoFileClip(video_path)
    duration = clip.duration
    timestamps = np.linspace(0, duration, max_frames).tolist()
    clip.reader.close()
    if clip.audio:
        clip.audio.reader.close_proc()

    queue = Queue()
    p = Process(target=extract_valid_frames_worker, args=(video_path, timestamps, queue))
    p.start()
    p.join(timeout)

    if p.is_alive():
        print("❌ 超过整体 60 秒限制，强制退出抽帧任务", flush=True)
        p.terminate()
        p.join()
        return []

    results = queue.get()
    print(f"🎞️ 成功提取 {len(results)} 张有效画面（耗时 {timeout:.1f}s）", flush=True)
    return results

def cluster_faces(faces):
    embeddings = [f.embedding for f in faces]
    if len(embeddings) == 0:
        return [], []
    X = np.stack(embeddings)
    clustering = DBSCAN(eps=12, min_samples=2, metric="cosine").fit(X)
    return clustering.labels_, X

def make_grid(images, positions, grid_size=(3, 3), image_size=(320, 180)):
    grid_width = grid_size[0] * image_size[0]
    grid_height = grid_size[1] * image_size[1]
    grid_img = Image.new("RGB", (grid_width, grid_height), color=(0, 0, 0))

    for idx, img in enumerate(images):
        if idx >= grid_size[0] * grid_size[1]:
            break
        resized = img.resize(image_size)
        x = (idx % grid_size[0]) * image_size[0]
        y = (idx // grid_size[0]) * image_size[1]
        grid_img.paste(resized, (x, y))
    return grid_img

def make_smart_keyframe_grid(video_path, output_dir="output", max_frames=60, detect_faces=True):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    valid_frames = safe_extract_valid_frames(video_path, max_frames=max_frames, timeout=60)

    if not valid_frames:
        print("❌ 无有效帧，终止处理", flush=True)
        return None

    all_faces = []
    for t, img, faces in valid_frames:
        all_faces.extend(faces)

    labels, embeddings = cluster_faces(all_faces)
    if len(set(labels)) <= 1:
        print("⚠️ 无法识别主角，使用默认顺序", flush=True)
        selected = valid_frames[:9]
    else:
        largest_cluster = np.argmax(np.bincount(labels[labels >= 0]))
        main_faces = [f for f, l in zip(all_faces, labels) if l == largest_cluster]
        main_face_embeddings = set(f.embedding.tobytes() for f in main_faces)

        selected = []
        for t, img, faces in valid_frames:
            for f in faces:
                if f.embedding.tobytes() in main_face_embeddings:
                    selected.append((t, img))
                    break
        selected = selected[:9]

    images = [img for _, img in selected]
    grid_img = make_grid(images, positions=None)
    out_path = os.path.join(output_dir, f"preview_{Path(video_path).stem}.jpg")
    grid_img.save(out_path)
    print(f"✅ 九宫格预览图已保存至 {out_path}", flush=True)
    return out_path

if __name__ == "__main__":
    video_path = sys.argv[1] if len(sys.argv) > 1 else "video.mp4"
    print(f"📽️ 开始处理视频: {video_path}", flush=True)

    output_path = make_smart_keyframe_grid(
        video_path,
        output_dir="preview_outputs",
        max_frames=80,
        detect_faces=True
    )

    if output_path:
        print(f"✅ 完成生成关键帧预览图：{output_path}", flush=True)
    else:
        print("❌ 处理失败", flush=True)

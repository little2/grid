#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
album_preview.py

AlbumPreviewGenerator
---------------------
- 扫描文件夹中的图片 / 视频 / 其他文件
- 计算可取上限 = floor(N/2)，并与网格 col*row 取最小（N 为候选项数：图片 + 视频帧）
- 英雄图固定 2x2（左上角），辅图按优先级与差异性筛选
- 视频取前几帧非黑屏截图，右下角打播放符号
- 人脸 + 清晰度（Laplacian 方差）选英雄图；辅图优先 视频帧 > 有脸图片 > 其他图片（同优先级按创建时间）
- 所有图从中心裁方形到 128×128；底部添加 30px 留白并写 **文件类型数量统计** 作为水印文字
- 返回统计信息与布局坐标，并输出最终预览图

依赖：opencv-python、Pillow、numpy
pip install opencv-python pillow numpy
"""

from __future__ import annotations
import os
import math
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()  # 让 Pillow 能直接打开 .heic/.heif
except Exception:
    # 没装也不报错；如果目录含 heic 而未安装，会走 OpenCV 分支并失败 -> 返回 None 被跳过
    pass

class AlbumPreviewGenerator:
    # ===== 默认配置 =====
    IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.heic', '.heif'}
    VID_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.m4v', '.webm', '.ts'}


 


    def __init__(
        self,
        tile_size: int = 128,
        bottom_strip: int = 30,
        nonblack_mean_threshold: float = 8.0,
        video_frame_step: int = 5,
        diversity_ahash_min_dist: int = 10,
        haar_face_path: Optional[str] = None,
        watermark_text: Optional[str] = None,  # 若为 None，则自动写入“各类型文件数量”
        font_path: Optional[str] = None,
        font_size: int = 16,
        recursive: bool = True, max_images: int | None = None, max_videos: int | None = None,
    ):
        self.TILE = tile_size
        self.BOTTOM_STRIP = bottom_strip
        self.NONBLACK_MEAN_THRESHOLD = nonblack_mean_threshold
        self.VIDEO_FRAME_STEP = video_frame_step
        self.DIVERSITY_AHASH_MIN_DIST = diversity_ahash_min_dist
        self.watermark_text = watermark_text
        self.font_path = font_path
        self.font_size = font_size
        self.recursive = recursive
        self.max_images = max_images
        self.max_videos = max_videos
        
        if haar_face_path is None:
            haar_face_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(haar_face_path)

        # 预载字体对象，失败则在绘制时回退到 PIL 默认字体
        self._font_obj = self._find_font(self.font_path, self.font_size)

    # ===== 字体查找（新增） =====
    @staticmethod
    def _find_font(font_path: Optional[str], size: int) -> Optional[ImageFont.FreeTypeFont]:
        # 指定字体优先
        if font_path and os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size=size)
            except Exception:
                pass
        # 常见系统字体与本地字体候选
        candidates = [
            "fonts/msyh.ttc",  # 项目内置字体
            "fonts/Roboto_Condensed-Regular.ttf",  # 项目内置字体
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",  # macOS 常见路径
            "/Library/Fonts/Arial.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]
        for p in candidates:
            if os.path.exists(p):
                try:
                    return ImageFont.truetype(p, size=size)
                except Exception:
                    continue
        return None  # 由 PIL 默认字体兜底

    # ===== 网格推导 =====
    def max_grid_from_limit(self, M: int) -> Tuple[int, int, int]:
        """
        给定最大图格值 M，返回 (N_max, col, row)，其中 col=N_max, row=N_max-2。
        同时可用 S = col*row 得到实际图格数（<= M）。
        """
        if M < 1:
            return (0, 0, 0)
        # 逐步下降以确保 N*(N-2) <= M（避免边界误差）。
        # 初值估算：N ≈ floor((2 + sqrt(4 + 4M)) / 2) = 1 + sqrt(1+M)
        N_max = int(math.floor(1 + math.sqrt(1 + M)))
        while N_max > 0 and N_max * (N_max - 2) > M:
            N_max -= 1
        col, row = N_max, max(0, N_max - 2)
        return N_max, col, row

    def max_grid_from_total_files(self, num_images: int, num_videos: int) -> Tuple[int, int, int, int]:
        """
        最大图格值定义：
        ⌊图片数量/3 + 视频数量⌋
        返回 (M, N_max, col, row)
        """
        # ✅ 新算法
        M = int(num_images / 3 + num_videos)
        if M < 1:
            return (0, 0, 0, 0)

        N_max, col, row = self.max_grid_from_limit(M)
        return M, N_max, col, row



    def _read_image_bgr(self, p: Path) -> Optional[np.ndarray]:
        # 先试 Pillow（支持 HEIC）
        try:
            pil_img = Image.open(str(p))
            pil_img.load()
            # 转成 RGB ndarray，再转 BGR 以匹配 OpenCV 习惯
            arr = np.array(pil_img.convert("RGB"))
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except Exception:
            pass
        # 退回 OpenCV（jpg/png/webp等常见格式OK）
        try:
            data = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
            return data
        except Exception:
            return None


    def _iter_files(self, folder: Path):
        if self.recursive:
            for root, _, files in os.walk(folder):
                for fn in files:
                    # 跳过隐藏/系统文件
                    if fn.startswith('.'):
                        continue
                    yield Path(root) / fn
        else:
            for p in folder.iterdir():
                if p.is_file():
                    yield p


    # ======== Public API ========
    def generate_preview(
        self,
        folder: str | Path,
        out_path: str | Path,
    ) -> Dict:
        """
        扫描 folder，计算应取的英雄/辅图数量与位置，并合成输出预览图。
        返回：
        {
          "counts": {"images": int, "videos": int, "others": int, "total": int},
          "grid": {"col": int, "row": int, "capacity": int, "limit": int, "planned": int},
          "hero": {"count": 0/1, "pos": (0,0) 或 None, "path": str 或 None},
          "aux": [{"pos": (gx,gy), "path": str}, ...],
          "output": {"path": str, "size": (W, H+strip)}
        }
        """
        folder = Path(folder)
        out_path = Path(out_path)

        cands, num_img, num_vid, num_other = self._load_candidates(folder)
        # 网格依据“候选数=图片+视频帧”
        M, N_max, col, row = self.max_grid_from_total_files(num_img , num_vid)
        hero, aux, info = self._select_items(cands, col, row)

        # 渲染 + 动态水印（改为各类型文件数量）
        placements = self._render_preview(hero, aux, col, row, out_path,
                                          counts={
                                              "images": num_img,
                                              "videos": num_vid,
                                              "others": num_other,
                                              "total": num_img + num_vid + num_other,
                                          })

        W, H = col * self.TILE, row * self.TILE
        result = {
            "counts": {
                "images": num_img,
                "videos": num_vid,
                "others": num_other,
                "total": num_img + num_vid + num_other
            },
            "grid": {
                "col": col,
                "row": row,
                "capacity": col * row,
                "limit": info.get("limit", 0),
                "planned": info.get("planned", 0),
            },
            "hero": {
                "count": 1 if hero else 0,
                "pos": (0, 0) if hero else None,  # 固定左上角 2x2
                "path": str(hero.src_path) if hero else None
            },
            "aux": [{"pos": pos, "path": path} for path, pos in placements.get("aux", [])],
            "output": {
                "path": str(out_path),
                "size": (W, H + self.BOTTOM_STRIP)
            }
        }
        return result

    # ======== Internal: 文件与候选构建 ========
    class _Candidate:
        def __init__(self, kind: str, src_path: Path, thumb_bgr: np.ndarray, ctime: float, is_from_video: bool):
            self.kind = kind  # 'image' or 'video'
            self.src_path = src_path
            self.thumb_bgr = thumb_bgr
            self.ctime = ctime
            self.is_from_video = is_from_video
            self.faces: List[Tuple[int,int,int,int]] = []
            self.sharpness: float = 0.0
            self.ahash: Optional[int] = None

    def _is_image(self, p: Path) -> bool:
        return p.suffix.lower() in self.IMG_EXTS

    def _is_video(self, p: Path) -> bool:
        return p.suffix.lower() in self.VID_EXTS

    def _file_ctime(self, p: Path) -> float:
        try:
            return p.stat().st_ctime
        except Exception:
            return time.time()

    def _center_crop_square(self, img: np.ndarray, size: int) -> np.ndarray:
        h, w = img.shape[:2]
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        crop = img[y0:y0+side, x0:x0+side]
        return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)

    def _compute_sharpness(self, img_gray: np.ndarray) -> float:
        return float(cv2.Laplacian(img_gray, cv2.CV_64F).var())

    def _detect_faces(self, img_gray: np.ndarray) -> List[Tuple[int,int,int,int]]:
        faces = self.face_cascade.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=4, minSize=(24,24))
        if faces is None or len(faces) == 0:
            return []
        # ensure Python list of tuples
        return [tuple(map(int, f)) for f in faces]

    def _ahash(self, img_bgr: np.ndarray, hash_size: int = 8) -> int:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
        avg = small.mean()
        bits = (small > avg).astype(np.uint8).flatten()
        val = 0
        for b in bits:
            val = (val << 1) | int(b)
        return val

    def _hamming_distance(self, a: int, b: int) -> int:
        return (a ^ b).bit_count()

    def _draw_play_icon(self, img_bgr: np.ndarray, margin: int = 8, size: int = 22) -> None:
        h, w = img_bgr.shape[:2]
        overlay = img_bgr.copy()
        center = (w - margin - size//2, h - margin - size//2)
        cv2.circle(overlay, center, size//2 + 6, (255,255,255), thickness=-1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, img_bgr, 1-alpha, 0, img_bgr)
        cv2.circle(img_bgr, center, size//2 + 6, (0,0,0), thickness=1)
        pts = np.array([
            [center[0] - size//4, center[1] - size//3],
            [center[0] - size//4, center[1] + size//3],
            [center[0] + size//3, center[1]]
        ], dtype=np.int32)
        cv2.fillPoly(img_bgr, [pts], color=(0,0,0))

    def _first_nonblack_frame(self, path: Path) -> Optional[np.ndarray]:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return None
        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % self.VIDEO_FRAME_STEP == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if gray.mean() >= self.NONBLACK_MEAN_THRESHOLD:
                        return frame
                frame_idx += 1
        finally:
            cap.release()
        return None

    # === 动态水印：文件类型数量统计 ===
    def _compose_watermark_text(self, counts: Dict[str, int]) -> str:
        if self.watermark_text:
            return self.watermark_text  # 用户强制指定
        return f"图片 {counts.get('images',0)} | 视频 {counts.get('videos',0)} | 其他 {counts.get('others',0)} | 总计 {counts.get('total',0)}"

    def _put_watermark_bottom(self, canvas: Image.Image, text: str) -> Image.Image:
        w, h = canvas.size
        new_img = Image.new("RGB", (w, h + self.BOTTOM_STRIP), "white")
        new_img.paste(canvas, (0, 0))
        draw = ImageDraw.Draw(new_img)
        # 字体：优先用预载字体
        font = self._font_obj or ImageFont.load_default()
        x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
        tw, th = x1 - x0, y1 - y0
        draw.text(((w - tw) // 2, h + (self.BOTTOM_STRIP - th) // 2), text, fill=(0, 0, 0), font=font)
        return new_img

    def _analyze_candidate(self, c: "_Candidate") -> None:
        g = cv2.cvtColor(c.thumb_bgr, cv2.COLOR_BGR2GRAY)
        c.sharpness = self._compute_sharpness(g)
        c.faces = self._detect_faces(g)
        c.ahash = self._ahash(c.thumb_bgr)

    def _load_candidates(self, folder: Path) -> Tuple[List["_Candidate"], int, int, int]:
        images, videos, others = [], [], []
        for p in sorted(self._iter_files(folder)):
            if self._is_image(p):
                images.append(p)
            elif self._is_video(p):
                videos.append(p)
            else:
                others.append(p)

        # 可选：限制最大处理量，避免超大目录卡顿
        if self.max_images is not None:
            images = images[: self.max_images]
        if self.max_videos is not None:
            videos = videos[: self.max_videos]

        candidates: List[AlbumPreviewGenerator._Candidate] = []

        # 图片
        for p in images:
            data = self._read_image_bgr(p)
            if data is None:
                continue
            thumb = self._center_crop_square(data, self.TILE)
            c = self._Candidate('image', p, thumb, self._file_ctime(p), is_from_video=False)
            self._analyze_candidate(c)
            candidates.append(c)

        # 视频 → 抓首个非黑屏帧 + 播放符号
        for p in videos:
            frame = self._first_nonblack_frame(p)
            if frame is None:
                continue
            snap = self._center_crop_square(frame, self.TILE)
            self._draw_play_icon(snap)
            c = self._Candidate('video', p, snap, self._file_ctime(p), is_from_video=True)
            self._analyze_candidate(c)
            candidates.append(c)

        return candidates, len(images), len(videos), len(others)
        

    # ======== Internal: 选择逻辑 ========
    def _hero_score(self, c: "_Candidate") -> float:
        if not c.faces:
            return 0.0
        max_area = max([w*h for (_, _, w, h) in c.faces])
        return float(max_area) * (c.sharpness + 1.0)

    def _select_items(
        self,
        cands: List["_Candidate"],
        col: int,
        row: int
    ) -> Tuple[Optional["_Candidate"], List["_Candidate"], Dict[str, int]]:
        N = len(cands)
        S = col * row
        M = N // 2
        K = min(S, M)
        if K <= 0:
            return None, [], dict(total=N, cap=S, limit=M, planned=0)

        # 需要至少 2x2 才能放英雄图
        hero = None
        if K >= 4 and col >= 2 and row >= 2:
            face_cands = [c for c in cands if len(c.faces) > 0]
            hero = max(face_cands, key=self._hero_score) if face_cands else max(cands, key=lambda c: c.sharpness)

        # 辅图优先级：视频 > 有脸图片 > 其他图片；同优先级按创建时间（早到晚）
        def aux_key(c: AlbumPreviewGenerator._Candidate):
            if c.is_from_video:
                prio = 0
            elif len(c.faces) > 0:
                prio = 1
            else:
                prio = 2
            return (prio, c.ctime)

        pool = [c for c in cands if c is not hero]
        pool.sort(key=aux_key)

        need_aux = max(0, K - (4 if hero else 0))
        selected: List[AlbumPreviewGenerator._Candidate] = []
        hashes: List[Optional[int]] = []
        for c in pool:
            if len(selected) >= need_aux:
                break
            if c.ahash is None:
                selected.append(c)
                hashes.append(None)
                continue
            ok = True
            for h in hashes:
                if h is None:
                    continue
                if self._hamming_distance(c.ahash, h) < self.DIVERSITY_AHASH_MIN_DIST:
                    ok = False
                    break
            if ok:
                selected.append(c)
                hashes.append(c.ahash)

        # 不足则放宽填满
        if len(selected) < need_aux:
            for c in pool:
                if len(selected) >= need_aux:
                    break
                if c not in selected:
                    selected.append(c)

        info = dict(
            total=N,
            cap=S,
            limit=M,
            planned=min(K, (4 if hero else 0) + len(selected)),
            hero=1 if hero else 0,
            aux=len(selected),
        )
        return hero, selected, info

    # ======== Internal: 布局与渲染 ========
    def _grid_positions(self, col: int, row: int, has_hero: bool) -> List[Tuple[int,int,int,int]]:
        """
        返回每个放置区域的像素矩形：(x0, y0, x1, y1)。
        第一项为 hero 的 2x2 区（若 has_hero=True），之后是所有单格，从左到右、从上到下。
        """
        TILE = self.TILE

        def cell_rect(r, c):
            x0 = c * TILE
            y0 = r * TILE
            return (x0, y0, x0 + TILE, y0 + TILE)

        rects: List[Tuple[int,int,int,int]] = []
        if has_hero and row >= 2 and col >= 2:
            rects.append((0, 0, 2*TILE, 2*TILE))
        # 单格
        for r in range(row):
            for c in range(col):
                if has_hero and r < 2 and c < 2:
                    continue
                rects.append(cell_rect(r, c))
        return rects

    def _render_preview(
        self,
        hero: Optional["_Candidate"],
        aux: List["_Candidate"],
        col: int,
        row: int,
        out_path: Path,
        counts: Optional[Dict[str, int]] = None,
    ) -> Dict[str, List[Tuple[str, Tuple[int,int]]]]:
        W, H = col * self.TILE, row * self.TILE
        canvas = Image.new("RGB", (W, H), (245, 245, 245))

        rects = self._grid_positions(col, row, has_hero=hero is not None)
        placements: Dict[str, List[Tuple[str, Tuple[int,int]]]] = {"hero": [], "aux": []}

        idx = 0
        # 英雄图（放大到 2x2 尺寸以保证清晰度）
        if hero:
            hx0, hy0, hx1, hy1 = rects[0]
            hero_img = Image.fromarray(cv2.cvtColor(cv2.resize(hero.thumb_bgr, (hx1-hx0, hy1-hy0)), cv2.COLOR_BGR2RGB))
            canvas.paste(hero_img, (hx0, hy0))
            placements["hero"].append((str(hero.src_path), (0, 0)))
            idx = 1

        # 辅图
        for c in aux:
            if idx >= len(rects):
                break
            x0, y0, x1, y1 = rects[idx]
            tile = Image.fromarray(cv2.cvtColor(c.thumb_bgr, cv2.COLOR_BGR2RGB))
            canvas.paste(tile, (x0, y0))
            placements["aux"].append((str(c.src_path), (x0 // self.TILE, y0 // self.TILE)))
            idx += 1

        # 底部留白 + 动态水印（各类型数量）
        text = self._compose_watermark_text(counts or {})
        final_img = self._put_watermark_bottom(canvas, text=text)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        final_img.save(out_path, quality=92)
        return placements



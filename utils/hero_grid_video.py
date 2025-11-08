# utils/hero_grid_video.py
import os
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy import VideoFileClip  # ✅ 2.x 推荐写法
import imagehash
from insightface.app import FaceAnalysis
from datetime import datetime

import sys  # ✅ 新增


class HeroGridVideo:
    """
    生成「主图 2x2 左上 + 其余按行填充」的视频九宫格/多宫格预览图。
    - 自动根据视频时长决定网格规模与采样数量
    - 主图以「人脸面积 × 清晰度(拉普拉斯方差)」评分，附近精修
    - 辅助图用 dhash 做多样性筛选，避免与主图/彼此过近
    - 自动检测统一黑边（letterbox），一致时统一去黑边再排版
    """

    # =========================
    # 初始化 & 基本输出
    # =========================
    def __init__(
        self,
        font_path: Optional[str] = "fonts/Roboto_Condensed-Regular.ttf",
        providers: Optional[list] = None,
        det_size: Tuple[int, int] = (640, 640),
        verbose: bool = True,
    ):
        """
        :param font_path: 水印字体路径
        :param providers: InsightFace 推理后端（默认 CPU）
        :param det_size: InsightFace 检测尺寸
        :param verbose: 控制台阶段提示与进度条
        """
        self.font_path = font_path
        self.providers = providers or ["CPUExecutionProvider"]
        self.det_size = det_size
        self.verbose = verbose

        self._stage("初始化 InsightFace 模型中 …")
        self.app = FaceAnalysis(providers=self.providers)
        # ctx_id=0 表示第一个设备，ONNX Runtime CPU 时无影响；如使用 GPU 可自行调整
        self.app.prepare(ctx_id=0, det_size=self.det_size)
        self._stage("模型初始化完成。")

    # =========================
    # 对外主方法
    # =========================
    def generate(
        self,
        video_path: str,
        preview_basename: str = None,
        sample_count: Optional[int] = None,          # 允许外部覆盖；不传则按时长自动
        num_aux: Optional[int] = None,               # 允许外部覆盖；不传则按时长自动
        manual_times: Optional[List[str | float | int]] = None,
        content_id: Optional[int] = None,            # ✅ 新增参数
    ) -> Dict[str, Any]:
        """
        生成网格图并返回元数据（输出路径、主图时间、辅助帧时间等）
        """
        ts_prefix = ''

        # ✅ 改：输出位置改为视频所在目录下的 preview 子目录
        video_path_obj = Path(video_path)
        video_stem = video_path_obj.stem
        video_dir = video_path_obj.parent
        preview_dir = video_dir / "preview"
        preview_dir.mkdir(parents=True, exist_ok=True)

        # ✅ 拼接输出路径（大拼图）
        out_dir = preview_dir
        out_path = str(out_dir / f"{video_stem}.jpg")

        # ✅ 若传入 content_id，则浮水印用 content_id，否则用视频文件名
        watermark_text = str(content_id) if content_id else video_stem

        # ts_prefix = datetime.now().strftime("%Y%m%d%H%M%S_")
        # out_dir = Path(preview_basename).parent
        # base_name = Path(preview_basename).name
        # # out_dir.mkdir(parents=True, exist_ok=True)
        # # out_path = str(out_dir / f"{ts_prefix}{base_name}.jpg")

        # video_stem = Path(video_path).stem           # 🔹从视频名取主干名
        # out_dir.mkdir(parents=True, exist_ok=True)
        # out_path = str(out_dir / f"{video_stem}.jpg")  # 🔹改成以视频名命名

        with VideoFileClip(video_path, audio=False) as clip:
            duration = max(float(clip.duration or 0.01), 0.01)

            # 自动布局与采样配置
            auto_cfg = self._decide_layout_by_duration(duration)
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
            hero_times = np.linspace(0, duration - (duration / (sample_count + 1)), sample_count, endpoint=True)
            hero_frames: List[Tuple[float, Image.Image]] = []
            for t in hero_times:
                try:
                    hero_frames.append((float(t), Image.fromarray(clip.get_frame(float(t)))))
                except Exception:
                    pass
            if not hero_frames:
                hero_frames = [(0.0, Image.fromarray(clip.get_frame(0.0)))]

            # 选主图（粗选 + 精修）
            hero_img, hero_meta = self._choose_hero_frame(hero_frames)
            refined_img, refined_meta = self._refine_hero_nearby(
                clip,
                hero_time=hero_meta.get("time", 0.0),
                window=3,
                step=0.3,
                min_face_area=1e-4,
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
                        sec = self._parse_time_to_seconds(t)
                        parsed.append(min(max(0.0, float(sec)), duration_eps))
                    except Exception:
                        continue
                parsed = sorted(set(round(x, 3) for x in parsed))
                for t in parsed:
                    img = self._safe_get_frame(clip, t)
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
                auto_frames = self._extract_diverse_frames(
                    clip,
                    num_frames=need_auto,
                    extra=6,
                    exclude_hashes=[hero_hash] + manual_hashes,
                    exclude_thr=6,
                )

            aux_frames = (manual_frames + auto_frames)[:num_aux]

            # === 若黑边一致：统一裁切到固定长宽比（去黑边），再进入排版 ===
            judge_imgs = [hero_img] + [img for _, img in aux_frames[:15]]
            lt_rb_frac = self._auto_detect_uniform_letterbox(
                judge_imgs,
                thr=32,
                ratio=0.95,
                max_frac=0.35,
                tolerance_px=8,
                min_consensus=0.6,
            )

            if lt_rb_frac is not None:
                self._stage("检测到一致黑边，将统一去除并保持固定长宽比。")
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

                hero_img = _apply_frac_crop(hero_img, lt_rb_frac)
                aux_frames = [(t, _apply_frac_crop(img, lt_rb_frac)) for (t, img) in aux_frames]
            else:
                self._stage("未检测到一致黑边（可能是灰边/共识不足/厚度差异大），保持原图比例。")


            # ======== 逐帧单独保存（主图 + 辅助帧） ========
            video_stem = Path(video_path).stem
            safe_stem = self._sanitize_filename(video_stem)
            single_dir = video_dir / f"stills"
            single_dir.mkdir(parents=True, exist_ok=True)

            # 保存主图
            hero_time_sec = float(hero_meta.get("time", 0.0))
            hero_time_str = self._format_zh_time(hero_time_sec)
            hero_text = f"▶️ {hero_time_str}"
            hero_img_wm = self._add_watermark(hero_img.copy(), hero_text)
            hero_out = single_dir / f"{hero_text}.jpg"
            hero_img_wm.save(hero_out, quality=90, optimize=True)

            # 保存辅助帧
            for t, img in aux_frames:
                ts_str = self._format_zh_time(float(t))
                aux_text = f"{safe_stem}_{ts_str}"
                aux_img_wm = self._add_watermark(img.copy(), aux_text)
                aux_out = single_dir / f"{aux_text}.jpg"
                aux_img_wm.save(aux_out, quality=90, optimize=True)

        # ======== 排版输出 ========
        if aux_frames:
            base_w, base_h = aux_frames[0][1].width, aux_frames[0][1].height
        else:
            base_w, base_h = hero_img.size

        cell_w, cell_h = base_w, base_h
        hero_w, hero_h = cell_w * 2, cell_h * 2

        grid_w, grid_h = cell_w * cols, cell_h * rows
        grid = Image.new("RGB", (grid_w, grid_h), (0, 0, 0))

        # 主图填 2×2 区域（letterbox，不裁切）
        hero_canvas = self._fit_into_cell(hero_img, hero_w, hero_h, bg=(0, 0, 0))
        grid.paste(hero_canvas, (0, 0))

        # 余格坐标顺序
        coords = self._grid_coords_with_hero_first(cols, rows)

        # 逐格贴图（每格 letterbox）
        for (gx, gy), (_, img) in zip(coords, aux_frames):
            cell_canvas = self._fit_into_cell(img, cell_w, cell_h, bg=(0, 0, 0))
            grid.paste(cell_canvas, (gx * cell_w, gy * cell_h))

        # 水印文字
        self._stage("拼接网格与水印 …")
        draw = ImageDraw.Draw(grid)
        font = self._load_msyh_font(size=max(14, int(cell_h * 0.20)))
        bbox = draw.textbbox((0, 0), watermark_text, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        margin = 64
        x = grid.width - text_w - margin
        y = grid.height - text_h - margin
        self._draw_text_with_outline(draw, (x, y), watermark_text, font)

        grid.save(out_path, quality=90, optimize=True)
        self._stage(f"已保存：{out_path}")


        return {
            "output_path": out_path,
            "grid_cols": cols, "grid_rows": rows,
            "hero_time": round(hero_meta.get("time", 0.0), 3),
            "hero_score": round(hero_meta.get("score", 0.0), 6),
            "aux_times": [round(t, 3) for t, _ in aux_frames],
            "manual_used_times": [round(t, 3) for t in manual_used_times] if manual_times else [],
            "sample_count_used": sample_count,
        }

    # =========================
    # 内部：布局与抽帧
    # =========================
    @staticmethod
    def _decide_layout_by_duration(duration_sec: float) -> dict:
        """
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

    @staticmethod
    def _grid_coords_with_hero_first(cols: int, rows: int) -> List[Tuple[int, int]]:
        coords = []
        for y in range(rows):
            for x in range(cols):
                if x < 2 and y < 2:  # 跳过主图 2x2 占位
                    continue
                coords.append((x, y))
        return coords

    # =========================
    # 内部：工具函数
    # =========================
    @staticmethod
    def _fit_into_cell(img: Image.Image, cell_w: int, cell_h: int, bg=(0, 0, 0)) -> Image.Image:
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

    @staticmethod
    def _parse_time_to_seconds(t) -> float:
        """
        支持: float/int（秒）、"mm:ss"、"hh:mm:ss"、"ss"（可含小数）
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

    @staticmethod
    def _safe_get_frame(clip: VideoFileClip, t: float) -> Optional[Image.Image]:
        try:
            return Image.fromarray(clip.get_frame(float(t)))
        except Exception:
            return None

    @staticmethod
    def _lap_var(pil_img: Image.Image) -> float:
        arr = np.array(pil_img.convert("L"), dtype=np.float32)
        k = np.array([[0, 1, 0],
                      [1,-4, 1],
                      [0, 1, 0]], dtype=np.float32)
        h, w = arr.shape
        pad = 1
        padded = np.pad(arr, pad, mode="edge")
        out = np.zeros_like(arr)
        for y in range(h):
            for x in range(w):
                region = padded[y:y+3, x:x+3]
                out[y, x] = (region * k).sum()
        return float(out.var())

    def _choose_hero_frame(self, frames: List[Tuple[float, Image.Image]]):
        self._stage(f"开始粗选主图：共 {len(frames)} 帧做人脸检测")
        start_ts = time.time()
        best = None
        for idx, (t, img) in enumerate(frames, start=1):
            arr = np.array(img)
            faces = self.app.get(arr)
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

            self._progress("粗选进度", idx, len(frames), start_ts, every=5)

        self._stage(f"粗选完成：主图时间 {best[4]['time']:.3f}s，得分 {best[4]['score']:.6f}")
        return best[3], best[4]

    def _refine_hero_nearby(
        self,
        clip: VideoFileClip,
        hero_time: float,
        window: float = 1.0,
        step: float = 0.1,
        min_face_area: float = 1e-4,
    ) -> Tuple[Optional[Image.Image], Optional[dict]]:
        """
        在主图附近二次搜索更优主图：
        评分 = 人脸总面积(归一化) * 清晰度(拉普拉斯方差)
        """
        duration = max(clip.duration, 0.01)
        t0 = max(0.0, hero_time - window)
        t1 = min(duration - 1e-3, hero_time + window)

        times = np.arange(t0, t1 + 1e-9, step, dtype=np.float64)
        total = len(times)
        if total <= 0:
            self._stage("精修区间为空，跳过。")
            return None, None

        self._stage(f"开始精修主图：窗口 {t0:.2f}s~{t1:.2f}s，步长 {step}s，共 {total} 帧")
        start_ts = time.time()
        last_tick = start_ts

        best = None
        cnt_scanned = 0
        cnt_has_face = 0
        cnt_filtered_small = 0
        last_best_score = -1.0

        frame0 = Image.fromarray(clip.get_frame(float(times[0])))
        _ = np.array(frame0).shape[:2]  # 预热

        for i, t in enumerate(times, start=1):
            self._progress("精修进度", i, total, start_ts, every=10)
            cnt_scanned += 1

            try:
                img = Image.fromarray(clip.get_frame(float(t)))
            except Exception:
                now = time.time()
                if now - last_tick >= 1.0:
                    self._stage(f"扫描中… {i}/{total} | 有脸 {cnt_has_face} | 过滤(小脸) {cnt_filtered_small} | 当前最佳 {max(0.0,last_best_score):.2f}")
                    last_tick = now
                continue

            arr = np.array(img)
            faces = self.app.get(arr)
            if not faces:
                now = time.time()
                if now - last_tick >= 1.0:
                    self._stage(f"扫描中… {i}/{total} | 有脸 {cnt_has_face} | 过滤(小脸) {cnt_filtered_small} | 当前最佳 {max(0.0,last_best_score):.2f}")
                    last_tick = now
                continue

            H, W = arr.shape[:2]
            area_sum = 0.0
            for f in faces:
                x1, y1, x2, y2 = map(int, f.bbox)
                x1, y1, x2, y2 = max(0,x1), max(0,y1), min(W,x2), min(H,y2)
                a = max(0, x2-x1) * max(0, y2-y1) / (W*H)
                if a >= min_face_area:
                    area_sum += a

            if area_sum > 0:
                cnt_has_face += 1
                sharp = self._lap_var(img)
                score = area_sum * sharp
                candidate = (score, -abs(t-hero_time), img, {"time": float(t), "score_face_area": area_sum, "sharp": sharp, "score": float(score)})

                if best is None or candidate > best:
                    best = candidate
                    last_best_score = best[0]
                    meta = best[3]
                    self._stage(f"↑ 新最佳：t={meta['time']:.3f}s  面积={meta['score_face_area']:.5f}  清晰={meta['sharp']:.2f}  分数={meta['score']:.2f}")
            else:
                cnt_filtered_small += 1

            now = time.time()
            if now - last_tick >= 1.0:
                self._stage(f"扫描中… {i}/{total} | 有脸 {cnt_has_face} | 过滤(小脸) {cnt_filtered_small} | 当前最佳 {max(0.0,last_best_score):.2f}")
                last_tick = now

        if best is None:
            self._stage("精修未找到更优帧，沿用粗选主图")
            return None, None

        meta = best[3]
        self._stage(f"精修完成：新主图时间 {meta['time']:.3f}s，综合分 {meta['score']:.2f} | 总扫 {cnt_scanned}，有脸 {cnt_has_face}，小脸过滤 {cnt_filtered_small}")
        return best[2], meta

    def _extract_diverse_frames2(
        self,
        clip: VideoFileClip,
        num_frames: int,
        extra: int = 6,
        exclude_hashes: Optional[List[imagehash.ImageHash]] = None,
        exclude_thr: int = 6,
    ) -> List[Tuple[float, Image.Image]]:
        """
        从视频中均匀抽样（num_frames + extra），用感知哈希贪心选取多样帧。
        - exclude_hashes / exclude_thr：用于排除与主图（或其它）过相近的帧。
        - 自适应阈值：根据候选帧间哈希距离的分位数调节多样性筛选强度。
        """
        self._stage(f"开始抽辅助帧：候选 {num_frames + extra}，目标 {num_frames}")

        duration = max(float(clip.duration or 0.01), 0.01)
        times = np.linspace(0.0, duration, num_frames + extra, endpoint=False, dtype=np.float64)

        raw_frames: List[Tuple[float, Image.Image]] = []
        start_ts = time.time()

        for k, t in enumerate(times, start=1):
            try:
                img = Image.fromarray(clip.get_frame(float(t)))
                raw_frames.append((float(t), img))
            except Exception:
                pass
            self._progress("多样性筛选", k, len(times), start_ts, every=10)

        if not raw_frames:
            try:
                return [(0.0, Image.fromarray(clip.get_frame(0.0)))]
            except Exception:
                return []

        hashes: List[imagehash.ImageHash] = [imagehash.dhash(img) for _, img in raw_frames]

        dists = []
        for i in range(len(hashes)):
            hi = hashes[i]
            for j in range(i + 1, len(hashes)):
                dists.append(abs(hi - hashes[j]))
        adaptive_thr = max(5, int(np.percentile(dists, 25))) if dists else 6

        selected: List[Tuple[float, Image.Image]] = []
        selected_hashes: List[imagehash.ImageHash] = []

        for (t, img), h in zip(raw_frames, hashes):
            if exclude_hashes and any(abs(h - eh) < exclude_thr for eh in exclude_hashes):
                continue
            if selected_hashes and any(abs(h - sh) < adaptive_thr for sh in selected_hashes):
                continue

            selected.append((t, img))
            selected_hashes.append(h)
            if len(selected) >= num_frames:
                break

        if len(selected) < num_frames:
            for idx, ((t, img), h) in enumerate(zip(raw_frames, hashes)):
                if len(selected) >= num_frames:
                    break
                if (t, img) in selected:
                    continue
                if exclude_hashes and any(abs(h - eh) < exclude_thr for eh in exclude_hashes):
                    continue
                if not selected_hashes or all(abs(h - sh) >= max(4, adaptive_thr - 2) for sh in selected_hashes):
                    selected.append((t, img))
                    selected_hashes.append(h)

        if len(selected) < num_frames:
            for (t, img), h in zip(raw_frames, hashes):
                if len(selected) >= num_frames:
                    break
                if (t, img) in selected:
                    continue
                if exclude_hashes and any(abs(h - eh) < exclude_thr for eh in exclude_hashes):
                    continue
                selected.append((t, img))

        selected.sort(key=lambda x: x[0])
        selected = selected[:num_frames]

        self._stage(f"辅助帧完成：实际选取 {len(selected)} 张")
        return selected


    def _extract_diverse_frames(
        self,
        clip: VideoFileClip,
        num_frames: int,
        extra: int = 6,
        exclude_hashes: Optional[List[imagehash.ImageHash]] = None,
        exclude_thr: int = 6,
    ) -> List[Tuple[float, Image.Image]]:
        """
        从视频中均匀抽样（num_frames + extra），用感知哈希贪心选取多样帧。
        必达目标数：不足时逐步放宽，最后用均匀采样回填，确保返回长度 == num_frames。
        """
        self._stage(f"开始抽辅助帧：目标 {num_frames}，候选 {num_frames + extra}")

        duration = max(float(clip.duration or 0.01), 0.01)
        # 第一轮候选时间（均匀取样，不含末尾端点，减少黑帧/转场干扰）
        times = np.linspace(0.0, duration, num_frames + extra, endpoint=False, dtype=np.float64)

        raw_frames: List[Tuple[float, Image.Image]] = []
        start_ts = time.time()
        for k, t in enumerate(times, start=1):
            try:
                img = Image.fromarray(clip.get_frame(float(t)))
                raw_frames.append((float(t), img))
            except Exception:
                pass
            self._progress("多样性筛选", k, len(times), start_ts, every=10)

        # 若完全抓不到帧，尝试 0 秒兜底
        if not raw_frames:
            try:
                return [(0.0, Image.fromarray(clip.get_frame(0.0)))] * max(1, num_frames)
            except Exception:
                return []

        # 预计算哈希
        hashes: List[imagehash.ImageHash] = [imagehash.dhash(img) for _, img in raw_frames]

        # 计算候选帧之间的距离用于自适应阈值
        dists = []
        for i in range(len(hashes)):
            hi = hashes[i]
            for j in range(i + 1, len(hashes)):
                dists.append(abs(hi - hashes[j]))
        adaptive_thr = max(5, int(np.percentile(dists, 25))) if dists else 6

        # -----------------------
        # Pass 1：严格多样性选取
        # -----------------------
        def greedy_select(thr: int) -> List[Tuple[float, Image.Image]]:
            selected: List[Tuple[float, Image.Image]] = []
            selected_hashes: List[imagehash.ImageHash] = []
            for (t, img), h in zip(raw_frames, hashes):
                # 排除与指定哈希过近的帧（通常是主图等）
                if exclude_hashes and any(abs(h - eh) < exclude_thr for eh in exclude_hashes):
                    continue
                # 与已选过近则跳过
                if selected_hashes and any(abs(h - sh) < thr for sh in selected_hashes):
                    continue
                selected.append((t, img))
                selected_hashes.append(h)
                if len(selected) >= num_frames:
                    break
            return selected

        selected = greedy_select(adaptive_thr)
        if len(selected) < num_frames:
            # -----------------------
            # Pass 2：放宽多样性阈值
            # -----------------------
            loose_thr = max(3, adaptive_thr - 2)
            selected = greedy_select(loose_thr)

        if len(selected) < num_frames:
            # -----------------------
            # Pass 3：无差别补齐（仅避开 exclude_hashes）
            # -----------------------
            picked = set(id(img) for _, img in selected)
            for (t, img), h in zip(raw_frames, hashes):
                if len(selected) >= num_frames:
                    break
                if id(img) in picked:
                    continue
                if exclude_hashes and any(abs(h - eh) < exclude_thr for eh in exclude_hashes):
                    continue
                selected.append((t, img))
                picked.add(id(img))

        if len(selected) < num_frames:
            # -----------------------
            # Pass 4：均匀采样回填（避免已选时间点）
            # -----------------------
            want = num_frames - len(selected)
            # 生成二轮均匀时间（错位一点点，避开与第一轮完全重复）
            times2 = np.linspace(0.0 + duration/(2*(num_frames+extra)), duration, want * 2, endpoint=False, dtype=np.float64)
            already_times = {round(t, 3) for t, _ in selected}
            filled = 0
            for t in times2:
                if filled >= want:
                    break
                t_ = float(np.clip(t, 0.0, max(duration - 1e-3, 0.0)))
                if round(t_, 3) in already_times:
                    continue
                img = self._safe_get_frame(clip, t_)
                if img is None:
                    continue
                h = imagehash.dhash(img)
                if exclude_hashes and any(abs(h - eh) < exclude_thr for eh in exclude_hashes):
                    continue
                selected.append((t_, img))
                already_times.add(round(t_, 3))
                filled += 1

        if len(selected) < num_frames:
            # -----------------------
            # Pass 5：极端兜底（允许重复最后一帧填满）
            # -----------------------
            if selected:
                last_t, last_img = selected[-1]
                while len(selected) < num_frames:
                    selected.append((last_t, last_img))
            else:
                # 理论上不会到这里
                try:
                    img0 = Image.fromarray(clip.get_frame(0.0))
                    selected = [(0.0, img0)] * num_frames
                except Exception:
                    selected = []

        # 规范化输出（按时间排序，截断到目标数）
        selected.sort(key=lambda x: x[0])
        selected = selected[:num_frames]

        self._stage(f"辅助帧完成：实际选取 {len(selected)} / 目标 {num_frames}")
        return selected


    # =========================
    # 内部：黑边检测
    # =========================
    @staticmethod
    def _is_near_black_line(arr_row_or_col: np.ndarray, thr: int = 16, ratio: float = 0.98) -> bool:
        if arr_row_or_col.ndim == 1:
            arr = arr_row_or_col
        else:
            arr = arr_row_or_col
        if arr.ndim == 2 and arr.shape[1] == 3:
            v = arr.max(axis=1)
        else:
            v = arr
        return (v < thr).mean() >= ratio

    def _detect_letterbox_bbox(self, img: Image.Image,
                               thr: int = 16,
                               ratio: float = 0.98,
                               max_frac: float = 0.20) -> Optional[tuple[int, int, int, int]]:
        arr = np.array(img.convert("RGB"))
        H, W = arr.shape[:2]

        top = 0
        while top < H and self._is_near_black_line(arr[top, :, :], thr=thr, ratio=ratio):
            top += 1
        bottom = H - 1
        while bottom > top and self._is_near_black_line(arr[bottom, :, :], thr=thr, ratio=ratio):
            bottom -= 1
        left = 0
        while left < W and self._is_near_black_line(arr[:, left, :], thr=thr, ratio=ratio):
            left += 1
        right = W - 1
        while right > left and self._is_near_black_line(arr[:, right, :], thr=thr, ratio=ratio):
            right -= 1

        crop_w = right - left + 1
        crop_h = bottom - top + 1
        if crop_w <= 0 or crop_h <= 0:
            return None

        if (top / H > max_frac) or ((H - 1 - bottom) / H > max_frac) or \
           (left / W > max_frac) or ((W - 1 - right) / W > max_frac):
            return None

        if top == 0 and left == 0 and bottom == H - 1 and right == W - 1:
            return None

        return (left, top, right, bottom)

    def _auto_detect_uniform_letterbox(self,
                                       frames: List[Image.Image],
                                       thr: int = 16,
                                       ratio: float = 0.98,
                                       max_frac: float = 0.20,
                                       tolerance_px: int = 4,
                                       min_consensus: float = 0.8) -> Optional[tuple[float, float, float, float]]:
        if not frames:
            return None

        sample = frames[:min(20, len(frames))]
        W0, H0 = sample[0].size
        boxes = []
        for img in sample:
            if img.size != (W0, H0):
                tmp = img.resize((W0, H0), Image.LANCZOS)
                box = self._detect_letterbox_bbox(tmp, thr=thr, ratio=ratio, max_frac=max_frac)
            else:
                box = self._detect_letterbox_bbox(img, thr=thr, ratio=ratio, max_frac=max_frac)
            if box:
                boxes.append(box)

        if not boxes:
            return None

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
            return None

        ls, ts, rs, bs = zip(*best_group)
        l_med, t_med, r_med, b_med = int(np.median(ls)), int(np.median(ts)), int(np.median(rs)), int(np.median(bs))
        l_frac = l_med / W0
        t_frac = t_med / H0
        r_frac = r_med / W0
        b_frac = b_med / H0
        print(f"[letterbox] use frac box = {l_frac:.4f},{t_frac:.4f},{r_frac:.4f},{b_frac:.4f}", flush=True)
        return (l_frac, t_frac, r_frac, b_frac)

    # =========================
    # 内部：绘制与字体
    # =========================
    @staticmethod
    def _safe_load_font(font_path: Optional[str], size: int):
        if font_path and os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size=size)
            except Exception:
                pass
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size=size)
        except Exception:
            return ImageFont.load_default()

    @staticmethod
    def _draw_text_with_outline(draw, xy, text, font, fill=(255, 255, 255, 220)):
        x, y = xy
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(-1,1),(1,1)]:
            draw.text((x+dx, y+dy), text, font=font, fill=(0, 0, 0, 200))
        draw.text((x, y), text, font=font, fill=fill)

    
    def _add_watermark(self, img: Image.Image, text: str) -> Image.Image:
        """
        在图像底部中央添加浮水印文字。
        自动根据图片宽度调整字体大小，确保文字完整呈现且不过小。
        """
        draw = ImageDraw.Draw(img)

        # 初始字号（随图像高度定一个上限）
        base_size = max(14, int(img.height * 0.08))
        font = self._load_msyh_font(size=base_size)

        # 计算目标最大宽度（图宽的 90%）
        max_text_width = int(img.width * 0.9)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]

        # 若太宽则逐步减小字号直到合适
        while text_w > max_text_width and base_size > 10:
            base_size -= 2
            font = self._load_msyh_font(size=base_size)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]

        # 若太窄可略微放大一点，提升可读性
        while text_w < max_text_width * 0.6 and base_size < int(img.height * 0.12):
            base_size += 2
            font = self._load_msyh_font(size=base_size)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]

        text_h = bbox[3] - bbox[1]
        x = (img.width - text_w) // 2
        y = img.height - text_h - int(img.height * 0.04)

        self._draw_text_with_outline(draw, (x, y), text, font)
        return img



    # =========================
    # 控制台输出
    # =========================
    def _stage(self, msg: str):
        if self.verbose:
            print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def _progress(self, prefix: str, i: int, total: int, start_ts: float, every: int = 10):
        if not self.verbose:
            return
        if i % every != 0 and i != total:
            return
        elapsed = time.time() - start_ts
        pct = 100.0 * i / total if total else 100.0
        eta = self._fmt_eta(elapsed, i, total)
        print(f"\r{prefix}: {i}/{total} ({pct:5.1f}%)  ETA {eta}", end="", flush=True)

    @staticmethod
    def _fmt_eta(elapsed, done, total):
        if done == 0:
            return "--:--"
        rate = elapsed / done
        remain = rate * (total - done)
        m, s = divmod(int(remain), 60)
        return f"{m:02d}:{s:02d}"

    @staticmethod
    def _format_zh_time(seconds: float) -> str:
        """把秒数格式化为中文：mm分ss秒 或 hh时mm分ss秒。"""
        total = int(round(max(0, seconds)))
        h, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """简单清理为可用文件名（跨平台安全一点）。"""
        keep = "-_.()（）·＋+&@"
        return "".join(ch if ch.isalnum() or ch in keep else "_" for ch in name)


    

    @staticmethod
    def _load_msyh_font(size: int) -> ImageFont.FreeTypeFont:
        """
        强制使用微软雅黑（msyh.ttc），避免中文乱码。
        若路径不存在则退回默认字体。
        """
        import sys
        cand = [
            "fonts/msyh.ttc",
            "fonts/msyh.ttf",
        ]
        # Windows 系统字体路径兜底
        if os.name == "nt":
            win_fonts = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
            cand += [
                os.path.join(win_fonts, "msyh.ttc"),
                os.path.join(win_fonts, "msyh.ttf"),
            ]
        # macOS/Linux 若你有 fonts 目录
        cand += [
            "/usr/share/fonts/truetype/microsoft/msyh.ttc",
            "/usr/share/fonts/truetype/msyh.ttc",
            "/Library/Fonts/msyh.ttc",
        ]

        for fp in cand:
            if os.path.exists(fp):
                try:
                    return ImageFont.truetype(fp, size=size)
                except Exception:
                    continue
        # 兜底
        return ImageFont.load_default()



# =========================
# 示例用法（可直接运行调试）
# =========================
if __name__ == "__main__":
    import traceback

    try:
        hg = HeroGridVideo(
            font_path="fonts/Roboto_Condensed-Regular.ttf",
            providers=["CPUExecutionProvider"],  # 如需 GPU 可按环境改为 ["CUDAExecutionProvider", "CPUExecutionProvider"]
            det_size=(640, 640),
            verbose=True,
        )
        meta = hg.generate(
            video_path="video/s6614244fe4b06d7f37acee3b.mp4",
            preview_basename="previews/370854",
            # manual_times=["01:34", "04:08", "04:37", "05:11", "08:33"],
            # sample_count=180,
            # num_aux=12,
        )
        print("✅ 网格已生成：", meta["output_path"])
        print("   主角帧时间(s)：", meta["hero_time"], " 评分：", meta["hero_score"])
        print("   辅助帧时间(s)：", meta["aux_times"])
    except Exception as e:
        print(f"❌ 错误：{e}")
        traceback.print_exc()

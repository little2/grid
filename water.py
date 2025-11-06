# -*- coding: utf-8 -*-
from moviepy import VideoFileClip, ImageClip, CompositeVideoClip, vfx, AudioArrayClip
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import textwrap
from pathlib import Path
import math


def add_watermark_with_audio_check(
    input_path: str,
    output_path: str,
    text: str = (
        "夜爱君作品由 AI 合成，仅限本课程学习使用。\n"
        "擅自公开传播者将承担全部法律责任并被依法追究。"
    ),
    font_path: str = "fonts/Roboto_Condensed-Regular.ttf",  # 可传中文字体，如 fonts/msyh.ttc
    font_size: int = 30,             # 固定字号
    safe_margin: int = 28,           # 四边安全边距（像素）
    bar_alpha: int = 110,            # 半透明底条不透明度 0-255（0=无底条）
    bar_pad_x: int = 12,             # 底条左右内边距
    bar_pad_y: int = 12,             # 底条上下内边距
    fill_rgba=(255, 255, 255, 215),  # 文本颜色（含透明度）
    stroke_fill=(0, 0, 0, 160),      # 描边颜色（含透明度）
    stroke_width: int = 2,
    position=("center", "bottom"),   # 固定模式下的位置
    y_offset: int = 0,               # 固定模式下的额外纵向偏移（像素）
    fade_duration: float = 0.6,      # 淡入淡出时长（秒），0 关闭
    diagonal_tile: bool = False,     # True = 斜向平铺整屏水印（仅固定模式有效）
    tile_gap: int = 160,             # 平铺间距
    # —— 新增：随机闪现模式 ——
    flash_interval: float | None = None,  # 每隔 N 秒闪现；<=0 或 None 表示关闭
    flash_duration: float = 1.2,          # 每次出现的持续时长
    rand_seed: int | None = None,         # 随机种子（可复现）
    # —— 其他 —— 
    add_silent_if_short: bool = True,
    short_threshold: int = 30,       # <30s 且无音轨 → 自动补静音
    silent_sr: int = 44100,
    silent_channels: int = 1,
    ffmpeg_crf: str = "22",
    ffmpeg_preset: str = "medium",
    threads: int = 4,
):
    """MoviePy v2：加水印；可选随机闪现模式；若视频<short_threshold秒且无音轨→自动加静音音轨。"""

    # ---------- 字体加载 ----------
    def _require_font(size: int):
        p = Path(font_path)
        if not p.exists():
            raise FileNotFoundError(f"Font not found: {font_path}")
        try:
            return ImageFont.truetype(str(p), size=size)
        except Exception:
            return ImageFont.load_default()

    # ---------- 文本渲染（固定字号，不随视频宽度放大） ----------
    def _layout_text_to_rgba(text_, max_width, base_font_size=font_size):
        scale_font = base_font_size
        font_obj = _require_font(scale_font)

        # 依据宽度粗估换行（按字符数）
        probe_chars = int(max(8, max_width / (scale_font * 1.9)))
        wrapped = []
        for para in text_.splitlines():
            if not para.strip():
                wrapped.append("")
                continue
            wrapped.extend(textwrap.wrap(para, width=probe_chars))

        # 量度行宽高
        dummy = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
        draw = ImageDraw.Draw(dummy)
        line_sizes, max_w, line_h = [], 0, 0
        for line in wrapped:
            _, _, w, h = draw.textbbox((0, 0), line, font=font_obj, stroke_width=stroke_width)
            line_sizes.append((w, h))
            max_w = max(max_w, w)
            line_h = max(line_h, h)

        text_w = min(max_width, max_w)
        text_h = line_h * len(wrapped)
        canvas_w = int(text_w + bar_pad_x * 2)
        canvas_h = int(text_h + bar_pad_y * 2)

        # 绘制
        img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        if bar_alpha > 0:
            draw.rectangle([(0, 0), (canvas_w, canvas_h)], fill=(0, 0, 0, bar_alpha))

        y = bar_pad_y
        for (line, (w, h)) in zip(wrapped, line_sizes):
            x = (canvas_w - w) // 2
            draw.text(
                (x, y), line, font=font_obj,
                fill=fill_rgba, stroke_width=stroke_width, stroke_fill=stroke_fill
            )
            y += line_h

        return np.array(img)

    # ---------- 斜向平铺（仅固定模式下使用） ----------
    def _tiled_canvas(base_rgba, W, H, gap=tile_gap):
        tile = Image.fromarray(base_rgba)
        canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        tile = tile.rotate(30, expand=True, resample=Image.BICUBIC)
        tw, th = tile.size
        for y in range(-th, H + th, th + gap):
            for x in range(-tw, W + tw, tw + gap):
                canvas.alpha_composite(tile, (x, y))
        return np.array(canvas)

    # ---------- 位置解析（固定模式用） ----------
    def _resolve_position(pos, vw, vh, wm_w, wm_h, margin, y_offset=0):
        x_spec, y_spec = pos

        def _coord(spec, total, size, m, axis):
            if isinstance(spec, (int, float)):
                return int(spec)
            s = str(spec).lower()
            if axis == "x":
                if s == "left":   return m
                if s == "center": return (total - size) // 2
                if s == "right":  return total - size - m
            else:
                if s == "top":    return m
                if s == "center": return (total - size) // 2
                if s == "bottom": return total - size - m
            return 0

        x = _coord(x_spec, vw, wm_w, margin, "x")
        y = _coord(y_spec, vh, wm_h, margin, "y") + y_offset
        return (x, y)

    # ---------- 随机位置（闪现模式用） ----------
    def _random_xy(vw, vh, wm_w, wm_h, margin, rng: np.random.Generator):
        max_x = max(margin, vw - wm_w - margin)
        max_y = max(margin, vh - wm_h - margin)
        x = int(rng.integers(margin, max_x + 1))
        y = int(rng.integers(margin, max_y + 1))
        return (x, y)

    # ---------- 静音音轨（v2：AudioArrayClip） ----------
    def _silent_audio(duration: float):
        samples = int(round(duration * silent_sr))
        if silent_channels == 1:
            arr = np.zeros((samples, 1), dtype=np.float32)
        else:
            arr = np.zeros((samples, silent_channels), dtype=np.float32)
        return AudioArrayClip(arr, fps=silent_sr)

    # ---------- 主流程（v2 API） ----------
    with VideoFileClip(input_path) as video:
        vw, vh = video.w, video.h
        max_text_w = max(200, vw - safe_margin * 2)

        # 基础水印图
        wm_base = _layout_text_to_rgba(text, max_text_w)
        wm_h, wm_w = wm_base.shape[0], wm_base.shape[1]

        wm_clips: list[ImageClip] = []

        if flash_interval and flash_interval > 0:
            # —— 随机闪现模式 ——
            rng = np.random.default_rng(rand_seed)
            # 生成每次出现的开始时间
            n_flashes = max(1, math.ceil(video.duration / flash_interval))
            starts = [i * flash_interval for i in range(n_flashes)]
            # 避免最后一次超出视频时长
            starts = [t for t in starts if t < video.duration]

            for t0 in starts:
                dur = min(flash_duration, max(0.05, video.duration - t0))  # 至少 0.05s
                # 随机位置（保证在画面内并留出安全边距）
                pos_xy = _random_xy(vw, vh, wm_w, wm_h, safe_margin, rng)

                clip = ImageClip(wm_base, duration=dur).with_position(pos_xy)
                if fade_duration and fade_duration > 0:
                    fd = min(fade_duration, dur / 2.0)  # 避免淡入淡出超过片段时长
                    clip = clip.with_effects([vfx.FadeIn(fd), vfx.FadeOut(fd)])
                clip = clip.with_start(t0)  # 指定出现时间
                wm_clips.append(clip)
        else:
            # —— 固定模式（全程显示 / 可选择斜向平铺）——
            if diagonal_tile:
                wm_rgba = _tiled_canvas(wm_base, vw, vh, gap=tile_gap)
                wm_clip = ImageClip(wm_rgba, duration=video.duration).with_position((0, 0))
            else:
                pos_xy = _resolve_position(position, vw, vh, wm_w, wm_h, safe_margin, y_offset=y_offset)
                wm_clip = ImageClip(wm_base, duration=video.duration).with_position(pos_xy)

            if fade_duration and fade_duration > 0:
                wm_clip = wm_clip.with_effects([vfx.FadeIn(fade_duration), vfx.FadeOut(fade_duration)])
            wm_clips.append(wm_clip)

        # 合成
        final = CompositeVideoClip([video, *wm_clips]).with_duration(video.duration)

        # 短视频且无音轨 → 自动加静音
        if add_silent_if_short:
            try:
                has_audio = (video.audio is not None)
            except Exception:
                has_audio = False
            if video.duration < short_threshold and not has_audio:
                print("⚠️ 视频短且无音轨，自动添加静音音轨")
                final = final.with_audio(_silent_audio(video.duration))

        final.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            threads=threads,
            preset=ffmpeg_preset,
            ffmpeg_params=["-crf", ffmpeg_crf],
        )
        final.close()


# ===== 示例：随机闪现模式 =====
if __name__ == "__main__":
    file_name = "11月6日(1).mp4"
    add_watermark_with_audio_check(
        f"./video/{file_name}",
        f"./watermark/w_{file_name}",
        font_path="fonts/msyh.ttc",
        font_size=36,
        safe_margin=10,
        # 开启随机闪现：每 2.5 秒出现一次，每次 1.2 秒
        flash_interval=0,
        # 如果你想用固定模式，请把 flash_interval 设为 0 或 None
        flash_duration=1.2,
        rand_seed=0,               # 可选：固定随机种子，便于复现
        
        position=("center", "bottom"),
        y_offset=20,                # 仅固定模式有效；随机模式忽略
        fade_duration=0.4,
    )

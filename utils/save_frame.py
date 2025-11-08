import os, sys
from pathlib import Path
from typing import Optional, Tuple
from moviepy import VideoFileClip
from PIL import Image, ImageDraw, ImageFont
import numpy as np


class FirstFrameWatermarker:
    def __init__(
        self,
        font_path: Optional[str] = None,
        font_size: int = 28,
        margin: Tuple[int, int] = (18, 14),
        box_pad: Tuple[int, int] = (12, 8),
        pos: str = "bottom-right",
        jpeg_quality: int = 90,
    ):
        self.font_path = font_path
        self.font_size = font_size
        self.margin = margin
        self.box_pad = box_pad
        self.pos = pos
        self.jpeg_quality = jpeg_quality

    # ====== 原工具函数改成 class 内部静态/类方法（几乎不改动） ======

    @staticmethod
    def _format_mmss(seconds: float) -> str:
        """把秒数格式化成 mm:ss（>=1小时显示 hh:mm:ss）。"""
        total = int(round(seconds))
        h, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
        return f"{m:02d}:{s:02d}" if h == 0 else f"{h:02d}:{m:02d}:{s:02d}"

    @staticmethod
    def _pick_cjk_font(user_font_path: Optional[str], size: int) -> ImageFont.FreeTypeFont:
        """优先使用用户传入；否则按平台/常见路径挑一款含中文字形的字体。找不到再抛异常由上层处理。"""
        cand = []

        # 1) 用户显式传入
        if user_font_path:
            cand.append(user_font_path)

        # 2) 项目内常见 CJK 字体（若有请放到 ./fonts/）
        cand += [
            "fonts/NotoSansCJK-Regular.ttc",
            "fonts/SourceHanSansSC-Regular.otf",
            "fonts/SourceHanSansTC-Regular.otf",
            "fonts/MicrosoftYaHei.ttf",
            "fonts/SimHei.ttf",
            "fonts/SimSun.ttc",
            "fonts/PingFang.ttc",
        ]

        # 3) 系统字体兜底
        if sys.platform == "darwin":  # macOS
            cand += [
                "/System/Library/Fonts/PingFang.ttc",
                "/System/Library/Fonts/STHeiti Medium.ttc",
                "/System/Library/Fonts/Hiragino Sans GB W3.otf",
            ]
        elif os.name == "nt":  # Windows
            fdir = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
            cand += [
                os.path.join(fdir, "msyh.ttc"),     # 微软雅黑
                os.path.join(fdir, "msyh.ttf"),
                os.path.join(fdir, "simhei.ttf"),
                os.path.join(fdir, "simsun.ttc"),
                os.path.join(fdir, "Deng.ttf"),
            ]
        else:  # Linux
            cand += [
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                "/usr/share/fonts/truetype/arphic/ukai.ttc",
                "/usr/share/fonts/truetype/arphic/uming.ttc",
            ]

        for fp in cand:
            try:
                if fp and os.path.exists(fp):
                    return ImageFont.truetype(fp, size)
            except Exception:
                continue

        # 4) 英文字体兜底（可能仍不含CJK）
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except Exception:
            raise

    # ====== 主流程：与原函数几乎一致 ======

    def save(
        self,
        video_path: str,
        epsilon: float = 0.0,              # 首帧黑场可设 0.05~0.2
        doc_name: Optional[str] = None,    # 不传则用文件名（不含扩展名）
        out_name: Optional[str] = None,    # 输出文件名；默认与视频同名 .jpg
    ) -> Path:
        vp = Path(video_path)
        if not vp.exists():
            raise FileNotFoundError(f"Video not found: {vp}")

        # 输出路径
        if out_name:
            out_path = vp.with_name(out_name) if "." in out_name else vp.with_name(out_name).with_suffix(".jpg")
        else:
            out_path = vp.with_suffix(".jpg")

        # 抽帧
        with VideoFileClip(str(vp)) as clip:
            t = min(max(epsilon, 0.0), max(clip.duration - 1e-3, 0.0))
            frame = clip.get_frame(t)  # ndarray, RGB
            ts_text = self._format_mmss(t)

        # 文档名称 + 时间
        doc = doc_name if doc_name else vp.stem
        watermark_text = f"{doc}  |  {ts_text}"

        # 转 PIL + 画布
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img, mode="RGBA")

        # 字体（中文不乱码）
        try:
            font = self._pick_cjk_font(self.font_path, self.font_size)
        except Exception:
            font = ImageFont.load_default()  # 兜底，建议仍放一款 CJK 字体至 ./fonts/

        # 文本尺寸
        l, t0, r, b = draw.textbbox((0, 0), watermark_text, font=font)
        tw, th = r - l, b - t0
        bw, bh = tw + self.box_pad[0] * 2, th + self.box_pad[1] * 2

        # 位置
        W, H = img.size
        mx, my = self.margin
        pos = self.pos
        if pos == "bottom-right":
            box_xy = (W - bw - mx, H - bh - my)
        elif pos == "bottom-left":
            box_xy = (mx, H - bh - my)
        elif pos == "top-left":
            box_xy = (mx, my)
        elif pos == "top-right":
            box_xy = (W - bw - mx, my)
        else:
            box_xy = (W - bw - mx, H - bh - my)

        # 背景 + 文本（带描边）
        box_x, box_y = box_xy
        draw.rectangle([box_x, box_y, box_x + bw, box_y + bh], fill=(0, 0, 0, int(255 * 0.42)))
        tx = box_x + self.box_pad[0]
        ty = box_y + self.box_pad[1]
        draw.text((tx, ty), watermark_text, font=font, fill=(255, 255, 255, 255),
                  stroke_width=2, stroke_fill=(0, 0, 0, 200))

        # 保存
        img.save(out_path, format="JPEG", quality=self.jpeg_quality, optimize=True)
        return out_path

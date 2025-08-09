from moviepy.editor import VideoFileClip
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path


from moviepy.editor import VideoFileClip
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path


def make_hero_grid_3x4(
    video_path: str,
    preview_basename: str,
    font_path: str = "fonts/Roboto_Condensed-Regular.ttf"
) -> str:
    """
    主图+辅图九宫格（3x4）：主图占左上2x2，其余填入剩下8格
    """

    clip = VideoFileClip(video_path)
    duration = clip.duration
    if duration <= 0:
        raise RuntimeError("视频时长为 0，无法处理。")

    # 抽取 9 张图（含主图）
    times = [(i + 1) * duration / 10 for i in range(9)]
    frames = [Image.fromarray(clip.get_frame(t)) for t in times]

    # 以缩图作为基准单位格
    unit_w, unit_h = frames[1].size
    hero_w, hero_h = unit_w * 2, unit_h * 2
    grid_w, grid_h = unit_w * 3, unit_h * 4  # 3列×4行

    grid_img = Image.new("RGBA", (grid_w, grid_h))

    # 主图：占左上2x2，贴在(0,0)
    grid_img.paste(frames[0].resize((hero_w, hero_h)), (0, 0))

    # 定义其余 8 张图的位置（以单位格为基础坐标）
    # 注意：0 已占 (0,0)、(1,0)、(0,1)、(1,1)
    grid_positions = [
        (2, 0),  # index=1
        (2, 1),  # index=2
        (0, 2),  # index=3
        (1, 2),  # index=4
        (2, 2),  # index=5
        (0, 3),  # index=6
        (1, 3),  # index=7
        (2, 3),  # index=8
    ]

    for idx, (gx, gy) in enumerate(grid_positions):
        img = frames[idx + 1].resize((unit_w, unit_h))
        x = gx * unit_w
        y = gy * unit_h
        grid_img.paste(img, (x, y))

    # 加水印文字
    draw = ImageDraw.Draw(grid_img)
    font_size = int(unit_h * 0.3)
    font = ImageFont.truetype(font_path, size=font_size)

    text = Path(preview_basename).name
    if text.startswith("preview_"):
        text = text[len("preview_"):]

    try:
        text_width, text_height = font.getsize(text)
    except AttributeError:
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
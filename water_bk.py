from moviepy import VideoFileClip, ImageClip, CompositeVideoClip
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# 1. 创建文字浮水印图像
def create_watermark_img(text, size=(800, 100), font_size=40):
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    try:
        font_path = "fonts/msyh.ttc"
        font_size = 30
        font = ImageFont.truetype(font_path, size=font_size)
       
    except:
        font = ImageFont.load_default()
    draw.text((10, 10), text, fill=(255, 255, 255, 180), font=font)
    return np.array(img)

# 2. 加载视频
video = VideoFileClip("input.mp4")

# 3. 生成水印图
text = "本作品由 AI 合成，仅限本课程学习使用。\n擅自公开传播者将承担全部法律责任并被依法追究。\n ."
wm_img = create_watermark_img(text, size=(video.w, 100))

# 4. 创建 ImageClip 水印
watermark_clip = ImageClip(wm_img, ismask=False).set_duration(video.duration)
watermark_clip = watermark_clip.set_position(("center", "bottom")).margin(bottom=0)

# 5. 合成与导出
final = CompositeVideoClip([video, watermark_clip])
final.write_videofile("output_watermarked.mp4", codec="libx264", audio_codec="aac")

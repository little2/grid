from utils.hero_grid_video import HeroGridVideo
import os
from pathlib import Path

def process_videos_in_dir(folder_path: str, hg):
    """
    遍历指定目录，当检测到有视频文件时执行：
        meta = hg.generate(video_path="路径/文件名")
    """
    folder = Path(folder_path)
    if not folder.exists():
        print(f"❌ 路径不存在: {folder}")
        return

    # 支持的视频扩展名
    video_exts = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv"}

    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() in video_exts:
            video_path = str(file.resolve())
            print(f"🎬 检测到视频：{video_path}")
            try:
                meta = hg.generate(video_path=video_path)
                print(f"✅ 生成完成：{meta}")
            except Exception as e:
                print(f"⚠️ 处理失败：{file.name}, 错误：{e}")

if __name__ == "__main__":
    # 假设 hg 是你的 HeroGridVideo 或类似对象
    
    hg = HeroGridVideo(font_path="fonts/Roboto_Condensed-Regular.ttf",
                   providers=["CPUExecutionProvider"],  # 或按需改为 GPU
                   det_size=(640, 640),
                   verbose=True)

    process_videos_in_dir("./video/409451", hg)



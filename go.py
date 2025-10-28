from utils.hero_grid_video import HeroGridVideo

hg = HeroGridVideo(font_path="fonts/Roboto_Condensed-Regular.ttf",
                   providers=["CPUExecutionProvider"],  # 或按需改为 GPU
                   det_size=(640, 640),
                   verbose=True)


meta = hg.generate(
    video_path="video/宇宙棒哥哥系列 三个正太弟弟激啪.mp4",
    preview_basename="previews/71121",
    # manual_times=["07:13"],
    # sample_count=180,
    # num_aux=12,
)

print(meta)



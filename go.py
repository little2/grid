from utils.hero_grid_video import HeroGridVideo

hg = HeroGridVideo(font_path="fonts/Roboto_Condensed-Regular.ttf",
                   providers=["CPUExecutionProvider"],  # 或按需改为 GPU
                   det_size=(640, 640),
                   verbose=True)


meta = hg.generate(
    video_path="video/20241215_002503.MP4",
    preview_basename="previews/51027",
    # manual_times=["07:13"],
    # sample_count=180,
    # num_aux=12,
)

print(meta)



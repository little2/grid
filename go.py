from utils.hero_grid_video import HeroGridVideo

hg = HeroGridVideo(font_path="fonts/Roboto_Condensed-Regular.ttf",
                   providers=["CPUExecutionProvider"],  # 或按需改为 GPU
                   det_size=(640, 640),
                   verbose=True)


meta = hg.generate(
    video_path="video/s6614244fe4b06d7f37acee3b.mp4",
    preview_basename="previews/AgADoAQAApidSEU",
    # manual_times=["01:34","04:08","04:37","05:11","08:33"],
    # sample_count=180,
    # num_aux=12,
)

print(meta)



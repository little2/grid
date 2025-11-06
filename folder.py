from utils.hero_grid_folder import AlbumPreviewGenerator
# from utils.archive_extractor import ArchiveExtractor

# extractor = ArchiveExtractor(common_passwords={
#     "empty": "",
#     "p1": "123456",
#     "p2": "password",
#     "tpv": "tpv",
#     "y2024": "2024",
#     "000":"000"
# })

# res = extractor.extract("video/小峰.zip", password="000", prefer_pwd_key="tpv")
# folder_url = res.get('out_dir')

folder_url = "video/s4"

gen = AlbumPreviewGenerator(tile_size=256, bottom_strip=30)
info = gen.generate_preview(folder_url, out_path="out_s4w.jpg")

import os
import asyncio
from dotenv import load_dotenv
from aiogram import Bot
from aiogram.enums import ParseMode
from aiogram.methods import GetUpdates

from aiogram.types import Update, Message, FSInputFile
from aiogram.client.default import DefaultBotProperties
from aiogram.exceptions import TelegramConflictError
from grid_db import MySQLManager
from pathlib import Path
from moviepy import VideoFileClip
import json
from PIL import Image, ImageDraw, ImageFont
import imagehash

import shutil
import subprocess


from telethon import TelegramClient
from telethon.sessions import StringSession
from telethon.errors import FloodWaitError
from telethon.errors import FileMigrateError
from telethon.tl.functions.upload import GetFileRequest
from telethon.tl.types import InputDocumentFileLocation, InputPeerChannel

load_dotenv()

current_job_id = None

config = {}
# 嘗試載入 JSON 並合併參數
try:
    configuration_json = json.loads(os.getenv('CONFIGURATION', '') or '{}')
    if isinstance(configuration_json, dict):
        config.update(configuration_json)  # 將 JSON 鍵值對合併到 config 中
except Exception as e:
    print(f"⚠️ 無法解析 CONFIGURATION：{e}")

BOT_TOKEN =  config.get('bot_token', os.getenv('BOT_TOKEN'))
API_ID = int(config.get('api_id', os.getenv('API_ID', 0)))
API_HASH = config.get('api_hash', os.getenv('API_HASH', ''))
TELEGROUP_THUMB = int(config.get('telegroup_thumb', os.getenv('TELEGROUP_THUMB', 0)))
TELEGROUP_ARCHIVE = int(config.get('telegroup_archive', os.getenv('TELEGROUP_ARCHIVE', 0)))
TELEGROUP_RELY_BOT = int(config.get('telegroup_rely_bot', os.getenv('TELEGROUP_RELY_BOT', 0)))

bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
tele_client = TelegramClient(StringSession(), API_ID, API_HASH)

db = MySQLManager({
    "host": config.get("db_host", os.getenv("MYSQL_DB_HOST", "localhost")),
    "port": int(config.get('db_port', int(os.getenv('MYSQL_DB_PORT', 3306)))),
    "user": config.get('db_user', os.getenv('MYSQL_DB_USER')),
    "password": config.get('db_password', os.getenv('MYSQL_DB_PASSWORD')),
    "db": config.get('db_name', os.getenv('MYSQL_DB_NAME')),
    "autocommit": True
})



DOWNLOAD_DIR = Path("downloads")
DOWNLOAD_DIR.mkdir(exist_ok=True)
shutdown_event = asyncio.Event()
BOT_NAME = None
BOT_ID = None


async def start_telethon():
    if not tele_client.is_connected():
        await tele_client.connect()
    try:
        await tele_client.start(bot_token=BOT_TOKEN)
    except FloodWaitError as e:
        print(f"⚠️ 导入 Bot 授权被限流 {e.seconds}s，跳过",flush=True)
        await asyncio.sleep(min(e.seconds, 60))
    except Exception as e:
        print(f"❌ 导入 Bot 授权失败：{e}",flush=True)

        


async def download_from_file_id3(
    file_id: str,
    save_path: str,
    chat_id: int,
    message_id: int
):
    # Ensure Telethon logged in
    await start_telethon()

    # Fetch message
    msg = await tele_client.get_messages(chat_id, ids=message_id)
    if not msg or not msg.media:
        raise RuntimeError(f"❌ 获取消息失败: {msg} {chat_id}/{message_id}")
    # Resume support
    start = os.path.getsize(save_path) if os.path.exists(save_path) else 0
    total = getattr(msg.media, 'size', None) or getattr(msg.document, 'size', None)
    mode = 'ab' if start else 'wb'
    with open(save_path, mode) as f:
        def prog(cur, tot):
            pct = (start + cur) / total * 100 if total else 0
            print(f"\r📥 下载进度：{start+cur}/{total} bytes ({pct:.1f}%)", end='', flush=True)
        await tele_client.download_file(
            msg,
            file=f,
            offset=start,
            limit=(total - start) if total else None,
            progress_callback=prog
        )
    print(f"\n✔️ 下载完成：{save_path}",flush=True)


async def safe_download(msg, save_path, try_resume: bool = False):
    doc = getattr(msg.media, 'document', None)
    
    if not doc or not getattr(doc, 'file_reference', None):
        print("⚠️ file_reference 缺失或不是文档类型，使用 fallback 方式下载",flush=True)
        await msg.download_media(file=save_path)
        return

    if not try_resume:
        print("⏬ 已强制禁用断点续传，使用 download_media",flush=True)
        await msg.download_media(file=save_path)
        return

    # 尝试 resume 模式
    try:
        await download_with_resume(msg, save_path)
    except FileMigrateError as e:
        print(f"🌐 DC迁移提示: 文件在 DC{e.new_dc}，尝试切换…",flush=True)
        await tele_client._switch_dc(e.new_dc)
        await download_with_resume(msg, save_path)
    except Exception as e:
        print(f"⚠️ resume下载失败，尝试 fallback download_media: {e}",flush=True)
        await msg.download_media(file=save_path)


async def download_from_file_id(file_id, save_path, chat_id, message_id):
    await start_telethon()
    msg = await tele_client.get_messages(chat_id, ids=message_id)
    if not msg or not msg.media:
        raise RuntimeError(f"❌ 获取消息失败: {chat_id}/{message_id}")
    # Delegate to your chunked downloader:
    await safe_download(msg, save_path)
    return True

async def download_with_resume(msg, save_path, chunk_size: int = 128 * 1024):
    """
    用 MTProto 分块下载并支持续传。
    chunk_size 必须满足：
      - 可被 4096 整除
      - 1048576 (1 MiB) 可被 chunk_size 整除
    128 KiB = 131072 bytes 符合要求（1 MiB / 128 KiB = 8）。
    """
    doc = msg.media.document
    total = doc.size

    # 构造文件位置
    location = InputDocumentFileLocation(
        id=doc.id,
        access_hash=doc.access_hash,
        file_reference=doc.file_reference,
        thumb_size=b""      # 原始文件
    )

    # 计算已下载字节
    start = os.path.getsize(save_path) if os.path.exists(save_path) else 0
    mode = 'ab' if start else 'wb'
    print(f"⏯️ 从 {start}/{total} 处续传…", flush=True)

    with open(save_path, mode) as f:
        offset = start
        while offset < total:
            # 始终使用固定 chunk_size
            resp = await tele_client(GetFileRequest(
                location=location,
                offset=offset,
                limit=chunk_size
            ))
            data = resp.bytes
            if not data:
                break  # 没数据就结束
            f.write(data)
            offset += len(data)

            # 打印进度
            pct = offset / total * 100
            print(f"\r📥 {offset}/{total} bytes ({pct:.1f}%)", end="", flush=True)

    print(f"\n✔️ 下载完成: {save_path}", flush=True)



async def make_keyframe_grid(
    video_path: str,
    preview_basename: str,
    rows: int = 3,
    cols: int = 3
) -> str:
    print(f"👉 Generated keyframe grid starting", flush=True)
    # 1. 抽帧并拼成网格
    clip = VideoFileClip(video_path)
    n = rows * cols
    times = [(i + 1) * clip.duration / (n + 1) for i in range(n)]
    imgs = [Image.fromarray(clip.get_frame(t)) for t in times]

    w, h = imgs[0].size
    grid_img = Image.new('RGB', (w * cols, h * rows))
    for idx, img in enumerate(imgs):
        x = (idx % cols) * w
        y = (idx // cols) * h
        grid_img.paste(img, (x, y))

    # 2. 添加文字浮水印
    draw = ImageDraw.Draw(grid_img)
    # 确保 Roboto_Condensed-Regular.ttf 在你的项目 fonts/ 目录下
    font_path = "fonts/Roboto_Condensed-Regular.ttf"
    font_size = int(h * 0.05)
    font = ImageFont.truetype(font_path, size=font_size)
    # text = 移置 preview_basename 中的 temp/preview_ 前缀
    text = Path(preview_basename).name  # 获取文件名
    if text.startswith("preview_"):
        text = text[len("preview_"):]
    
    # 兼容不同 Pillow 版本计算尺寸
    try:
        text_width, text_height = font.getsize(text)
    except AttributeError:
        # Pillow >= 8.0 推荐用 textbbox
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

    # 放在右下角，留 10px 边距
    x = grid_img.width - text_width - 10
    y = grid_img.height - text_height - 10

    # 半透明白字
    draw.text((x, y), text, fill=(255, 255, 255, 128), font=font)

    # 3. 保存并返回路径
    output_path = f"{preview_basename}.jpg"
    grid_img.save(output_path)
    print(f"✔️ Generated keyframe grid with watermark: {output_path}", flush=True)
    return output_path


def fast_zip_with_password(file_paths: list[str], dest_zip: str, password: str):
    """
    使用系统自带的 zip 工具，以“存储”模式（-0）打包不压缩并设置密码。
    - file_paths: 要打包的文件全路径列表
    - dest_zip: 输出的 zip 路径
    - password: zip 密码
    """
    # 1. 如果已存在同名 zip，则先删掉
    try:
        os.remove(dest_zip)
    except FileNotFoundError:
        pass

    # 2. 确认系统里有 zip 命令
    if not shutil.which("zip"):
        raise RuntimeError("未找到系统 zip 命令，请安装 zip 或在 PATH 中可用。")

    # 3. 构造命令：-0 存储模式（不压缩）、-P 明文密码
    cmd = ["zip", "-0", "-P", password, dest_zip] + file_paths
    subprocess.run(cmd, check=True)

async def handle_video(message: Message):
    print("Starting to handle video", flush=True)
    video = message.video
    file_unique_id = video.file_unique_id
    file_id = video.file_id
    await db.init()

    print("- create/update video, file_extension", flush=True)
    await db.execute("""
        INSERT INTO video (file_unique_id, file_size, duration, width, height, mime_type, create_time, update_time)
        VALUES (%s, %s, %s, %s, %s, %s, NOW(), NOW())
        ON DUPLICATE KEY UPDATE 
            file_size=VALUES(file_size),
            duration=VALUES(duration),
            width=VALUES(width),
            height=VALUES(height),
            mime_type=VALUES(mime_type),
            update_time=NOW()
    """, (file_unique_id, video.file_size, video.duration, video.width, video.height, video.mime_type))

    
    await db.execute("""
        INSERT IGNORE INTO file_extension (file_type, file_unique_id, file_id, bot, create_time)
        VALUES ('video', %s, %s, %s, NOW())
    """, (file_unique_id, file_id, BOT_NAME))

    print("- create/update bid_thumbnail", flush=True)
    thumb_row = await db.fetchone("""
        SELECT thumb_file_unique_id FROM bid_thumbnail WHERE file_unique_id=%s
    """, (file_unique_id,))
    if thumb_row and thumb_row[0]:
        thumb_file_unique_id = thumb_row[0]

        

        rows = await db.fetchall("""
            SELECT file_id, bot FROM file_extension WHERE file_unique_id=%s
        """, (thumb_file_unique_id,))
        if rows:
            for file_id_result, bot_name in rows:
                if bot_name == BOT_NAME:
                    print("-- ✅ 縮圖已存在",flush=True)
                    await message.answer_photo(file_id_result, caption="✅ 縮圖已存在")
                    return
                else:
                    print("-- ✅ 縮圖已存在,但是在别的BOT,不用生成",flush=True)
                    # await bypass(file_id_result, bot_name, BOT_NAME)
                    return
        else:
            print("-- No existing thumbnail found, will create a new one")
            pass
            #await db.execute("DELETE FROM bid_thumbnail WHERE thumb_file_unique_id=%s", (thumb_file_unique_id,))
    else:
        print("- Create grid_jobs", flush=True)
        # 在 handle_video 或者你插入 grid_jobs 的地方，把 message.chat.id、message.message_id 也传进去
        await db.execute("""
            INSERT INTO grid_jobs (
                file_id,
                file_unique_id,
                file_type,
                bot_name,
                job_state,
                scheduled_at,
                retry_count,
                source_chat_id,
                source_message_id
            )
            VALUES (%s, %s, 'video', %s, 'pending', NOW(), 0, %s, %s)
            ON DUPLICATE KEY UPDATE
                job_state      = 'pending',
                scheduled_at   = NOW(),
                retry_count    = retry_count + 1,
                source_chat_id = VALUES(source_chat_id),
                source_message_id = VALUES(source_message_id)
        """, (
            file_id,
            file_unique_id,
            BOT_NAME,
            message.chat.id,
            message.message_id
        ))

    await message.answer("🌀 已加入關鍵幀任務排程",reply_to_message_id=message.message_id)


async def handle_document(message: Message):
    """处理收到的 document：入库 document 表和 file_extension 表"""
    doc = message.document
    try:
        # 1. 写入或更新 document 表
        await db.execute("""
            INSERT INTO document (
                file_unique_id,
                file_size,
                file_name,
                mime_type,
                caption,
                create_time
            )
            VALUES (%s, %s, %s, %s, %s, NOW())
            ON DUPLICATE KEY UPDATE
                file_size = VALUES(file_size),
                file_name = VALUES(file_name),
                mime_type = VALUES(mime_type),
                caption = VALUES(caption),
                create_time = NOW()
        """, (
            doc.file_unique_id,
            doc.file_size,
            doc.file_name,
            doc.mime_type,
            message.caption or None
        ))

        # 2. 写入或更新 file_extension 表
        await db.execute("""
            INSERT INTO file_extension (
                file_type,
                file_unique_id,
                file_id,
                bot,
                create_time
            )
            VALUES ('document', %s, %s, %s, NOW())
            ON DUPLICATE KEY UPDATE
                file_id      = VALUES(file_id),
                bot          = VALUES(bot),
                create_time  = NOW()
        """, (
            doc.file_unique_id,
            doc.file_id,
            BOT_NAME
        ))

        await message.reply("✅ 文档已入库")
    except Exception as e:
        print(f"[Error] handle_document: {e}",flush=True)

async def get_last_update_id() -> int:
    await db.init()
    row = await db.fetchone("SELECT message_id FROM scrap_progress WHERE api_id=%s AND chat_id=0", (API_ID,))
    return int(row[0]) if row else 0

async def update_scrap_progress(new_update_id: int):
    await db.execute("""
        INSERT INTO scrap_progress (chat_id, api_id, message_id, update_datetime)
        VALUES (0, %s, %s, NOW())
        ON DUPLICATE KEY UPDATE 
            message_id=VALUES(message_id),
            update_datetime=NOW()
    """, (API_ID, new_update_id))






async def limited_polling():
    last_update_id = await get_last_update_id()
    print(f"📥 Polling from offset={last_update_id + 1}",flush=True)

    while not shutdown_event.is_set():
        try:
            updates: list[Update] = await bot(GetUpdates(
                offset=last_update_id + 1,
                limit=100,
                timeout=5
            ))
        except TelegramConflictError:
            # 一旦捕获 Conflict，就把当前正在处理的 job 标记为失败
            if 'job_id' in locals():  # 确保 job_id 在作用域内
                await db.execute("""
                    UPDATE grid_jobs
                    SET job_state='failed',
                        error_message='Conflict'
                    WHERE id=%s
                """, (current_job_id,))
            # 跳出轮询或做其他善后处理
            print("❌ 轮询被中断，Conflict 错误已写入数据库", flush=True)
            shutdown_event.set()
            break

        if not updates:
            await asyncio.sleep(600)
            continue

        max_update_id = last_update_id
        for update in updates:
            print(f"📬 Received update: {update.update_id}")
            max_update_id = max(max_update_id, update.update_id)
            if update.message and update.message.video:
                try:
                    await handle_video(update.message)
                except Exception as e:
                    print(f"[Error] handle_video: {e}")

            # 改为调用封装好的 handle_document
            elif update.message and update.message.document:
                await handle_document(update.message)

        if max_update_id != last_update_id:
            await update_scrap_progress(max_update_id)
            last_update_id = max_update_id

        await asyncio.sleep(600)

    print("🛑 Polling stopped",flush=True)



async def process_one_grid_job():
    global current_job_id  # 声明这里要用到模块级的全局变量

    job = await db.fetchone("""
        SELECT id, file_id, file_unique_id, source_chat_id, source_message_id
        FROM grid_jobs
        WHERE job_state='pending' AND bot_name=%s
        ORDER BY scheduled_at ASC
        LIMIT 1
    """, (BOT_NAME,))

    if not job:
        print("📭 No Pending Job Found")
        await asyncio.sleep(60)
        shutdown_event.set()
        return

    job_id, file_id, file_unique_id, chat_id, message_id = job
    print(f"🔧 Processing job ID={job_id}",flush=True)
    current_job_id = job_id  # 更新全局变量

    await db.execute("""
        UPDATE grid_jobs
        SET job_state='processing',started_at=NOW() 
        WHERE id=%s
    """, (job_id))

   
    photo_file_id = None
    photo_unique_id = None


    # 1) 准备临时目录
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)


    # 2) 下载视频
    try:
        video_path = str(temp_dir / f"{file_unique_id}.mp4")
        print(f"📥 开始下载视频: {video_path}", flush=True)
        await download_from_file_id(file_id, video_path, chat_id, message_id)
    except Exception as e:
        print(f"❌ 下载视频失败471: {e} {file_unique_id} ({file_id})", flush=True)
        # 抛出错误
        await db.execute("""
            UPDATE grid_jobs
            SET job_state='failed',error_message='下载视频失败'
            WHERE id=%s
        """, (job_id))
        shutdown_event.set()
        return
        
    # 让主循环继续等待下一个任务
    # 这里可以选择等待一段时间再重试
    
    # 3) 生成预览图
    try:
        preview_basename = str(temp_dir / f"preview_{file_unique_id}")
        preview_path = await make_keyframe_grid(video_path, preview_basename)
    except Exception as e:
        await db.execute("""
            UPDATE grid_jobs
            SET job_state='failed',error_message='生成预览图失败'
            WHERE id=%s
        """, (job_id))
        shutdown_event.set()
        return

    # 4) 之后再计算 pHash、上传、更新数据库……
    try:
        phash_str = None
        with Image.open(preview_path) as img:
            phash_str = str(imagehash.phash(img))

        input_file = FSInputFile(preview_path)

        # 9)  备份:
        sent2 = await bot.send_photo(
            chat_id=TELEGROUP_RELY_BOT,
            photo=input_file,
            caption=f"|_forward_|-100{TELEGROUP_THUMB}",
        )
        photo_file_id = sent2.photo[-1].file_id
        photo_unique_id = sent2.photo[-1].file_unique_id
        photo_file_size = sent2.photo[-1].file_size
        photo_width = sent2.photo[-1].width
        photo_height= sent2.photo[-1].height
        print(f"✔️ 透过RELY发送预览图到分镜图群成功: {e}", flush=True)

    except Exception as e:
        print(f"❌ 透过RELY发送预览图到分镜图群失败: {e} {TELEGROUP_RELY_BOT} {TELEGROUP_THUMB}", flush=True)
        
    try:
        sent = await bot.send_photo(
            chat_id=chat_id,
            photo=input_file,
            reply_to_message_id=message_id
        )

        photo_file_id = sent.photo[-1].file_id
        photo_unique_id = sent.photo[-1].file_unique_id
        photo_file_size = sent.photo[-1].file_size
        photo_width = sent.photo[-1].width
        photo_height= sent.photo[-1].height

        print(f"✔️ 回覆预览图成功: {photo_file_id} {photo_unique_id}", flush=True)
    except Exception as e:
        print(f"❌ 回覆预览图失败: {e}", flush=True)
        await db.execute("""
            UPDATE grid_jobs
            SET job_state='failed',error_message='回覆预览图失败'
            WHERE id=%s
        """, (job_id))

    if photo_file_id is None:
        shutdown_event.set()
        
    await db.execute("""
        INSERT INTO photo (
            file_unique_id, file_size, width, height, file_name,
            caption, root_unique_id, create_time, files_drive,
            hash, same_fuid
        )
        VALUES (%s, %s, %s, %s, NULL, NULL, NULL, NOW(), NULL, %s, NULL)
        ON DUPLICATE KEY UPDATE
            file_size=VALUES(file_size),
            width=VALUES(width),
            height=VALUES(height),
            create_time=NOW(),
            hash=VALUES(hash)         
    """, (
        photo_unique_id,
        photo_file_size,
        photo_width,
        photo_height,
        phash_str
    ))




    await db.execute("""
        INSERT INTO file_extension (file_type, file_unique_id, file_id, bot, create_time)
        VALUES ('photo', %s, %s, %s, NOW())
        ON DUPLICATE KEY UPDATE
            file_id=VALUES(file_id),
            bot=VALUES(bot),
            create_time=NOW()
    """, (photo_unique_id, photo_file_id, BOT_NAME))

    await db.execute(
        """
        INSERT INTO bid_thumbnail (
            file_unique_id,
            thumb_file_unique_id,
            bot_name,
            file_id,
            confirm_status,
            uploader_id,
            status,
            t_update
        )
        VALUES (%s, %s, %s, %s, 0, 0, 1, 1)
        ON DUPLICATE KEY UPDATE
            file_id          = VALUES(file_id),
            confirm_status   = VALUES(confirm_status),
            uploader_id      = VALUES(uploader_id),
            status           = VALUES(status),
            t_update         = 1
        """,
        (
            file_unique_id,
            photo_unique_id,
            BOT_NAME,
            photo_file_id,       # 这里加上 photo_file_id
        )
    )

    await db.execute(
        """
        INSERT INTO sora_content (source_id, thumb_file_unique_id, stage)
        VALUES (%s, %s, 'pending')
        ON DUPLICATE KEY UPDATE
            thumb_file_unique_id = VALUES(thumb_file_unique_id),
            stage = 'pending'
        """,
        (
            file_unique_id,
            photo_unique_id
        )
    )

    print(f"✔️ 预览图已入库: {photo_file_id} {photo_unique_id}", flush=True)

    # 6) 更新任务状态
    await db.execute("""
        UPDATE grid_jobs
        SET job_state='done',
            finished_at=NOW(),
            grid_file_id=%s
        WHERE id=%s
    """, (photo_file_id, job_id))


    # 7)  —— 新增：打包 ZIP —— 

    zip_path = str(temp_dir / f"{file_unique_id}.zip")
    # 把下载的视频和生成的预览图，一次性传给 fast_zip_with_password
    await asyncio.to_thread(
        fast_zip_with_password,
        [video_path, preview_path],
        zip_path,
        file_unique_id
    )
    print(f"✔️ Created ZIP archive: {zip_path}")

    # 8)  备份:上传 ZIP 到指定 chat_id（优先环境变量，否则原 chat），并显示上传进度
    await start_telethon()
 


    #如果你仍想先拿到 `Entity` 对象，也可以这样做：```python
    # 1) 构造完整 ID
    CHANNEL_ID = int(f"-100{TELEGROUP_ARCHIVE}")



    try:
        # 2) 获取实体
        chat_entity = await tele_client.get_entity(CHANNEL_ID)
        
        # 3) 发送文件
        await tele_client.send_file(
            chat_entity,
            file=zip_path,
            caption=f"🔒 已打包并加密：{file_unique_id}.zip",
            force_document=True,
            progress_callback=lambda cur, tot: telethon_upload_progress(cur, tot, zip_path)
        )

    except Exception as e:
        await bot.send_document(
            chat_id=TELEGROUP_ARCHIVE,
            document=FSInputFile(zip_path),
            caption=f"🔒 已打包并加密：{file_unique_id}.zip",
            reply_to_message_id=message_id
        )



    print()
    print(f"✅ ZIP 已发送到 chat_id={chat_id}",flush=True)




    print(f"✅ Job ID={job_id} completed",flush=True)
    shutdown_event.set()

        
        
        


# 进度回调
def telethon_upload_progress(current: int, total: int, zip_path: str):
    pct = (current / total * 100) if total else 0
    print(f"\r📤 上传 {zip_path}: {current}/{total} bytes ({pct:.1f}%)", end="", flush=True)

async def shutdown():
    # 1) 关闭 aiogram 内部的 HTTP session
    await bot.session.close()
    # 2) 关闭你的 MySQL 连接池
    await db.close()
    await tele_client.disconnect()

async def main():
    global BOT_NAME, BOT_ID, API_ID
    me = await bot.get_me()
    BOT_NAME = me.username
    BOT_ID = me.id
    print(f"🤖 Logged in as @{BOT_NAME} (BOT_ID={BOT_ID}, API_ID={API_ID})")

    await start_telethon()
    

   
    task2 = asyncio.create_task(limited_polling())

     # 并行启动，两者谁先结束，就取消另一个
    await asyncio.sleep(10)  # 等待 5 秒，确保 Telethon 完全连接
    task1 = asyncio.create_task(process_one_grid_job())

    try:
        done, pending = await asyncio.wait(
            [task1, task2],
            return_when=asyncio.FIRST_COMPLETED
        )
        # 取消还在跑的任务
        for t in pending:
            t.cancel()
    finally:
        # 不管如何，都优雅地关掉 session 和连接池
        await shutdown()


if __name__ == "__main__":
    asyncio.run(main())

import os
import asyncio
import json
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from aiogram import Bot
from aiogram.enums import ParseMode
from aiogram.methods import GetUpdates
from aiogram.types import Update, Message, FSInputFile
from aiogram.client.default import DefaultBotProperties

from telethon import TelegramClient
from telethon.sessions import StringSession
from moviepy.editor import VideoFileClip
from PIL import Image, ImageDraw, ImageFont
import imagehash

from grid_db import MySQLManager

# Load environment variables
load_dotenv()

# Try loading JSON config
config = {}
try:
    cfg_json = json.loads(os.getenv('CONFIGURATION', '') or '{}')
    if isinstance(cfg_json, dict):
        config.update(cfg_json)
except Exception as e:
    print(f"⚠️ 无法解析 CONFIGURATION：{e}", flush=True)

# Bot and Telethon credentials
BOT_TOKEN = config.get('bot_token', os.getenv('BOT_TOKEN'))
API_ID = int(config.get('api_id', os.getenv('API_ID', 0)))
API_HASH = config.get('api_hash', os.getenv('API_HASH', ''))

# Initialize aiogram Bot
bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

# Initialize Telethon for large file downloads
tele_client = TelegramClient(StringSession(), API_ID, API_HASH)

# Database manager
db = MySQLManager({
    "host": config.get("db_host", os.getenv("MYSQL_DB_HOST", "localhost")),
    "port": int(config.get('db_port', os.getenv('MYSQL_DB_PORT', 3306))),
    "user": config.get('db_user', os.getenv('MYSQL_DB_USER')),
    "password": config.get('db_password', os.getenv('MYSQL_DB_PASSWORD')),
    "db": config.get('db_name', os.getenv('MYSQL_DB_NAME')),
    "autocommit": True
})

# Directories and events
DOWNLOAD_DIR = Path("downloads")
DOWNLOAD_DIR.mkdir(exist_ok=True)
shutdown_event = asyncio.Event()
BOT_NAME = None
API_ID_BOT = None

async def start_telethon():
    if not tele_client.is_connected():
        await tele_client.start(bot_token=BOT_TOKEN)

async def download_from_file_id(file_id: str, save_path: str):
    # Use Telethon to download large files
    print(f"👉 Download starting via Telethon", flush=True)
    await start_telethon()
    await tele_client.download_media(file_id, save_path)
    print(f"✔️ Download completed: {save_path}", flush=True)

async def make_keyframe_grid(
    video_path: str,
    preview_basename: str,
    rows: int = 3,
    cols: int = 3
) -> str:
    print(f"👉 Generating keyframe grid", flush=True)
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

    # Add text watermark
    draw = ImageDraw.Draw(grid_img)
    font_path = "fonts/Roboto_Condensed-Regular.ttf"
    font_size = int(h * 0.05)
    try:
        font = ImageFont.truetype(font_path, size=font_size)
    except (IOError, OSError):
        font = ImageFont.load_default()

    text = Path(preview_basename).name.replace("preview_", "")
    try:
        tw, th = font.getsize(text)
    except AttributeError:
        bbox = draw.textbbox((0,0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    x = grid_img.width - tw - 10
    y = grid_img.height - th - 10
    draw.text((x, y), text, fill=(255,255,255,128), font=font)

    output_path = f"{preview_basename}.jpg"
    grid_img.save(output_path)
    print(f"✔️ Generated keyframe grid with watermark: {output_path}", flush=True)
    return output_path

async def handle_video(message: Message):
    video = message.video
    file_unique_id = video.file_unique_id
    file_id = video.file_id
    await db.init()

    # Insert or update video meta
    await db.execute(
        """
        INSERT INTO video (file_unique_id, file_size, duration, width, height, mime_type, create_time, update_time)
        VALUES (%s, %s, %s, %s, %s, %s, NOW(), NOW())
        ON DUPLICATE KEY UPDATE 
            file_size=VALUES(file_size),
            duration=VALUES(duration),
            width=VALUES(width),
            height=VALUES(height),
            mime_type=VALUES(mime_type),
            update_time=NOW()
        """,
        (file_unique_id, video.file_size, video.duration, video.width, video.height, video.mime_type)
    )

    # Insert into file_extension
    await db.execute(
        """
        INSERT IGNORE INTO file_extension (file_type, file_unique_id, file_id, bot, create_time)
        VALUES ('video', %s, %s, %s, NOW())
        """,
        (file_unique_id, file_id, BOT_NAME)
    )

    # Enqueue grid job
    await db.execute(
        """
        INSERT INTO grid_jobs (
            file_id, file_unique_id, file_type, bot_name,
            job_state, scheduled_at, retry_count,
            source_chat_id, source_message_id
        ) VALUES (%s, %s, 'video', %s, 'pending', NOW(), 0, %s, %s)
        ON DUPLICATE KEY UPDATE
            job_state='pending', scheduled_at=NOW(), retry_count=retry_count+1,
            source_chat_id=VALUES(source_chat_id), source_message_id=VALUES(source_message_id)
        """,
        (file_id, file_unique_id, BOT_NAME, message.chat.id, message.message_id)
    )

    await message.answer("🌀 已加入关键帧任务排程")

async def handle_document(message: Message):
    doc = message.document
    await db.init()
    # Insert/update document meta
    await db.execute(
        """
        INSERT INTO document (
            file_unique_id, file_size, file_name, mime_type, caption, create_time)
        VALUES (%s, %s, %s, %s, %s, NOW())
        ON DUPLICATE KEY UPDATE
            file_size=VALUES(file_size), file_name=VALUES(file_name),
            mime_type=VALUES(mime_type), caption=VALUES(caption), create_time=NOW()
        """,
        (doc.file_unique_id, doc.file_size, doc.file_name, doc.mime_type, message.caption or None)
    )
    # Insert/update extension
    await db.execute(
        """
        INSERT INTO file_extension (file_type, file_unique_id, file_id, bot, create_time)
        VALUES ('document', %s, %s, %s, NOW())
        ON DUPLICATE KEY UPDATE file_id=VALUES(file_id), bot=VALUES(bot), create_time=NOW()
        """,
        (doc.file_unique_id, doc.file_id, BOT_NAME)
    )

    await message.reply("✅ 文档已入库")

async def get_last_update_id() -> int:
    await db.init()
    row = await db.fetchone("SELECT message_id FROM scrap_progress WHERE api_id=%s AND chat_id=0", (API_ID_BOT,))
    return int(row[0]) if row else 0

async def update_scrap_progress(new_id: int):
    await db.execute(
        """
        INSERT INTO scrap_progress (chat_id, api_id, message_id, update_datetime)
        VALUES (0, %s, %s, NOW())
        ON DUPLICATE KEY UPDATE message_id=VALUES(message_id), update_datetime=NOW()
        """,
        (API_ID_BOT, new_id)
    )

async def limited_polling():
    last_id = await get_last_update_id()
    print(f"📥 Polling from offset={last_id+1}", flush=True)
    while not shutdown_event.is_set():
        updates = await bot(GetUpdates(offset=last_id+1, limit=100, timeout=5))
        if not updates:
            await asyncio.sleep(1)
            continue
        max_id = last_id
        for upd in updates:
            max_id = max(max_id, upd.update_id)
            if upd.message and upd.message.video:
                await handle_video(upd.message)
            elif upd.message and upd.message.document:
                await handle_document(upd.message)
        if max_id != last_id:
            await update_scrap_progress(max_id)
            last_id = max_id
        await asyncio.sleep(1)
    print("🛑 Polling stopped", flush=True)

async def process_one_grid_job():
    job = await db.fetchone(
        """
        SELECT id, file_id, file_unique_id, source_chat_id, source_message_id
        FROM grid_jobs WHERE job_state='pending'
        ORDER BY scheduled_at ASC LIMIT 1
        """
    )
    if not job:
        print("📭 No pending job found", flush=True)
        await asyncio.sleep(30)
        shutdown_event.set()
        return
    job_id, file_id, file_unique_id, chat_id, msg_id = job
    print(f"🔧 Processing job ID={job_id}", flush=True)
    await db.execute("UPDATE grid_jobs SET job_state='processing', started_at=NOW() WHERE id=%s", (job_id,))

    try:
        video_path = f"temp/{file_unique_id}.mp4"
        preview_base = f"temp/preview_{file_unique_id}"
        os.makedirs("temp", exist_ok=True)

        await download_from_file_id(file_id, video_path)
        preview_path = await make_keyframe_grid(video_path, preview_base)

        # Compute pHash
        with Image.open(preview_path) as img:
            phash_str = str(imagehash.phash(img))

        sent = await bot.send_photo(chat_id=chat_id, photo=FSInputFile(preview_path), reply_to_message_id=msg_id)
        photo_id = sent.photo[-1].file_id
        photo_uid = sent.photo[-1].file_unique_id

        # Update grid_jobs
        await db.execute(
            "UPDATE grid_jobs SET job_state='done', finished_at=NOW(), grid_file_id=%s, phash=%s WHERE id=%s",
            (photo_id, phash_str, job_id)
        )

        # Insert photo record
        await db.execute(
            """
            INSERT INTO photo (file_unique_id, file_size, width, height, create_time, hash)
            VALUES (%s, %s, %s, %s, NOW(), %s)
            ON DUPLICATE KEY UPDATE file_size=VALUES(file_size), width=VALUES(width), height=VALUES(height), create_time=NOW(), hash=VALUES(hash)
            """,
            (photo_uid, sent.photo[-1].file_size, sent.photo[-1].width, sent.photo[-1].height, phash_str)
        )

        # Update file_extension & bid_thumbnail
        await db.execute(
            "INSERT INTO file_extension (file_type, file_unique_id, file_id, bot, create_time) VALUES ('photo', %s, %s, %s, NOW()) ON DUPLICATE KEY UPDATE file_id=VALUES(file_id), bot=VALUES(bot), create_time=NOW()",
            (photo_uid, photo_id, BOT_NAME)
        )
        await db.execute(
            """
            INSERT INTO bid_thumbnail (file_unique_id, thumb_file_unique_id, bot_name, file_id, confirm_status, uploader_id, status, t_update)
            VALUES (%s, %s, %s, %s, 0, 0, 1, 1)
            ON DUPLICATE KEY UPDATE file_id=VALUES(file_id), t_update=1
            """,
            (file_unique_id, photo_uid, BOT_NAME, photo_id)
        )

        print(f"✅ Job ID={job_id} completed (pHash={phash_str})", flush=True)
    except Exception as e:
        print(f"❌ Job ID={job_id} failed: {e}", flush=True)
    finally:
        shutdown_event.set()

async def shutdown():
    await bot.session.close()
    await db.close()

async def main():
    global BOT_NAME, API_ID_BOT
    me = await bot.get_me()
    BOT_NAME = me.username
    API_ID_BOT = me.id
    print(f"🤖 Logged in as @{BOT_NAME} (API_ID={API_ID_BOT})", flush=True)

    # Start Telethon
    await start_telethon()

    # Parallel tasks
    task1 = asyncio.create_task(process_one_grid_job())
    task2 = asyncio.create_task(limited_polling())
    done, pending = await asyncio.wait([task1, task2], return_when=asyncio.FIRST_COMPLETED)
    for t in pending:
        t.cancel()
    await shutdown()

if __name__ == "__main__":
    asyncio.run(main())

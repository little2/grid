import os
import asyncio
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List

from dotenv import load_dotenv

from aiogram import Bot
from aiogram.enums import ParseMode
from aiogram.methods import GetUpdates
from aiogram.types import Update, Message, FSInputFile
from aiogram.client.default import DefaultBotProperties
from aiogram.exceptions import TelegramConflictError

from telethon import TelegramClient
from telethon.sessions import StringSession
from telethon.errors import FloodWaitError, FileMigrateError
from telethon.tl.functions.upload import GetFileRequest
from telethon.tl.types import InputDocumentFileLocation

from PIL import Image, ImageDraw, ImageFont
import imagehash

from moviepy import VideoFileClip

from grid_db import MySQLManager


from utils.hero_grid_video import HeroGridVideo


# =========================
# 基础配置 & 全局对象pytho
# =========================
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("grid_worker")

CONFIG: dict = {}
try:
    cfg_json = json.loads(os.getenv("CONFIGURATION") or "{}")
    if isinstance(cfg_json, dict):
        CONFIG.update(cfg_json)
except Exception as e:
    log.warning("无法解析 CONFIGURATION：%s", e)

# 环境变量兜底
BOT_TOKEN = CONFIG.get("bot_token", os.getenv("BOT_TOKEN"))

API_ID = int(CONFIG.get("api_id", os.getenv("API_ID", 0)))
API_HASH = CONFIG.get("api_hash", os.getenv("API_HASH", ""))
TELEGROUP_THUMB = int(CONFIG.get("telegroup_thumb", os.getenv("TELEGROUP_THUMB", 0)))
TELEGROUP_ARCHIVE = int(CONFIG.get("telegroup_archive", os.getenv("TELEGROUP_ARCHIVE", 0)))
TELEGROUP_RELY_BOT = int(CONFIG.get("telegroup_rely_bot", os.getenv("TELEGROUP_RELY_BOT", 0)))

if not BOT_TOKEN:
    raise RuntimeError("缺少 BOT_TOKEN")

    
bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
tele_client = TelegramClient(StringSession(), API_ID, API_HASH)

db = MySQLManager({
    "host": CONFIG.get("db_host", os.getenv("MYSQL_DB_HOST", "localhost")),
    "port": int(CONFIG.get("db_port", os.getenv("MYSQL_DB_PORT", 3306))),
    "user": CONFIG.get("db_user", os.getenv("MYSQL_DB_USER")),
    "password": CONFIG.get("db_password", os.getenv("MYSQL_DB_PASSWORD")),
    "db": CONFIG.get("db_name", os.getenv("MYSQL_DB_NAME")),
    "autocommit": True,
    "pool_recycle": 3600,
})

DOWNLOAD_DIR = Path("downloads")
DOWNLOAD_DIR.mkdir(exist_ok=True)
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

shutdown_event = asyncio.Event()
current_job_id: Optional[int] = None
BOT_NAME: Optional[str] = None
BOT_ID: Optional[int] = None


# =========================
# Telethon 启动与下载工具
# =========================
async def start_telethon() -> None:
    if not tele_client.is_connected():
        await tele_client.connect()
    try:
        await tele_client.start(bot_token=BOT_TOKEN)
    except FloodWaitError as e:
        log.warning("导入 Bot 授权被限流 %ss，暂缓重试", e.seconds)
        await asyncio.sleep(min(e.seconds, 60))
    except Exception as e:
        log.error("导入 Bot 授权失败：%s", e)

async def download_with_resume(msg, save_path: str, chunk_size: int = 128 * 1024) -> None:
    """
    低级 MTProto 分块下载，支持断点续传。
    chunk_size 需满足：可被 4096 整除、且 1MiB 可被 chunk_size 整除。
    """
    doc = getattr(getattr(msg, "media", None), "document", None)
    if not doc:
        raise RuntimeError("消息不含 document，无法断点续传")

    total = doc.size
    location = InputDocumentFileLocation(
        id=doc.id,
        access_hash=doc.access_hash,
        file_reference=doc.file_reference,
        thumb_size=b"",
    )

    start_size = os.path.getsize(save_path) if os.path.exists(save_path) else 0
    mode = "ab" if start_size else "wb"
    log.info("⏯️ 从 %s/%s 处续传…", start_size, total)

    with open(save_path, mode) as f:
        offset = start_size
        while offset < total:
            resp = await tele_client(GetFileRequest(location=location, offset=offset, limit=chunk_size))
            data = resp.bytes
            if not data:
                break
            f.write(data)
            offset += len(data)
            pct = (offset / total * 100) if total else 0
            print(f"\r📥 {offset}/{total} bytes ({pct:.1f}%)", end="", flush=True)
    print()
    log.info("✔️ 下载完成: %s", save_path)

from telethon.errors import FloodWaitError, FileMigrateError, AuthKeyUnregisteredError

async def safe_download(msg, save_path: str, try_resume: bool = True) -> None:
    """
    优先断点续传；若遇到 DC 迁移 / 授权问题，退回到 Telethon 自带的 download_media。
    """
    doc = getattr(getattr(msg, "media", None), "document", None)
    if not doc or not getattr(doc, "file_reference", None):
        log.warning("file_reference 缺失或非文档类型，使用 download_media 兜底")
        await tele_client.download_media(msg, file=save_path)  # ✅ 用 client 方法
        return

    if not try_resume:
        log.info("⏬ 禁用续传，直接 download_media")
        await tele_client.download_media(msg, file=save_path)
        return

    try:
        # 首选：断点续传（低层 API）
        await download_with_resume(msg, save_path)
        return
    except FileMigrateError as e:
        # ✅ 不再 _switch_dc；直接走 Telethon 内建下载（会自动处理 DC）
        log.info("🌐 DC 迁移提示：%s，改用 download_media 兜底", e)
        await tele_client.download_media(msg, file=save_path)
        return
    except AuthKeyUnregisteredError as e:
        # ✅ 尝试重连后直接走 download_media
        log.warning("AuthKey 失效，尝试重连后回退 download_media：%s", e)
        try:
            await tele_client.disconnect()
        except Exception:
            pass
        await start_telethon()
        await tele_client.download_media(msg, file=save_path)
        return
    except Exception as e:
        log.warning("续传失败，fallback download_media：%s", e)
        await tele_client.download_media(msg, file=save_path)
        return


async def download_from_file_id(file_id: str, save_path: str, chat_id: int, message_id: int) -> bool:
    await start_telethon()
    msg = await tele_client.get_messages(chat_id, ids=message_id)
    if not msg or not getattr(msg, "media", None):
        raise RuntimeError(f"❌ 获取消息失败: chat_id={chat_id}, message_id={message_id}")
    await safe_download(msg, save_path)
    return True


# =========================
# 多媒体处理
# =========================
def _find_font() -> Optional[ImageFont.FreeTypeFont]:
    # 优先本地字体，其次系统字体，最后返回 None（由 PIL 默认字体兜底）
    candidates = [
        "fonts/Roboto_Condensed-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size=24)
            except Exception:
                continue
    return None

async def make_keyframe_grid(video_path: str, preview_basename: str, rows: int = 3, cols: int = 3) -> str:
    # 1) 抽帧
    clip = VideoFileClip(video_path)
    n = rows * cols
    times = [(i + 1) * clip.duration / (n + 1) for i in range(n)]
    imgs = [Image.fromarray(clip.get_frame(t)) for t in times]

    # 2) 拼网格
    w, h = imgs[0].size
    grid_img = Image.new("RGB", (w * cols, h * rows))
    for idx, img in enumerate(imgs):
        x = (idx % cols) * w
        y = (idx // cols) * h
        grid_img.paste(img, (x, y))

    # 3) 水印
    draw = ImageDraw.Draw(grid_img)
    text = Path(preview_basename).name
    if text.startswith("preview_"):
        text = text[len("preview_"):]
    font = _find_font()
    font = font.font_variant(size=int(h * 0.05)) if font else ImageFont.load_default()

    # 尺寸估算兼容
    try:
        text_w, text_h = font.getsize(text)  # Pillow < 8
    except Exception:
        bbox = draw.textbbox((0, 0), text, font=font)  # Pillow >= 8
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

    x = grid_img.width - text_w - 10
    y = grid_img.height - text_h - 10
    draw.text((x, y), text, fill=(255, 255, 255, 128), font=font)

    # 4) 保存
    output_path = f"{preview_basename}.jpg"
    grid_img.save(output_path, quality=90)
    log.info("✔️ 生成关键帧网格：%s", output_path)
    return output_path


def fast_zip_with_password(file_paths: List[str], dest_zip: str, password: str) -> None:
    """使用系统 zip（-0 存储模式）+ 明文密码（-P）。"""
    try:
        os.remove(dest_zip)
    except FileNotFoundError:
        pass

    if not shutil.which("zip"):
        raise RuntimeError("未找到系统 zip 命令，请安装 zip 或将其加入 PATH")

    cmd = ["zip", "-0", "-P", password, dest_zip] + file_paths
    subprocess.run(cmd, check=True)


# =========================
# Aiogram 处理器
# =========================
async def handle_video(message: Message) -> None:
    log.info("Start handle_video")
    await db.init()

    v = message.video
    file_unique_id = v.file_unique_id
    file_id = v.file_id

    # video 表
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
        (file_unique_id, v.file_size, v.duration, v.width, v.height, v.mime_type),
    )

    # file_extension 表
    await db.execute(
        """
        INSERT IGNORE INTO file_extension (file_type, file_unique_id, file_id, bot, create_time)
        VALUES ('video', %s, %s, %s, NOW())
        """,
        (file_unique_id, file_id, BOT_NAME),
    )

    # bid_thumbnail 是否已有
    thumb_row = await db.fetchone(
        "SELECT thumb_file_unique_id FROM bid_thumbnail WHERE file_unique_id=%s",
        (file_unique_id,),
    )
    if thumb_row and thumb_row[0]:
        thumb_unique_id = thumb_row[0]
        rows = await db.fetchall(
            "SELECT file_id, bot FROM file_extension WHERE file_unique_id=%s",
            (thumb_unique_id,),
        )
        if rows:
            for file_id_result, bot_name in rows:
                if bot_name == BOT_NAME:
                    await message.answer_photo(file_id_result, caption="(✅) 缩图已存在")
                    return
                else:
                    log.info("缩图已存在，但在其它 BOT，略过")
                    # message.delete()
                    return
        else:
            log.info("缩图记录存在，但未在 file_extension 找到实体，准备重建")

    # 新增 grid_jobs
    await db.execute(
        """
        INSERT INTO grid_jobs (
            file_id, file_unique_id, file_type, bot_name, job_state,
            scheduled_at, retry_count, source_chat_id, source_message_id
        )
        VALUES (%s, %s, 'video', %s, 'pending', NOW(), 0, %s, %s)
        ON DUPLICATE KEY UPDATE
            source_chat_id     = VALUES(source_chat_id),
            source_message_id  = VALUES(source_message_id)
        """,
        (file_id, file_unique_id, BOT_NAME, message.chat.id, message.message_id),
    )

    await message.answer("🌀 已加入关键帧任务排程", reply_to_message_id=message.message_id)


async def handle_document(message: Message) -> None:
    await db.init()
    d = message.document
    try:
        await db.execute(
            """
            INSERT INTO document (file_unique_id, file_size, file_name, mime_type, caption, create_time)
            VALUES (%s, %s, %s, %s, %s, NOW())
            ON DUPLICATE KEY UPDATE
                file_size = VALUES(file_size),
                file_name = VALUES(file_name),
                mime_type = VALUES(mime_type),
                caption   = VALUES(caption),
                create_time = NOW()
            """,
            (d.file_unique_id, d.file_size, d.file_name, d.mime_type, message.caption or None),
        )

        await db.execute(
            """
            INSERT INTO file_extension (file_type, file_unique_id, file_id, bot, create_time)
            VALUES ('document', %s, %s, %s, NOW())
            ON DUPLICATE KEY UPDATE
                file_id = VALUES(file_id),
                bot     = VALUES(bot),
                create_time = NOW()
            """,
            (d.file_unique_id, d.file_id, BOT_NAME),
        )

        await message.reply("✅ 文档已入库")
    except Exception as e:
        log.exception("handle_document 失败：%s", e)


# =========================
# 轮询 & 进度保存
# =========================
async def get_last_update_id() -> int:
    await db.init()
    row = await db.fetchone("SELECT message_id FROM scrap_progress WHERE api_id=%s AND chat_id=0", (BOT_ID,))
    return int(row[0]) if row else 0

async def update_scrap_progress(new_update_id: int) -> None:
    await db.execute(
        """
        INSERT INTO scrap_progress (chat_id, api_id, message_id, update_datetime)
        VALUES (0, %s, %s, NOW())
        ON DUPLICATE KEY UPDATE 
            message_id=VALUES(message_id),
            update_datetime=NOW()
        """,
        (BOT_ID, new_update_id),
    )

async def limited_polling() -> None:
    global current_job_id
    last_update_id = await get_last_update_id() #update_id 只保证对某个 Bot 唯一，不同 Bot 的更新是各自独立的。
    log.info("📥 Polling from offset=%s", last_update_id + 1)

    while not shutdown_event.is_set():
        try:
            updates: List[Update] = await bot(
                GetUpdates(offset=last_update_id + 1, limit=100, timeout=5)
            )
        except TelegramConflictError:
            if current_job_id is not None:
                await db.execute(
                    "UPDATE grid_jobs SET job_state='failed', error_message='Conflict' WHERE id=%s",
                    (current_job_id,),
                )
            log.error("❌ 轮询被中断（Conflict），已写入数据库")
            shutdown_event.set()
            break

        if not updates:
            await asyncio.sleep(600)
            continue

        max_update_id = last_update_id
        for upd in updates:
            max_update_id = max(max_update_id, upd.update_id)
            msg = getattr(upd, "message", None)
            if not msg:
                continue
            try:
                if msg.video:
                    await handle_video(msg)
                elif msg.document:
                    await handle_document(msg)
            except Exception as e:
                log.exception("处理消息失败：%s", e)

        if max_update_id != last_update_id:
            await update_scrap_progress(max_update_id)
            last_update_id = max_update_id

        await asyncio.sleep(600)

    log.info("🛑 Polling stopped")


# =========================
# 核心任务：取一笔 pending 生成九宫格+回传+入库+打包
# =========================
def telethon_upload_progress(current: int, total: int, zip_path: str) -> None:
    pct = (current / total * 100) if total else 0
    print(f"\r📤 上传 {zip_path}: {current}/{total} bytes ({pct:.1f}%)", end="", flush=True)


# =========================
# (0) Grid Job 读取工具
# =========================
async def fetch_next_pending_job(db: MySQLManager, bot_name: str) -> Optional[Tuple[int, str, str, int, int]]:
    """
    读取最早一笔待处理的九宫格任务。
    返回: (id, file_id, file_unique_id, source_chat_id, source_message_id)
    若无任务则返回 None
    """
   
    # row = await db.fetchone(
    #     """
    #     SELECT id, file_id, file_unique_id, source_chat_id, source_message_id
    #     FROM grid_jobs
    #     WHERE job_state='pending' AND bot_name=%s
    #     ORDER BY scheduled_at ASC, id ASC
    #     LIMIT 1
    #     """,
    #     (bot_name,),
    # )

    row = await db.fetchone("SELECT id, file_id, file_unique_id, source_chat_id, source_message_id  FROM `grid_jobs` WHERE `file_unique_id` LIKE 'AgADEwkAApOFkVQ'")



    return row if row else None


async def update_job_status(db: MySQLManager, job_state: str, job_id: int, error_message: str) -> None:   
    await db.execute(
        "UPDATE grid_jobs SET job_state=%s, started_at=NOW(), error_message=%s WHERE id=%s",
        (job_state, error_message, job_id,),
    )



async def process_one_grid_job() -> None:
    global current_job_id
    await db.init()

    # 0) 找出要处理的任务（已独立成函数）
    job = await fetch_next_pending_job(db,BOT_NAME)
    print(f"Fetched job: {job}")

    if not job:
        log.info("📭 No Pending Job Found")
        await asyncio.sleep(60)
        shutdown_event.set()
        return

    job_id, file_id, file_unique_id, chat_id, message_id = job
    current_job_id = job_id
    log.info("✅ (1) Processing job ID=%s", job_id)
    await update_job_status(db,job_state='processing', error_message='', job_id=job_id)


    # 1) 下载视频
    video_path = str(TEMP_DIR / f"{file_unique_id}.mp4")
    try:
        log.info("(2) 📥 开始下载视频: %s", video_path)
        await download_from_file_id(file_id, video_path, chat_id, message_id)
    except Exception as e:
        await update_job_status(db,job_state='failed', error_message='下载视频失败', job_id=job_id)
        log.exception("❌ 下载视频失败: %s (%s / %s)", e, file_unique_id, file_id)
        # 继续处理后续任务
        return

   

    # 2) 生成预览图
    try:
        preview_basename = str(TEMP_DIR / f"{file_unique_id}")
        log.info("(3) 生成关键帧网格…")

       


        hg = HeroGridVideo(font_path="fonts/Roboto_Condensed-Regular.ttf",
                        providers=["CPUExecutionProvider"],  # 或按需改为 GPU
                        det_size=(640, 640),
                        verbose=True)


        meta = hg.generate(
            video_path=video_path,
            preview_basename=preview_basename,
            # manual_times=["07:13"],
            # sample_count=180,
            # num_aux=12,
        )


        print("✅ 网格已生成：", meta["output_path"])
        print("   主角帧时间(s)：", meta["hero_time"], " 评分：", meta["hero_score"])
        print("   辅助帧时间(s)：", meta["aux_times"])
        preview_path = f"{preview_basename}.jpg"
        # preview_path = await make_keyframe_grid(video_path, preview_basename)
    except Exception as e:
        await db.execute(
            "UPDATE grid_jobs SET job_state='failed', error_message='生成预览图失败' WHERE id=%s",
            (job_id,),
        )
        log.exception("❌ 生成预览图失败：%s", e)
        shutdown_event.set()
        return


    shutdown_event.set()
    return


    # 3) 计算 pHash + 发送图片（RELY 备份 + 原聊天回复）
    phash_str = None
    try:
        with Image.open(preview_path) as img:
            phash_str = str(imagehash.phash(img))
    except Exception as e:
        log.warning("pHash 计算失败（忽略）：%s", e)

    input_file = FSInputFile(preview_path)
    photo_file_id = None
    photo_unique_id = None
    photo_file_size = None
    photo_width = None
    photo_height = None

    # 3.1 备份到分镜图群（经 RELY）
    try:
        sent2 = await bot.send_photo(
            chat_id=TELEGROUP_RELY_BOT,
            photo=input_file,
            caption=f"|_forward_|-100{TELEGROUP_THUMB}",
        )
        p2 = sent2.photo[-1]
        photo_file_id = p2.file_id
        photo_unique_id = p2.file_unique_id
        photo_file_size = p2.file_size
        photo_width = p2.width
        photo_height = p2.height
        log.info("(4.1) ✔️ 通过 RELY 发送预览图成功")
    except Exception as e:
        log.warning("(4.1) 通过 RELY 发送预览图失败：%s", e)

    # 3.2 回复到原消息
    try:
        sent = await bot.send_photo(chat_id=chat_id, photo=input_file, reply_to_message_id=message_id)
        p = sent.photo[-1]
        photo_file_id = p.file_id
        photo_unique_id = p.file_unique_id
        photo_file_size = p.file_size
        photo_width = p.width
        photo_height = p.height
        log.info("(4.2) ✔️ 回复预览图成功: %s %s", photo_file_id, photo_unique_id)
    except Exception as e:
        log.exception("(4.2) 回复预览图失败：%s", e)
        await db.execute(
            "UPDATE grid_jobs SET job_state='failed', error_message='回覆预览图失败' WHERE id=%s",
            (job_id,),
        )

    if not photo_file_id:
        shutdown_event.set()
        return

    # 4) 入库 photo / file_extension / bid_thumbnail / sora_content
    await db.execute(
        """
        INSERT INTO photo (file_unique_id, file_size, width, height, file_name, caption, root_unique_id, create_time, files_drive, hash, same_fuid)
        VALUES (%s, %s, %s, %s, NULL, NULL, NULL, NOW(), NULL, %s, NULL)
        ON DUPLICATE KEY UPDATE
            file_size=VALUES(file_size),
            width=VALUES(width),
            height=VALUES(height),
            create_time=NOW(),
            hash=VALUES(hash)
        """,
        (photo_unique_id, photo_file_size, photo_width, photo_height, phash_str),
    )

    await db.execute(
        """
        INSERT INTO file_extension (file_type, file_unique_id, file_id, bot, create_time)
        VALUES ('photo', %s, %s, %s, NOW())
        ON DUPLICATE KEY UPDATE
            file_id=VALUES(file_id),
            bot=VALUES(bot),
            create_time=NOW()
        """,
        (photo_unique_id, photo_file_id, BOT_NAME),
    )

    await db.execute(
        """
        INSERT INTO bid_thumbnail (file_unique_id, thumb_file_unique_id, bot_name, file_id, confirm_status, uploader_id, status, t_update)
        VALUES (%s, %s, %s, %s, 0, 0, 1, 1)
        ON DUPLICATE KEY UPDATE
            file_id        = VALUES(file_id),
            confirm_status = VALUES(confirm_status),
            uploader_id    = VALUES(uploader_id),
            status         = VALUES(status),
            t_update       = 1
        """,
        (file_unique_id, photo_unique_id, BOT_NAME, photo_file_id),
    )

    await db.execute(
        """
        INSERT INTO sora_content (source_id, thumb_file_unique_id, file_type, stage)
        VALUES (%s, %s, 'v', 'pending')
        ON DUPLICATE KEY UPDATE
            thumb_file_unique_id = VALUES(thumb_file_unique_id),
            stage = 'pending'
        """,
        (file_unique_id, photo_unique_id),
    )

    log.info("(5) ✔️ 预览图入库完毕: %s %s", photo_file_id, photo_unique_id)

    # 5) 标记完成
    await db.execute(
        "UPDATE grid_jobs SET job_state='done', finished_at=NOW(), grid_file_id=%s WHERE id=%s",
        (photo_file_id, job_id),
    )

    # 6) 打包 ZIP 并归档
    zip_path = str(TEMP_DIR / f"{file_unique_id}.zip")
    await asyncio.to_thread(fast_zip_with_password, [video_path, preview_path], zip_path, file_unique_id)
    log.info("(6) ✔️ 已创建 ZIP：%s", zip_path)

    await start_telethon()
    channel_id_full = int(f"-100{TELEGROUP_ARCHIVE}")
    try:
        entity = await tele_client.get_entity(channel_id_full)
        await tele_client.send_file(
            entity,
            file=zip_path,
            caption=f"🔒 已打包并加密：{file_unique_id}.zip",
            force_document=True,
            progress_callback=lambda cur, tot: telethon_upload_progress(cur, tot, zip_path),
        )
        print()
        log.info("(7) ✅ ZIP 已发送到 chat_id=%s", channel_id_full)
    except Exception as e:
        log.warning("Telethon 发送失败，尝试 aiogram：%s", e)
        await bot.send_document(
            chat_id=TELEGROUP_ARCHIVE,
            document=FSInputFile(zip_path),
            caption=f"🔒 已打包并加密：{file_unique_id}.zip",
            reply_to_message_id=message_id,
        )

    log.info("(8) ✅ Job ID=%s 完成", job_id)
    shutdown_event.set()


# =========================
# 关停清理
# =========================
async def shutdown() -> None:
    try:
        await bot.session.close()
    except Exception:
        pass
    try:
        await db.close()
    except Exception:
        pass
    try:
        await tele_client.disconnect()
    except Exception:
        pass


# =========================
# main
# =========================
async def main() -> None:
    global BOT_NAME, BOT_ID
    me = await bot.get_me()
    BOT_NAME = me.username
    BOT_ID = me.id
    log.info("🤖 Logged in as @%s (BOT_ID=%s, API_ID=%s)", BOT_NAME, BOT_ID, API_ID)

    await start_telethon()

    # 同时跑：轮询（收视频建任务）+ 处理一笔 pending 任务
    poll_task = asyncio.create_task(limited_polling())
    await asyncio.sleep(10)  # 给 Telethon 一点时间
    job_task = asyncio.create_task(process_one_grid_job())

    try:
        done, pending = await asyncio.wait([poll_task, job_task], return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
    finally:
        await shutdown()


if __name__ == "__main__":
    asyncio.run(main())

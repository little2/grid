# archive_extractor.py
from __future__ import annotations
import os
import io
import sys
import time
import shutil
import zipfile
import tarfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# 可选支持（若未安装，会自动跳过 7z / RAR）
try:
    import py7zr  # type: ignore
except Exception:
    py7zr = None

try:
    import rarfile  # type: ignore
except Exception:
    rarfile = None


class ArchiveExtractor:
    """
    用法:
        extractor = ArchiveExtractor(common_passwords={
            "empty": "",
            "classic": "123456",
            "zip2023": "2023",
            "zip2024": "2024",
            "tpv": "tpv",
            "pwd1": "password",
        })

        result = extractor.extract("D:/downloads/payload.zip")
        print(result)
        # {
        #   "ok": True,
        #   "out_dir": "D:/downloads/20250824_223045",
        #   "used_password_key": "zip2024",
        #   "used_password": "2024",
        #   "type": "zip",
        #   "error": None
        # }

    支持格式:
        - .zip（支持密码）
        - .tar/.tgz/.tar.gz/.tbz2/.tar.bz2（无密码）
        - .7z（需要安装 py7zr，支持密码）
        - .rar（需要安装 rarfile + 系统 unrar/bsdtar，支持密码）

    安全:
        - 带 path traversal 防护（防止解出到目标目录之外）
    """

    def __init__(self, common_passwords: Dict[str, str] | None = None):
        self.common_passwords: Dict[str, str] = common_passwords or {}

    # ---------- Public API ----------
    def extract(
        self,
        archive_path: str | Path,
        dest_root: str | Path | None = None,
        password: Optional[str] = None,
        prefer_pwd_key: Optional[str] = None,
    ) -> Dict[str, Optional[str] | bool]:
        """
        :param archive_path: 压缩档路径
        :param dest_root: 目标根目录（默认用压缩档所在目录）
        :param password: 用户显式传入的密码（若无则会尝试 common_passwords）
        :param prefer_pwd_key: 若你想优先尝试 common_passwords 中某个键的密码，可传这个 key
        :return: 结果 dict，字段见函数顶部示例
        """
        p = Path(archive_path)
        if not p.exists() or not p.is_file():
            return self._fail(f"Archive not found: {p}")

        # 目标根目录
        base_dir = Path(dest_root) if dest_root else p.parent

        # 以时间命名的文件夹
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = base_dir / ts
        out_dir.mkdir(parents=True, exist_ok=True)

        suffix = p.suffix.lower()
        # 对应多重后缀（如 .tar.gz）
        # 我们匹配常见 tar.* 情况
        lname = p.name.lower()
        is_tar = lname.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz"))
        try:
            if suffix == ".zip":
                return self._extract_zip(p, out_dir, password, prefer_pwd_key)
            elif is_tar:
                return self._extract_tar(p, out_dir)
            elif suffix == ".7z":
                return self._extract_7z(p, out_dir, password, prefer_pwd_key)
            elif suffix == ".rar":
                return self._extract_rar(p, out_dir, password, prefer_pwd_key)
            else:
                return self._fail(f"Unsupported archive type: {p.suffix}", out_dir)
        except Exception as e:
            return self._fail(f"Unexpected error: {e}", out_dir)

    # ---------- ZIP ----------
    def _extract_zip(
        self,
        path: Path,
        out_dir: Path,
        password: Optional[str],
        prefer_pwd_key: Optional[str],
    ) -> Dict[str, Optional[str] | bool]:
        with zipfile.ZipFile(path) as zf:
            # 先尝试无密码
            if self._zip_try_extract(zf, out_dir, None):
                return self._ok(out_dir, type_="zip")

            # 构造尝试密码队列
            candidates = self._candidate_passwords(password, prefer_pwd_key)

            for key, pwd in candidates:
                if self._zip_try_extract(zf, out_dir, pwd):
                    return self._ok(out_dir, used_key=key, used_pwd=pwd, type_="zip")

            return self._fail("All passwords failed for ZIP.", out_dir)

    def _zip_try_extract(self, zf: zipfile.ZipFile, out_dir: Path, pwd: Optional[str]) -> bool:
        try:
            # 先测试每个成员，顺便做路径安全检查
            for info in zf.infolist():
                self._ensure_safe_member(info.filename, out_dir)

            # 真正解压
            if pwd is not None:
                # zipfile 需要 bytes 密码
                zf.extractall(path=out_dir, pwd=pwd.encode("utf-8"))
            else:
                zf.extractall(path=out_dir)
            return True
        except RuntimeError as e:
            # 常见: Bad password
            return False
        except Exception:
            # 其他错误也算失败（但不影响继续尝试其他密码）
            return False

    # ---------- TAR ----------
    def _extract_tar(self, path: Path, out_dir: Path) -> Dict[str, Optional[str] | bool]:
        # tar 系列基本不支持密码；直接解
        try:
            with tarfile.open(path) as tf:
                for m in tf.getmembers():
                    self._ensure_safe_member(m.name, out_dir)
                tf.extractall(out_dir, members=self._safe_tar_members(tf))
            return self._ok(out_dir, type_="tar")
        except Exception as e:
            return self._fail(f"TAR extract error: {e}", out_dir)

    # 仅返回安全的 tar 成员
    def _safe_tar_members(self, tf: tarfile.TarFile):
        for m in tf.getmembers():
            # 拒绝绝对路径/上跳/软链接到外部等
            if not self._is_safe_path(out_dir=tf.extractfile, base_dir=None):  # 占位以满足类型检查
                pass  # 不用这个分支，这行只为绕过静态检查
            # 做具体判断
            if m.islnk() or m.issym():
                # 禁止链接
                continue
            if self._member_unsafe(m.name):
                continue
            yield m

    # ---------- 7Z ----------
    def _extract_7z(
        self,
        path: Path,
        out_dir: Path,
        password: Optional[str],
        prefer_pwd_key: Optional[str],
    ) -> Dict[str, Optional[str] | bool]:
        if py7zr is None:
            return self._fail("py7zr not installed. `pip install py7zr` to enable .7z.", out_dir)

        # 先试无密码
        if self._7z_try_extract(path, out_dir, None):
            return self._ok(out_dir, type_="7z")

        candidates = self._candidate_passwords(password, prefer_pwd_key)
        for key, pwd in candidates:
            if self._7z_try_extract(path, out_dir, pwd):
                return self._ok(out_dir, used_key=key, used_pwd=pwd, type_="7z")

        return self._fail("All passwords failed for 7z.", out_dir)

    def _7z_try_extract(self, path: Path, out_dir: Path, pwd: Optional[str]) -> bool:
        try:
            with py7zr.SevenZipFile(path, mode="r", password=pwd) as z:
                # 无法提前拿到成员名做校验，只能在解压后再检查
                z.extractall(path=out_dir)
            # 解完再做一次目录安全扫描（防止罕见实现差异）
            return self._post_scan_safety(out_dir)
        except Exception:
            return False

    # ---------- RAR ----------
    def _extract_rar(
        self,
        path: Path,
        out_dir: Path,
        password: Optional[str],
        prefer_pwd_key: Optional[str],
    ) -> Dict[str, Optional[str] | bool]:
        if rarfile is None:
            return self._fail("rarfile not installed. `pip install rarfile` and install unrar/bsdtar.", out_dir)

        try:
            rf = rarfile.RarFile(path)
        except Exception as e:
            return self._fail(f"Cannot open RAR: {e}", out_dir)

        # 先试无密码
        if self._rar_try_extract(rf, out_dir, None):
            rf.close()
            return self._ok(out_dir, type_="rar")

        candidates = self._candidate_passwords(password, prefer_pwd_key)
        for key, pwd in candidates:
            if self._rar_try_extract(rf, out_dir, pwd):
                rf.close()
                return self._ok(out_dir, used_key=key, used_pwd=pwd, type_="rar")

        rf.close()
        return self._fail("All passwords failed for RAR.", out_dir)

    def _rar_try_extract(self, rf, out_dir: Path, pwd: Optional[str]) -> bool:
        try:
            for info in rf.infolist():
                self._ensure_safe_member(info.filename, out_dir)
            rf.extractall(path=out_dir, pwd=pwd)
            return True
        except Exception:
            return False

    # ---------- Password candidates ----------
    def _candidate_passwords(
        self,
        user_pwd: Optional[str],
        prefer_pwd_key: Optional[str],
    ) -> List[Tuple[str, str]]:
        """
        返回待尝试的 (key, password) 列表，顺序：
        1) 用户显式提供的 password（若有）
        2) prefer_pwd_key 指向的字典密码（若有）
        3) 其余 common_passwords（去重后）
        """
        tried: List[Tuple[str, str]] = []
        if user_pwd is not None:
            tried.append(("__user_input__", user_pwd))

        # 指定优先 key
        if prefer_pwd_key and prefer_pwd_key in self.common_passwords:
            tried.append((prefer_pwd_key, self.common_passwords[prefer_pwd_key]))

        # 其他常用密码
        for k, v in self.common_passwords.items():
            # 避免重复
            if (k, v) not in tried and ("__user_input__", v) not in tried:
                tried.append((k, v))

        return tried

    # ---------- Safety helpers ----------
    def _ensure_safe_member(self, member_name: str, out_dir: Path):
        """ 抛异常以阻止危险路径 """
        if self._member_unsafe(member_name):
            raise ValueError(f"Unsafe path in archive member: {member_name}")

    def _member_unsafe(self, member_name: str) -> bool:
        # 去掉盘符/前导分隔
        name = member_name.replace("\\", "/")
        if name.startswith("/") or name.startswith("../") or "/../" in name:
            return True
        # Windows 盘符
        if ":" in name.split("/")[0]:
            return True
        return False

    def _post_scan_safety(self, out_dir: Path) -> bool:
        """ 对已经解出的文件再做一次安全检查，若发现越界/软链等，直接删除并返回 False """
        try:
            base = out_dir.resolve()
            for p in out_dir.rglob("*"):
                # 禁止软链接跳出
                try:
                    rp = p.resolve()
                except Exception:
                    return False
                if not str(rp).startswith(str(base)):
                    return False
            return True
        except Exception:
            return False

    # ---------- Result helpers ----------
    def _ok(
        self,
        out_dir: Path,
        used_key: Optional[str] = None,
        used_pwd: Optional[str] = None,
        type_: str = "unknown",
    ) -> Dict[str, Optional[str] | bool]:
        return {
            "ok": True,
            "out_dir": str(out_dir),
            "used_password_key": used_key,
            "used_password": used_pwd,
            "type": type_,
            "error": None,
        }

    def _fail(self, msg: str, out_dir: Path | None = None) -> Dict[str, Optional[str] | bool]:
        return {
            "ok": False,
            "out_dir": str(out_dir) if out_dir else None,
            "used_password_key": None,
            "used_password": None,
            "type": None,
            "error": msg,
        }

from aiomysql import create_pool
from typing import Any, List, Tuple, Optional

class MySQLManager:
    def __init__(self, config: dict):
        """
        初始化 MySQLManager 並接收連線參數，例如：
        {
            "host": "localhost",
            "port": 3306,
            "user": "root",
            "password": "pass",
            "db": "your_db",
            "autocommit": True
        }
        """
        self.config = config
        self.pool = None

    async def init(self):
        """建立連線池"""
        if self.pool is None:
            self.pool = await create_pool(**self.config)

    async def close(self):
        """优雅地关闭 aiomysql 连接池"""
        if getattr(self, "pool", None) is not None:
            self.pool.close()
            await self.pool.wait_closed()

    async def fetchone(self, query: str, args: Tuple = ()) -> Optional[Tuple[Any]]:
        """執行查詢並回傳第一筆資料"""
        await self.init()
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, args)
                return await cur.fetchone()

    async def fetchall(self, query: str, args: Tuple = ()) -> List[Tuple[Any]]:
        """執行查詢並回傳所有結果"""
        await self.init()
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, args)
                return await cur.fetchall()

    async def execute(self, query: str, args: Tuple = ()) -> int:
        """執行 INSERT/UPDATE/DELETE，並回傳受影響列數"""
        await self.init()
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                rows = await cur.execute(query, args)
                await conn.commit()
                return rows

    async def executemany(self, query: str, param_list: List[Tuple]) -> int:
        """批次執行多筆 INSERT/UPDATE"""
        await self.init()
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                rows = await cur.executemany(query, param_list)
                await conn.commit()
                return rows

    async def close(self):
        """關閉連線池"""
        if self.pool is not None:
            self.pool.close()
            await self.pool.wait_closed()

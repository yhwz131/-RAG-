"""
对话记忆模块
维护多轮对话的上下文，支持持久化存储
支持按 token 数控制历史长度
"""
import os
import json
import time
import re
from typing import List, Dict, Optional
from config.settings import settings
from utils.logger import get_logger
from utils.tokens import estimate_tokens

logger = get_logger("memory")

# 内存中最多保留的会话数，超出后淘汰最旧的
MAX_IN_MEMORY_SESSIONS = 100


class ConversationMemory:
    """对话记忆管理器（支持持久化）"""

    def __init__(self, max_rounds: int = None, max_tokens: int = None):
        self.max_rounds = max_rounds or settings.max_history_rounds
        self.max_tokens = max_tokens or settings.max_history_tokens
        self.sessions_dir = settings.sessions_dir
        self.histories: Dict[str, List[Dict]] = {}
        self.meta: Dict[str, Dict] = {}  # session_id -> {title, created_at, updated_at}
        os.makedirs(self.sessions_dir, exist_ok=True)
        self._load_all()
        self._evict_oldest()
        logger.info(f"对话记忆初始化: max_rounds={self.max_rounds}, max_tokens={self.max_tokens}, 已加载 {len(self.histories)} 个会话")

    # ---- 读写 ----

    def add(self, session_id: str, role: str, content: str):
        """添加一条对话记录（同时按条数和 token 数控制历史长度）"""
        # 如果会话已被淘汰出内存，先从磁盘加载
        if session_id not in self.histories and session_id in self.meta:
            self._load_session(session_id)
        if session_id not in self.histories:
            self.histories[session_id] = []
            self.meta[session_id] = {
                "session_id": session_id,
                "title": self._generate_title(role, content),
                "created_at": time.time(),
                "updated_at": time.time(),
            }
        self.histories[session_id].append({"role": role, "content": content})
        self.meta[session_id]["updated_at"] = time.time()
        
        # 按条数裁剪：只保留最近 N 轮
        if len(self.histories[session_id]) > self.max_rounds * 2:
            self.histories[session_id] = self.histories[session_id][-self.max_rounds * 2:]
        
        # 按 token 数裁剪：从最新消息向前，超出 token 限制的消息被移除
        self._trim_by_tokens(session_id)
        
        self._save(session_id)
        self._evict_oldest()
        logger.debug(f"会话 [{session_id}] 添加 {role} 消息")

    def get_context(self, session_id: str) -> List[Dict]:
        """获取对话上下文（供 LLM 使用，按需从磁盘加载）"""
        if session_id not in self.histories and session_id in self.meta:
            self._load_session(session_id)
        return self.histories.get(session_id, [])

    def get_full_history(self, session_id: str) -> List[Dict]:
        """获取完整对话历史（含 assistant 原始内容）"""
        return self.histories.get(session_id, [])

    def clear(self, session_id: str):
        """清空指定会话的对话历史"""
        self._validate_session_id(session_id)
        self.histories.pop(session_id, None)
        self.meta.pop(session_id, None)
        path = os.path.join(self.sessions_dir, f"{session_id}.json")
        if os.path.exists(path):
            os.remove(path)
        logger.info(f"会话 [{session_id}] 已删除")

    # ---- 会话列表 ----

    def list_sessions(self) -> List[Dict]:
        """返回所有会话的元信息，按更新时间倒序"""
        sessions = list(self.meta.values())
        sessions.sort(key=lambda s: s.get("updated_at", 0), reverse=True)
        return sessions

    # ---- 内部方法 ----

    @staticmethod
    def _validate_session_id(session_id: str):
        """校验 session_id，防止路径穿越攻击"""
        if not session_id or not re.match(r'^[a-zA-Z0-9_-]{1,64}$', session_id):
            raise ValueError(f"非法 session_id（仅允许字母数字下划线连字符，1-64字符）: {session_id}")

    def _generate_title(self, role: str, content: str) -> str:
        """从第一条用户消息生成标题"""
        if role == "user":
            title = content.strip().replace("\n", " ")
            return title[:40] + ("..." if len(title) > 40 else "")
        return "新对话"

    def _trim_by_tokens(self, session_id: str):
        """按 token 数裁剪对话历史，保留最新的消息"""
        messages = self.histories.get(session_id, [])
        if not messages:
            return
        
        # 从最新消息向前计算 token 数
        total_tokens = 0
        keep_from = len(messages)
        
        for i in range(len(messages) - 1, -1, -1):
            msg_tokens = estimate_tokens(messages[i].get("content", ""))
            if total_tokens + msg_tokens > self.max_tokens:
                break
            total_tokens += msg_tokens
            keep_from = i
        
        if keep_from > 0:
            removed = keep_from
            # 确保从 user 消息开始（角色配对完整性）
            if keep_from < len(messages) and messages[keep_from].get("role") != "user":
                keep_from += 1
            self.histories[session_id] = messages[keep_from:]
            logger.info(
                f"会话 [{session_id}] token裁剪: 移除 {removed} 条旧消息, "
                f"保留 {len(self.histories[session_id])} 条, 估算 {total_tokens} tokens"
            )

    def _save(self, session_id: str):
        """持久化单个会话到 JSON（原子写入：先写临时文件再 rename）"""
        self._validate_session_id(session_id)
        path = os.path.join(self.sessions_dir, f"{session_id}.json")
        tmp_path = path + ".tmp"
        data = {
            "meta": self.meta.get(session_id, {}),
            "messages": self.histories.get(session_id, []),
        }
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, path)  # 原子替换
        except Exception as e:
            logger.error(f"保存会话 {session_id} 失败: {e}")
            # 清理可能残留的临时文件
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _load_all(self):
        """启动时从磁盘加载所有会话元信息，仅加载最近 N 个的消息"""
        if not os.path.isdir(self.sessions_dir):
            return
        # 第一遍：加载所有元信息
        file_entries = []
        for fname in os.listdir(self.sessions_dir):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(self.sessions_dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                sid = data["meta"]["session_id"]
                self.meta[sid] = data["meta"]
                updated_at = data["meta"].get("updated_at", 0)
                file_entries.append((updated_at, sid, path))
            except Exception as e:
                logger.warning(f"加载会话 {fname} 失败: {e}")
        # 第二遍：只加载最近 N 个会话的消息到内存
        file_entries.sort(key=lambda x: x[0], reverse=True)
        for _, sid, path in file_entries[:MAX_IN_MEMORY_SESSIONS]:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.histories[sid] = data.get("messages", [])
            except Exception:
                pass

    def _load_session(self, session_id: str):
        """按需从磁盘加载单个会话的消息（懒加载）"""
        path = os.path.join(self.sessions_dir, f"{session_id}.json")
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.histories[session_id] = data.get("messages", [])
        except Exception as e:
            logger.warning(f"懒加载会话 {session_id} 失败: {e}")

    def _evict_oldest(self):
        """淘汰最旧的会话（仅释放内存，不删除磁盘文件）"""
        while len(self.histories) > MAX_IN_MEMORY_SESSIONS:
            oldest_sid = min(self.histories, key=lambda sid: self.meta.get(sid, {}).get("updated_at", 0))
            del self.histories[oldest_sid]
            logger.debug(f"淘汰会话 {oldest_sid} 出内存")
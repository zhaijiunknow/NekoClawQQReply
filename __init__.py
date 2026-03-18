
from __future__ import annotations

import asyncio
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional
from plugin.sdk.base import NekoPluginBase
from plugin.sdk.decorators import lifecycle, neko_plugin, plugin_entry
from plugin.sdk import ok, fail

from .qq_client import QQClient
from .permission import PermissionManager
from .group_permission import GroupPermissionManager

try:
    import tomli
    import tomli_w
except ImportError:
    import tomllib as tomli
    import tomli_w


@neko_plugin
class QQAutoReplyPlugin(NekoPluginBase):

    def __init__(self, ctx):
        super().__init__(ctx)
        self.file_logger = self.enable_file_logging(log_level="INFO")
        self.logger = self.file_logger

        self.qq_client: Optional[QQClient] = None
        self.permission_mgr: Optional[PermissionManager] = None
        self.group_permission_mgr: Optional[GroupPermissionManager] = None

        self._running = False
        self._message_task: Optional[asyncio.Task] = None

        # Normal 权限转述功能
        self._admin_qq: Optional[str] = None
        self._normal_relay_probability: float = 0.3

    @lifecycle(id="startup")
    async def startup(self, **_):
        """插件启动时初始化"""
        cfg = await self.config.dump(timeout=5.0)
        cfg = cfg if isinstance(cfg, dict) else {}
        qq_cfg = cfg.get("qq_auto_reply", {})

        # 初始化权限管理器
        trusted_users = qq_cfg.get("trusted_users", [])
        self.permission_mgr = PermissionManager(trusted_users)

        # 初始化群聊权限管理器
        trusted_groups = qq_cfg.get("trusted_groups", [])
        self.group_permission_mgr = GroupPermissionManager(trusted_groups)

        # 获取管理员 QQ（用于转述）
        for user in trusted_users:
            if user.get("level") == "admin":
                self._admin_qq = user.get("qq")
                break

        # 获取转述概率
        self._normal_relay_probability = qq_cfg.get("normal_relay_probability", 0.3)

        # 初始化 QQ 客户端
        onebot_url = qq_cfg.get("onebot_url", "ws://127.0.0.1:3001")
        token = qq_cfg.get("token")
        self.qq_client = QQClient(onebot_url, token, self.logger)

        self.logger.info(f"QQ Auto Reply Plugin started (Admin QQ: {self._admin_qq}, Relay Probability: {self._normal_relay_probability})")
        return ok(data={"status": "running"})

    @lifecycle(id="shutdown")
    async def shutdown(self, **_):
        """插件关闭时清理资源"""
        await self.stop_auto_reply()
        if self.qq_client:
            await self.qq_client.disconnect()
        self.logger.info("QQ Auto Reply Plugin shutdown")
        return ok(data={"status": "shutdown"})

    @plugin_entry(
        id="start_auto_reply",
        name="启动自动回复",
        description="开始监听 QQ 消息并自动回复。连接到 OneBot 服务（NapCat/LLOneBot），接收消息后根据权限等级生成 AI 回复。",
        input_schema={
            "type": "object",
            "properties": {},
        },
    )
    async def start_auto_reply(self, **_):
        """启动自动回复功能"""
        if self._running:
            return fail("ALREADY_RUNNING", "自动回复已在运行中")

        if not self.qq_client:
            return fail("NOT_INITIALIZED", "QQ 客户端未初始化")

        try:
            # 连接到 OneBot 服务（NapCat/LLOneBot 等）
            await self.qq_client.connect()

            # 启动消息处理任务
            self._running = True
            self._message_task = asyncio.create_task(self._process_messages())

            self.logger.info("Auto reply started")
            return ok(data={"status": "started"})
        except Exception as e:
            self.logger.exception("Failed to start auto reply")
            return fail("START_ERROR", f"启动失败: {e}")

    @plugin_entry(
        id="stop_auto_reply",
        name="停止自动回复",
        description="停止监听 QQ 消息，断开与 OneBot 服务的连接。",
        input_schema={
            "type": "object",
            "properties": {},
        },
    )
    async def stop_auto_reply(self, **_):
        """停止自动回复功能"""
        if not self._running:
            return ok(data={"status": "not_running"})

        self._running = False
        if self._message_task:
            self._message_task.cancel()
            try:
                await self._message_task
            except asyncio.CancelledError:
                pass
            self._message_task = None

        self.logger.info("Auto reply stopped")
        return ok(data={"status": "stopped"})

    async def _process_messages(self):
        """处理接收到的 QQ 消息"""
        while self._running:
            try:
                message = await self.qq_client.receive_message()
                if message:
                    await self._handle_message(message)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error processing message: {e}")
                await asyncio.sleep(1)

    async def _handle_message(self, message: Dict[str, Any]):
        """处理单条消息，通过 OneBot API 回复"""
        message_type = message.get("message_type")
        sender_id = message.get("user_id")
        message_text = message.get("content", "")
        user_nickname = message.get("user_nickname")  # QQ 昵称

        # 私聊消息处理
        if message_type == "private":
            await self._handle_private_message(sender_id, message_text, user_nickname)

        # 群聊消息处理
        elif message_type == "group":
            group_id = message.get("group_id")
            is_at_bot = message.get("is_at_bot", False)
            await self._handle_group_message(group_id, sender_id, message_text, is_at_bot, user_nickname)

    async def _handle_private_message(self, sender_id: str, message_text: str, user_nickname: Optional[str] = None):
        """处理私聊消息"""
        # 检查权限
        permission_level = self.permission_mgr.get_permission_level(sender_id)
        if permission_level == "none":
            self.logger.debug(f"Ignored message from untrusted user: {sender_id}")
            return

        self.logger.info(
            f"Received private message from {sender_id} (level: {permission_level}): {message_text}"
        )

        # Normal 权限：不直接回复，概率转述给管理员
        if permission_level == "normal":
            await self._handle_normal_relay(
                message_text,
                sender_id,
                source_type="private",
                source_id=sender_id
            )
            return

        # Admin 和 Trusted 权限：正常回复
        reply_text = await self._generate_reply(
            message_text, permission_level, sender_id,
            is_group=False,
            user_nickname=user_nickname
        )

        if reply_text:
            try:
                await self.qq_client.send_message(sender_id, reply_text)
                self.logger.info(f"Sent reply to {sender_id}: {reply_text}")
            except Exception as e:
                self.logger.error(f"Failed to send message via OneBot: {e}")

    async def _handle_group_message(self, group_id: str, sender_id: str, message_text: str, is_at_bot: bool, user_nickname: Optional[str] = None):
        """处理群聊消息"""
        # 检查群聊权限
        group_level = self.group_permission_mgr.get_group_level(group_id)
        if group_level == "none":
            self.logger.debug(f"Ignored message from untrusted group: {group_id}")
            return

        self.logger.info(
            f"Received group message from group {group_id}, user {sender_id} (group level: {group_level}): {message_text}"
        )

        # Normal 群聊：不响应 @，概率转述给管理员
        if group_level == "normal":
            await self._handle_normal_relay(
                message_text,
                sender_id,
                source_type="group",
                source_id=group_id
            )
            return

        # Trusted 群聊：只响应 @ 机器人的消息
        if not is_at_bot:
            self.logger.debug(f"Ignored group message without @: {group_id}")
            return

        # 生成回复（群聊中不检查用户权限）
        reply_text = await self._generate_reply(
            message_text, "group", sender_id,
            is_group=True,
            group_id=group_id,
            user_nickname=user_nickname
        )

        if reply_text:
            try:
                await self.qq_client.send_group_message(group_id, reply_text)
                self.logger.info(f"Sent group reply to {group_id}: {reply_text}")
            except Exception as e:
                self.logger.error(f"Failed to send group message via OneBot: {e}")

    async def _handle_normal_relay(self, message_text: str, sender_id: str, source_type: str, source_id: str):
        """处理 Normal 权限的转述逻辑"""
        # 检查是否有管理员
        if not self._admin_qq:
            self.logger.debug("No admin QQ configured, skipping relay")
            return

        # 概率触发
        if random.random() > self._normal_relay_probability:
            self.logger.debug(f"Relay not triggered (probability: {self._normal_relay_probability})")
            return

        self.logger.info(f"🔔 Relay triggered for {source_type} {source_id}, user {sender_id}")

        # 生成转述给主人的回复
        try:
            from main_logic.omni_offline_client import OmniOfflineClient
            from utils.config_manager import get_config_manager
            from config.prompts_sys import SESSION_INIT_PROMPT
            from utils.language_utils import get_global_language

            config_manager = get_config_manager()
            master_name, her_name, _, catgirl_data, _, lanlan_prompt_map, _, _, _ = config_manager.get_character_data()

            # 获取角色核心提示词
            character_prompt = lanlan_prompt_map.get(her_name, "你是一个友好的AI助手")

            # 获取角色卡额外字段
            current_character = catgirl_data.get(her_name, {})
            character_card_fields = {}
            for key, value in current_character.items():
                if key not in ['_reserved', 'voice_id', 'system_prompt', 'model_type',
                               'live2d', 'vrm', 'vrm_animation', 'lighting', 'vrm_rotation',
                               'live2d_item_id', 'item_id', 'idleAnimation']:
                    if isinstance(value, (str, int, float, bool)) and value:
                        character_card_fields[key] = value

            # 获取对话模型配置
            conversation_config = config_manager.get_model_api_config('conversation')
            base_url = conversation_config.get('base_url', '')
            api_key = conversation_config.get('api_key', '')
            model = conversation_config.get('model', '')

            # 创建临时会话
            reply_chunks = []

            async def on_text_delta(text: str, is_first: bool):
                reply_chunks.append(text)

            temp_session = OmniOfflineClient(
                base_url=base_url,
                api_key=api_key,
                model=model,
                on_text_delta=on_text_delta
            )

            # 构建转述提示词
            user_language = get_global_language()
            init_prompt = SESSION_INIT_PROMPT.get(user_language, SESSION_INIT_PROMPT['zh'])
            init_prompt = init_prompt.format(name=her_name)

            system_prompt_parts = [
                init_prompt,
                character_prompt
            ]

            if character_card_fields:
                system_prompt_parts.append("\n======角色卡额外设定======")
                for field_name, field_value in character_card_fields.items():
                    system_prompt_parts.append(f"{field_name}: {field_value}")
                system_prompt_parts.append("======角色卡设定结束======")

            # 转述场景说明
            source_desc = f"QQ 群 {source_id}" if source_type == "group" else f"QQ 用户 {source_id}"
            system_prompt_parts.append(f"""
======转述场景======
- 你在 {source_desc} 中看到了用户 {sender_id} 的发言
- 发言内容："{message_text}"
- 现在你要把这个有趣的内容转述给{master_name if master_name else "主人"}
- 请用简短自然的话（不超过50字）告诉{master_name if master_name else "主人"}这件事
- 不要使用 Markdown 格式，不要使用表情符号
- 记住你是 {her_name}，以 {her_name} 的身份转述
- 例如："主人，我刚看到有人说xxx很好吃呢，要不要试试？"
======场景说明结束======""")

            system_prompt = "\n".join(system_prompt_parts)

            await temp_session.connect(instructions=system_prompt)

            # 发送转述请求
            relay_prompt = f"请把这个内容转述给{master_name if master_name else '主人'}：{message_text}"
            await temp_session.stream_text(relay_prompt)

            # 等待回复完成
            for i in range(30):
                await asyncio.sleep(1)
                if not temp_session._is_responding:
                    break

            # 组合回复
            relay_text = ''.join(reply_chunks).strip()

            if relay_text:
                # 发送给管理员
                try:
                    await self.qq_client.send_message(self._admin_qq, relay_text)
                    self.logger.info(f"✅ Relayed to admin {self._admin_qq}: {relay_text[:50]}...")
                except Exception as e:
                    self.logger.error(f"Failed to relay to admin: {e}")

            # 断开临时会话
            await temp_session.disconnect()

        except Exception as e:
            self.logger.error(f"Failed to generate relay message: {e}")


    async def _generate_reply(
        self, message: str, permission_level: str, sender_id: str,
        is_group: bool = False, group_id: str = None, user_nickname: Optional[str] = None
    ) -> Optional[str]:
        """生成回复内容（使用 OmniOfflineClient + Memory Server 同步）"""
        # 私聊：只为 admin 和 trusted 用户生成 AI 回复
        # 群聊：所有 @ 机器人的用户都生成回复
        if not is_group and permission_level not in ["admin", "trusted"]:
            return None

        try:
            from main_logic.omni_offline_client import OmniOfflineClient
            from utils.config_manager import get_config_manager
            import httpx
            from config import MEMORY_SERVER_PORT

            config_manager = get_config_manager()

            # 获取角色完整数据
            master_name, her_name, _, catgirl_data, _, lanlan_prompt_map, _, _, _ = config_manager.get_character_data()

            # 🔥 获取用户称呼
            # 1. 优先使用插件设置的昵称
            custom_nickname = self.permission_mgr.get_nickname(sender_id)

            # 2. 根据场景确定称呼
            if is_group:
                # 群聊中：自定义昵称 > QQ昵称 > QQ号
                if custom_nickname:
                    user_title = custom_nickname
                elif user_nickname:
                    user_title = user_nickname
                else:
                    user_title = f"QQ用户{sender_id}"
            else:
                # 私聊中：根据权限等级确定称呼
                if permission_level == "admin":
                    # 管理员：使用 master_name
                    user_title = master_name if master_name else "主人"
                else:
                    # 其他用户：自定义昵称 > QQ昵称 > QQ号
                    if custom_nickname:
                        user_title = custom_nickname
                    elif user_nickname:
                        user_title = user_nickname
                    else:
                        user_title = f"QQ用户{sender_id}"

            # 获取当前角色的完整配置
            current_character = catgirl_data.get(her_name, {})

            # 获取角色核心提示词（system_prompt）
            character_prompt = lanlan_prompt_map.get(her_name, "你是一个友好的AI助手")

            # 获取角色卡的额外字段（如果有）
            character_card_fields = {}
            for key, value in current_character.items():
                # 排除系统保留字段
                if key not in ['_reserved', 'voice_id', 'system_prompt', 'model_type',
                               'live2d', 'vrm', 'vrm_animation', 'lighting', 'vrm_rotation',
                               'live2d_item_id', 'item_id', 'idleAnimation']:
                    if isinstance(value, (str, int, float, bool)) and value:
                        character_card_fields[key] = value

            self.logger.info(f"使用角色: {her_name}, 额外字段: {list(character_card_fields.keys())}")

            # 获取对话模型配置
            conversation_config = config_manager.get_model_api_config('conversation')
            base_url = conversation_config.get('base_url', '')
            api_key = conversation_config.get('api_key', '')
            model = conversation_config.get('model', '')

            # 为每个 QQ 用户维护独立的对话客户端
            if not hasattr(self, '_user_sessions'):
                self._user_sessions = {}

            # 获取或创建用户的 session
            if sender_id not in self._user_sessions:
                self.logger.info(f"为用户 {sender_id} 创建新的对话 session")

                # 创建回复收集器
                reply_chunks = []

                async def on_text_delta(text: str, is_first: bool):
                    reply_chunks.append(text)

                # 创建用户专属 OmniOfflineClient
                user_session = OmniOfflineClient(
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    on_text_delta=on_text_delta
                )

                # 🔥 使用与前端完全一致的提示词结构
                from config.prompts_sys import SESSION_INIT_PROMPT
                from utils.language_utils import get_global_language

                # 获取用户语言
                user_language = get_global_language()

                # 构建初始提示（与前端一致）
                init_prompt = SESSION_INIT_PROMPT.get(user_language, SESSION_INIT_PROMPT['zh'])
                init_prompt = init_prompt.format(name=her_name)

                # 构建完整系统提示（与前端架构一致）
                system_prompt_parts = [
                    init_prompt,  # "你是一个角色扮演大师。请按要求扮演以下角色（{name}）。"
                    character_prompt  # 角色核心提示词
                ]

                # 注入角色卡额外字段
                if character_card_fields:
                    system_prompt_parts.append("\n======角色卡额外设定======")
                    for field_name, field_value in character_card_fields.items():
                        system_prompt_parts.append(f"{field_name}: {field_value}")
                    system_prompt_parts.append("======角色卡设定结束======")

                # 添加 QQ 对话特定说明
                if is_group:
                    system_prompt_parts.append(f"""
======QQ 群聊环境======
- 你正在 QQ 群 {group_id} 中与用户 {sender_id} 对话
- 对方的称呼是：{user_title}
- 这是群聊环境，有多个用户在场
- 请保持角色设定，用简短自然的话回复（不超过50字）
- 不要使用 Markdown 格式，不要使用表情符号
- 记住你是 {her_name}，始终以 {her_name} 的身份回复
- 在回复中自然地称呼对方为"{user_title}"
======环境说明结束======""")
                else:
                    system_prompt_parts.append(f"""
======QQ 私聊环境======
- 你正在通过 QQ 与用户 {sender_id} 私聊
- 对方的称呼是：{user_title}
- 请保持角色设定，用简短自然的话回复（不超过50字）
- 不要使用 Markdown 格式，不要使用表情符号
- 记住你是 {her_name}，始终以 {her_name} 的身份回复
- 在回复中自然地称呼对方为"{user_title}"
======环境说明结束======""")

                system_prompt = "\n".join(system_prompt_parts)

                self.logger.info(f"系统提示词长度: {len(system_prompt)} 字符")
                self.logger.info(f"使用语言: {user_language}, 初始提示: {init_prompt[:50]}...")

                await user_session.connect(instructions=system_prompt)

                self._user_sessions[sender_id] = {
                    'session': user_session,
                    'reply_chunks': reply_chunks,
                    'her_name': her_name,
                    'character_fields': character_card_fields
                }

            # 获取用户 session
            user_data = self._user_sessions[sender_id]
            user_session = user_data['session']
            reply_chunks = user_data['reply_chunks']
            her_name = user_data['her_name']

            # 清空之前的回复
            reply_chunks.clear()

            # 发送消息到 AI（通过 OmniOfflineClient.stream_text）
            self.logger.info(f"发送消息到 AI: {message[:50]}...")
            await user_session.stream_text(message)

            # 等待回复完成
            for i in range(30):
                await asyncio.sleep(1)
                if not user_session._is_responding:
                    break

            # 组合回复
            ai_reply = ''.join(reply_chunks).strip()

            if ai_reply:
                # 🔥 只有私聊且管理员权限的对话才同步到记忆系统
                # 群聊消息不进入记忆
                if not is_group and permission_level == "admin":
                    try:
                        # 获取 OmniOfflineClient 维护的完整对话历史
                        conversation_history = user_session._conversation_history

                        # 只同步最新的用户消息和 AI 回复（增量同步）
                        # conversation_history 格式: [SystemMessage, HumanMessage, AIMessage, ...]
                        # 我们需要最后两条消息（user + assistant）
                        if len(conversation_history) >= 2:
                            # 转换为 Memory Server 期望的格式
                            recent_messages = []
                            for msg in conversation_history[-2:]:  # 最后两条消息
                                if hasattr(msg, 'type'):
                                    role = 'user' if msg.type == 'human' else 'assistant'
                                    content = msg.content
                                    recent_messages.append({
                                        'role': role,
                                        'content': [{'type': 'text', 'text': content}]
                                    })

                            # 调用 Memory Server 的 /cache 端点
                            async with httpx.AsyncClient() as client:
                                response = await client.post(
                                    f"http://localhost:{MEMORY_SERVER_PORT}/cache/{her_name}",
                                    json={'input_history': json.dumps(recent_messages, ensure_ascii=False)},
                                    timeout=5.0
                                )

                                if response.status_code == 200:
                                    result = response.json()
                                    count = result.get('count', 0)
                                    self.logger.info(f"✅ [管理员] 成功同步 {count} 条消息到 Memory Server (用户: {sender_id})")
                                else:
                                    self.logger.warning(f"⚠️ Memory Server 返回错误: {response.status_code}")

                    except Exception as e:
                        self.logger.error(f"⚠️ 记忆同步失败（不影响回复）: {e}")
                else:
                    if is_group:
                        self.logger.info(f"ℹ️ [群聊] 跳过记忆同步 (群: {group_id}, 用户: {sender_id})")
                    else:
                        self.logger.info(f"ℹ️ [非管理员] 跳过记忆同步 (用户: {sender_id}, 权限: {permission_level})")

                self.logger.info(f"AI 生成回复: {ai_reply[:50]}...")
                return ai_reply
            else:
                self.logger.warning("AI 未生成回复")
                return f"收到你的消息: {message}"

        except Exception as e:
            self.logger.exception(f"AI 生成回复失败: {e}")
            return f"收到你的消息: {message}"


    async def _save_trusted_users_to_config(self):
        """持久化信任用户列表到 plugin.toml"""
        try:
            # 获取 plugin.toml 路径
            plugin_dir = Path(__file__).parent
            config_path = plugin_dir / "plugin.toml"

            # 读取现有配置
            with open(config_path, "rb") as f:
                config = tomli.load(f)

            # 获取当前用户列表
            users = self.permission_mgr.list_users()

            # 更新配置
            if "qq_auto_reply" not in config:
                config["qq_auto_reply"] = {}

            config["qq_auto_reply"]["trusted_users"] = users

            # 写回文件
            with open(config_path, "wb") as f:
                tomli_w.dump(config, f)

            self.logger.info(f"✅ 成功持久化 {len(users)} 个信任用户到配置文件")
            return True

        except Exception as e:
            self.logger.error(f"❌ 持久化配置失败: {e}")
            return False

    async def _save_trusted_groups_to_config(self):
        """持久化信任群聊列表到 plugin.toml"""
        try:
            # 获取 plugin.toml 路径
            plugin_dir = Path(__file__).parent
            config_path = plugin_dir / "plugin.toml"

            # 读取现有配置
            with open(config_path, "rb") as f:
                config = tomli.load(f)

            # 获取当前群聊列表
            groups = self.group_permission_mgr.list_groups()

            # 更新配置
            if "qq_auto_reply" not in config:
                config["qq_auto_reply"] = {}

            config["qq_auto_reply"]["trusted_groups"] = groups

            # 写回文件
            with open(config_path, "wb") as f:
                tomli_w.dump(config, f)

            self.logger.info(f"✅ 成功持久化 {len(groups)} 个信任群聊到配置文件")
            return True

        except Exception as e:
            self.logger.error(f"❌ 持久化群聊配置失败: {e}")
            return False


    @plugin_entry(
        id="add_trusted_user",
        name="添加信任用户",
        description="添加一个信任的 QQ 号到白名单。支持三种权限等级：admin（管理员）、trusted（信任用户）、normal（普通用户）。可选设置昵称。",
        input_schema={
            "type": "object",
            "properties": {
                "qq_number": {
                    "type": "string",
                    "description": "QQ 号",
                },
                "level": {
                    "type": "string",
                    "description": "权限等级: admin, trusted, normal",
                    "default": "trusted",
                },
                "nickname": {
                    "type": "string",
                    "description": "用户昵称（可选，管理员无需设置）",
                    "default": "",
                },
            },
            "required": ["qq_number"],
        },
    )
    async def add_trusted_user(self, qq_number: str, level: str = "trusted", nickname: str = "", **_):
        """添加信任用户并持久化到配置文件"""
        if not self.permission_mgr:
            return fail("NOT_INITIALIZED", "权限管理器未初始化")

        # 添加到内存（管理员不设置昵称）
        user_nickname = "" if level == "admin" else nickname
        self.permission_mgr.add_user(qq_number, level, user_nickname)
        self.logger.info(f"Added trusted user: {qq_number} with level {level}" +
                        (f" and nickname {user_nickname}" if user_nickname else ""))

        # 持久化到配置文件
        success = await self._save_trusted_users_to_config()

        if success:
            result_data = {
                "qq_number": qq_number,
                "level": level,
                "persisted": True
            }
            if user_nickname:
                result_data["nickname"] = user_nickname
            return ok(data=result_data)
        else:
            result_data = {
                "qq_number": qq_number,
                "level": level,
                "persisted": False,
                "warning": "已添加到内存，但持久化失败"
            }
            if user_nickname:
                result_data["nickname"] = user_nickname
            return ok(data=result_data)

    @plugin_entry(
        id="remove_trusted_user",
        name="移除信任用户",
        description="从白名单中移除一个 QQ 号，移除后该用户将无法触发自动回复。",
        input_schema={
            "type": "object",
            "properties": {
                "qq_number": {
                    "type": "string",
                    "description": "QQ 号",
                },
            },
            "required": ["qq_number"],
        },
    )
    async def remove_trusted_user(self, qq_number: str, **_):
        """移除信任用户并持久化到配置文件"""
        if not self.permission_mgr:
            return fail("NOT_INITIALIZED", "权限管理器未初始化")

        # 从内存中移除
        self.permission_mgr.remove_user(qq_number)
        self.logger.info(f"Removed trusted user: {qq_number}")

        # 持久化到配置文件
        success = await self._save_trusted_users_to_config()

        if success:
            return ok(data={
                "qq_number": qq_number,
                "persisted": True
            })
        else:
            return ok(data={
                "qq_number": qq_number,
                "persisted": False,
                "warning": "已从内存移除，但持久化失败"
            })

    @plugin_entry(
        id="list_trusted_users",
        name="列出信任用户",
        description="列出所有在白名单中的 QQ 号及其权限等级。",
        input_schema={
            "type": "object",
            "properties": {},
        },
    )
    async def list_trusted_users(self, **_):
        """列出所有信任用户"""
        if not self.permission_mgr:
            return fail("NOT_INITIALIZED", "权限管理器未初始化")

        users = self.permission_mgr.list_users()
        return ok(data={"users": users})

    @plugin_entry(
        id="set_user_nickname",
        name="设置用户昵称",
        description="为信任用户设置专属称呼。管理员始终被称为主人，其他用户可以设置自定义昵称。",
        input_schema={
            "type": "object",
            "properties": {
                "qq_number": {
                    "type": "string",
                    "description": "QQ 号",
                },
                "nickname": {
                    "type": "string",
                    "description": "昵称（留空则清除昵称）",
                },
            },
            "required": ["qq_number"],
        },
    )
    async def set_user_nickname(self, qq_number: str, nickname: str = "", **_):
        """设置用户昵称并持久化到配置文件"""
        if not self.permission_mgr:
            return fail("NOT_INITIALIZED", "权限管理器未初始化")

        # 检查用户是否存在
        permission_level = self.permission_mgr.get_permission_level(qq_number)
        if permission_level == "none":
            return fail("USER_NOT_FOUND", f"用户 {qq_number} 不在信任列表中")

        # 管理员不能设置昵称（始终是主人）
        if permission_level == "admin":
            return fail("ADMIN_NO_NICKNAME", "管理员始终被称为主人，无法设置昵称")

        # 设置昵称
        success = self.permission_mgr.set_nickname(qq_number, nickname)

        if not success:
            return fail("SET_FAILED", "设置昵称失败")

        # 持久化到配置文件
        persist_success = await self._save_trusted_users_to_config()

        action = "清除" if not nickname else "设置"
        self.logger.info(f"{action}用户 {qq_number} 的昵称: {nickname}")

        if persist_success:
            return ok(data={
                "qq_number": qq_number,
                "nickname": nickname if nickname else None,
                "persisted": True
            })
        else:
            return ok(data={
                "qq_number": qq_number,
                "nickname": nickname if nickname else None,
                "persisted": False,
                "warning": "已在内存中更新，但持久化失败"
            })

    @plugin_entry(
        id="add_trusted_group",
        name="添加信任群聊",
        description="添加一个信任的 QQ 群到白名单。支持两种等级：trusted（信任群聊）、normal（普通群聊）。",
        input_schema={
            "type": "object",
            "properties": {
                "group_id": {
                    "type": "string",
                    "description": "群号",
                },
                "level": {
                    "type": "string",
                    "description": "权限等级: trusted, normal",
                    "default": "normal",
                },
            },
            "required": ["group_id"],
        },
    )
    async def add_trusted_group(self, group_id: str, level: str = "normal", **_):
        """添加信任群聊并持久化到配置文件"""
        if not self.group_permission_mgr:
            return fail("NOT_INITIALIZED", "群聊权限管理器未初始化")

        # 添加到内存
        self.group_permission_mgr.add_group(group_id, level)
        self.logger.info(f"Added trusted group: {group_id} with level {level}")

        # 持久化到配置文件
        success = await self._save_trusted_groups_to_config()

        if success:
            return ok(data={
                "group_id": group_id,
                "level": level,
                "persisted": True
            })
        else:
            return ok(data={
                "group_id": group_id,
                "level": level,
                "persisted": False,
                "warning": "已添加到内存，但持久化失败"
            })

    @plugin_entry(
        id="remove_trusted_group",
        name="移除信任群聊",
        description="从白名单中移除一个 QQ 群，移除后该群将无法触发自动回复。",
        input_schema={
            "type": "object",
            "properties": {
                "group_id": {
                    "type": "string",
                    "description": "群号",
                },
            },
            "required": ["group_id"],
        },
    )
    async def remove_trusted_group(self, group_id: str, **_):
        """移除信任群聊并持久化到配置文件"""
        if not self.group_permission_mgr:
            return fail("NOT_INITIALIZED", "群聊权限管理器未初始化")

        # 从内存中移除
        self.group_permission_mgr.remove_group(group_id)
        self.logger.info(f"Removed trusted group: {group_id}")

        # 持久化到配置文件
        success = await self._save_trusted_groups_to_config()

        if success:
            return ok(data={
                "group_id": group_id,
                "persisted": True
            })
        else:
            return ok(data={
                "group_id": group_id,
                "persisted": False,
                "warning": "已从内存移除，但持久化失败"
            })

    @plugin_entry(
        id="list_trusted_groups",
        name="列出信任群聊",
        description="列出所有在白名单中的 QQ 群及其权限等级。",
        input_schema={
            "type": "object",
            "properties": {},
        },
    )
    async def list_trusted_groups(self, **_):
        """列出所有信任群聊"""
        if not self.group_permission_mgr:
            return fail("NOT_INITIALIZED", "群聊权限管理器未初始化")

        groups = self.group_permission_mgr.list_groups()
        return ok(data={"groups": groups})
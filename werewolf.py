import asyncio
import json
import random
import re
from collections import Counter
from typing import Optional

import requests
import agentscope
from agentscope.agent import ReActAgent
from agentscope.formatter import DashScopeMultiAgentFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.model import DashScopeChatModel
from agentscope.pipeline import MsgHub
from agentscope.tracing import trace_reply


# =========================
# 基础配置
# =========================

API_URL = "http://10.2.13.134:16001/v1/chat/completions"
MODEL_NAME = "Qwen2.5-32B"

PLAYER_COUNT = 8
ROLE_DECK = ["狼人"] * 2 + ["预言家"] + ["女巫"] + ["村民"] * 4

MAX_RETRY = 3


# =========================
# 打印工具
# =========================

def print_section(title: str):
    print(f"\n========== {title} ==========")


def print_dm(text: str):
    print(f"DM: {text}")


def role_tag(name: str, role_assignment: dict) -> str:
    return f"{name}（{role_assignment.get(name, '未知')}）"


def print_player_line(name: str, text: str, role_assignment: dict):
    print(f"{role_tag(name, role_assignment)}: {text}")


def print_action(name: str, text: str, role_assignment: dict):
    print(f"{role_tag(name, role_assignment)}: {text}")


# =========================
# 本地模型 Agent
# =========================

class LocalQwenAgent(ReActAgent):
    def __init__(self, name: str, sys_prompt: str, api_url: str):
        formatter = DashScopeMultiAgentFormatter()
        placeholder_model = DashScopeChatModel(
            model_name="qwen-plus",
            api_key="none",
        )

        super().__init__(
            name=name,
            sys_prompt=sys_prompt,
            model=placeholder_model,
            formatter=formatter,
            memory=InMemoryMemory(),
        )
        self.api_url = api_url

    @trace_reply
    async def go_direct(self, msg: Msg | list[Msg] | None = None) -> Msg | None:
        # 不自动打印，统一交给外层 print_* 控制
        return None

    @trace_reply
    async def reply(self, msg: Msg | list[Msg] | None = None) -> Msg:
        if msg is not None:
            await self.memory.add(msg)

        history = await self.memory.get_memory()
        formatted_messages = await self.formatter.format(history)

        payload = {
            "model": MODEL_NAME,
            "messages": formatted_messages,
            "temperature": 0.5,
        }

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=(30, 300),
            )
            response.raise_for_status()
            data = response.json()
            answer = extract_text_from_response(data)
        except Exception as e:
            answer = f"[模型调用异常] {type(e).__name__}: {e}"

        reply_msg = Msg(self.name, answer, "assistant")
        await self.memory.add(reply_msg)
        return reply_msg


def extract_text_from_response(data) -> str:
    if isinstance(data, dict):
        if "choices" in data:
            try:
                return data["choices"][0]["message"]["content"]
            except Exception:
                pass

        if "output" in data and isinstance(data["output"], str):
            return data["output"]

        if "text" in data and isinstance(data["text"], str):
            return data["text"]

        if "response" in data and isinstance(data["response"], str):
            return data["response"]

        if "message" in data:
            if isinstance(data["message"], str):
                return data["message"]
            if isinstance(data["message"], dict) and "content" in data["message"]:
                return str(data["message"]["content"])

    try:
        return f"[未知返回格式] {json.dumps(data, ensure_ascii=False)}"
    except Exception:
        return f"[未知返回格式] {str(data)}"


# =========================
# 通用工具
# =========================

async def safe_add_memory(agent, msg: Msg):
    if hasattr(agent, "memory") and getattr(agent, "memory") is not None:
        await agent.memory.add(msg)


def alive_names(alive_players):
    return [p.name for p in alive_players]


def find_player(players, name: str):
    for p in players:
        if p.name == name:
            return p
    return None


def assign_roles(players):
    roles = ROLE_DECK[:]
    random.shuffle(roles)
    return {p.name: role for p, role in zip(players, roles)}


def extract_first_number(text: str) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"\d+", text)
    if not m:
        return None
    try:
        return int(m.group())
    except Exception:
        return None


def most_common_or_none(items):
    if not items:
        return None
    return Counter(items).most_common(1)[0][0]


def extract_player_mentions(text: str):
    return re.findall(r"player\d+", text)


def extract_public_claims(name: str, speech: str):
    claims = []

    if "我是预言家" in speech or "我跳预言家" in speech:
        claims.append({
            "type": "claim_seer",
            "speaker": name,
            "text": speech,
        })

    for target in re.findall(r"player\d+", speech):
        if f"{target}是狼人" in speech or f"{target} 是狼人" in speech:
            claims.append({
                "type": "seer_result",
                "speaker": name,
                "target": target,
                "result": "狼人",
                "text": speech,
            })
        elif f"{target}是好人" in speech or f"{target} 是好人" in speech:
            claims.append({
                "type": "seer_result",
                "speaker": name,
                "target": target,
                "result": "好人",
                "text": speech,
            })

    return claims


def build_public_summary(game_state, alive_name_set: set[str]) -> str:
    events = game_state.get("public_events", [])
    if not events:
        return "目前没有明确的公开跳身份、报查验或投票总结。"

    lines = []
    for event in events[-20:]:
        t = event.get("type")
        speaker = event.get("speaker")

        if t == "night_result":
            dead = event.get("dead", [])
            lines.append(f"上一夜公开结果：死亡玩家为 {dead if dead else '无人'}。")

        elif t == "claim_seer":
            if speaker in alive_name_set:
                lines.append(f"{speaker} 白天公开声称自己是预言家。")

        elif t == "seer_result":
            target = event.get("target")
            result = event.get("result")
            if speaker in alive_name_set:
                lines.append(f"{speaker} 白天声称查验 {target} 是{result}。")

        elif t == "vote_out":
            target = event.get("target")
            role = event.get("role")
            lines.append(f"上一轮白天被流放的是 {target}，身份为 {role}。")

    return "\n".join(lines) if lines else "目前没有明确的公开跳身份、报查验或投票总结。"


# =========================
# 角色提示词
# =========================
# 
# def build_role_prompt(role_name: str) -> str:
#     BASE_RULES = (
#         "你正在参加一局狼人杀。\n"
#         "请严格遵守当前阶段要求，不要编造不存在的历史。\n"
#         "白天发言时，只能基于当前公开信息、之前已经出现的发言、死亡信息和投票信息做判断。\n"
#         "不要提及已经死亡或流放的玩家，除非是在复盘公共事件。\n"
#         "不要怀疑自己，也不要投票给自己。\n"
#     )
#     if role_name == "狼人":
#         return (
#             BASE_RULES +
#             "你的角色是【狼人】。\n"
#             "你的胜利条件：让狼人阵营存活人数达到或超过场上其他所有存活玩家人数。\n"
#             "你的策略要求：\n"
#             "1. 夜晚你会与其他狼人讨论并决定击杀目标。优先考虑击杀：真预言家、带队能力强的人、发言最像好人的人，或对狼人威胁最大的玩家。\n"
#             "2. 白天你必须伪装成好人。你的发言要自然，像在认真找狼，而不是一味乱踩人。\n"
#             "3. 可以采用这些伪装策略：\n"
#             "   - 跟随多数人的怀疑方向，避免自己过于突兀；\n"
#             "   - 适度质疑队友，制造不像同伴的感觉；\n"
#             "   - 如果有预言家跳身份，可根据局势选择站边、对跳、倒钩、冲锋，但发言要合理。\n"
#             "4. 不要轻易暴露过强的攻击性，也不要连续发言逻辑混乱。你的核心任务是误导好人投错票。\n"
#             "5. 如果你被多人怀疑，优先自圆其说、转移焦点、指出别人逻辑漏洞，而不是硬认。\n"
#             "输出要求：\n"
#             "1. 夜晚若被要求选择击杀目标，只输出玩家编号数字，例如：3。\n"
#             "2. 白天若被要求投票，只输出玩家编号数字，例如：5。\n"
#             "3. 白天若被要求发言，请输出1到3句话，内容应包含怀疑、判断、站边或投票倾向。\n"
#         )

#     if role_name == "预言家":
#         return (
#             BASE_RULES +
#             "你的角色是【预言家】。\n"
#             "你的胜利条件：帮助好人阵营找出并投出所有狼人。\n"
#             "你的能力：每晚可以查验一名玩家身份，得知其是狼人或好人。\n"
#             "你的策略要求：\n"
#             "1. 夜晚优先查验发言最可疑、带节奏能力强、身份不清但影响局势的人。\n"
#             "2. 白天你的核心任务是建立可信度，并用查验信息带领好人站对边。\n"
#             "3. 如果你选择跳预言家，发言要尽量清晰：报查验、报立场、给出怀疑链条和出人建议。\n"
#             "4. 如果场上出现对跳预言家，你要抓住对方发言漏洞、查验矛盾、视角问题，争取好人信任。\n"
#             "5. 不要只机械报结果，要结合场上发言说明为什么某人更像狼或更像好人。\n"
#             "6. 若局势不明，也可以暂时隐藏身份，但一旦需要带队，应明确表达立场。\n"
#             "输出要求：\n"
#             "1. 夜晚查验时，只输出玩家编号数字，例如：4。\n"
#             "2. 白天投票时，只输出玩家编号数字。\n"
#             "3. 白天发言时，用1到3句话说明你的判断。若适合公开身份，可以说明自己的查验结果，但不要说废话。\n"
#         )

#     if role_name == "女巫":
#         return (
#             BASE_RULES +
#             "你的角色是【女巫】。\n"
#             "你的胜利条件：帮助好人阵营找出并投出所有狼人。\n"
#             "你的能力：你有一瓶解药和一瓶毒药，各只能使用一次。解药可救当夜被杀者，毒药可在夜晚毒杀一名玩家。\n"
#             "你的策略要求：\n"
#             "1. 使用解药要谨慎。优先考虑是否值得救关键身份、强发言玩家，或是否会暴露女巫信息。\n"
#             "2. 使用毒药也要谨慎。只有当你对某人狼人身份有较强把握，或局势需要强行打断狼人节奏时再毒。\n"
#             "3. 白天你不一定要暴露自己是女巫，除非公开信息能帮助好人建立正确判断。\n"
#             "4. 你要结合夜晚信息、死亡信息、白天发言和投票倾向来判断谁更像狼。\n"
#             "5. 不要轻易把自己的药使用情况全部交代清楚，除非这样做对好人明显更有利。\n"
#             "输出要求：\n"
#             "1. 夜晚行动时，只能输出以下格式之一：救 / 不救 / 毒3 / 不行动。\n"
#             "2. 白天投票时，只输出玩家编号数字。\n"
#             "3. 白天发言时，用1到3句话表达你的怀疑、立场和理由。是否公开自己是女巫，由你根据局势判断。\n"
#         )

#     return (
#         BASE_RULES +
#         "你的角色是【村民】。\n"
#         "你的胜利条件：通过白天讨论与投票，帮助好人阵营找出并投出所有狼人。\n"
#         "你的策略要求：\n"
#         "1. 首先你不是狼人，在别人怀疑你时你可以自证也可以怀疑别人。你没有夜晚能力，因此要重点依靠发言逻辑、投票行为、身份跳法和玩家之间的相互关系来找狼。\n"
#         "2. 白天发言时，应尽量给出明确站边和怀疑对象，不要只说模糊废话。\n"
#         "3. 重点关注这些信息：谁发言前后矛盾、谁刻意带节奏、谁在保护可疑对象、谁投票不合逻辑。\n"
#         "4. 如果场上有人跳预言家或女巫，你要判断其发言是否真实可信，而不是盲信。\n"
#         "5. 你的任务不是乱冲，而是通过稳定推理帮助好人阵营形成正确共识。\n"
#         "输出要求：\n"
#         "1. 白天投票时，只输出玩家编号数字，例如：5。\n"
#         "2. 白天发言时，输出1到3句话，尽量包含怀疑对象、理由和投票倾向。\n"
#     )


def build_role_prompt(role_name: str) -> str:
    BASE_RULES = (
        "你正在参加一局狼人杀。\n"
        "请严格遵守当前阶段要求，不要编造不存在的历史。\n"
        "你会被明确告知自己的玩家编号，例如 player3。这个编号就是你自己。\n"
        "白天发言时，只能基于当前公开信息、之前已经出现的发言、死亡信息和投票信息做判断。\n"
        "不要提及已经死亡或流放的玩家，除非是在复盘公共事件。\n"
        "不要怀疑自己，也不要投票给自己。\n"
    )
    if role_name == "狼人":
        return (
            BASE_RULES +
            "你的角色是【狼人】。\n"
            "你的胜利条件：让狼人阵营存活人数达到或超过场上其他所有存活玩家人数。\n"
            "夜晚你会与其他狼人讨论并共同决定击杀对象。\n"
            "优先考虑击杀：真预言家、带队能力强的人、已经报查验的人、明显威胁狼人的人。\n"
            "白天你必须伪装成好人，发言自然，像在认真找狼。\n"
            "如果要求白天发言，请输出 1 到 3 句话。\n"
            "如果要求投票或击杀，只输出玩家编号数字，例如：3。\n"
        )

    if role_name == "预言家":
        return (
            BASE_RULES +
            "你的角色是【预言家】。\n"
            "你的胜利条件：帮助好人阵营找出并投出所有狼人。\n"
            "每晚你可以查验一名玩家，得知其是狼人还是好人。\n"
            "白天你可以根据局势决定是否公开自己身份和查验结果。\n"
            "如果要求白天发言，请输出 1 到 3 句话。\n"
            "如果要求投票或查验，只输出玩家编号数字，例如：4。\n"
        )

    if role_name == "女巫":
        return (
            BASE_RULES +
            "你的角色是【女巫】。\n"
            "你的胜利条件：帮助好人阵营找出并投出所有狼人。\n"
            "你有一瓶解药和一瓶毒药，各只能使用一次。\n"
            "解药可以在夜晚救活被狼刀杀的人，毒药可以在夜晚毒杀你认为是狼人的人\n"
            "夜晚行动时，只能输出：救 / 不救 / 毒 / 不行动 之一。\n"
            "使用毒药时，请带上想到毒杀的玩家编号，比如毒4\n"
            "白天不一定要暴露自己是女巫。\n"
            "白天发言请输出 1 到 3 句话；投票时只输出数字。\n"
        )

    return (
        BASE_RULES +
        "你的角色是【村民】。\n"
        "你的胜利条件：通过白天讨论与投票，帮助好人阵营找出并投出所有狼人。\n"
        "你的策略要求：\n"
        "1. 首先你不是狼人，在别人怀疑你时你可以自证也可以怀疑别人。你没有夜晚能力，因此要重点依靠发言逻辑、投票行为、身份跳法和玩家之间的相互关系来找狼。\n"
        "2. 白天发言时，应尽量给出明确站边和怀疑对象，不要只说模糊废话。\n"
        "3. 重点关注这些信息：谁发言前后矛盾、谁刻意带节奏、谁在保护可疑对象、谁投票不合逻辑。\n"
        "4. 如果场上有人跳预言家或女巫，你要判断其发言是否真实可信，而不是盲信。\n"
        "5. 你的任务不是乱冲，而是通过稳定推理帮助好人阵营形成正确共识。\n"
        "输出要求：\n"
        "1. 白天投票时，只输出玩家编号数字，例如：5。\n"
        "2. 白天发言时，输出1到3句话，尽量包含怀疑对象、理由和投票倾向。\n"
    )


# =========================
# 初始化
# =========================

async def init_agents():
    players = []
    for i in range(PLAYER_COUNT):
        p = LocalQwenAgent(
            name=f"player{i}",
            sys_prompt=(
                f"你是 {f'player{i}'}，正在参加一局狼人杀。"
                "你必须遵守游戏规则和阶段要求。"
                "如果要求发言，保持简短；如果要求投票、击杀或查验，只输出编号数字。"
            ),
            api_url=API_URL,
        )
        players.append(p)

    dm = LocalQwenAgent(
        name="DM",
        sys_prompt=(
            "你是狼人杀主持人。"
            "你负责推进流程、记录公共事件、汇总投票、宣布死亡和胜负。"
            "你不参与玩家推理。"
        ),
        api_url=API_URL,
    )

    return dm, players


async def inject_private_role_info(players, role_assignment):
    for p in players:
        role_name = role_assignment[p.name]

        await safe_add_memory(
            p,
            Msg(
                "DM",
                (
                    f"你的玩家编号是：{p.name}。\n"
                    f"你的身份是：{role_name}。\n"
                    f"请牢记：你就是 {p.name}，不是其他玩家。\n"
                    f"无论在白天发言、夜晚讨论、投票还是技能行动中，"
                    f"都不能把自己当作怀疑对象，也不能投票给自己。"
                ),
                "system",
            ),
        )

        await safe_add_memory(
            p,
            Msg("DM", build_role_prompt(role_name), "system"),
        )


# =========================
# 输入约束助手
# =========================

async def ask_for_target_number(agent, prompt_text: str, valid_targets: set[str], max_retries: int = MAX_RETRY):
    for _ in range(max_retries):
        resp = await agent(Msg("DM", prompt_text, "system"))
        num = extract_first_number(resp.content)
        if num is None:
            await safe_add_memory(agent, Msg("DM", "请只输出玩家编号数字，例如：3", "system"))
            continue

        target = f"player{num}"
        if target in valid_targets:
            return target

        await safe_add_memory(
            agent,
            Msg("DM", f"目标无效。当前合法目标只有：{sorted(valid_targets)}。请只输出数字。", "system"),
        )
    return None


async def ask_for_short_speech(agent, prompt_text: str, max_retries: int = MAX_RETRY):
    for _ in range(max_retries):
        resp = await agent(Msg("DM", prompt_text, "system"))
        text = resp.content.strip()

        if re.fullmatch(r"\d+", text):
            await safe_add_memory(
                agent,
                Msg("DM", "当前是发言阶段，不是投票阶段。请用 1 到 3 句话表达看法，不要只输出数字。", "system"),
            )
            continue

        if text:
            return text

    return "我暂时没有更多确定信息，先继续观察。"


async def ask_witch_action(witch_agent, killed_target_name: Optional[str], antidote_available: bool,
                           poison_available: bool, alive_name_set: set[str], max_retries: int = MAX_RETRY):
    prompt = (
        f"现在是夜晚。今晚被狼人击杀的玩家是：{killed_target_name if killed_target_name else '无人'}。\n"
        f"你的救药是否可用：{'是' if antidote_available else '否'}。\n"
        f"你的毒药是否可用：{'是' if poison_available else '否'}。\n"
        f"请只用以下格式之一回答：救 / 不救 / 毒 / 不行动"
    )

    for _ in range(max_retries):
        resp = await witch_agent(Msg("DM", prompt, "system"))
        content = resp.content.strip()

        if content == "救":
            if antidote_available and killed_target_name is not None:
                return {"save": True, "poison_target": None}
            await safe_add_memory(witch_agent, Msg("DM", "当前不能使用救药，请重新选择。", "system"))
            continue

        if content == "不救":
            return {"save": False, "poison_target": None}

        if content == "不行动":
            return {"save": False, "poison_target": None}

        if content.startswith("毒"):
            if not poison_available:
                await safe_add_memory(witch_agent, Msg("DM", "你的毒药已经用完，请重新选择。", "system"))
                continue

            num = extract_first_number(content)
            if num is None:
                await safe_add_memory(witch_agent, Msg("DM", "格式错误。示例：毒3", "system"))
                continue

            target = f"player{num}"
            if target not in alive_name_set:
                await safe_add_memory(witch_agent, Msg("DM", "毒杀目标无效，请重新选择。", "system"))
                continue

            return {"save": False, "poison_target": target}

        await safe_add_memory(witch_agent, Msg("DM", "回答无效。只允许：救 / 不救 / 毒 / 不行动", "system"))

    return {"save": False, "poison_target": None}


# =========================
# 胜负判断
# =========================

def judge_winner(alive_players, role_assignment):
    alive_name_set = {p.name for p in alive_players}

    alive_wolves = [n for n in alive_name_set if role_assignment[n] == "狼人"]
    alive_gods = [n for n in alive_name_set if role_assignment[n] in {"预言家", "女巫"}]
    alive_villagers = [n for n in alive_name_set if role_assignment[n] == "村民"]

    # 狼人死光，好人赢
    if len(alive_wolves) == 0:
        return "good"

    # 屠神：神职死光，狼人赢
    if len(alive_gods) == 0:
        return "wolf"

    # 屠民：村民死光，狼人赢
    if len(alive_villagers) == 0:
        return "wolf"

    return None


# =========================
# 夜晚阶段
# =========================

async def wolf_discussion_phase(wolves, alive_players, role_assignment, game_state):
    if not wolves:
        return []

    print("\n【狼人讨论】")

    alive_name_set = set(alive_names(alive_players))
    public_summary = build_public_summary(game_state, alive_name_set)
    discussion_records = []

    async with MsgHub(
        participants=wolves,
        announcement=Msg(
            "DM",
            "现在是夜晚狼人讨论阶段。你们都是狼人，可以共享观点并讨论今晚刀谁。",
            "system",
        ),
    ):
        for wolf in wolves:
            speech = await ask_for_short_speech(
                wolf,
                f"当前存活玩家：{sorted(alive_name_set)}。\n"
                f"白天关键公开信息如下：\n{public_summary}\n"
                "请结合这些信息发言，分析谁像真预言家、谁在带队、今晚优先刀谁。1 到 3 句话。",
            )
            discussion_records.append((wolf.name, speech))
            print_player_line(wolf.name, speech, role_assignment)

    return discussion_records


async def wolf_vote_phase(wolves, alive_players, role_assignment, game_state):
    alive_name_set = set(alive_names(alive_players))
    wolf_names = {w.name for w in wolves}
    valid_targets = alive_name_set - wolf_names

    if not wolves or not valid_targets:
        return None

    print("\n【狼人投票】")

    public_summary = build_public_summary(game_state, alive_name_set)
    votes = []

    for wolf in wolves:
        target = await ask_for_target_number(
            wolf,
            f"当前合法击杀目标：{sorted(valid_targets)}。\n"
            f"白天关键公开信息如下：\n{public_summary}\n"
            "请结合白天跳预言家、报查验、带队投票的人，决定今晚击杀对象。只输出玩家编号数字。",
            valid_targets,
        )
        if target is not None:
            votes.append(target)
            print_action(wolf.name, f"刀 {target}", role_assignment)

    target = most_common_or_none(votes)
    if target is None:
        target = random.choice(list(valid_targets))
        print_dm(f"狼人投票无有效结果，系统随机指定击杀目标：{target}")
    else:
        print_dm(f"狼人最终击杀目标：{target}")

    return target


async def seer_action_phase(alive_players, role_assignment, game_state):
    seers = [p for p in alive_players if role_assignment[p.name] == "预言家"]
    if not seers:
        return None

    seer = seers[0]
    alive_name_set = set(alive_names(alive_players))
    valid_targets = alive_name_set - {seer.name}
    if not valid_targets:
        return None

    print("\n【预言家行动】")

    target = await ask_for_target_number(
        seer,
        f"现在是夜晚。当前可查验玩家：{sorted(valid_targets)}。请只输出玩家编号数字。",
        valid_targets,
    )
    if target is None:
        print_action(seer.name, "查验失败", role_assignment)
        return None

    result = "狼人" if role_assignment[target] == "狼人" else "好人"

    print_action(seer.name, f"查验 {target}", role_assignment)
    print_dm(f"{role_tag(seer.name, role_assignment)} 查验结果：{target} 是{result}")

    await safe_add_memory(seer, Msg("DM", f"查验结果：{target} 的身份是【{result}】。", "system"))

    game_state["private_events"].append({
        "type": "seer_check",
        "round_id": game_state["round_id"],
        "seer": seer.name,
        "target": target,
        "result": result,
    })

    return {"seer": seer.name, "target": target, "result": result}


async def witch_action_phase(alive_players, role_assignment, killed_target_name, game_state):
    witches = [p for p in alive_players if role_assignment[p.name] == "女巫"]
    if not witches:
        return {"save": False, "poison_target": None}

    witch = witches[0]
    alive_name_set = set(alive_names(alive_players))

    print("\n【女巫行动】")
    action = await ask_witch_action(
        witch_agent=witch,
        killed_target_name=killed_target_name,
        antidote_available=game_state["witch_antidote_available"],
        poison_available=game_state["witch_poison_available"],
        alive_name_set=alive_name_set,
    )

    if action["save"] and game_state["witch_antidote_available"]:
        game_state["witch_antidote_available"] = False
        print_action(witch.name, "救", role_assignment)

    elif action["poison_target"] is not None and game_state["witch_poison_available"]:
        game_state["witch_poison_available"] = False
        print_action(witch.name, f"毒 {action['poison_target']}", role_assignment)

    else:
        print_action(witch.name, "不行动", role_assignment)

    return action


async def run_night_round(dm, players, alive_players, role_assignment, game_state):
    print_section("天黑请闭眼")

    wolves = [p for p in alive_players if role_assignment[p.name] == "狼人"]

    await wolf_discussion_phase(wolves, alive_players, role_assignment, game_state)
    wolf_kill_target = await wolf_vote_phase(wolves, alive_players, role_assignment, game_state)
    await seer_action_phase(alive_players, role_assignment, game_state)
    witch_action = await witch_action_phase(alive_players, role_assignment, wolf_kill_target, game_state)

    final_deaths = set()
    if wolf_kill_target is not None:
        final_deaths.add(wolf_kill_target)

    if witch_action["save"] and wolf_kill_target is not None:
        final_deaths.discard(wolf_kill_target)

    if witch_action["poison_target"] is not None:
        final_deaths.add(witch_action["poison_target"])

    dead_players = [p for p in alive_players if p.name in final_deaths]
    new_alive_players = [p for p in alive_players if p.name not in final_deaths]

    dead_names = [p.name for p in dead_players]
    game_state["public_events"].append({
        "type": "night_result",
        "round_id": game_state["round_id"],
        "dead": dead_names,
    })

    if dead_players:
        print_dm(f"夜晚结束。死亡玩家：{'、'.join(dead_names)}")
    else:
        print_dm("夜晚结束。无人死亡。")

    return dead_players, new_alive_players


# =========================
# 白天阶段
# =========================

async def day_speech_phase(alive_players, dead_players, role_assignment, game_state):
    print("\n【白天发言】")

    alive_name_set = set(alive_names(alive_players))
    dead_text = "、".join([p.name for p in dead_players]) if dead_players else "无人"
    public_summary = build_public_summary(game_state, alive_name_set)

    speeches = []

    async with MsgHub(
        participants=alive_players,
        announcement=Msg(
            "DM",
            f"现在天亮了。昨夜死亡玩家：{dead_text}。现在进入白天讨论阶段。",
            "system",
        ),
    ):
        for speaker in alive_players:
            speech = await ask_for_short_speech(
                speaker,
                f"当前是第 {game_state['round_id']} 轮白天。\n"
                f"当前存活玩家：{sorted(alive_name_set)}。\n"
                f"公开关键信息总结：\n{public_summary}\n"
                "请轮到你发言。你只能基于已经公开的信息发言，不能编造前几轮不存在的信息。"
                "不能怀疑自己。请用 1 到 3 句话表达你的怀疑、判断或站边。",
            )
            speeches.append((speaker.name, speech))
            print_player_line(speaker.name, speech, role_assignment)

            claims = extract_public_claims(speaker.name, speech)
            for c in claims:
                c["round_id"] = game_state["round_id"]
                game_state["public_events"].append(c)

    return speeches


async def day_vote_phase(alive_players, role_assignment, game_state):
    print_section("开始投票")

    alive_name_set = set(alive_names(alive_players))
    public_summary = build_public_summary(game_state, alive_name_set)

    votes = []

    for voter in alive_players:
        valid_targets = alive_name_set - {voter.name}
        if not valid_targets:
            continue

        target = await ask_for_target_number(
            voter,
            f"当前存活玩家：{sorted(alive_name_set)}。\n"
            f"你不能投自己。当前合法投票目标：{sorted(valid_targets)}。\n"
            f"公开关键信息总结：\n{public_summary}\n"
            "请投票给你认为最可能是狼人的玩家，只输出玩家编号数字。",
            valid_targets,
        )
        if target is not None:
            votes.append(target)
            print_player_line(voter.name, target.replace("player", ""), role_assignment)

    print_section("投票结束")

    if not votes:
        print_dm("本轮无人成功投票，流放失败。")
        return None

    target = most_common_or_none(votes)
    print_dm(f"本轮得票最高者：{target}")
    return target


async def run_day_round(dm, alive_players, dead_players, role_assignment, game_state):
    print_section("天亮了")

    dead_text = "、".join([p.name for p in dead_players]) if dead_players else "无人"
    print_dm(f"昨夜死亡：{dead_text}")

    await day_speech_phase(alive_players, dead_players, role_assignment, game_state)
    executed_name = await day_vote_phase(alive_players, role_assignment, game_state)

    if executed_name is None:
        return alive_players, None

    executed_player = find_player(alive_players, executed_name)
    if executed_player is None:
        print_dm("投票结果异常，本轮流放失败。")
        return alive_players, None

    new_alive_players = [p for p in alive_players if p.name != executed_name]

    game_state["public_events"].append({
        "type": "vote_out",
        "round_id": game_state["round_id"],
        "target": executed_name,
        "role": role_assignment[executed_name],
    })

    print_dm(f"{role_tag(executed_name, role_assignment)} 被流放。")
    return new_alive_players, executed_player


# =========================
# 主流程
# =========================

async def run_conversation():
    agentscope.init(project="rolePlay_test")

    dm, players = await init_agents()
    role_assignment = assign_roles(players)

    print("角色分配结果：", role_assignment)

    await inject_private_role_info(players, role_assignment)
    await safe_add_memory(dm, Msg("DM", f"全局身份表：{json.dumps(role_assignment, ensure_ascii=False)}", "system"))

    game_state = {
        "round_id": 1,
        "witch_antidote_available": True,
        "witch_poison_available": True,
        "public_events": [],
        "private_events": [],
    }

    print_dm(
        f"游戏开始。共有 {PLAYER_COUNT} 名玩家，其中 "
        f"{ROLE_DECK.count('狼人')} 名狼人、"
        f"{ROLE_DECK.count('预言家')} 名预言家、"
        f"{ROLE_DECK.count('女巫')} 名女巫、"
        f"{ROLE_DECK.count('村民')} 名村民。现在进入第一夜。"
    )

    alive_players = list(players)

    while True:
        print(f"\n========== 第 {game_state['round_id']} 轮 ==========")

        winner = judge_winner(alive_players, role_assignment)
        if winner is not None:
            break

        dead_players, alive_players = await run_night_round(
            dm=dm,
            players=players,
            alive_players=alive_players,
            role_assignment=role_assignment,
            game_state=game_state,
        )

        winner = judge_winner(alive_players, role_assignment)
        if winner is not None:
            break

        alive_players, executed_player = await run_day_round(
            dm=dm,
            alive_players=alive_players,
            dead_players=dead_players,
            role_assignment=role_assignment,
            game_state=game_state,
        )

        if executed_player is not None:
            print(f"白天被流放：{executed_player.name}，身份：{role_assignment[executed_player.name]}")

        winner = judge_winner(alive_players, role_assignment)
        if winner is not None:
            break

        alive_info = {p.name: role_assignment[p.name] for p in alive_players}
        print("当前存活：", alive_info)

        game_state["round_id"] += 1

    print_section("游戏结束")
    if winner == "good":
        print_dm("好人阵营获胜！")
    else:
        print_dm("狼人阵营获胜！")

    print("\n【最终身份表】")
    for p in players:
        print(f"{p.name} -> {role_assignment[p.name]}")


if __name__ == "__main__":
    asyncio.run(run_conversation())
# -*- coding: utf-8 -*-
"""Super Agent BETA for AgentScope official example workspace.

This example intentionally follows the patterns shown in:
- examples/quickstart_agent.py
- examples/task_tool.py
- examples/workflow_handoffs.py
- examples/workflow_routing.py

Capabilities:
- Web search for public information such as weather
- View / write / insert text file
- Execute shell command
- Delegate XiaoHongShu drafting and one-click posting

Run example:
    python examples/super_agent_beta.py --task "帮我搜索上海明天的天气，并保存成 markdown 摘要"

Environment examples:
    set DASHSCOPE_API_KEY=...
    set SUPER_AGENT_MODEL_PROVIDER=dashscope

    set OPENAI_API_KEY=...
    set SUPER_AGENT_MODEL_PROVIDER=openai
    set OPENAI_BASE_URL=https://api.openai.com/v1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from agentscope import tool as agentscope_tool
from agentscope.agent import ReActAgent
from agentscope.formatter import DashScopeChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg, TextBlock
from agentscope.model import DashScopeChatModel, OpenAIChatModel
from agentscope.tool import ToolResponse, Toolkit


WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_DIR = WORKSPACE_ROOT / "examples" / "runtime"
XIAOHONGSHU_POSTER = Path(__file__).with_name("xiaohongshu_playwright_poster.py")
DEFAULT_XIAOHONGSHU_PAYLOAD = (
    WORKSPACE_ROOT / "examples" / "xiaohongshu_post_payload.example.json"
)


def _tool(name: str):
    """Safely get a builtin tool function from agentscope.tool."""
    return getattr(agentscope_tool, name, None)


VIEW_TEXT_FILE = _tool("view_text_file")
WRITE_TEXT_FILE = _tool("write_text_file")
INSERT_TEXT_FILE = _tool("insert_text_file")
EXECUTE_SHELL_COMMAND = _tool("execute_shell_command")


@dataclass(slots=True)
class SuperAgentToolConfig:
    enable_browser_mcp: bool = False
    enable_websearch: bool = True
    enable_view_text_file: bool = True
    enable_write_text_file: bool = True
    enable_insert_text_file: bool = True
    enable_execute_shell_command: bool = True


DEFAULT_TOOL_CONFIG = SuperAgentToolConfig()
ACTIVE_TOOL_CONFIG = DEFAULT_TOOL_CONFIG


def _ensure_runtime_dir() -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)


def _split_lines_or_csv(value: str) -> list[str]:
    if not value.strip():
        return []
    parts = re.split(r"[\r\n,;]+", value)
    return [part.strip() for part in parts if part.strip()]


def _extract_text_blocks(msg: Msg) -> list[TextBlock]:
    blocks = msg.get_content_blocks("text")
    if blocks:
        return blocks
    text = getattr(msg, "content", "")
    if isinstance(text, str):
        return [TextBlock(type="text", text=text)]
    return [TextBlock(type="text", text=str(text))]


def build_model():
    """Create a chat model from environment variables."""
    provider = os.environ.get("SUPER_AGENT_MODEL_PROVIDER", "dashscope").lower()

    if provider == "openai":
        api_key = os.environ["OPENAI_API_KEY"]
        model_name = os.environ.get("OPENAI_MODEL_NAME", "gpt-4.1-mini")
        client_kwargs = {}
        base_url = os.environ.get("OPENAI_BASE_URL")
        if base_url:
            client_kwargs["base_url"] = base_url
        return OpenAIChatModel(
            model_name=model_name,
            api_key=api_key,
            stream=True,
            client_kwargs=client_kwargs or None,
        )

    api_key = os.environ["DASHSCOPE_API_KEY"]
    model_name = os.environ.get("DASHSCOPE_MODEL_NAME", "qwen-max")
    return DashScopeChatModel(
        model_name=model_name,
        api_key=api_key,
        stream=True,
        enable_thinking=False,
    )


def _http_json(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    payload: dict | None = None,
) -> dict:
    data = None
    request_headers = headers.copy() if headers else {}

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")

    request = urllib.request.Request(
        url=url,
        data=data,
        headers=request_headers,
        method=method,
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _fallback_duckduckgo_search(query: str, max_results: int) -> str:
    url = f"https://duckduckgo.com/html/?q={urllib.parse.quote_plus(query)}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        html = response.read().decode("utf-8", errors="ignore")

    matches = re.findall(
        r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not matches:
        return "未解析到搜索结果。建议配置 TAVILY_API_KEY 或 SERPER_API_KEY 提升稳定性。"

    lines = []
    for index, (link, title_html) in enumerate(matches[:max_results], start=1):
        title = re.sub(r"<.*?>", "", title_html)
        title = re.sub(r"\s+", " ", title).strip()
        lines.append(f"{index}. {title}\n链接: {link}")
    return "\n\n".join(lines)


def web_search(query: str, max_results: int = 5) -> ToolResponse:
    """Search the public web and return concise text snippets.

    Args:
        query (str):
            Search query, for example "上海 明天 天气" or "AgentScope toolkit example".
        max_results (int):
            Maximum number of results to return. Keep this small for concise answers.
    """
    try:
        if os.environ.get("TAVILY_API_KEY"):
            data = _http_json(
                "https://api.tavily.com/search",
                method="POST",
                payload={
                    "api_key": os.environ["TAVILY_API_KEY"],
                    "query": query,
                    "max_results": max_results,
                    "topic": "general",
                },
            )
            lines = []
            for index, item in enumerate(data.get("results", [])[:max_results], start=1):
                lines.append(
                    "\n".join(
                        [
                            f"{index}. {item.get('title', '').strip()}",
                            f"链接: {item.get('url', '').strip()}",
                            f"摘要: {item.get('content', '').strip()}",
                        ],
                    ),
                )
            text = "\n\n".join(lines) or "没有拿到 Tavily 搜索结果。"
        elif os.environ.get("SERPER_API_KEY"):
            data = _http_json(
                "https://google.serper.dev/search",
                method="POST",
                headers={"X-API-KEY": os.environ["SERPER_API_KEY"]},
                payload={"q": query, "num": max_results},
            )
            lines = []
            for index, item in enumerate(data.get("organic", [])[:max_results], start=1):
                lines.append(
                    "\n".join(
                        [
                            f"{index}. {item.get('title', '').strip()}",
                            f"链接: {item.get('link', '').strip()}",
                            f"摘要: {item.get('snippet', '').strip()}",
                        ],
                    ),
                )
            answer_box = data.get("answerBox") or {}
            if answer_box:
                lines.insert(0, f"快速答案: {json.dumps(answer_box, ensure_ascii=False)}")
            text = "\n\n".join(lines) or "没有拿到 Serper 搜索结果。"
        else:
            text = _fallback_duckduckgo_search(query, max_results)
    except Exception as exc:
        text = f"网页搜索失败: {exc}"

    return ToolResponse(content=[TextBlock(type="text", text=text)])


def save_xiaohongshu_post_payload(
    title: str,
    content: str,
    output_file: str = "",
    image_paths: str = "",
    topics: str = "",
    mode: str = "draft",
    location: str = "",
) -> ToolResponse:
    """Save a XiaoHongShu post payload as JSON for later automated posting.

    Args:
        title (str):
            Post title.
        content (str):
            Main post body.
        output_file (str):
            Output json path. Empty means examples/runtime/xiaohongshu_post.json.
        image_paths (str):
            Image file paths separated by comma, semicolon, or new line.
        topics (str):
            Topics separated by comma, semicolon, or new line.
        mode (str):
            "draft" or "publish".
        location (str):
            Optional location text.
    """
    _ensure_runtime_dir()
    payload_path = Path(output_file).expanduser() if output_file else RUNTIME_DIR / "xiaohongshu_post.json"
    payload_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "title": title.strip(),
        "content": content.strip(),
        "images": _split_lines_or_csv(image_paths),
        "topics": _split_lines_or_csv(topics),
        "mode": mode.strip().lower() or "draft",
        "location": location.strip(),
    }
    payload_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    text = (
        "已生成小红书发帖载荷文件。\n"
        f"payload_file: {payload_path}\n"
        f"mode: {payload['mode']}\n"
        f"images_count: {len(payload['images'])}\n"
        f"topics_count: {len(payload['topics'])}"
    )
    return ToolResponse(content=[TextBlock(type="text", text=text)])


async def run_xiaohongshu_playwright(
    payload_file: str = "",
    mode: str = "draft",
    headless: bool = False,
    user_data_dir: str = "",
    publish_url: str = "https://creator.xiaohongshu.com/publish/publish",
) -> ToolResponse:
    """Run the bundled Playwright script to draft or publish a XiaoHongShu post.

    Args:
        payload_file (str):
            JSON payload file path. Empty means the bundled example payload.
        mode (str):
            "draft" or "publish".
        headless (bool):
            Whether to run the browser in headless mode.
        user_data_dir (str):
            Persistent browser profile directory for keeping the login session.
        publish_url (str):
            Publish page URL. Adjust if XiaoHongShu changes the creator URL.
    """
    payload_path = (
        Path(payload_file).expanduser()
        if payload_file
        else DEFAULT_XIAOHONGSHU_PAYLOAD
    )
    profile_path = (
        Path(user_data_dir).expanduser()
        if user_data_dir
        else RUNTIME_DIR / ".xhs_chromium_profile"
    )

    cmd = [
        sys.executable,
        str(XIAOHONGSHU_POSTER),
        "--payload-file",
        str(payload_path),
        "--mode",
        mode.strip().lower() or "draft",
        "--user-data-dir",
        str(profile_path),
        "--publish-url",
        publish_url,
    ]
    if headless:
        cmd.append("--headless")

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(WORKSPACE_ROOT),
    )
    stdout, stderr = await process.communicate()
    output = stdout.decode("utf-8", errors="ignore").strip()
    err_output = stderr.decode("utf-8", errors="ignore").strip()

    if process.returncode != 0:
        text = (
            f"小红书自动发帖脚本执行失败，退出码 {process.returncode}。\n"
            f"stdout:\n{output or '(empty)'}\n\n"
            f"stderr:\n{err_output or '(empty)'}"
        )
    else:
        text = (
            "小红书自动发帖脚本执行完成。\n"
            f"stdout:\n{output or '(empty)'}\n\n"
            f"stderr:\n{err_output or '(empty)'}"
        )
    return ToolResponse(content=[TextBlock(type="text", text=text)])


def _register_tool_if_enabled(
    toolkit: Toolkit,
    enabled: bool,
    func,
    *,
    group_name: str,
) -> None:
    if enabled and func is not None:
        toolkit.register_tool_function(func, group_name=group_name)


def build_office_toolkit(config: SuperAgentToolConfig) -> Toolkit:
    """Create a Toolkit with the requested office tools."""
    toolkit = Toolkit()

    toolkit.create_tool_group(
        group_name="office_info",
        description="Information lookup tools.",
        active=config.enable_websearch,
        notes="Use these tools for weather, public info, and quick online lookup.",
    )
    toolkit.create_tool_group(
        group_name="office_files",
        description="Text file viewing and editing tools.",
        active=(
            config.enable_view_text_file
            or config.enable_write_text_file
            or config.enable_insert_text_file
        ),
        notes="Prefer reading before writing, and only edit the target file that is needed.",
    )
    toolkit.create_tool_group(
        group_name="office_system",
        description="Local command execution tools.",
        active=config.enable_execute_shell_command,
        notes="Prefer safe, inspectable commands and keep output concise.",
    )
    toolkit.create_tool_group(
        group_name="xhs_publish",
        description="XiaoHongShu drafting and publishing tools.",
        active=True,
        notes="Use draft mode first. Publish only when the content and images are ready.",
    )

    _register_tool_if_enabled(
        toolkit,
        config.enable_websearch,
        web_search,
        group_name="office_info",
    )
    _register_tool_if_enabled(
        toolkit,
        config.enable_view_text_file,
        VIEW_TEXT_FILE,
        group_name="office_files",
    )
    _register_tool_if_enabled(
        toolkit,
        config.enable_write_text_file,
        WRITE_TEXT_FILE,
        group_name="office_files",
    )
    _register_tool_if_enabled(
        toolkit,
        config.enable_insert_text_file,
        INSERT_TEXT_FILE,
        group_name="office_files",
    )
    _register_tool_if_enabled(
        toolkit,
        config.enable_execute_shell_command,
        EXECUTE_SHELL_COMMAND,
        group_name="office_system",
    )

    toolkit.register_tool_function(
        save_xiaohongshu_post_payload,
        group_name="xhs_publish",
    )
    toolkit.register_tool_function(
        run_xiaohongshu_playwright,
        group_name="xhs_publish",
    )

    if config.enable_browser_mcp:
        toolkit.create_tool_group(
            group_name="browser_mcp",
            description="Reserved group for browser MCP integration.",
            active=True,
            notes="Browser MCP is enabled in config, but you still need to register a real MCP client.",
        )

    return toolkit


def build_specialist_agent(
    *,
    name: str,
    sys_prompt: str,
    config: SuperAgentToolConfig,
) -> ReActAgent:
    return ReActAgent(
        name=name,
        sys_prompt=sys_prompt,
        model=build_model(),
        formatter=DashScopeChatFormatter(),
        toolkit=build_office_toolkit(config),
        memory=InMemoryMemory(),
    )


async def delegate_office_task(task_description: str) -> ToolResponse:
    """Delegate a practical office task to a specialist agent.

    Args:
        task_description (str):
            A concrete office task such as search, summarization, file editing, or command execution.
    """
    specialist = build_specialist_agent(
        name="OfficeSpecialist",
        sys_prompt=(
            "你是一名基础办公智能体，负责信息检索、天气查询、文档整理、"
            "文本文件编辑、命令执行与结果交付。"
            "当需要修改文件时，先阅读上下文，再进行最小修改。"
        ),
        config=ACTIVE_TOOL_CONFIG,
    )
    result = await specialist(Msg("user", task_description, "user"))
    return ToolResponse(content=_extract_text_blocks(result))


async def delegate_xiaohongshu_task(task_description: str) -> ToolResponse:
    """Delegate a XiaoHongShu content or publishing task to a specialist agent.

    Args:
        task_description (str):
            Tasks such as topic research, post drafting, payload generation, or one-click posting.
    """
    specialist = build_specialist_agent(
        name="XiaoHongShuSpecialist",
        sys_prompt=(
            "你是一名小红书运营智能体，负责选题调研、文案生成、"
            "标签整理、发帖 payload 生成与自动发帖执行。"
            "默认先走 draft 模式；只有当任务明确要求发布，且素材准备完毕时，才走 publish 模式。"
        ),
        config=ACTIVE_TOOL_CONFIG,
    )
    result = await specialist(Msg("user", task_description, "user"))
    return ToolResponse(content=_extract_text_blocks(result))


def build_super_agent(config: SuperAgentToolConfig = DEFAULT_TOOL_CONFIG) -> ReActAgent:
    """Create the top-level BETA super agent."""
    global ACTIVE_TOOL_CONFIG
    ACTIVE_TOOL_CONFIG = config

    toolkit = Toolkit()
    toolkit.register_tool_function(delegate_office_task)
    toolkit.register_tool_function(delegate_xiaohongshu_task)

    if config.enable_websearch:
        toolkit.register_tool_function(web_search)
    if config.enable_view_text_file and VIEW_TEXT_FILE is not None:
        toolkit.register_tool_function(VIEW_TEXT_FILE)
    if config.enable_write_text_file and WRITE_TEXT_FILE is not None:
        toolkit.register_tool_function(WRITE_TEXT_FILE)
    if config.enable_insert_text_file and INSERT_TEXT_FILE is not None:
        toolkit.register_tool_function(INSERT_TEXT_FILE)
    if config.enable_execute_shell_command and EXECUTE_SHELL_COMMAND is not None:
        toolkit.register_tool_function(EXECUTE_SHELL_COMMAND)
    toolkit.register_tool_function(save_xiaohongshu_post_payload)
    toolkit.register_tool_function(run_xiaohongshu_playwright)

    sys_prompt = f"""
你是 SuperAgent BETA。

你的定位：
1. 处理基础办公任务，包括网页搜索、天气查询、文件查看、写作整理、文本文件修改、shell 命令执行。
2. 处理小红书任务，包括选题调研、文案生成、发帖 payload 生成，以及调用本地 Playwright 脚本自动发帖。

当前工具配置：
{json.dumps(asdict(config), ensure_ascii=False, indent=2)}

工作原则：
1. 简单任务可以直接回答或直接调用工具。
2. 复杂办公任务优先调用 delegate_office_task。
3. 小红书任务优先调用 delegate_xiaohongshu_task。
4. 涉及真实发帖时，默认先保存草稿；只有用户明确要求发布时才 publish。
5. enable_browser_mcp 当前为 {config.enable_browser_mcp}，不要假装具备 Browser MCP 能力。
6. 做文件修改或命令执行前，先说清楚要做什么，再执行。
""".strip()

    return ReActAgent(
        name="SuperAgentBETA",
        sys_prompt=sys_prompt,
        model=build_model(),
        formatter=DashScopeChatFormatter(),
        toolkit=toolkit,
        memory=InMemoryMemory(),
    )


async def run_task(task: str) -> None:
    agent = build_super_agent()
    result = await agent(Msg("user", task, "user"))
    print(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AgentScope Super Agent BETA example.")
    parser.add_argument(
        "--task",
        default=(
            "先用网页搜索帮我查北京明天的天气，再输出一个 5 行内的中文摘要。"
        ),
        help="Task to send to the super agent.",
    )
    return parser.parse_args()


def main() -> None:
    _ensure_runtime_dir()
    args = parse_args()
    asyncio.run(run_task(args.task))


if __name__ == "__main__":
    main()

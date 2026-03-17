# -*- coding: utf-8 -*-
"""Playwright-based XiaoHongShu posting example.

This script is intentionally standalone so it can be:
1. run directly for debugging;
2. called by examples/super_agent_beta.py;
3. triggered through AgentScope's execute_shell_command tool.

Typical usage:
    python examples/xiaohongshu_playwright_poster.py ^
      --payload-file examples/xiaohongshu_post_payload.example.json ^
      --mode draft ^
      --user-data-dir examples/runtime/.xhs_chromium_profile

Notes:
- First run is usually non-headless so you can complete login manually.
- Selectors on XiaoHongShu may change over time; update the selector lists below if needed.
- Use only on your own account and follow XiaoHongShu platform rules.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path


DEFAULT_PUBLISH_URL = "https://creator.xiaohongshu.com/publish/publish"
ARTIFACT_DIR = Path(__file__).resolve().parent / "runtime" / "xhs_artifacts"

TITLE_SELECTORS = [
    "input[placeholder*='标题']",
    "textarea[placeholder*='标题']",
    "[contenteditable='true'][data-placeholder*='标题']",
]
CONTENT_SELECTORS = [
    "textarea[placeholder*='正文']",
    "div[contenteditable='true']",
    "[contenteditable='true'][data-placeholder*='正文']",
    "[contenteditable='true'][data-placeholder*='添加正文']",
]
UPLOAD_SELECTORS = [
    "input[type='file']",
]
DRAFT_BUTTON_SELECTORS = [
    "button:has-text('保存到草稿')",
    "button:has-text('保存草稿')",
    "text=保存到草稿",
    "text=保存草稿",
]
PUBLISH_BUTTON_SELECTORS = [
    "button:has-text('发布')",
    "text=发布",
]
LOCATION_SELECTORS = [
    "input[placeholder*='地点']",
    "input[placeholder*='位置']",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automate XiaoHongShu web posting with Playwright.")
    parser.add_argument("--payload-file", required=True, help="Path to the payload JSON file.")
    parser.add_argument("--mode", choices=["draft", "publish"], default="draft")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--user-data-dir",
        required=True,
        help="Persistent Chromium profile directory.",
    )
    parser.add_argument("--publish-url", default=DEFAULT_PUBLISH_URL)
    return parser.parse_args()


def load_payload(path: str) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    payload.setdefault("title", "")
    payload.setdefault("content", "")
    payload.setdefault("images", [])
    payload.setdefault("topics", [])
    payload.setdefault("location", "")
    return payload


def build_content(payload: dict) -> str:
    lines = [payload["content"].strip()]
    topics = [topic.strip().lstrip("#") for topic in payload.get("topics", []) if topic.strip()]
    if topics:
        lines.append("")
        lines.append(" ".join(f"#{topic}" for topic in topics))
    return "\n".join(lines).strip()


async def wait_for_first_locator(page, selectors: list[str], timeout_ms: int = 2000):
    for selector in selectors:
        locator = page.locator(selector).first
        try:
            await locator.wait_for(timeout=timeout_ms)
            return locator, selector
        except Exception:
            continue
    return None, None


async def fill_field(page, locator, value: str) -> None:
    await locator.click()
    try:
        await locator.fill(value)
        return
    except Exception:
        pass

    await page.keyboard.press("Control+A")
    await page.keyboard.type(value)


async def set_input_files(locator, files: list[str]) -> None:
    await locator.set_input_files(files)


async def maybe_manual_login(page) -> None:
    if "login" not in page.url.lower():
        return

    print("检测到登录页，请在打开的浏览器中完成登录。登录完成后回到终端按 Enter 继续。")
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, input)


async def post_note(args: argparse.Namespace) -> None:
    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise RuntimeError(
            "缺少 playwright，请先安装: pip install playwright && playwright install chromium"
        ) from exc

    payload = load_payload(args.payload_file)
    payload_mode = payload.get("mode", args.mode)
    mode = args.mode or payload_mode or "draft"

    Path(args.user_data_dir).mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as playwright:
        context = await playwright.chromium.launch_persistent_context(
            user_data_dir=args.user_data_dir,
            headless=args.headless,
            viewport={"width": 1440, "height": 960},
        )
        page = context.pages[0] if context.pages else await context.new_page()

        await page.goto(args.publish_url, wait_until="domcontentloaded")
        await maybe_manual_login(page)
        await page.goto(args.publish_url, wait_until="domcontentloaded")
        await page.wait_for_timeout(3000)

        image_files = [str(Path(image).expanduser()) for image in payload.get("images", [])]
        if image_files:
            upload_locator, upload_selector = await wait_for_first_locator(page, UPLOAD_SELECTORS, 4000)
            if upload_locator is None:
                raise RuntimeError("未找到图片上传控件，请检查当前页面结构或更新上传选择器。")
            await set_input_files(upload_locator, image_files)
            print(f"已使用选择器上传图片: {upload_selector}")
            await page.wait_for_timeout(5000)

        title_locator, title_selector = await wait_for_first_locator(page, TITLE_SELECTORS, 4000)
        if title_locator is None:
            raise RuntimeError("未找到标题输入框，请更新 TITLE_SELECTORS。")
        await fill_field(page, title_locator, payload["title"])
        print(f"已填写标题: {title_selector}")

        content_text = build_content(payload)
        if content_text:
            content_locator, content_selector = await wait_for_first_locator(page, CONTENT_SELECTORS, 4000)
            if content_locator is None:
                raise RuntimeError("未找到正文输入框，请更新 CONTENT_SELECTORS。")
            await fill_field(page, content_locator, content_text)
            print(f"已填写正文: {content_selector}")

        if payload.get("location"):
            location_locator, location_selector = await wait_for_first_locator(page, LOCATION_SELECTORS, 1500)
            if location_locator is not None:
                await fill_field(page, location_locator, payload["location"])
                print(f"已填写地点: {location_selector}")

        screenshot_before = ARTIFACT_DIR / "xhs_before_submit.png"
        await page.screenshot(path=str(screenshot_before), full_page=True)
        print(f"已保存截图: {screenshot_before}")

        if mode == "publish":
            target_selectors = PUBLISH_BUTTON_SELECTORS
        else:
            target_selectors = DRAFT_BUTTON_SELECTORS

        button_locator, button_selector = await wait_for_first_locator(page, target_selectors, 4000)
        if button_locator is None:
            raise RuntimeError(
                f"未找到 {'发布' if mode == 'publish' else '草稿'} 按钮，请更新按钮选择器。"
            )
        await button_locator.click()
        print(f"已点击按钮: {button_selector}")

        await page.wait_for_timeout(5000)
        screenshot_after = ARTIFACT_DIR / "xhs_after_submit.png"
        await page.screenshot(path=str(screenshot_after), full_page=True)
        print(f"已保存截图: {screenshot_after}")

        await context.close()
        print(f"小红书 {mode} 流程执行完成。")


def main() -> None:
    args = parse_args()
    asyncio.run(post_note(args))


if __name__ == "__main__":
    main()

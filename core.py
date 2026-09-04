# -*- coding: utf-8 -*-
"""
腳本拆解機 — 核心生成邏輯（provider-agnostic）

被 app.py（網頁介面）與 script_breaker_cli.py（命令列）共用。
支援三種 API 供應商：
  - anthropic：Claude，走 tool use 強制結構化輸出
  - openai：官方 OpenAI，走 Structured Outputs (json_schema, strict)
  - custom： 任何 OpenAI 相容 API（DeepSeek、Kimi/Moonshot、Qwen、自架的 vLLM/Ollama 等），
             先嘗試 json_schema，若該服務不支援則自動退回 json_object + 手動解析，
             並在解析失敗時重試——這一段就是取代原本工具「伺服器回應無法解析」的地方。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 1.5

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-5-20250929",
    "openai": "gpt-4.1",
    "custom": "",
}

# 原工具裡觀察到的常用情緒標籤。模型仍可視內容自訂，但優先從這裡挑，
# 方便日後統計哪種標籤的影片轉換率比較好。
EMOTION_TAG_VOCAB = [
    "故事", "稀缺感", "認同", "恐懼", "慾望", "利益", "權威感", "真實感", "社會證明",
]


class GenerationError(RuntimeError):
    """所有生成失敗情境的統一例外，訊息用中文講清楚發生什麼事。"""


@dataclass
class ProviderConfig:
    provider: str = "anthropic"  # anthropic | openai | custom
    api_key: str = ""
    model: str = ""
    base_url: str = ""  # 僅 custom 需要

    def resolved_model(self) -> str:
        return self.model or DEFAULT_MODELS.get(self.provider, "")


@dataclass
class ProductInfo:
    name: str
    details: str
    audience: str = ""
    price: str = ""


@dataclass
class PipelineResult:
    product: dict
    positioning: dict = field(default_factory=dict)
    audience_confirmation: dict = field(default_factory=dict)
    emotion_and_title: dict = field(default_factory=dict)
    script: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ── JSON 擷取工具（給 json_object fallback 用）────────────────────────

def _extract_json(text: str) -> dict:
    """
    從模型回覆的自由文字中盡量擷取出合法 JSON。
    處理常見的三種情況：純 JSON、包在 ```json ... ``` code fence 裡、
    前後夾雜說明文字。全部失敗才拋出 ValueError。
    """
    text = text.strip()

    # 1) 直接整段就是合法 JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2) 去除 markdown code fence
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        inner = fence_match.group(1).strip()
        try:
            return json.loads(inner)
        except json.JSONDecodeError:
            text = inner  # 繼續往下嘗試

    # 3) 找第一個 { 到最後一個 }，用 raw_decode 抓出第一個合法物件
    start = text.find("{")
    if start != -1:
        decoder = json.JSONDecoder()
        try:
            obj, _ = decoder.raw_decode(text[start:])
            return obj
        except json.JSONDecodeError:
            pass

    raise ValueError("回覆內容無法解析為 JSON")


def _with_additional_properties_false(schema: dict) -> dict:
    """遞迴替 object 型別的 schema 補上 additionalProperties: false，
    這是 OpenAI Structured Outputs 的 strict 模式需要的。"""
    schema = dict(schema)
    if schema.get("type") == "object":
        schema.setdefault("additionalProperties", False)
        props = schema.get("properties")
        if props:
            schema["properties"] = {k: _with_additional_properties_false(v) for k, v in props.items()}
    if schema.get("type") == "array" and "items" in schema:
        schema["items"] = _with_additional_properties_false(schema["items"])
    return schema


# ── 各 provider 的單次呼叫（不含重試，重試在 call_structured 統一處理）──

def _attempt_anthropic(cfg: ProviderConfig, system: str, user: str, schema: dict, tool_name: str) -> dict:
    try:
        import anthropic
    except ImportError as e:
        raise GenerationError("缺少 anthropic 套件，請先執行：pip install -r requirements.txt") from e

    client = anthropic.Anthropic(api_key=cfg.api_key)
    tool_def = {
        "name": tool_name,
        "description": f"回傳符合結構的「{tool_name}」分析結果",
        "input_schema": schema,
    }
    resp = client.messages.create(
        model=cfg.resolved_model(),
        max_tokens=4096,
        system=system,
        tools=[tool_def],
        tool_choice={"type": "tool", "name": tool_name},
        messages=[{"role": "user", "content": user}],
    )
    for block in resp.content:
        if block.type == "tool_use" and block.name == tool_name:
            return block.input
    raise GenerationError("模型回覆中沒有找到預期的 tool_use 區塊")


def _attempt_openai_compatible(cfg: ProviderConfig, system: str, user: str, schema: dict, tool_name: str) -> dict:
    try:
        from openai import OpenAI
    except ImportError as e:
        raise GenerationError("缺少 openai 套件，請先執行：pip install -r requirements.txt") from e

    if cfg.provider == "custom" and not cfg.base_url:
        # 沒填 Base URL 時絕對不能讓它悄悄打去官方 OpenAI——之前就是這樣才會出現
        # 「明明選了自訂／填了 Gemini 金鑰，錯誤訊息卻叫你去 platform.openai.com 開金鑰」的怪狀況。
        raise GenerationError(
            "供應商選了「自訂 OpenAI 相容 API」，但 Base URL 是空的，所以請求被送到官方 OpenAI 去了。"
            "請在設定裡填入該服務的 Base URL（例如 Gemini 是 "
            "https://generativelanguage.googleapis.com/v1beta/openai/，或用設定畫面裡的快速填入按鈕）。"
        )

    kwargs: dict[str, Any] = {"api_key": cfg.api_key}
    if cfg.provider == "custom":
        kwargs["base_url"] = cfg.base_url
    client = OpenAI(**kwargs)

    strict_schema = _with_additional_properties_false(schema)

    # 先嘗試 Structured Outputs（json_schema, strict）。官方 OpenAI 一定支援；
    # 其他 OpenAI 相容服務不一定支援，失敗就退回 json_object 模式。
    try:
        resp = client.chat.completions.create(
            model=cfg.resolved_model(),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": tool_name, "schema": strict_schema, "strict": True},
            },
        )
        content = resp.choices[0].message.content
        return json.loads(content)
    except Exception as e:  # noqa: BLE001 — json_schema 不支援時的服務會丟各種錯誤，統一接住後退回
        if cfg.provider == "openai":
            # 官方 OpenAI 理論上一定支援 json_schema，這裡失敗多半是別的問題（額度/網路/模型名稱），
            # 直接往外拋讓外層重試機制處理，不需要再退回 json_object。
            raise
        fallback_note = f"（json_schema 模式失敗，改用 json_object 模式重試：{e}）"

    # ── json_object fallback：多加一句「只能回傳合法 JSON」的提示，並自己解析 ──
    schema_hint = json.dumps(strict_schema, ensure_ascii=False)
    resp = client.chat.completions.create(
        model=cfg.resolved_model(),
        messages=[
            {
                "role": "system",
                "content": system + "\n\n請只回傳一個合法的 JSON 物件，不要加任何說明文字或 markdown 標記，"
                f"JSON 需符合此 schema：{schema_hint}",
            },
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content
    try:
        return _extract_json(content)
    except ValueError as e:
        raise GenerationError(f"{fallback_note}；json_object 模式回覆仍無法解析：{e}") from e


def call_structured(
    cfg: ProviderConfig,
    system: str,
    user: str,
    schema: dict,
    tool_name: str,
    max_retries: int = MAX_RETRIES,
    on_progress=None,
) -> dict:
    """統一的重試包裝：依 cfg.provider 分派到對應的單次呼叫函式，失敗時重試並用指數退避。"""
    if not cfg.api_key:
        raise GenerationError("尚未設定 API 金鑰，請先在設定裡填入。")
    if cfg.provider == "custom" and not cfg.base_url:
        # 這種是設定沒填對，重試也不會變好，直接不進重試迴圈、立刻讓使用者知道要填 Base URL。
        raise GenerationError(
            "供應商選了「自訂 OpenAI 相容 API」，但 Base URL 是空的——沒填的話請求會被送到官方 OpenAI，"
            "所以你貼的金鑰（例如 Gemini 的 AIzaSy... 開頭）在那邊一定會被拒絕。"
            "請在設定裡填入該服務的 Base URL，或用設定畫面裡的快速填入按鈕。"
        )

    attempt_fn = _attempt_anthropic if cfg.provider == "anthropic" else _attempt_openai_compatible

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            return attempt_fn(cfg, system, user, schema, tool_name)
        except Exception as e:  # noqa: BLE001
            last_err = e
            msg = f"[嘗試 {attempt}/{max_retries}] {tool_name} 失敗：{e}"
            if on_progress:
                on_progress(msg)
            if attempt < max_retries:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
            else:
                raise GenerationError(
                    f"{tool_name} 產生失敗，已重試 {max_retries} 次。最後錯誤：{e}"
                ) from last_err
    raise GenerationError(f"{tool_name} 產生失敗：未知原因")


def test_connection(cfg: ProviderConfig) -> tuple[bool, str]:
    """送一個最小請求驗證 API 金鑰/設定是否可用，回傳 (是否成功, 訊息)。"""
    tiny_schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
    }
    try:
        result = call_structured(
            cfg,
            system="請直接回傳 ok=true。",
            user="ping",
            schema=tiny_schema,
            tool_name="connection_test",
            max_retries=1,
        )
        if result.get("ok"):
            return True, f"連線成功（模型：{cfg.resolved_model()}）"
        return True, "連線成功，但回覆內容不如預期，仍可正常使用。"
    except GenerationError as e:
        return False, str(e)


# ── 各步驟 schema／prompt ─────────────────────────────────────────────

def step_positioning(cfg: ProviderConfig, product: ProductInfo, on_progress=None) -> dict:
    schema = {
        "type": "object",
        "properties": {
            "core_angle": {"type": "string", "description": "這支影片的核心切角"},
            "differentiation": {"type": "string", "description": "跟同類產品的差異化定位"},
            "format_suggestion": {
                "type": "string",
                "description": "建議的影片形式，例如：真人口播、情境劇、開箱、前後對比",
            },
            "key_selling_points": {
                "type": "array",
                "items": {"type": "string"},
                "description": "3-5 個最值得放進腳本的賣點，依重要性排序",
            },
        },
        "required": ["core_angle", "differentiation", "format_suggestion", "key_selling_points"],
    }
    system = (
        "你是專門為短影音（TikTok/Reels/抖音）廣告寫腳本的內容策略師，"
        "擅長從產品資訊中找出最能打中受眾的切角。全程使用繁體中文回覆。"
    )
    user = (
        f"產品名稱：{product.name}\n"
        f"產品細節：{product.details}\n"
        f"目標受眾：{product.audience or '未提供，請自行合理推測'}\n"
        f"價格帶：{product.price or '未提供'}\n\n"
        "請分析這個產品最適合的內容定位。"
    )
    return call_structured(cfg, system, user, schema, "content_positioning", on_progress=on_progress)


def step_audience_confirmation(cfg: ProviderConfig, product: ProductInfo, positioning: dict, on_progress=None) -> dict:
    schema = {
        "type": "object",
        "properties": {
            "audience_profile": {"type": "string", "description": "具體的目標受眾輪廓描述"},
            "pain_points": {"type": "array", "items": {"type": "string"}, "description": "受眾的痛點"},
            "desires": {"type": "array", "items": {"type": "string"}, "description": "受眾的渴望/想達成的狀態"},
            "objections": {
                "type": "array",
                "items": {"type": "string"},
                "description": "受眾可能有的顧慮或抗拒購買的理由",
            },
        },
        "required": ["audience_profile", "pain_points", "desires", "objections"],
    }
    system = "你是消費者洞察分析師，擅長把模糊的目標受眾描述，具體化成可以寫進腳本的痛點與渴望。全程使用繁體中文回覆。"
    user = (
        f"產品名稱：{product.name}\n"
        f"產品細節：{product.details}\n"
        f"已知目標受眾：{product.audience or '未提供'}\n"
        f"內容定位：{json.dumps(positioning, ensure_ascii=False)}\n\n"
        "請具體化這支影片的目標受眾，並列出他們的痛點、渴望與購買前的顧慮。"
    )
    return call_structured(cfg, system, user, schema, "audience_confirmation", on_progress=on_progress)


def step_emotion_and_title(
    cfg: ProviderConfig, product: ProductInfo, positioning: dict, audience: dict, on_progress=None
) -> dict:
    schema = {
        "type": "object",
        "properties": {
            "emotion_arc": {
                "type": "array",
                "items": {"type": "string"},
                "description": "整支影片的情緒轉折順序，例如：懸念→痛點→焦慮→解方→信任→行動",
            },
            "title_options": {
                "type": "array",
                "items": {"type": "string"},
                "description": "4-6 個可作為影片開頭鉤子/標題的句子選項",
            },
            "recommended_title": {"type": "string", "description": "最推薦的一個標題/開頭鉤子"},
        },
        "required": ["emotion_arc", "title_options", "recommended_title"],
    }
    system = "你是短影音腳本的鉤子設計師，擅長設計開頭 3 秒的鉤子與整支影片的情緒節奏。全程使用繁體中文回覆。"
    user = (
        f"產品名稱：{product.name}\n"
        f"內容定位：{json.dumps(positioning, ensure_ascii=False)}\n"
        f"受眾洞察：{json.dumps(audience, ensure_ascii=False)}\n\n"
        "請設計這支影片的情緒轉折順序，並提供幾個開頭鉤子/標題選項。"
    )
    return call_structured(cfg, system, user, schema, "emotion_and_title", on_progress=on_progress)


def _sentence_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "index": {"type": "integer", "description": "第幾句，從 1 開始"},
            "text": {"type": "string", "description": "這一句的逐字稿內容"},
            "purpose": {"type": "string", "description": "這一句的設計目的"},
            "emotion_tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "1-2 個情緒標籤，優先從常用集合挑選：" + "、".join(EMOTION_TAG_VOCAB),
            },
            "shot_suggestion": {
                "type": "string",
                "description": (
                    "這一句實際拍攝時的畫面建議，要具體到可以直接照著拍，"
                    "例如：手機畫面滑動顯示多筆消費截圖、手拿產品對鏡頭特寫、真人對鏡頭口播搭配字幕"
                ),
            },
        },
        "required": ["index", "text", "purpose", "emotion_tags", "shot_suggestion"],
    }


def step_script(
    cfg: ProviderConfig,
    product: ProductInfo,
    positioning: dict,
    audience: dict,
    emotion_and_title: dict,
    sentence_count: int = 12,
    on_progress=None,
) -> dict:
    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "採用的標題/開頭鉤子"},
            "sentences": {
                "type": "array",
                "description": f"逐字稿，依序約 {sentence_count} 句",
                "items": _sentence_schema(),
            },
        },
        "required": ["title", "sentences"],
    }
    system = (
        "你是短影音廣告的逐字稿寫手兼分鏡師，擅長用真實、口語化的第一人稱口吻寫腳本，"
        "每一句都要有明確的設計目的、情緒標籤，以及具體到可以直接照著拍的畫面建議。全程使用繁體中文回覆。"
    )
    user = (
        f"產品名稱：{product.name}\n"
        f"產品細節：{product.details}\n"
        f"價格帶：{product.price or '未提供'}\n"
        f"內容定位：{json.dumps(positioning, ensure_ascii=False)}\n"
        f"受眾洞察：{json.dumps(audience, ensure_ascii=False)}\n"
        f"情緒與標題設計：{json.dumps(emotion_and_title, ensure_ascii=False)}\n\n"
        f"請依照以上資訊，寫出約 {sentence_count} 句的完整逐字稿，"
        "每句都要標註設計目的、使用的情緒催化劑（情緒標籤），以及這一句實際拍攝時的畫面建議，"
        "最後一句要導向明確的行動呼籲（例如購買連結、限時優惠）。"
    )
    return call_structured(cfg, system, user, schema, "full_script", on_progress=on_progress)


def step_rewrite_sentence(
    cfg: ProviderConfig,
    product: ProductInfo,
    script: dict,
    sentence_index: int,
    instruction: str = "",
    on_progress=None,
) -> dict:
    """只重寫逐字稿裡的其中一句，其餘句子維持不變——原工具沒有這個功能，
    但改單句遠比整篇重跑省時間也省 token。"""
    schema = _sentence_schema()
    system = (
        "你是短影音廣告的逐字稿寫手兼分鏡師。現在只需要重寫指定的那一句（含畫面建議），"
        "維持與前後句的語氣和情節連貫。全程使用繁體中文回覆。"
    )
    target = next((s for s in script.get("sentences", []) if s.get("index") == sentence_index), None)
    user = (
        f"產品名稱：{product.name}\n"
        f"完整逐字稿（供參考上下文）：{json.dumps(script, ensure_ascii=False)}\n"
        f"要重寫的是第 {sentence_index} 句，原句：{json.dumps(target, ensure_ascii=False)}\n"
        f"額外要求：{instruction or '（無，請直接換一個寫法，維持原本的設計目的）'}\n\n"
        "請重寫這一句。"
    )
    result = call_structured(cfg, system, user, schema, "rewrite_sentence", on_progress=on_progress)
    result["index"] = sentence_index
    return result


def step_summary(cfg: ProviderConfig, product: ProductInfo, audience: dict, script: dict, on_progress=None) -> dict:
    schema = {
        "type": "object",
        "properties": {
            "resonance_reasons": {
                "type": "array",
                "items": {"type": "string"},
                "description": "這支影片會打中受眾的原因，3 條左右",
            },
            "suggested_next_steps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "投放/優化這支影片的下一步建議",
            },
        },
        "required": ["resonance_reasons", "suggested_next_steps"],
    }
    system = "你是廣告成效顧問，負責在腳本完成後做總結，並給出可執行的下一步建議。全程使用繁體中文回覆。"
    user = (
        f"產品名稱：{product.name}\n"
        f"受眾洞察：{json.dumps(audience, ensure_ascii=False)}\n"
        f"完整逐字稿：{json.dumps(script, ensure_ascii=False)}\n\n"
        "請總結這支影片會打中受眾的原因，並給出後續優化建議。"
    )
    return call_structured(cfg, system, user, schema, "script_summary", on_progress=on_progress)


# ── Mock 模式：不打任何 API，跑假資料驗證整條流程與 UI ─────────────────

def mock_pipeline(product: ProductInfo) -> PipelineResult:
    positioning = {
        "core_angle": f"以「{product.name}」解決受眾最在意的核心痛點",
        "differentiation": "強調成分/規格單純、使用門檻低",
        "format_suggestion": "真人口播＋情境對比",
        "key_selling_points": [s.strip() for s in product.details.split("、") if s.strip()][:5] or ["核心賣點"],
    }
    audience = {
        "audience_profile": product.audience or "對此類產品有明確需求的一般消費者",
        "pain_points": ["找不到安全/有效的解方", "資訊太多不知道怎麼選"],
        "desires": ["快速看到效果", "使用起來安心無負擔"],
        "objections": ["價格是否合理", "會不會踩雷"],
    }
    emotion = {
        "emotion_arc": ["懸念", "痛點", "焦慮", "解方", "信任", "行動"],
        "title_options": [
            f"{product.name}，我只敢用這一個方法",
            f"用了{product.name}之後才發現以前都繞遠路",
            f"沒有人告訴我{product.name}可以這樣用",
        ],
        "recommended_title": f"{product.name}，我只敢用這一個方法",
    }
    purposes = ["拋出懸念鉤子", "喚起同感痛點", "強化找不到解的焦慮", "引出產品解方", "消除核心顧慮",
                "建立品牌信任", "說明吸收/使用優勢", "降低使用門檻", "疊加價值感", "真人見證強化信任",
                "情感共鳴升溫慾望", "導向行動呼籲"]
    tag_cycle = [["故事", "稀缺感"], ["認同", "恐懼"], ["恐懼", "慾望"], ["慾望", "利益"], ["利益", "認同"],
                 ["權威感", "真實感"], ["利益", "慾望"], ["利益", "真實感"], ["慾望", "利益"],
                 ["真實感", "社會證明"], ["慾望", "認同"], ["利益", "稀缺感"]]
    shot_cycle = [
        "手機畫面滑動顯示多筆消費/相關截圖", "真人對鏡頭口播，表情略顯煩惱", "手拿產品或帳單特寫，搭配字幕強調數字",
        "情境重現：使用產品前的困擾畫面", "手拿產品對鏡頭展示包裝/成分標示", "產品使用過程特寫（例如倒出/塗抹/操作）",
        "前後對比畫面（左右分割或淡入淡出）", "手指點選/翻看產品說明或介面", "多個賣點以字卡快速閃過搭配產品畫面",
        "真人見證口吻，搭配日常生活情境畫面", "情感畫面：微笑、放鬆的生活情境", "產品包裝＋購買連結字卡，搭配倒數/庫存字樣",
    ]
    sentences = [
        {
            "index": i + 1,
            "text": f"（模擬句 {i + 1}）關於「{product.name}」的示範文案。",
            "purpose": purposes[i % len(purposes)],
            "emotion_tags": tag_cycle[i % len(tag_cycle)],
            "shot_suggestion": shot_cycle[i % len(shot_cycle)],
        }
        for i in range(12)
    ]
    script = {"title": emotion["recommended_title"], "sentences": sentences}
    summary = {
        "resonance_reasons": [
            "用真實口吻串全程，讓目標受眾高度代入",
            "每個功能點都對應受眾最深的恐懼與慾望",
            "以信任線索與見證，推動轉換",
        ],
        "suggested_next_steps": ["可測試不同開頭鉤子", "針對不同受眾輪廓製作變體"],
    }
    return PipelineResult(
        product=asdict(product),
        positioning=positioning,
        audience_confirmation=audience,
        emotion_and_title=emotion,
        script=script,
        summary=summary,
    )


# ── Markdown 匯出（跟原工具逐字稿卡片呈現一致）─────────────────────────

def render_markdown(result: PipelineResult) -> str:
    p = result.product
    lines = [f"# {result.script.get('title', p.get('name', ''))}", ""]
    lines.append(f"產品：{p.get('name', '')}｜價格帶：{p.get('price') or '（未提供）'}")
    lines.append("")

    lines.append("## 內容定位")
    pos = result.positioning
    lines.append(f"- 核心切角：{pos.get('core_angle', '')}")
    lines.append(f"- 差異化定位：{pos.get('differentiation', '')}")
    lines.append(f"- 建議形式：{pos.get('format_suggestion', '')}")
    for kp in pos.get("key_selling_points", []):
        lines.append(f"- 賣點：{kp}")
    lines.append("")

    lines.append("## 受眾洞察")
    aud = result.audience_confirmation
    lines.append(f"- 受眾輪廓：{aud.get('audience_profile', '')}")
    for k in aud.get("pain_points", []):
        lines.append(f"- 痛點：{k}")
    for k in aud.get("desires", []):
        lines.append(f"- 渴望：{k}")
    for k in aud.get("objections", []):
        lines.append(f"- 顧慮：{k}")
    lines.append("")

    lines.append("## 完整逐字稿")
    for s in result.script.get("sentences", []):
        lines.append(f"**第 {s.get('index')} 句**")
        lines.append(s.get("text", ""))
        lines.append(f"設計目的：{s.get('purpose', '')}")
        lines.append(f"情緒標籤：{'、'.join(s.get('emotion_tags', []))}")
        if s.get("shot_suggestion"):
            lines.append(f"畫面建議：{s.get('shot_suggestion', '')}")
        lines.append("")

    lines.append("## 這支影片會打中受眾的原因")
    for r in result.summary.get("resonance_reasons", []):
        lines.append(f"- {r}")
    lines.append("")
    if result.summary.get("suggested_next_steps"):
        lines.append("## 後續優化建議")
        for r in result.summary.get("suggested_next_steps", []):
            lines.append(f"- {r}")
        lines.append("")

    return "\n".join(lines)

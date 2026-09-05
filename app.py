#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文案產生器 — 網頁版後端 (Flask)

本機執行：
    pip install -r requirements.txt
    python app.py
然後瀏覽器打開 http://127.0.0.1:5000

部署到雲端（例如 Render）：
    用 gunicorn 啟動：gunicorn app:app
    強烈建議設定環境變數 SCRIPT_BREAKER_PASSWORD，網頁才會要求輸入密碼才能使用——
    否則網址一旦外流，任何人都能用你設定好的 API 金鑰生成內容。
    建議一併設定 SCRIPT_BREAKER_SECRET_KEY（任意一串隨機字串），
    這樣重新部署／重啟服務時，大家不用重新登入。

這支後端只會把 API 金鑰從瀏覽器傳到這支後端、再傳給你選的 LLM 供應商，不會經過任何
其他第三方伺服器。金鑰若勾選「記住在本機」會寫進同目錄下的 config.local.json（明文）；
部署在雲端時，這個檔案在服務重啟/重新部署後可能會消失，建議改用環境變數
（ANTHROPIC_API_KEY / OPENAI_API_KEY / SCRIPT_BREAKER_PROVIDER / SCRIPT_BREAKER_MODEL /
SCRIPT_BREAKER_BASE_URL）設定預設值，比較穩定。
"""

from __future__ import annotations

import json
import os
import secrets
from dataclasses import asdict
from datetime import timedelta
from functools import wraps
from pathlib import Path

from flask import Flask, jsonify, redirect, request, send_from_directory, session, url_for

import core

app = Flask(__name__, static_folder="static", template_folder="templates")

# 密碼保護：只有設定了 SCRIPT_BREAKER_PASSWORD 環境變數才會啟用登入頁，
# 本機用 啟動.bat 執行時不會設這個變數，所以不會多一道登入手續。
APP_PASSWORD = os.environ.get("SCRIPT_BREAKER_PASSWORD", "")
app.secret_key = os.environ.get("SCRIPT_BREAKER_SECRET_KEY") or secrets.token_hex(32)
app.permanent_session_lifetime = timedelta(days=30)

CONFIG_PATH = Path(__file__).parent / "config.local.json"

# 記憶體中的目前設定；若本機有存檔則開機時載入
_current_cfg = core.ProviderConfig()


def _load_config_from_disk() -> None:
    global _current_cfg
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            _current_cfg = core.ProviderConfig(
                provider=data.get("provider", "anthropic"),
                api_key=data.get("api_key", ""),
                model=data.get("model", ""),
                base_url=data.get("base_url", ""),
            )
        except (json.JSONDecodeError, OSError):
            pass
    else:
        # 沒有本機存檔時，退回看環境變數。雲端部署（沒有可靠的本機硬碟）建議都用這個方式設定，
        # 而不是靠設定畫面的「記住在本機」。SCRIPT_BREAKER_API_KEY 對任何 provider 都通用；
        # ANTHROPIC_API_KEY / OPENAI_API_KEY 是額外方便沿用習慣命名的備援。
        env_provider = os.environ.get("SCRIPT_BREAKER_PROVIDER", "anthropic")
        api_key = (
            os.environ.get("SCRIPT_BREAKER_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY" if env_provider == "anthropic" else "OPENAI_API_KEY")
            or ""
        )
        _current_cfg = core.ProviderConfig(
            provider=env_provider,
            api_key=api_key,
            model=os.environ.get("SCRIPT_BREAKER_MODEL", ""),
            base_url=os.environ.get("SCRIPT_BREAKER_BASE_URL", ""),
        )


_load_config_from_disk()


# ── 密碼保護（僅在部署到公開網址、設定了 SCRIPT_BREAKER_PASSWORD 時啟用）──────

@app.before_request
def _require_login():
    if not APP_PASSWORD:
        return  # 沒設密碼＝本機個人使用，不強制登入
    if request.endpoint in ("login", "static"):
        return
    if session.get("authed"):
        return
    if request.path.startswith("/api/"):
        return jsonify({"ok": False, "error": "登入已過期，請重新整理頁面登入。"}), 401
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        submitted = request.form.get("password", "")
        # 用 compare_digest 避免時間差攻擊猜密碼，雖然對這種個人小工具來說是多慮，但順手做對。
        if APP_PASSWORD and secrets.compare_digest(submitted, APP_PASSWORD):
            session.clear()
            session["authed"] = True
            session.permanent = True
            return redirect(url_for("index"))
        error = "密碼錯誤，請再試一次。"
    return send_from_directory(app.template_folder, "login.html") if error is None else _login_page_with_error(error)


def _login_page_with_error(error: str):
    html = (app.template_folder and (Path(app.template_folder) / "login.html").read_text(encoding="utf-8")) or ""
    html = html.replace("<!--ERROR-->", f'<div class="err">{error}</div>')
    return html, 401


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


def _product_from_json(data: dict) -> core.ProductInfo:
    return core.ProductInfo(
        name=(data.get("name") or "").strip(),
        details=(data.get("details") or "").strip(),
        audience=(data.get("audience") or "").strip(),
        price=(data.get("price") or "").strip(),
        emotion=(data.get("emotion") or "").strip(),
        story_request=(data.get("story_request") or "").strip(),
        story_length=(data.get("story_length") or "").strip(),
    )


def _handle(fn):
    """統一的錯誤處理：GenerationError 轉成人看得懂的 JSON 錯誤訊息，而不是原始 HTTP 例外。"""
    try:
        return jsonify({"ok": True, "data": fn()})
    except core.GenerationError as e:
        return jsonify({"ok": False, "error": str(e)}), 200  # 200 但 ok=false，前端一律看 ok 欄位
    except Exception as e:  # noqa: BLE001
        return jsonify({"ok": False, "error": f"未預期的錯誤：{e}"}), 200


# ── 靜態頁面 ───────────────────────────────────────────────────────

@app.get("/")
def index():
    return send_from_directory(app.template_folder, "index.html")


# ── 設定 ──────────────────────────────────────────────────────────

@app.get("/api/config")
def get_config():
    return jsonify({
        "provider": _current_cfg.provider,
        "model": _current_cfg.resolved_model(),
        "base_url": _current_cfg.base_url,
        "has_api_key": bool(_current_cfg.api_key),
        "persisted": CONFIG_PATH.exists(),
    })


@app.post("/api/config")
def save_config():
    global _current_cfg
    data = request.get_json(force=True) or {}
    provider = data.get("provider", "anthropic")
    if provider not in ("anthropic", "openai", "custom"):
        return jsonify({"ok": False, "error": "不支援的 provider"}), 400

    api_key = data.get("api_key", "")
    # 空字串代表「沿用目前已儲存的金鑰」，避免每次存設定都要重貼一次金鑰
    if not api_key and _current_cfg.provider == provider:
        api_key = _current_cfg.api_key

    _current_cfg = core.ProviderConfig(
        provider=provider,
        api_key=api_key,
        model=data.get("model", ""),
        base_url=data.get("base_url", ""),
    )

    if data.get("persist"):
        CONFIG_PATH.write_text(
            json.dumps(asdict(_current_cfg), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    elif CONFIG_PATH.exists() and data.get("persist") is False:
        CONFIG_PATH.unlink()

    return jsonify({"ok": True})


@app.post("/api/test-connection")
def test_connection():
    def run():
        ok, msg = core.test_connection(_current_cfg)
        if not ok:
            raise core.GenerationError(msg)
        return {"message": msg}

    return _handle(run)


# ── 生成流程 ───────────────────────────────────────────────────────

@app.post("/api/step/positioning")
def api_positioning():
    data = request.get_json(force=True) or {}
    product = _product_from_json(data.get("product", {}))

    def run():
        if data.get("mock"):
            return core.mock_pipeline(product).positioning
        return core.step_positioning(_current_cfg, product)

    return _handle(run)


@app.post("/api/step/audience")
def api_audience():
    data = request.get_json(force=True) or {}
    product = _product_from_json(data.get("product", {}))
    positioning = data.get("positioning", {})

    def run():
        if data.get("mock"):
            return core.mock_pipeline(product).audience_confirmation
        return core.step_audience_confirmation(_current_cfg, product, positioning)

    return _handle(run)


@app.post("/api/step/emotion_title")
def api_emotion_title():
    data = request.get_json(force=True) or {}
    product = _product_from_json(data.get("product", {}))
    positioning = data.get("positioning", {})
    audience = data.get("audience", {})

    def run():
        if data.get("mock"):
            return core.mock_pipeline(product).emotion_and_title
        return core.step_emotion_and_title(_current_cfg, product, positioning, audience)

    return _handle(run)


@app.post("/api/step/script")
def api_script():
    data = request.get_json(force=True) or {}
    product = _product_from_json(data.get("product", {}))
    positioning = data.get("positioning", {})
    audience = data.get("audience", {})
    emotion_title = data.get("emotion_title", {})
    sentence_count = int(data.get("sentence_count") or 12)

    def run():
        if data.get("mock"):
            return core.mock_pipeline(product).script
        return core.step_script(_current_cfg, product, positioning, audience, emotion_title, sentence_count)

    return _handle(run)


@app.post("/api/step/rewrite_sentence")
def api_rewrite_sentence():
    data = request.get_json(force=True) or {}
    product = _product_from_json(data.get("product", {}))
    script = data.get("script", {})
    index = int(data.get("index"))
    instruction = data.get("instruction", "")

    def run():
        if data.get("mock"):
            sentences = script.get("sentences", [])
            target = next((s for s in sentences if s.get("index") == index), None)
            if target:
                return {**target, "text": target["text"] + "（已重寫-模擬）"}
            return {"index": index, "text": "（模擬重寫句）", "purpose": "示範", "emotion_tags": ["利益"], "shot_suggestion": "（模擬畫面建議）"}
        return core.step_rewrite_sentence(_current_cfg, product, script, index, instruction)

    return _handle(run)


@app.post("/api/step/revise_script")
def api_revise_script():
    data = request.get_json(force=True) or {}
    product = _product_from_json(data.get("product", {}))
    positioning = data.get("positioning", {})
    audience = data.get("audience", {})
    emotion_title = data.get("emotion_title", {})
    script = data.get("script", {})
    feedback = (data.get("feedback") or "").strip()

    def run():
        if not feedback:
            raise core.GenerationError("請先在右邊輸入你想調整的想法，再送出。")
        if data.get("mock"):
            sentences = script.get("sentences", [])
            updated = [{**s, "text": (s.get("text") or "") + "（已依回饋調整-模擬）"} for s in sentences]
            return {**script, "sentences": updated}
        return core.step_revise_script(_current_cfg, product, positioning, audience, emotion_title, script, feedback)

    return _handle(run)


@app.post("/api/step/summary")
def api_summary():
    data = request.get_json(force=True) or {}
    product = _product_from_json(data.get("product", {}))
    audience = data.get("audience", {})
    script = data.get("script", {})

    def run():
        if data.get("mock"):
            return core.mock_pipeline(product).summary
        return core.step_summary(_current_cfg, product, audience, script)

    return _handle(run)


@app.post("/api/step/competitor_research")
def api_competitor_research():
    data = request.get_json(force=True) or {}
    product = _product_from_json(data.get("product", {}))
    positioning = data.get("positioning", {})

    def run():
        if data.get("mock"):
            return core.mock_competitor_references(product)
        return core.search_competitor_references(_current_cfg, product, positioning)

    return _handle(run)


# ── 匯出 ──────────────────────────────────────────────────────────

@app.post("/api/export/markdown")
def export_markdown():
    data = request.get_json(force=True) or {}
    result = core.PipelineResult(
        product=data.get("product", {}),
        positioning=data.get("positioning", {}),
        audience_confirmation=data.get("audience", {}),
        emotion_and_title=data.get("emotion_title", {}),
        script=data.get("script", {}),
        summary=data.get("summary", {}),
    )
    md = core.render_markdown(result)
    return app.response_class(md, mimetype="text/markdown")


def _open_browser():
    import webbrowser
    webbrowser.open("http://127.0.0.1:5000")


if __name__ == "__main__":
    import threading
    print("文案產生器網頁版啟動中…瀏覽器會自動開啟，若沒有請手動開啟 http://127.0.0.1:5000")
    print("=" * 60)
    print("⚠️  這個黑色視窗就是伺服器本體，使用網頁的時候請不要關閉！")
    print("    可以把它縮到最小，但按下 X 關閉會讓網頁變成「無法連線」。")
    print("    要停止伺服器時，再回來這裡按 Ctrl+C 或關閉視窗即可。")
    print("=" * 60)
    threading.Timer(1.2, _open_browser).start()
    # debug=False：一般使用者雙擊執行用，避免自動重載造成瀏覽器開兩次、也避免暴露除錯器
    app.run(host="127.0.0.1", port=5000, debug=False)

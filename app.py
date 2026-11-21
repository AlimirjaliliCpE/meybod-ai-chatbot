
import ollama
from flask import Flask, request, jsonify, render_template
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from pathlib import Path

app = Flask(__name__)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["30 per hour"],
    storage_uri="memory://"
)

MODEL = "qwen2.5:3b"

# Read university knowledge
KNOWLEDGE_FILE = Path(__file__).parent / "knowledge.txt"

if KNOWLEDGE_FILE.exists():
    UNIVERSITY_INFO = KNOWLEDGE_FILE.read_text(
        encoding="utf-8"
    )
else:
    UNIVERSITY_INFO = "اطلاعات دانشگاه در دسترس نیست."


SYSTEM_PROMPT = f"""
تو دستیار هوشمند دانشگاه میبد هستی.

اطلاعات معتبر دانشگاه که باید مبنای پاسخ‌ها قرار گیرد:

{UNIVERSITY_INFO}

قوانین پاسخ‌گویی:

1. پاسخ‌های مربوط به دانشگاه را فقط بر اساس اطلاعات
   موجود در بخش اطلاعات دانشگاه ارائه بده.

2. اگر پاسخ سؤال در اطلاعات موجود نیست، حدس نزن
   و اطلاعات ساختگی ارائه نکن.

3. در صورت نبود اطلاعات کافی، بگو:
   «متأسفانه اطلاعات دقیقی درباره این موضوع در
   پایگاه اطلاعاتی من ثبت نشده است. لطفاً با
   واحد مربوطه دانشگاه تماس بگیرید.»

4. پاسخ‌ها فارسی، محترمانه و مختصر باشند.

5. اطلاعات دانشگاه را با دانسته‌های عمومی خودت
   جایگزین نکن.

6. خودت را دستیار هوشمند دانشگاه میبد معرفی کن.
"""


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
@limiter.limit("5 per minute")
def chat():

    try:

        data = request.get_json(silent=True) or {}

        message = data.get("message", "").strip()
        history = data.get("history", [])

        if not message:
            return jsonify({"error": "پیام خالی است."}), 400

        if len(message) > 2000:
            return jsonify({"error": "پیام بیش از حد طولانی است."}), 400

        safe_history = []

        if isinstance(history, list):

            for item in history[-6:]:

                if not isinstance(item, dict):
                    continue

                role = item.get("role")
                content = item.get("content")

                if role not in ["user", "assistant"]:
                    continue

                if not isinstance(content, str):
                    continue

                content = content.strip()

                if content and len(content) <= 2000:
                    safe_history.append({
                        "role": role,
                        "content": content
                    })

        conversation = safe_history + [{
            "role": "user",
            "content": message
        }]

        response = ollama.chat(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                *conversation
            ],
            options={
                "temperature": 0.2,
                "num_predict": 300,
                "num_ctx": 4096
            }
        )

        answer = response.message.content

        return jsonify({"answer": answer})

    except Exception:

        app.logger.exception("Chat error")

        return jsonify({
            "error": "خطا در پاسخ‌گویی. لطفاً دوباره تلاش کنید."
        }), 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )

# -*- coding: utf-8 -*-
"""
rewrite_core.py — نسخة مبسطة بدون تحقق رقمي
تُعيد الصياغة فقط باستخدام OpenAI وتحافظ على الهيكل الصحفي للسيو.
"""

from openai import OpenAI, APIError, APITimeoutError, RateLimitError

SYSTEM_PROMPT = """أنت محرر اقتصادي محترف متخصص في الصياغة الصحفية الموجهة للجمهور العام.
مهمتك إعادة كتابة المقال بلغة عربية مبسّطة وواضحة دون أي حشو أو تحليل، مع الحفاظ الكامل على جميع الأرقام والتواريخ والأسماء كما هي في النص الأصلي.

إرشادات إلزامية:
1) لا تغيّر أي رقم أو نسبة أو تاريخ أو اسم بنك أو جهة مالية.
2) وحّد الأرقام العشرية لتُكتب بنقطة (مثل 47.40). لا تضف كسورًا للأعداد الصحيحة.
3) لا تضف أي توقعات أو تحليل أو تعليقات تفسيرية.
4) لا تحذف أي بنك أو جهة مذكورة في المصدر.
5) استخدم جُملاً قصيرة وواضحة، وفقرات من 2–3 جمل كحد أقصى.
6) لا تبدأ الفقرات بعبارات آلية ثقيلة (مثل: في المقابل، من جهة أخرى، يجدر الذكر).
7) احذف أي تكرار أو حشو لغوي.
8) الأسلوب موضوعي إخباري قريب من القارئ.
9) لا تُدرج منهجية أو توقيت تحديث أو اسم الكاتب.
10) الناتج يجب أن يكون نصًا عاديًا فقط.

بعد إعادة الصياغة، أخرج العناصر التالية نصيًا وبالترتيب:

1) Title (≤ 60 حرفًا): 
- تتضمن عبارة "سعر الدولار اليوم في مصر" + التاريخ كما في المصدر.

2) Meta (≤ 150 حرفًا):
- تلخّص الحالة وتذكر وجود قائمة أسعار لجميع البنوك.

3) Body:
- مقدمة من 3–4 جمل توضّح الحدث بلغة مباشرة.
- فقرة تمهيدية قصيرة: "فيما يلي أحدث أسعار الدولار في البنوك المصرية اليوم (شراء/بيع):"
- قائمة بكل البنوك المذكورة بالترتيب نفسه كما وردت، مع الأسعار موحدة العلامة العشرية.
- خاتمة قصيرة (سطر واحد) دون تحليل أو توقّع.

4) SEO-Extras:
- Slug: سعر-الدولار-في-مصر-اليوم-[اليوم]-[الشهر]-[السنة]
- Keywords: سعر الدولار اليوم, الدولار في مصر, أسعار البنوك, الجنيه المصري
- AltText: سعر الدولار اليوم في مصر – قائمة أسعار البنوك
"""

USER_PROMPT_TEMPLATE = """هذا هو النص الأصلي الذي أريد إعادة صياغته بأسلوب مبسّط للجمهور العام، مع الحفاظ على الأرقام كما هي وتوحيد العلامة العشرية إلى النقطة دون تغيير القيم:

[note]
- لا تغيّر الأرقام، لا تضف حقائق جديدة.
- الأرقام التي تحتوي فاصلة عشرية تُكتب بنقطة وبخانتين عشريتين.
- الأعداد الصحيحة تُترك كما هي (دون .00).
- أعِد كل البنوك المذكورة كما وردت.

[النص الأصلي]
{original_text}
"""

def rewrite_with_openai(raw_text: str, api_key: str, base_url: str, model: str,
                        temperature: float = 0.3, max_output_tokens: int = 2000) -> str:
    """استدعاء OpenAI لإعادة الصياغة فقط."""
    client = OpenAI(api_key=api_key, base_url=base_url)
    user_prompt = USER_PROMPT_TEMPLATE.format(original_text=raw_text)

    params = dict(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=[{"role": "user", "content": [{"type": "input_text", "text": user_prompt}]}],
        max_output_tokens=max_output_tokens,
    )

    def _call(p):
        return client.responses.create(**p).output_text.strip()

    last_err = None
    for attempt in range(4):
        try:
            if attempt == 0:
                p = dict(params)
                p["temperature"] = float(temperature)
                return _call(p)
            else:
                return _call(params)
        except (RateLimitError, APITimeoutError):
            continue
        except APIError as e:
            msg = str(e)
            if "temperature" in msg or "Unsupported parameter" in msg:
                return _call(params)
            last_err = e
            continue
    raise RuntimeError(f"OpenAI call failed: {last_err}")

def process_article(raw_text: str, api_key: str, base_url: str,
                    model: str = "gpt-5-mini",
                    temperature: float = 0.3,
                    max_output_tokens: int = 2000):
    """يُعيد النص فقط دون أي تحقق."""
    rewritten = rewrite_with_openai(
        raw_text, api_key, base_url, model,
        temperature=temperature,
        max_output_tokens=max_output_tokens
    )
    return rewritten, {"status": "ok"}

# -*- coding: utf-8 -*-
"""
rewrite_core.py
نواة المنطق: قفل الأرقام الحساسة (أسعار/نسب) + توحيد العلامة العشرية + استدعاء OpenAI
يتجاهل التواريخ أو الأرقام غير المالية.
يدعم fallback تلقائي عند عدم دعم temperature.
"""

import re
from decimal import Decimal, InvalidOperation
from typing import List, Tuple
from openai import OpenAI, APIError, APITimeoutError, RateLimitError

# ----------------- إعداد التعبيرات والسياقات -----------------

NUM_REGEX = re.compile(
    r'(?<![\w.\-])(?:\d{1,3}(?:[.,]\d{3})+|\d+)(?:[.,]\d+)?(?![\w.\-])',
    re.UNICODE
)

CURRENCY_HINTS = [
    "جنيه", "EGP", "ج.م", "USD", "دولار", "ريال", "SAR", "EUR", "يورو",
    "شراء", "بيع", "%"
]

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

# ----------------- أدوات الأرقام -----------------

def _to_decimal(num_str: str) -> Decimal:
    s = num_str.strip()
    if "," in s and "." in s:
        last_comma = s.rfind(",")
        last_dot = s.rfind(".")
        if last_comma > last_dot:
            s = s.replace(".", "")
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    else:
        if "," in s and "." not in s:
            s = s.replace(",", ".")
    return Decimal(s)

def extract_numbers_with_spans(text: str) -> List[Tuple[str, Tuple[int, int]]]:
    return [(m.group(0), m.span()) for m in NUM_REGEX.finditer(text)]

def _is_protected_number(token: str, neighbor: str) -> bool:
    """نعتبر الرقم محميًا إذا كان كسريًا أو بجوار كلمات مالية (جنيه، دولار، شراء...)."""
    has_decimal = bool(re.search(r'[.,]\d+', token))
    if has_decimal:
        return True
    neighbor_norm = neighbor.replace("ـ", "")
    return any(hint in neighbor_norm for hint in CURRENCY_HINTS)

def extract_protected_decimals(text: str) -> List[Decimal]:
    """تُرجع فقط الأرقام الحساسة (أسعار/نِسَب)."""
    vals: List[Decimal] = []
    for token, (start, end) in extract_numbers_with_spans(text):
        neighbor = text[max(0, start-15):min(len(text), end+15)]
        if not _is_protected_number(token, neighbor):
            continue
        try:
            vals.append(_to_decimal(token))
        except InvalidOperation:
            continue
    return vals

def multiset_contains(small: List[Decimal], big: List[Decimal]) -> bool:
    """يتحقق أن جميع أرقام المصدر المحمية موجودة في المخرَج."""
    from collections import Counter
    cs, cb = Counter(small), Counter(big)
    for k, v in cs.items():
        if cb.get(k, 0) < v:
            return False
    return True

def normalize_decimal_token(token: str) -> str:
    """يوحّد الأرقام العشرية إلى نقطتين عشريتين؛ الصحيحة تُترك كما هي."""
    has_decimal = bool(re.search(r'[.,]\d+', token))
    if not has_decimal:
        return token
    try:
        dec = _to_decimal(token)
    except InvalidOperation:
        return token
    return f"{dec:.2f}"

def unify_decimal_points(text: str) -> str:
    out = []
    last_idx = 0
    for token, (start, end) in extract_numbers_with_spans(text):
        out.append(text[last_idx:start])
        out.append(normalize_decimal_token(token))
        last_idx = end
    out.append(text[last_idx:])
    return "".join(out)

# ----------------- استدعاء OpenAI -----------------

def rewrite_with_openai(raw_text: str, api_key: str, base_url: str, model: str,
                        temperature: float = 0.3, max_output_tokens: int = 2000) -> str:
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
        except (RateLimitError, APITimeoutError) as e:
            last_err = e
            continue
        except APIError as e:
            last_err = e
            msg = str(e)
            if "temperature" in msg or "Unsupported parameter" in msg:
                try:
                    return _call(params)
                except Exception as ee:
                    last_err = ee
                    continue
            continue
    raise RuntimeError(f"OpenAI call failed: {last_err}")

# ----------------- المعالجة الرئيسية -----------------

def process_article(raw_text: str, api_key: str, base_url: str,
                    model: str = "gpt-5-mini",
                    temperature: float = 0.3,
                    max_output_tokens: int = 2000):
    """يشغّل التسلسل الكامل: استخراج → إعادة صياغة → توحيد → تحقق رقمي."""
    original_protected = extract_protected_decimals(raw_text)
    rewritten = rewrite_with_openai(raw_text, api_key, base_url, model,
                                    temperature=temperature,
                                    max_output_tokens=max_output_tokens)
    rewritten_unified = unify_decimal_points(rewritten)
    rewritten_protected = extract_protected_decimals(rewritten_unified)

    if not multiset_contains(original_protected, rewritten_protected):
        from collections import Counter
        want = Counter(original_protected)
        got = Counter(rewritten_protected)
        missing = []
        for k, v in want.items():
            if got.get(k, 0) < v:
                missing.append((str(k), v, got.get(k, 0)))
        msg = "فشل التحقق الرقمي: بعض القيم الحساسة من المصدر غير موجودة في المخرَج أو ناقصة."
        info = {
            "status": "fail",
            "message": msg,
            "missing": missing,
            "original_protected": [str(x) for x in original_protected],
            "rewritten_protected": [str(x) for x in rewritten_protected],
        }
        return None, info

    return rewritten_unified, {"status": "ok"}

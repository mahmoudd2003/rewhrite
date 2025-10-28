# -*- coding: utf-8 -*-
import streamlit as st
from rewrite_core import process_article

st.set_page_config(page_title="Arabic SEO Rewriter", page_icon="📰", layout="centered")
st.title("📰 Arabic SEO Rewriter")
st.caption("إعادة صياغة بأسلوب مبسّط للجمهور العام مع قفل الأرقام الحسّاسة وتوحيد العلامة العشرية – نص فقط جاهز للسيو.")

st.markdown("**الخطوة 1:** أدخل المقال الأصلي (لصقًا أو رفع ملف).")
input_mode = st.radio("طريقة الإدخال", ["لصق النص", "رفع ملف"], horizontal=True)
raw_text = ""

if input_mode == "لصق النص":
    raw_text = st.text_area("الصق النص الأصلي هنا:", height=300, placeholder="الصق المقال كاملاً…")
else:
    file = st.file_uploader("ارفع ملف نصي (.txt)", type=["txt"])
    if file:
        raw_text = file.read().decode("utf-8", errors="ignore")

st.markdown("---")
st.markdown("**الخطوة 2:** الإعدادات (اختياري).")
col1, col2 = st.columns(2)
with col1:
    model = st.selectbox("نموذج OpenAI", ["gpt-5-mini", "gpt-5"], index=0)
with col2:
    temperature = st.slider("Temperature (قد يتم تجاهله تلقائيًا إن لم يكن مدعومًا)", 0.0, 1.0, 0.3, 0.05)

max_output_tokens = st.number_input("الحد الأقصى للتوكنات", min_value=500, max_value=8000, value=2000, step=100)

st.markdown("---")
st.markdown("**الخطوة 3:** مفاتيح OpenAI (من أسرار Streamlit).")
api_key = st.secrets.get("OPENAI_API_KEY", "")
base_url = st.secrets.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

if not api_key:
    st.warning("الرجاء إضافة OPENAI_API_KEY إلى أسرار Streamlit (Settings → Secrets).")
    st.stop()

if not base_url.endswith("/v1"):
    st.info("تنبيه: يُفضّل أن ينتهي OPENAI_BASE_URL بـ /v1. القيمة الحالية: {}".format(base_url))

if st.button("▶️ إعادة الصياغة الآن", type="primary"):
    if not raw_text.strip():
        st.error("لم يتم إدخال نص.")
    else:
        with st.spinner("يتم المعالجة…"):
            output, info = process_article(
                raw_text,
                api_key=api_key,
                base_url=base_url,
                model=model,
                temperature=float(temperature),
                max_output_tokens=int(max_output_tokens),
            )
        if info["status"] != "ok":
            st.error(info["message"])
            with st.expander("تفاصيل التحقق"):
                st.write("Missing (value, needed, got):", info.get("missing", []))
                st.write("Original protected numbers:", info.get("original_protected", []))
                st.write("Rewritten protected numbers:", info.get("rewritten_protected", []))
        else:
            st.success("تمت إعادة الصياغة بنجاح (الأرقام الحسّاسة مطابقة للمصدر).")
            st.markdown("### ✅ النص النهائي (SEO-ready)")
            st.write(output)
            st.download_button(
                "⬇️ تحميل النص",
                data=output.encode("utf-8"),
                file_name="rewritten_article.txt",
                mime="text/plain"
            )

st.markdown("---")
st.caption("لا يتم نشر أي شيء تلقائيًا. التطبيق يُنتج نصًا فقط.")

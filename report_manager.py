import streamlit as st
from datetime import date
from supabase import create_client

# Supabase 연결
url = st.secrets["supabase"]["url"]
key = st.secrets["supabase"]["key"]
supabase = create_client(url, key)

st.title("📚 금융기관 레포트 요약 관리")

# 레포트 입력 폼
with st.form("report_form"):
    report_date = st.date_input("날짜", value=date.today())
    institution = st.text_input("기관명")
    summary = st.text_area("요약")
    submit = st.form_submit_button("저장")
    if submit:
        supabase.table("reports").insert({
            "date": report_date.isoformat(),
            "institution": institution,
            "summary": summary
        }).execute()
        st.success("저장되었습니다.")

# 레포트 목록 보기
st.subheader("📄 저장된 레포트 목록")
res = supabase.table("reports").select("*").order("date", desc=True).execute()
for r in res.data:
    st.markdown(f"**{r['date']} - {r['institution']}**")
    st.markdown(r["summary"])

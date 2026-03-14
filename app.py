import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Waze Churn Prediction", page_icon="🚗", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("waze_model.pkl")

model = load_model()

st.title("توقع بقاء/مغادرة مستخدمي Waze 🚗")
st.write("أدخل بيانات المستخدم لتوقع ما إذا كان سيستمر في استخدام التطبيق (Retained) أم سيغادر (Churned).")

col1, col2 = st.columns(2)

with col1:
    sessions = st.number_input("عدد الجلسات (sessions)", min_value=0, value=100)
    drives = st.number_input("عدد مرات القيادة (drives)", min_value=0, value=80)
    total_sessions = st.number_input("إجمالي الجلسات (total_sessions)", min_value=0.0, value=150.0)
    n_days_after_onboarding = st.number_input("عدد الأيام منذ التسجيل", min_value=0, value=1000)
    driven_km_drives = st.number_input("المسافة المقطوعة (كم)", min_value=0.0, value=3000.0)

with col2:
    duration_minutes_drives = st.number_input("مدة القيادة (بالدقائق)", min_value=0.0, value=1500.0)
    total_navigations_fav1 = st.number_input("الذهاب للمكان المفضل 1", min_value=0, value=50)
    total_navigations_fav2 = st.number_input("الذهاب للمكان المفضل 2", min_value=0, value=10)
    activity_days = st.number_input("أيام النشاط", min_value=0, max_value=31, value=15)
    driving_days = st.number_input("أيام القيادة", min_value=0, max_value=31, value=10)

device = st.selectbox("نوع الجهاز (Device)", ["iPhone", "Android"])

if st.button("توقع حالة المستخدم"):
    device_encoded = 1 if device == "iPhone" else 0 
    
    input_data = pd.DataFrame({
        'sessions': [sessions],
        'drives': [drives],
        'total_sessions': [total_sessions],
        'n_days_after_onboarding': [n_days_after_onboarding],
        'total_navigations_fav1': [total_navigations_fav1],
        'total_navigations_fav2': [total_navigations_fav2],
        'driven_km_drives': [driven_km_drives],
        'duration_minutes_drives': [duration_minutes_drives],
        'activity_days': [activity_days],
        'driving_days': [driving_days],
        'device': [device_encoded]
    })
    
    # الإصلاح الذكي من غير ما نغير شكل النتيجة بتاعتك
    probabilities = model.predict_proba(input_data)[0]
    churn_probability = probabilities[0] 
    
    st.markdown("---")
    
    # لو احتمالية المغادرة عدت 20% هيطبع الرسالة الحمراء بتاعتك
    if churn_probability > 0.20:
        st.error("⚠️ النتيجة: المستخدم سيغادر التطبيق (Churned)")
    else:
        st.success("🎉 النتيجة: المستخدم سيستمر في استخدام التطبيق (Retained)")

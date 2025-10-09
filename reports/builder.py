# ui/tab_planning.py
from __future__ import annotations
import streamlit as st
from datetime import datetime, timedelta

# Not: approvals.save_approval önceki mesajdaki haliyle event_id döndürüyorsa mesajda event_id göstereceğiz.
# Eğer senin sürümün hash döndürüyorsa alttaki yorumlu satırı aktif edip metni ona göre değiştir.
from patrol.approvals import save_approval  # def save_approval(payload: dict) -> str | event_id

def render():
    st.subheader("🚓 Devriye Planlama – Onay Kaydı")

    with st.form("approval_form", clear_on_submit=False):
        # Basit demo alanları (elindeki gerçek veriye göre doldurulabilir)
        alt_id = st.text_input("Plan/Alternatif ID", value="ALT-001")
        dev_no = st.text_input("Devriye/Atama Kodu", value="DV-12")
        teams = st.multiselect("Takımlar", options=["Alpha", "Bravo", "Charlie"], default=["Alpha"])

        col1, col2 = st.columns(2)
        with col1:
            t_start = st.datetime_input("Başlangıç", value=datetime.now())
        with col2:
            t_end = st.datetime_input("Bitiş", value=datetime.now() + timedelta(hours=4))

        approver = st.text_input("Onaylayan", value="komiser.kaya")

        submitted = st.form_submit_button("Onayı Kaydet")
        if submitted:
            payload = {
                "alt_id": alt_id,
                "assignment": dev_no,
                "teams": teams,
                "start": str(t_start),
                "end": str(t_end),
                "approver": approver,
            }
            # save_approval -> önceki önerime göre event_id döndürür
            approval_id_or_hash = save_approval(payload)

            # --- Eğer save_approval event_id döndürüyorsa:
            st.success(f"Onay kaydedildi • Kayıt ID: `{approval_id_or_hash}`")

            # --- Eğer senin sürümünde hash dönüyorsa, yukarıdaki satırı yorumlayıp şunu aç:
            # st.success(f"Onay kaydedildi • İçerik karması (hash): `{approval_id_or_hash}`")

            with st.expander("Gönderilen payload", expanded=False):
                st.json(payload)

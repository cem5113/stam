# ui/tab_reports.py
from __future__ import annotations
import streamlit as st
import pandas as pd
from datetime import timedelta
from typing import List, Dict, Tuple, Optional

from dataio.loaders import load_sf_crime_latest
from patrol.approvals import list_approvals
from features.stats_classic import time_distributions, spatial_top_geoid, offense_breakdown
from reports.builder import approvals_to_df, frames_to_csv_bytes, pack_zip

def _category_col(df: pd.DataFrame) -> Optional[str]:
    for c in ("category_grouped","category","subcategory_grouped","subcategory"):
        if c in df.columns: return c
    return None

def _date_bounds(df: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
    if "date" in df.columns:
        dd = pd.to_datetime(df["date"], errors="coerce").dropna()
        if len(dd) > 0:
            return dd.min().normalize(), dd.max().normalize()
    now = pd.Timestamp.today().normalize()
    return now - pd.Timedelta(days=30), now

def _slice_df(df: pd.DataFrame, d1: pd.Timestamp, d2: pd.Timestamp, cats: List[str], ccol: Optional[str]) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out[(out["date"] >= d1) & (out["date"] <= d2)]
    if ccol and cats:
        out = out[out[ccol].astype(str).isin(cats)]
    return out

def _period_choice_to_range(choice: str, end_max: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    end = end_max
    if choice == "Günlük":
        start = end
    elif choice == "Haftalık":
        start = end - pd.Timedelta(days=6)
    else:  # Aylık
        start = end - pd.Timedelta(days=29)
    return start.normalize(), end.normalize()

def render():
    st.subheader("🧾 Raporlar & Operasyonel Öneriler")

    # veri
    try:
        df, src = load_sf_crime_latest()
    except Exception as e:
        st.error("Veri yüklenemedi.")
        st.exception(e)
        return

    # sol panel
    left, mid, right = st.columns([1.15, 2.0, 1.2])
    with left:
        st.markdown("**Filtreler**")
        period = st.radio("Rapor türü", ["Günlük","Haftalık","Aylık"], index=0,
                          help="Günlük plan, haftalık özet, aylık değerlendirme")
        dmin, dmax = _date_bounds(df)
        start_def, end_def = _period_choice_to_range(period, dmax)
        d1 = st.date_input("Başlangıç", value=start_def.date(), min_value=dmin.date(), max_value=dmax.date())
        d2 = st.date_input("Bitiş", value=end_def.date(),   min_value=dmin.date(), max_value=dmax.date())
        ccol = _category_col(df)
        cats_all = sorted(df[ccol].dropna().astype(str).unique()) if ccol else []
        pick_cats = st.multiselect("Suç türü filtresi", cats_all, default=[])

        mode = st.radio("Gösterim", ["Planlanan devriyeler","Suç gerçekleşmeleri"], index=0,
                        help="İcra edilen devriyeler ayrı log gelince eklenecek")

    # veri kesiti
    d1_ts, d2_ts = pd.to_datetime(d1), pd.to_datetime(d2)
    dfw = _slice_df(df, d1_ts, d2_ts, pick_cats, ccol)

    # orta panel
    with mid:
        if mode == "Planlanan devriyeler":
            st.markdown("**Planlanan devriyeler (onay kayıtları)**")
            appr = list_approvals(limit=500)
            df_appr = approvals_to_df(appr)
            if len(df_appr) > 0:
                st.dataframe(df_appr.sort_values("ts_sf", ascending=False), use_container_width=True, height=380)
            else:
                st.info("Henüz onay kaydı yok.")
        else:
            st.markdown("**Suç yoğunluğu özeti**")
            # Top GEOID
            top = spatial_top_geoid(dfw, n=15)
            st.dataframe(top, use_container_width=True, height=220)
            # Zamansal özet
            sums = time_distributions(dfw)
            st.caption("Saatlik dağılım")
            st.line_chart(sums["by_hour"].set_index("event_hour"))
            st.caption("Gün × saat ısı")
            st.dataframe(sums["heat"].fillna(0).iloc[-7:, :], use_container_width=True, height=180)

    # sağ panel — öneriler + indirme
    with right:
        st.markdown("**Öneriler (otomatik cümleler)**")
        tips: List[str] = []
        # basit kural: en üst 3 GEOID toplamın %X üzerindeyse artırılabilir
        try:
            ycol = "crime_count" if "crime_count" in dfw.columns else None
            if mode != "Planlanan devriyeler" and ycol:
                g15 = spatial_top_geoid(dfw, n=15)
                total = float(dfw[ycol].sum())
                top3 = float(g15.head(3)["value"].sum()) if len(g15) >= 3 else float(g15["value"].sum())
                share = (top3 / total) if total > 0 else 0.0
                if share >= 0.25:
                    g_list = ", ".join(g15.head(3)["GEOID"].astype(str))
                    tips.append(f"**{g_list}** hücrelerinde yoğunluk yüksek (ilk 3 ≈ %{share*100:.0f}). "
                                f"Bu bölgelerde devriye **artırılabilir**.")
                elif share <= 0.10 and total > 100:
                    tips.append("Yoğunluk şehre daha dengeli yayılmış görünüyor; sabit plan **korunabilir**.")
        except Exception:
            pass

        if tips:
            for t in tips:
                st.markdown(f"- {t}")
        else:
            st.caption("Şimdilik otomatik öneri üretilmedi.")

        st.markdown("---")
        st.markdown("**Rapor Çıkışı**")
        # rapor dosyaları (CSV -> ZIP)
        files: Dict[str, pd.DataFrame] = {}

        # planlanan devriyeler
        appr = approvals_to_df(list_approvals(limit=1000))
        files["planned_approvals.csv"] = appr

        # suç özetleri
        try:
            sums = time_distributions(dfw)
            br = offense_breakdown(dfw)
            files["by_hour.csv"] = sums["by_hour"]
            files["by_dow.csv"] = sums["by_dow"]
            files["heatmatrix.csv"] = sums["heat"].reset_index().rename(columns={"index":"day"})
            files["offense_breakdown.csv"] = br
            top10 = spatial_top_geoid(dfw, n=10)
            files["top10_geoid.csv"] = top10
        except Exception:
            pass

        csv_blobs = frames_to_csv_bytes(files)
        zip_blob = pack_zip(csv_blobs)
        st.download_button(
            "⬇️ ZIP indir (CSV raporlar)",
            data=zip_blob,
            file_name=f"reports_{period.lower()}_{d1_ts.date()}_{d2_ts.date()}.zip",
            mime="application/zip",
            use_container_width=True
        )

        st.caption("PDF şablonu bir sonraki adımda eklenecek (builder.py genişletilecek).")

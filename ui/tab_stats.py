# ui/tab_stats.py
from __future__ import annotations
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from dataio.loaders import load_sf_crime_latest
from features.stats_classic import time_distributions, spatial_top_geoid, offense_breakdown

def _category_options(df: pd.DataFrame):
    for c in ("category_grouped","category","subcategory_grouped","subcategory"):
        if c in df.columns:
            return c, sorted(df[c].dropna().astype(str).unique())
    return None, []

def render():
    st.subheader("📊 Suç İstatistikleri (Geçmiş)")

    # veri
    try:
        df, src = load_sf_crime_latest()
    except Exception as e:
        st.error("Veri yüklenemedi.")
        st.exception(e)
        return

    left, mid, right = st.columns([1.1, 2.0, 1.2])

    # -------- Sol panel: filtreler --------
    with left:
        st.markdown("**Filtreler**")
        # tarih aralığı
        min_d = pd.to_datetime(df.get("date", pd.Timestamp.today())).min()
        max_d = pd.to_datetime(df.get("date", pd.Timestamp.today())).max()
        d1, d2 = st.date_input(
            "Tarih aralığı",
            value=(pd.to_datetime(max_d).date() - pd.Timedelta(days=30), pd.to_datetime(max_d).date()),
            min_value=pd.to_datetime(min_d).date() if pd.notna(min_d) else None,
            max_value=pd.to_datetime(max_d).date() if pd.notna(max_d) else None
        )
        date_range = (pd.to_datetime(d1), pd.to_datetime(d2))

        # kategori
        ccol, opts = _category_options(df)
        pick_cats = []
        if ccol and len(opts) > 0:
            pick_cats = st.multiselect("Suç türü", options=opts, default=[])

        mode = st.radio("Gösterim", ["Zamansal", "Mekânsal", "Suç Türleri"], index=0)

        st.caption("ℹ️ İpuçları: grafiklerin üzerine gelerek değerleri görebilirsiniz.")

    # -------- Orta panel: grafik/harita --------
    with mid:
        if mode == "Zamansal":
            st.markdown("**Zamansal dağılımlar**")
            sums = time_distributions(df, date_range=date_range, categories=pick_cats)

            st.caption("Saatlik dağılım")
            st.line_chart(sums["by_hour"].set_index("event_hour"))

            st.caption("Haftanın günlerine göre (0=Pzt)")
            st.bar_chart(sums["by_dow"].set_index("day_name"))

            st.caption("Gün × Saat ısı matrisi")
            fig, ax = plt.subplots(figsize=(6, 3))
            hm = sums["heat"].fillna(0)
            im = ax.imshow(hm.to_numpy(), aspect="auto")
            ax.set_yticks(range(len(hm.index)))
            ax.set_yticklabels(hm.index)
            ax.set_xticks(range(len(hm.columns)))
            ax.set_xticklabels(hm.columns, rotation=0, fontsize=8)
            ax.set_xlabel("Saat")
            ax.set_ylabel("Gün")
            st.pyplot(fig, clear_figure=True)

        elif mode == "Mekânsal":
            st.markdown("**GEOID — en yoğun bölgeler**")
            top = spatial_top_geoid(df, n=15, date_range=date_range, categories=pick_cats)
            st.dataframe(top, use_container_width=True, height=360)
            st.download_button("⬇️ Top GEOID listesi (CSV)", top.to_csv(index=False).encode("utf-8"),
                               file_name="top_geoid.csv", mime="text/csv")

        else:  # Suç Türleri
            st.markdown("**Suç türü dağılımları**")
            br = offense_breakdown(df, date_range=date_range)
            if not br.empty:
                st.bar_chart(br.set_index("offense").head(20))
                with st.expander("Tablo (ilk 100)"):
                    st.dataframe(br.head(100), use_container_width=True)
                st.download_button("⬇️ Tür dağılımı (CSV)", br.to_csv(index=False).encode("utf-8"),
                                   file_name="offense_breakdown.csv", mime="text/csv")
            else:
                st.info("Tür bilgisi bulunamadı.")

    # -------- Sağ panel: özet kutuları --------
    with right:
        st.markdown("**Özet**")
        # toplam olay (filtreli basit toplam)
        ycol = "crime_count" if "crime_count" in df.columns else None
        dmask = (pd.to_datetime(df.get("date", pd.NaT), errors="coerce") >= date_range[0]) & \
                (pd.to_datetime(df.get("date", pd.NaT), errors="coerce") <= date_range[1])
        sub = df[dmask].copy()
        if pick_cats and ccol:
            sub = sub[sub[ccol].astype(str).isin(pick_cats)]
        total = int(sub[ycol].sum()) if ycol in sub.columns else int(len(sub))
        st.metric("Toplam olay", f"{total:,}")

        # en yoğun gün/saat
        try:
            td = time_distributions(df, date_range=date_range, categories=pick_cats)
            top_hour = int(td["by_hour"].sort_values("value", ascending=False).iloc[0]["event_hour"])
            top_day = str(td["by_dow"].sort_values("value", ascending=False).iloc[0]["day_name"])
            st.metric("En yoğun saat", f"{top_hour:02d}:00")
            st.metric("En yoğun gün", top_day)
        except Exception:
            st.caption("Yoğunluk özetleri hesaplanamadı.")

        st.markdown("---")
        st.caption("📥 İndir")
        if mode == "Zamansal":
            st.download_button("Saatlik dağılım (CSV)",
                               td["by_hour"].to_csv(index=False).encode("utf-8"),
                               file_name="by_hour.csv", mime="text/csv")
            st.download_button("Gün × saat ısı (CSV)",
                               td["heat"].to_csv().encode("utf-8"),
                               file_name="heatmatrix.csv", mime="text/csv")

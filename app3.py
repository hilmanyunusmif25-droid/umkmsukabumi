"""
Dashboard Segmentasi UMKM Kabupaten Sukabumi
=============================================
Capstone Project - Magister Informatika (Data Science), Universitas Nusa Putra

Versi ini diselaraskan dengan TESIS Bab I-V dan notebook
"Clustering_UMKM_Sukabumi_FINAL.ipynb":
  2.057 data -> 0 duplikat -> 229 dihapus (29 non-numerik + 200 bernilai nol)
             -> 57 galat satuan (aset & omzet < Rp10.000) -> 1.771 data analisis
  log1p -> satu StandardScaler untuk semua algoritma
  K-Means (K=4) vs Hierarchical Ward (K=4) vs DBSCAN (eps=0,41; MinPts=5)
  Feasibility gate (cluster terkecil >= 5%, noise <= 10%) + skor komposit 30/30/20/20

Sumber data (pilih salah satu, letakkan satu folder dengan app.py):
  1. "UMKM 2057.xlsx"              -> data mentah, pipeline lengkap dijalankan (disarankan)
  2. "Data_UMKM_per_Cluster.xlsx"  -> data 1.771 hasil pembersihan dari notebook

Jalankan:  streamlit run app.py
"""

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

# =============================================================================
# KONFIGURASI
# =============================================================================
st.set_page_config(
    page_title="Dashboard Segmentasi UMKM Sukabumi",
    page_icon="📊",
    layout="wide",
)

FOLDER = Path(__file__).parent
FILE_MENTAH = FOLDER / "UMKM 2057.xlsx"
FILE_BERSIH = FOLDER / "Data_UMKM_per_Cluster.xlsx"

RANDOM_STATE = 42
K_FINAL = 4
DB_EPS = 0.41
DB_MINPTS = 5
MIN_SHARE = 0.05          # feasibility gate: cluster terkecil >= 5% data
MAX_NOISE = 0.10          # feasibility gate: noise <= 10% data
BATAS_GALAT = 10_000      # galat satuan: aset DAN omzet < Rp10.000 (sama dengan notebook)
BOBOT_DEFAULT = {"Silhouette": 0.30, "DBI": 0.30, "CHI": 0.20, "Balance": 0.20}

# Angka pembersihan data menurut tesis (Tabel 9). Dipakai bila sumber data
# adalah file yang sudah dibersihkan (tidak berisi data mentah 2.057 baris).
FUNNEL_TESIS = {"awal": 2057, "duplikat": 0, "non_numerik": 29, "nol": 200, "galat": 57}

# Skenario sensitivitas bobot (Tabel 6 dan Tabel 16 tesis)
SKENARIO_BOBOT = {
    "Utama (30/30/20/20)": (0.30, 0.30, 0.20, 0.20),
    "Setara (25/25/25/25)": (0.25, 0.25, 0.25, 0.25),
    "Dominan balance (20/20/20/40)": (0.20, 0.20, 0.20, 0.40),
    "Dominan Silhouette (50/20/15/15)": (0.50, 0.20, 0.15, 0.15),
    "Dominan DBI/CHI (20/40/30/10)": (0.20, 0.40, 0.30, 0.10),
    "Ekstrem (0/0/0/100)": (0.0, 0.0, 0.0, 1.0),
}

# Profil segmen (nama, interpretasi, dan rekomendasi mengikuti Tabel 17 & 19 tesis).
# Warna sudah divalidasi aman untuk buta warna (dengan penanda bentuk sebagai
# encoding kedua pada scatter).
SEGMEN = {
    "dagang": {
        "no": 1,
        "nama": "UMKM Perdagangan Beromzet Tinggi",
        "en": "High-Revenue Trading MSMEs",
        "warna": "#eda100",
        "simbol": "square",
        "ciri": "Aset kecil tetapi omzet tinggi (rasio omzet/aset ±40×) — pola usaha perdagangan/reseller dengan perputaran barang cepat.",
        "rekomendasi": [
            "Pembiayaan perdagangan (trade financing)",
            "Pendampingan digitalisasi, misalnya pendaftaran e-commerce",
            "Memanfaatkan kapasitas omzet yang besar meskipun asetnya kecil",
        ],
    },
    "berkembang": {
        "no": 2,
        "nama": "Usaha Mikro Bertumbuh",
        "en": "Growing Micro Enterprises",
        "warna": "#1baf7a",
        "simbol": "circle",
        "ciri": "Aset dan omzet seimbang (rasio omzet/aset ±1,3) — usaha mikro yang sedang berkembang, segmen terbesar.",
        "rekomendasi": [
            "Kredit modal kerja mikro",
            "Pembiayaan berbasis koperasi untuk mempercepat pertumbuhan",
            "Pelatihan manajemen usaha dan pencatatan keuangan",
        ],
    },
    "mapan": {
        "no": 3,
        "nama": "UMKM Mapan",
        "en": "Established MSMEs",
        "warna": "#2a78d6",
        "simbol": "diamond",
        "ciri": "Aset dan omzet tertinggi — paling berpotensi naik kelas.",
        "rekomendasi": [
            "Pendampingan peningkatan skala usaha (scale-up)",
            "Akses perbankan/KUR dengan plafon lebih besar",
            "Sertifikasi produk dan perluasan pasar",
        ],
    },
    "subsisten": {
        "no": 4,
        "nama": "Usaha Mikro Subsisten",
        "en": "Subsistence Micro Enterprises",
        "warna": "#e87ba4",
        "simbol": "triangle-up",
        "ciri": "Aset dan omzet paling kecil (median omzet Rp1,5 juta/tahun) — segmen paling rentan, didominasi pelaku usaha perempuan.",
        "rekomendasi": [
            "Kandidat utama hibah mikro (kapasitas belum memadai untuk kredit formal)",
            "Literasi keuangan dasar",
            "Pendampingan legalitas/registrasi usaha",
            "Pendekatan pemberdayaan perempuan pelaku usaha",
        ],
    },
}
URUTAN_SEGMEN = sorted(SEGMEN, key=lambda k: SEGMEN[k]["no"])
LABEL_SEGMEN = {k: f"Cluster {v['no']} · {v['nama']}" for k, v in SEGMEN.items()}
WARNA_LABEL = {LABEL_SEGMEN[k]: SEGMEN[k]["warna"] for k in SEGMEN}
SIMBOL_LABEL = {LABEL_SEGMEN[k]: SEGMEN[k]["simbol"] for k in SEGMEN}
URUTAN_LABEL = [LABEL_SEGMEN[k] for k in URUTAN_SEGMEN]
WARNA_ALGO = {"K-Means": "#2a78d6", "Hierarchical (Ward)": "#1baf7a", "DBSCAN": "#eda100"}

# Pengelompokan jenis usaha dari kolom Produksi (sama persis dengan notebook, subbab 4.8 tesis)
ATURAN_JENIS = [
    ("Kuliner", r"kue|keripik|kripik|makan|minum|kopi|coffee|snack|roti|sambal|frozen|kerupuk|opak|rangginang|dodol|madu|gula|jamu|herbal|teh|susu|bakso|baso|seblak|catering|katering|nasi|mie|cookies|cake|bolu|olahan|jajan|basreng|cireng|abon|sirup|jus|juice|tempe|tahu|ikan|ayam|jamur|pisang|singkong|ubi|kacang|camilan|cemilan|sale|wajit|ranginang|keju|stik|pangan|kuliner|food|bumbu|telur|rempeyek|peyek|manisan|selai|es |aren|cokelat|coklat|sate|pempek|risol|donat|lapis|beras|pecel|burger|pizza|martabak"),
    ("Fesyen & konveksi", r"baju|pakaian|konveksi|jahit|batik|fashion|hijab|kerudung|tas|sepatu|sandal|kaos|busana|bordir|rajut|tenun|mukena|sablon|garment"),
    ("Kerajinan", r"kerajinan|craft|anyam|bambu|kayu|souvenir|aksesoris|bros|manik|lukis|ukir|rotan|keramik|gerabah|tanaman hias|bunga|furniture|mebel"),
    ("Perdagangan/warung", r"warung|toko|sembako|jual|dagang|kelontong|grosir|agen|reseller|pulsa"),
    ("Jasa", r"jasa|salon|servis|service|laundry|cukur|barber|bengkel|print|percetakan|fotokopi|rias|travel|kursus|desain"),
    ("Pertanian/peternakan", r"tani|ternak|kambing|domba|sapi|bibit|pupuk|maggot|pakan|kolam|budidaya|sayur|buah|lele|nila|bebek|puyuh"),
]

# PRISMA (subbab 2.9, Gambar 7 tesis)
PRISMA = {
    "sumber": "Google Scholar melalui Publish or Perish",
    "tahun": "2021–2026",
    "kata_kunci": '"K-Means" "DBSCAN" "Hierarchical Clustering" "UMKM/MSME segmentation"',
    "tahap": [
        # (tahap, kotak utama, n, kotak dikeluarkan, n_keluar)
        ("Identifikasi", "Artikel hasil penelusuran Google Scholar melalui Publish or Perish (2021–2026)", 2740,
         "Duplikat/di luar cakupan dihapus", 2306),
        ("Penyaringan", "Artikel disaring berdasarkan judul dan abstrak", 434,
         "Dikeluarkan: tidak relevan dengan komparasi algoritma clustering/segmentasi UMKM/metrik evaluasi", 392),
        ("Kelayakan", "Artikel teks lengkap dinilai kelayakannya", 42,
         "Dikeluarkan: algoritma, objek, variabel, atau metrik tidak sesuai", 32),
        ("Inklusi", "Artikel yang disertakan dalam kajian", 10, None, None),
    ],
}

# Tabel 1 tesis: Rangkuman Penelitian Terdahulu
PENELITIAN_TERDAHULU = pd.DataFrame([
    (1, "Ramadhan et al. (2025) [14]", "K-Means, Hierarchical, DBSCAN", "UMKM pariwisata (Rembang)",
     "Membandingkan tiga algoritma, tetapi tidak mengkaji strategi prapemrosesan yang adil antaralgoritma maupun pendekatan evaluasi multi-metrik."),
    (2, "Wardani et al. (2023) [32]", "K-Means, DBSCAN, Hierarchical", "Segmentasi pasar",
     "Berfokus pada segmentasi pasar, bukan UMKM, dan tidak menggunakan CHI dalam evaluasi cluster."),
    (3, "Cahapin et al. (2023) [8]", "K-Means, Hierarchical, DBSCAN", "Data penerimaan mahasiswa",
     "Perbandingan dilakukan pada data pendidikan sehingga penerapannya pada segmentasi UMKM terbatas."),
    (4, "Fuchs & Höpken (2022) [33]", "K-Means, Hierarchical, DBSCAN", "Kajian clustering (pariwisata)",
     "Membahas algoritma secara teoretis dan demonstratif tanpa implementasi pada data UMKM riil."),
    (5, "Arrosyad et al. (2024) [34]", "K-Means", "Persebaran UMKM Jawa Barat",
     "Hanya menggunakan K-Means sehingga kinerjanya relatif terhadap algoritma lain tidak dapat dinilai."),
    (6, "Ameliana et al. (2024) [11]", "K-Means", "UMKM Jawa Barat",
     "Hanya berfokus pada K-Means tanpa membandingkan dengan Hierarchical Clustering atau DBSCAN."),
    (7, "Febriyani & Sari (2024) [35]", "Agglomerative Hierarchical", "Kata Kansei untuk konsep desain kemasan produk UMKM",
     "Hanya menggunakan Hierarchical Clustering dan objeknya preferensi konsumen, bukan data ekonomi UMKM."),
    (8, "Herdiana et al. (2025) [29]", "K-Means (Elbow)", "Penentuan jumlah cluster optimal",
     "Berfokus pada penentuan jumlah cluster, bukan perbandingan algoritma clustering."),
    (9, "Shutaywi & Kachouie (2021) [12]", "Silhouette Analysis", "Evaluasi clustering",
     "Berfokus pada metrik evaluasi tanpa penerapan pada segmentasi UMKM."),
    (10, "Ros et al. (2023) [30]", "PDBI", "Evaluasi clustering",
     "Mengusulkan metrik evaluasi baru tanpa membandingkan algoritma clustering."),
    (11, "Penelitian ini", "K-Means, Hierarchical (Ward), DBSCAN", "UMKM Kabupaten Sukabumi",
     "Membandingkan tiga algoritma pada dataset UMKM yang sama dengan penskalaan seragam, feasibility gate eksplisit, skor komposit multi-metrik, dan analisis sensitivitas bobot."),
], columns=["No", "Peneliti (Tahun)", "Metode", "Objek Penelitian", "Celah Penelitian"])

RESEARCH_GAP = [
    ("Celah algoritma", "Sebagian besar penelitian hanya memakai satu algoritma (terutama K-Means); studi komparatif ketiga algoritma umumnya di luar domain UMKM."),
    ("Celah evaluasi", "Evaluasi umumnya hanya memakai Silhouette Score, tanpa pemeriksaan keseimbangan ukuran cluster maupun kriteria kelayakan eksplisit."),
    ("Celah prapemrosesan", "Kemencengan aset/omzet dan keadilan normalisasi antaralgoritma belum banyak diperhatikan."),
    ("Celah wilayah", "Penelitian pemetaan UMKM dengan clustering yang berfokus pada Kabupaten Sukabumi masih terbatas."),
    ("Celah tujuan analisis", "Banyak penelitian berhenti pada pembentukan cluster tanpa interpretasi mendalam untuk rekomendasi kebijakan."),
]

# Kompatibilitas parameter lebar antar versi Streamlit
_ver = tuple(int(p) for p in st.__version__.split(".")[:2])
WIDE = {"width": "stretch"} if _ver >= (1, 49) else {"use_container_width": True}


# =============================================================================
# UTILITAS FORMAT
# =============================================================================
def angka(x, des=0):
    """Format angka gaya Indonesia: 1.234.567,89"""
    s = f"{x:,.{des}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


def rp(x):
    """Rupiah ringkas: Rp 1,3 jt / Rp 2,4 M"""
    if pd.isna(x):
        return "-"
    for batas, akhiran in [(1e12, "T"), (1e9, "M"), (1e6, "jt"), (1e3, "rb")]:
        if abs(x) >= batas:
            return f"Rp {angka(x / batas, 1)} {akhiran}"
    return f"Rp {angka(x)}"


def rp_penuh(x):
    return "-" if pd.isna(x) else f"Rp {angka(x)}"


LOG_TICKS = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]
LOG_TEXT = ["100", "1rb", "10rb", "100rb", "1jt", "10jt", "100jt", "1M", "10M"]


def sumbu_log(fig, x=True, y=True):
    if x:
        fig.update_xaxes(type="log", tickvals=LOG_TICKS, ticktext=LOG_TEXT)
    if y:
        fig.update_yaxes(type="log", tickvals=LOG_TICKS, ticktext=LOG_TEXT)
    return fig


def rapikan(fig, tinggi=420, legenda=True):
    fig.update_layout(
        height=tinggi,
        margin=dict(l=10, r=10, t=110 if legenda else 50, b=10),
        title=dict(y=0.985, yanchor="top", x=0, xanchor="left"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, title=None)
        if legenda else None,
        showlegend=legenda,
        hoverlabel=dict(namelength=-1),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridwidth=0.5)
    return fig


def tampil(fig):
    st.plotly_chart(fig, **WIDE)


def tabel(df, **kw):
    st.dataframe(df, hide_index=True, **WIDE, **kw)


# =============================================================================
# PIPELINE DATA & MODEL (di-cache)
# =============================================================================
def _norm_gender(x):
    if pd.isna(x):
        return "Tidak diketahui"
    s = str(x).strip().lower().replace(" ", "")
    if s.startswith("p"):
        return "Perempuan"
    if s.startswith("l"):
        return "Laki-laki"
    return "Tidak diketahui"


def jenis_usaha(teks):
    if pd.isna(teks) or str(teks).strip() == "":
        return "Tidak terisi"
    t = str(teks).lower()
    for nama, pola in ATURAN_JENIS:
        if re.search(pola, t):
            return nama
    return "Lainnya"


@st.cache_data(show_spinner="Membaca data UMKM...")
def baca_data(file_bytes: bytes) -> pd.DataFrame:
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    # File hasil notebook (Data_UMKM_per_Cluster.xlsx) menyimpan data di sheet "Semua Data"
    sheet = "Semua Data" if "Semua Data" in xls.sheet_names else xls.sheet_names[0]
    df = xls.parse(sheet)
    df.columns = df.columns.str.strip()
    return df


@st.cache_data(show_spinner="Membersihkan data dan melatih model...")
def jalankan_pipeline(raw: pd.DataFrame):
    df = raw.copy()
    # File yang sudah dibersihkan notebook memiliki kolom hasil cluster
    sudah_bersih = "Cluster_KMeans_raw" in df.columns
    kolom_turunan = ["Cluster_KMeans_raw", "Cluster", "Segmen", "Cluster_HC", "Cluster_DBSCAN",
                     "Jenis_Usaha", "JK", "Kec", "Program"]
    df = df.drop(columns=[c for c in kolom_turunan if c in df.columns])
    n_awal = len(df)
    n_kosong = df.isnull().sum()

    # 1. Duplikat identik (subbab 4.2)
    kolom_id = [c for c in ["Nama Lengkap", "Jenis Kelamin", "Usia", "Nama Usaha",
                            "Alamat", "Kecamatan", "Produksi", "Asset", "Omset"]
                if c in df.columns]
    n_dup = int(df.duplicated(subset=kolom_id, keep="first").sum())
    df = df.drop_duplicates(subset=kolom_id, keep="first").copy()

    # 2. Asset / Omset non-numerik atau bernilai nol
    a = pd.to_numeric(df["Asset"], errors="coerce")
    o = pd.to_numeric(df["Omset"], errors="coerce")
    non_num = a.isna() | o.isna()
    nol = (~non_num) & ((a <= 0) | (o <= 0))
    n_non_num, n_nol = int(non_num.sum()), int(nol.sum())
    df["Asset"], df["Omset"] = a, o
    df = df[(df["Asset"] > 0) & (df["Omset"] > 0)].copy()

    # 3. Galat satuan pencatatan: aset DAN omzet < Rp10.000
    galat = (df["Asset"] < BATAS_GALAT) & (df["Omset"] < BATAS_GALAT)
    df_salah = df[galat].copy()
    df = df[~galat].reset_index(drop=True)

    if sudah_bersih:          # angka pembersihan diambil dari tesis (Tabel 9)
        f = FUNNEL_TESIS
        n_awal, n_dup, n_non_num, n_nol = f["awal"], f["duplikat"], f["non_numerik"], f["nol"]
        n_galat = f["galat"]
        n_kosong = None
    else:
        n_galat = len(df_salah)

    funnel = pd.DataFrame({
        "Tahap": ["Data awal", "Duplikat identik", "Aset/omzet non-numerik",
                  "Aset/omzet bernilai nol", "Galat satuan pencatatan", "Data analisis akhir"],
        "Jumlah": [n_awal, -n_dup, -n_non_num, -n_nol, -n_galat, len(df)],
        "Keterangan": ["16 atribut",
                       "Tidak ada data yang dihapus" if n_dup == 0 else "Dihapus",
                       "Format tidak terbaca sebagai angka",
                       "Aset atau omzet = 0",
                       "Aset & omzet < Rp10.000 (maks Rp5.000 / Rp9.000)",
                       "Fitur: aset dan omzet tahunan"],
    })

    # Kolom kategorikal untuk tampilan dan profil sosial (aturan sama dengan notebook)
    df["Kecamatan"] = (df.get("Kecamatan", pd.Series(dtype=str)).fillna("Tidak diketahui")
                       .astype(str).str.strip().str.replace(r"\s+", " ", regex=True).str.title())
    if len(df_salah):
        df_salah["Kecamatan"] = df_salah["Kecamatan"].astype(str).str.strip().str.title()
    if "Jenis Kelamin" in df.columns:
        df["Jenis Kelamin"] = df["Jenis Kelamin"].map(_norm_gender)
    if "Pelatihan yang di ikuti" in df.columns:
        df["Pelatihan yang di ikuti"] = df["Pelatihan yang di ikuti"].fillna("-").astype(str).str.strip()
    df["Jenis Usaha"] = df["Produksi"].map(jenis_usaha) if "Produksi" in df.columns else "Tidak terisi"
    for c in ["Nama Usaha", "Produksi"]:
        if c in df.columns:
            df[c] = df[c].fillna("-").astype(str).str.strip()

    # 4. Transformasi: log1p -> StandardScaler (satu scaler untuk semua algoritma)
    X_log = np.log1p(df[["Asset", "Omset"]].values)
    scaler = StandardScaler().fit(X_log)
    X = scaler.transform(X_log)

    # 5. Model
    kmeans = KMeans(n_clusters=K_FINAL, init="k-means++", random_state=RANDOM_STATE, n_init=10).fit(X)
    lab_km = kmeans.labels_
    lab_hc = AgglomerativeClustering(n_clusters=K_FINAL, linkage="ward").fit_predict(X)
    lab_db = DBSCAN(eps=DB_EPS, min_samples=DB_MINPTS).fit_predict(X)

    # 6. Penamaan segmen (aturan sama dengan notebook): mapan = rata-rata aset tertinggi,
    #    subsisten = rata-rata omzet terendah, dagang = rasio omzet/aset tertinggi dari sisanya
    rata = df.groupby(lab_km)[["Asset", "Omset"]].mean()
    rata["rasio"] = rata["Omset"] / rata["Asset"]
    idx_mapan = int(rata["Asset"].idxmax())
    idx_sub = int(rata["Omset"].idxmin())
    sisa = [i for i in rata.index if i not in (idx_mapan, idx_sub)]
    idx_dagang = int(rata.loc[sisa, "rasio"].idxmax())
    idx_berk = int([i for i in sisa if i != idx_dagang][0])
    peta = {idx_mapan: "mapan", idx_sub: "subsisten", idx_dagang: "dagang", idx_berk: "berkembang"}

    df["Segmen_Key"] = [peta[l] for l in lab_km]
    df["Cluster"] = df["Segmen_Key"].map(lambda k: SEGMEN[k]["no"])
    df["Segmen"] = df["Segmen_Key"].map(LABEL_SEGMEN)

    # Nomor cluster Hierarchical diurutkan dari yang terbesar (719, 529, 290, 233 seperti tesis)
    urut_hc = pd.Series(lab_hc).value_counts().index.tolist()
    df["Cluster_HC"] = pd.Series(lab_hc).map({l: i + 1 for i, l in enumerate(urut_hc)}).values
    df["Cluster_DBSCAN"] = lab_db

    centroid = pd.DataFrame(np.expm1(scaler.inverse_transform(kmeans.cluster_centers_)),
                            columns=["Asset", "Omset"])
    centroid["Z_Asset"], centroid["Z_Omset"] = kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1]
    centroid["Segmen_Key"] = [peta[i] for i in range(K_FINAL)]
    centroid["Segmen"] = centroid["Segmen_Key"].map(LABEL_SEGMEN)

    return {
        "df": df, "df_salah": df_salah, "funnel": funnel, "n_awal": n_awal, "n_kosong": n_kosong,
        "sudah_bersih": sudah_bersih, "X": X, "X_log": X_log,
        "scaler": scaler, "kmeans": kmeans, "peta": peta, "centroid": centroid,
        "lab_km": lab_km, "lab_hc": lab_hc, "lab_db": lab_db,
    }


def metrik(X, label, n_total=None):
    """Silhouette, DBI, CHI, ukuran cluster, balance, dan feasibility gate."""
    label = np.asarray(label)
    n_total = n_total or len(label)
    mask = label != -1
    n_noise = int((~mask).sum())
    ukuran = np.bincount(label[mask]) if mask.any() else np.array([0])
    ukuran = ukuran[ukuran > 0]
    n_cl = len(ukuran)
    hasil = {"Cluster": n_cl, "Noise": n_noise,
             "Silhouette": np.nan, "DBI": np.nan, "CHI": np.nan,
             "Terkecil": int(ukuran.min()) if n_cl else 0,
             "Terbesar": int(ukuran.max()) if n_cl else 0}
    if n_cl >= 2:
        hasil["Silhouette"] = silhouette_score(X[mask], label[mask])
        hasil["DBI"] = davies_bouldin_score(X[mask], label[mask])
        hasil["CHI"] = calinski_harabasz_score(X[mask], label[mask])
    hasil["Balance"] = hasil["Terkecil"] / hasil["Terbesar"] if hasil["Terbesar"] else 0
    hasil["Lolos Gate"] = bool(hasil["Terkecil"] >= MIN_SHARE * n_total
                               and n_noise <= MAX_NOISE * n_total and n_cl >= 2)
    return hasil


def skor_komposit(tabel_m, bobot):
    """Persamaan (14): min-max antar kandidat; DBI dibalik (1 - DBI_norm); Balance dipakai langsung."""
    t = tabel_m.copy()

    def mm(s):
        rentang = s.max() - s.min()
        return pd.Series(1.0, index=s.index) if rentang == 0 else (s - s.min()) / rentang

    ws, wd, wc, wb = bobot
    t["Sil_norm"] = mm(t["Silhouette"])
    t["1-DBI_norm"] = 1 - mm(t["DBI"]) if t["DBI"].max() != t["DBI"].min() else 1.0
    t["CHI_norm"] = mm(t["CHI"])
    t["Skor"] = ws * t["Sil_norm"] + wd * t["1-DBI_norm"] + wc * t["CHI_norm"] + wb * t["Balance"]
    return t


@st.cache_data(show_spinner="Mengevaluasi K = 2 sampai 10...")
def evaluasi_k(X):
    baris = []
    for k in range(2, 11):
        lab = KMeans(n_clusters=k, init="k-means++", random_state=RANDOM_STATE, n_init=10).fit(X)
        m = metrik(X, lab.labels_)
        m.update({"K": k, "WCSS": lab.inertia_,
                  "Ukuran": ", ".join(str(v) for v in np.bincount(lab.labels_))})
        baris.append(m)
    t = pd.DataFrame(baris)
    t = skor_komposit(t, tuple(BOBOT_DEFAULT.values()))
    return t


@st.cache_data(show_spinner=False)
def k_distance(X, k):
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    d, _ = nn.kneighbors(X)
    return np.sort(d[:, k - 1])


@st.cache_data(show_spinner="Grid search DBSCAN (ε 0,05–1,00; MinPts 5/10/15/20)...")
def grid_dbscan(X):
    baris, n_kombinasi = [], 0
    for mp in [5, 10, 15, 20]:
        for eps in np.round(np.arange(0.05, 1.0001, 0.01), 2):
            n_kombinasi += 1
            lab = DBSCAN(eps=eps, min_samples=mp).fit_predict(X)
            n_cl = len(set(lab)) - (1 if -1 in lab else 0)
            n_noise = int((lab == -1).sum())
            if 2 <= n_cl <= 10 and n_noise <= MAX_NOISE * len(X):
                m = metrik(X, lab)
                m.update({"eps": eps, "MinPts": mp})
                baris.append(m)
    t = pd.DataFrame(baris)
    if len(t):
        t = skor_komposit(t, tuple(BOBOT_DEFAULT.values()))
    return t, n_kombinasi


@st.cache_data(show_spinner=False)
def matriks_linkage(X):
    return linkage(X, method="ward")


# =============================================================================
# SIDEBAR & MUAT DATA
# =============================================================================
with st.sidebar:
    st.markdown("## 📊 Segmentasi UMKM")
    st.caption("Kabupaten Sukabumi · Data 2024")
    halaman = st.radio(
        "Menu",
        ["🏠 Ringkasan", "📚 PRISMA & Penelitian Terdahulu", "🧹 Data & Preprocessing", "🔢 Penentuan Jumlah Cluster",
         "⚖️ Perbandingan Algoritma", "🧩 Profil Segmen", "🗺️ Sebaran Kecamatan",
         "🎯 Cek Segmen UMKM", "📄 Data & Unduhan"],
        label_visibility="collapsed",
    )
    st.divider()
    unggahan = st.file_uploader("Ganti sumber data (opsional)", type=["xlsx"],
                                help="Default: 'UMKM 2057.xlsx' (data mentah) atau 'Data_UMKM_per_Cluster.xlsx' "
                                     "di folder yang sama dengan app.py. File harus memiliki kolom Asset, Omset, dan Kecamatan.")
    st.caption(
        f"**Metode:** log1p → StandardScaler → K-Means (K={K_FINAL}, random_state={RANDOM_STATE}, n_init=10). "
        f"Pembanding: Hierarchical Ward (K={K_FINAL}) dan DBSCAN (ε={angka(DB_EPS, 2)}, MinPts={DB_MINPTS})."
    )

if unggahan is not None:
    raw = baca_data(unggahan.getvalue())
elif FILE_MENTAH.exists():
    raw = baca_data(FILE_MENTAH.read_bytes())
elif FILE_BERSIH.exists():
    raw = baca_data(FILE_BERSIH.read_bytes())
else:
    st.error(
        f"File data tidak ditemukan.\n\n"
        f"Letakkan **{FILE_MENTAH.name}** (disarankan) atau **{FILE_BERSIH.name}** "
        "satu folder dengan `app.py`, atau unggah file melalui sidebar."
    )
    st.stop()

if not {"Asset", "Omset"}.issubset(raw.columns):
    st.error("Kolom **Asset** dan **Omset** tidak ditemukan di file. Periksa nama kolom.")
    st.stop()

P = jalankan_pipeline(raw)
df, X = P["df"], P["X"]
N = len(df)
N_AWAL = P["n_awal"]

if P["sudah_bersih"]:
    st.sidebar.info("Sumber: data 1.771 hasil pembersihan notebook. Angka tahap pembersihan "
                    "diambil dari tesis (Tabel 9).")

# Tabel evaluasi algoritma (Tabel 13)
m_km = metrik(X, P["lab_km"])
m_hc = metrik(X, P["lab_hc"])
m_db = metrik(X, P["lab_db"])
eval_algo = pd.DataFrame([
    {"Algoritma": "K-Means", "Parameter": f"K={K_FINAL}", **m_km},
    {"Algoritma": "Hierarchical (Ward)", "Parameter": f"Ward, K={K_FINAL}", **m_hc},
    {"Algoritma": "DBSCAN", "Parameter": f"ε={angka(DB_EPS, 2)}, MinPts={DB_MINPTS}", **m_db},
])
layak = eval_algo[eval_algo["Lolos Gate"]].reset_index(drop=True)
skor_layak = skor_komposit(layak, tuple(BOBOT_DEFAULT.values())) if len(layak) else layak
terbaik = skor_layak.sort_values("Skor", ascending=False).iloc[0] if len(skor_layak) else None

# Profil segmen (Tabel 17)
profil = (
    df.groupby(["Cluster", "Segmen_Key", "Segmen"])
    .agg(Jumlah=("Asset", "size"),
         Asset_Rata2=("Asset", "mean"), Asset_Median=("Asset", "median"),
         Asset_Min=("Asset", "min"), Asset_Max=("Asset", "max"),
         Omset_Rata2=("Omset", "mean"), Omset_Median=("Omset", "median"),
         Omset_Min=("Omset", "min"), Omset_Max=("Omset", "max"))
    .reset_index()
    .sort_values("Cluster")
)
profil["Persen"] = profil["Jumlah"] / N * 100
profil["Rasio Omzet/Aset"] = profil["Omset_Rata2"] / profil["Asset_Rata2"]


def profil_sosial(data):
    """Tabel 18 tesis: jenis usaha, % perempuan, kecamatan & program terbanyak."""
    baris = []
    for key in URUTAN_SEGMEN:
        x = data[data["Segmen_Key"] == key]
        if x.empty:
            continue
        terisi = x[x["Jenis Usaha"] != "Tidak terisi"]
        top = (terisi["Jenis Usaha"].value_counts(normalize=True) * 100).head(2)
        baris.append({
            "Cluster": SEGMEN[key]["no"], "Segmen": SEGMEN[key]["nama"], "Jumlah": len(x),
            "Jenis usaha terisi": len(terisi) / len(x) * 100,
            "Jenis usaha dominan": "; ".join(f"{k} {angka(v, 1)}%" for k, v in top.items()),
            "Perempuan": (x["Jenis Kelamin"] == "Perempuan").mean() * 100,
            "Kecamatan terbanyak": ", ".join(x["Kecamatan"].value_counts().head(3).index),
            "Program pelatihan terbanyak": " | ".join(x["Pelatihan yang di ikuti"].value_counts().head(2).index),
        })
    return pd.DataFrame(baris)


def scatter_segmen(data, judul="", tinggi=520, centroid=True, ekstra=None, ruang="Rupiah"):
    """ruang='Rupiah' -> sumbu log rupiah; ruang='Standar' -> ruang fitur terstandardisasi (Gambar 11 tesis)."""
    hover = {c: True for c in ["Nama Usaha", "Kecamatan", "Produksi"] if c in data.columns}
    if ruang == "Standar":
        d = data.copy()
        z = P["scaler"].transform(np.log1p(d[["Asset", "Omset"]].values))
        d["Z_Asset"], d["Z_Omset"] = z[:, 0], z[:, 1]
        hover.update({"Asset": ":,.0f", "Omset": ":,.0f", "Segmen": False, "Z_Asset": ":.2f", "Z_Omset": ":.2f"})
        fig = px.scatter(
            d, x="Z_Asset", y="Z_Omset", color="Segmen", symbol="Segmen",
            color_discrete_map=WARNA_LABEL, symbol_map=SIMBOL_LABEL,
            category_orders={"Segmen": URUTAN_LABEL}, hover_data=hover, opacity=0.75,
            labels={"Z_Asset": "Aset (log1p, terstandardisasi)", "Z_Omset": "Omzet (log1p, terstandardisasi)"},
            title=judul,
        )
        cx, cy = "Z_Asset", "Z_Omset"
    else:
        hover.update({"Asset": ":,.0f", "Omset": ":,.0f", "Segmen": False})
        fig = px.scatter(
            data, x="Asset", y="Omset", color="Segmen", symbol="Segmen",
            color_discrete_map=WARNA_LABEL, symbol_map=SIMBOL_LABEL,
            category_orders={"Segmen": URUTAN_LABEL}, hover_data=hover, opacity=0.75,
            labels={"Asset": "Aset (Rp, skala log)", "Omset": "Omzet per tahun (Rp, skala log)"},
            title=judul,
        )
        cx, cy = "Asset", "Omset"
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color="rgba(255,255,255,0.8)")))
    if centroid:
        c = P["centroid"]
        fig.add_trace(go.Scatter(
            x=c[cx], y=c[cy], mode="markers", name="Centroid",
            marker=dict(symbol="x", size=16, color="#0b0b0b", line=dict(width=2, color="white")),
            text=c["Segmen"], customdata=c[["Asset", "Omset"]],
            hovertemplate="<b>Centroid</b><br>%{text}<br>Aset: Rp %{customdata[0]:,.0f}"
                          "<br>Omzet: Rp %{customdata[1]:,.0f}<extra></extra>",
        ))
    if ekstra is not None:
        fig.add_trace(ekstra)
    if ruang != "Standar":
        sumbu_log(fig)
    return rapikan(fig, tinggi)


def kartu_segmen(key, kolom):
    s = SEGMEN[key]
    baris = profil[profil["Segmen_Key"] == key].iloc[0]
    with kolom:
        with st.container(border=True):
            st.markdown(
                f"<div style='border-left:6px solid {s['warna']};padding-left:10px'>"
                f"<span style='font-size:0.8rem;opacity:0.7'>Cluster {s['no']}</span><br>"
                f"<b>{s['nama']}</b></div>",
                unsafe_allow_html=True,
            )
            st.metric("Jumlah UMKM", angka(baris["Jumlah"]))
            st.caption(f"**{angka(baris['Persen'], 2)}%** dari total UMKM")
            st.caption(f"Rata-rata aset **{rp(baris['Asset_Rata2'])}** · rata-rata omzet **{rp(baris['Omset_Rata2'])}**")
            st.caption(f"Median aset **{rp(baris['Asset_Median'])}** · median omzet **{rp(baris['Omset_Median'])}**")
            st.caption(s["ciri"])


def diagram_prisma():
    """Diagram alur PRISMA (Gambar 7 tesis) dengan shape & anotasi Plotly."""
    fig = go.Figure()
    tahap = PRISMA["tahap"]
    warna_tahap = ["#2a78d6", "#1baf7a", "#eda100", "#e87ba4"]
    tinggi, jarak = 1.5, 2.5
    y_pusat = [10 - jarak * i - 1.25 for i in range(len(tahap))]
    X_LBL, X_UT, X_KL = (0, 1.3), (1.8, 6.2), (7.0, 11.4)

    def kotak(x, yc, warna_garis, isi, tebal=False, latar="rgba(255,255,255,0)"):
        fig.add_shape(type="rect", x0=x[0], x1=x[1], y0=yc - tinggi / 2, y1=yc + tinggi / 2,
                      line=dict(color=warna_garis, width=2.5 if tebal else 1.5), fillcolor=latar)
        fig.add_annotation(x=(x[0] + x[1]) / 2, y=yc, text=isi, showarrow=False,
                           font=dict(size=13), align="center")

    def panah(x0, y0, x1, y1):
        fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=2, arrowsize=1.3, arrowwidth=1.8, arrowcolor="#52514e", text="")

    def bungkus(teks, lebar=42):
        kata, baris, skr = teks.split(), [], ""
        for k in kata:
            if len(skr) + len(k) + 1 > lebar:
                baris.append(skr)
                skr = k
            else:
                skr = f"{skr} {k}".strip()
        baris.append(skr)
        return "<br>".join(baris)

    for i, (nama, utama, n, keluar, n_keluar) in enumerate(tahap):
        yc = y_pusat[i]
        fig.add_shape(type="rect", x0=X_LBL[0], x1=X_LBL[1], y0=yc - tinggi / 2, y1=yc + tinggi / 2,
                      line=dict(width=0), fillcolor=warna_tahap[i])
        fig.add_annotation(x=sum(X_LBL) / 2, y=yc, text=f"<b>{nama}</b>", showarrow=False, textangle=-90,
                           font=dict(size=13, color="white"))
        kotak(X_UT, yc, warna_tahap[i], f"{bungkus(utama)}<br><b>(n = {angka(n)})</b>", tebal=True)
        if keluar:
            kotak(X_KL, yc, "#8a8a85", f"{bungkus(keluar)}<br><b>(n = {angka(n_keluar)})</b>",
                  latar="rgba(138,138,133,0.10)")
            panah(X_UT[1], yc, X_KL[0], yc)
        if i < len(tahap) - 1:
            panah(sum(X_UT) / 2, yc - tinggi / 2, sum(X_UT) / 2, y_pusat[i + 1] + tinggi / 2)

    fig.update_xaxes(visible=False, range=[-0.1, 11.5])
    fig.update_yaxes(visible=False, range=[-0.2, 10.1], scaleanchor=None)
    fig.update_layout(height=640, margin=dict(l=5, r=5, t=50, b=5), showlegend=False,
                      title=dict(text="Alur seleksi literatur berdasarkan metode PRISMA", x=0, xanchor="left"),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def pilih_ruang(key):
    opsi = ["Rupiah (skala log)", "Terstandardisasi"]
    v = st.radio("Sumbu scatter", opsi, horizontal=True, key=key)
    return "Standar" if v == opsi[1] else "Rupiah"


# =============================================================================
# HALAMAN 1 - RINGKASAN
# =============================================================================
if halaman == "🏠 Ringkasan":
    st.title("Dashboard Segmentasi UMKM Kabupaten Sukabumi")
    st.caption("Perbandingan K-Means, Hierarchical Clustering, dan DBSCAN berdasarkan aset dan omzet tahunan UMKM (data Pemerintah Kabupaten Sukabumi, 2024)")

    c = st.columns(5)
    c[0].metric("Data awal", angka(N_AWAL))
    c[1].metric("Data dianalisis", angka(N), help=f"{angka(N_AWAL - N)} data dikeluarkan saat pembersihan")
    c[2].metric("Jumlah segmen", K_FINAL)
    c[3].metric("Algoritma terbaik", terbaik["Algoritma"] if terbaik is not None else "-")
    c[4].metric("Skor komposit", angka(terbaik["Skor"], 4) if terbaik is not None else "-")

    st.subheader("Empat segmen UMKM")
    cols = st.columns(4)
    for i, key in enumerate(URUTAN_SEGMEN):
        kartu_segmen(key, cols[i])

    kiri, kanan = st.columns([3, 2])
    with kiri:
        tampil(scatter_segmen(df, "Peta aset vs omzet per segmen (K-Means, K=4)"))
    with kanan:
        fig = px.bar(
            profil, x="Jumlah", y="Segmen", orientation="h", color="Segmen",
            color_discrete_map=WARNA_LABEL, category_orders={"Segmen": URUTAN_LABEL},
            text=profil.apply(lambda r: f"{angka(r['Jumlah'])} ({angka(r['Persen'], 2)}%)", axis=1),
            title="Jumlah UMKM per segmen",
        )
        fig.update_traces(textposition="outside", marker_line_width=0, cliponaxis=False)
        fig.update_layout(yaxis_title=None, xaxis_title="Jumlah UMKM", bargap=0.35)
        tampil(rapikan(fig, 520, legenda=False))

    sub = profil[profil["Segmen_Key"] == "subsisten"].iloc[0]
    pr_sub = (df.loc[df["Segmen_Key"] == "subsisten", "Jenis Kelamin"] == "Perempuan").mean() * 100
    st.subheader("Temuan utama")
    st.markdown(
        f"""
- **K-Means terpilih** sebagai algoritma terbaik dengan skor komposit **{angka(terbaik['Skor'], 4)}**, unggul atas Hierarchical Clustering pada Silhouette ({angka(m_km['Silhouette'], 4)} vs {angka(m_hc['Silhouette'], 4)}), DBI ({angka(m_km['DBI'], 4)} vs {angka(m_hc['DBI'], 4)}), CHI ({angka(m_km['CHI'], 2)} vs {angka(m_hc['CHI'], 2)}), dan balance ({angka(m_km['Balance'], 4)} vs {angka(m_hc['Balance'], 4)}) pada K yang sama (K=4).
- **DBSCAN** memiliki Silhouette tertinggi ({angka(m_db['Silhouette'], 4)}), tetapi hanya memisahkan {m_db['Terkecil']} UMKM dari {angka(m_db['Terbesar'])} UMKM lainnya sehingga **tidak lolos feasibility gate** (cluster terkecil ≥ 5% = {int(np.ceil(MIN_SHARE * N))} UMKM).
- Segmen terbesar adalah **{profil.sort_values('Jumlah').iloc[-1]['Segmen']}** ({angka(profil['Persen'].max(), 2)}%).
- Segmen **Usaha Mikro Subsisten** ({angka(sub['Persen'], 2)}%) adalah kelompok paling rentan (median omzet {rp(sub['Omset_Median'])}/tahun) dan didominasi pelaku usaha perempuan ({angka(pr_sub, 1)}%).
"""
    )

# =============================================================================
# HALAMAN PRISMA - PENELITIAN TERDAHULU
# =============================================================================
elif halaman == "📚 PRISMA & Penelitian Terdahulu":
    st.title("PRISMA & Penelitian Terdahulu")
    st.caption("Seleksi literatur secara sistematis menggunakan PRISMA (Preferred Reporting Items for Systematic "
               "Reviews and Meta-Analyses) — subbab 2.9 dan Gambar 7 tesis")

    tahap = PRISMA["tahap"]
    c = st.columns(4)
    for col, (nama, _, n, _, _) in zip(c, tahap):
        col.metric(nama, f"{angka(n)} artikel")
    st.caption(f"**Sumber:** {PRISMA['sumber']} · **Tahun terbit:** {PRISMA['tahun']} · "
               f"**Kata kunci:** {PRISMA['kata_kunci']}")

    kiri, kanan = st.columns([3, 2])
    with kiri:
        tampil(diagram_prisma())
    with kanan:
        st.subheader("Penyusutan artikel per tahap")
        fn = pd.DataFrame({"Tahap": [t[0] for t in tahap], "Jumlah": [t[2] for t in tahap]})
        fig = go.Figure(go.Funnel(y=fn["Tahap"], x=fn["Jumlah"], textinfo="value+percent initial",
                                  texttemplate="%{value:,} artikel<br>(%{percentInitial:.2%} dari awal)",
                                  marker=dict(color=["#2a78d6", "#1baf7a", "#eda100", "#e87ba4"])))
        fig.update_layout(separators=",.")
        tampil(rapikan(fig, 380, legenda=False))
        ringkas = pd.DataFrame([{"Tahap": t[0], "Masuk": angka(t[2]),
                                 "Dikeluarkan": angka(t[4]) if t[4] else "-",
                                 "Lanjut ke tahap berikutnya": angka(t[2] - t[4]) if t[4] else "-",
                                 "Alasan dikeluarkan": t[3].replace("Dikeluarkan: ", "") if t[3] else "-"}
                                for t in tahap])
        tabel(ringkas)

    st.subheader("Rangkuman penelitian terdahulu (Tabel 1)")
    a, b, c = st.columns([2, 2, 3])
    filt_metode = a.multiselect("Filter algoritma", ["K-Means", "Hierarchical", "DBSCAN", "Metrik evaluasi"])
    hanya_umkm = b.toggle("Hanya objek UMKM", value=False)
    t = PENELITIAN_TERDAHULU.copy()
    if filt_metode:
        m = np.zeros(len(t), dtype=bool)
        for f in filt_metode:
            if f == "Metrik evaluasi":
                m |= t["Objek Penelitian"].str.contains("Evaluasi|jumlah cluster", case=False).values
            else:
                m |= t["Metode"].str.contains(f, case=False).values
        t = t[m]
    if hanya_umkm:
        t = t[t["Objek Penelitian"].str.contains("UMKM", case=False)]
    c.caption(f"Menampilkan {len(t)} dari {len(PENELITIAN_TERDAHULU)} baris (10 artikel hasil PRISMA + penelitian ini).")
    tabel(t, column_config={"Celah Penelitian": st.column_config.TextColumn(width="large")})

    st.subheader("Cakupan metode pada penelitian terdahulu")
    pt = PENELITIAN_TERDAHULU[PENELITIAN_TERDAHULU["No"] <= 10]
    cak = pd.DataFrame({
        "Aspek": ["Memakai K-Means", "Memakai Hierarchical", "Memakai DBSCAN",
                  "Membandingkan ketiga algoritma", "Objek UMKM", "Fokus metrik/penentuan K"],
        "Jumlah artikel": [
            pt["Metode"].str.contains("K-Means").sum(),
            pt["Metode"].str.contains("Hierarchical").sum(),
            pt["Metode"].str.contains("DBSCAN").sum(),
            (pt["Metode"].str.contains("K-Means") & pt["Metode"].str.contains("Hierarchical")
             & pt["Metode"].str.contains("DBSCAN")).sum(),
            pt["Objek Penelitian"].str.contains("UMKM").sum(),
            pt["Objek Penelitian"].str.contains("Evaluasi|jumlah cluster", case=False).sum(),
        ]})
    a, b = st.columns([3, 2])
    with a:
        fig = px.bar(cak, x="Jumlah artikel", y="Aspek", orientation="h", text="Jumlah artikel",
                     color_discrete_sequence=["#2a78d6"], title="Dari 10 artikel yang disertakan")
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(yaxis=dict(autorange="reversed", title=None), xaxis=dict(range=[0, 10.5]), bargap=0.3)
        tampil(rapikan(fig, 340, legenda=False))
    with b:
        st.markdown("**Research gap (subbab 2.10)**")
        for judul, isi in RESEARCH_GAP:
            st.markdown(f"- **{judul}:** {isi}")

    st.info("**Posisi penelitian ini:** membandingkan K-Means, Hierarchical Clustering (Ward), dan DBSCAN pada data UMKM "
            "Kabupaten Sukabumi dengan prapemrosesan seragam (log1p + satu StandardScaler), feasibility gate yang ditetapkan "
            "sejak awal, skor komposit multi-metrik, dan analisis sensitivitas bobot.")

# =============================================================================
# HALAMAN 2 - DATA & PREPROCESSING
# =============================================================================
elif halaman == "🧹 Data & Preprocessing":
    st.title("Data & Preprocessing")
    st.caption("Tahapan KDD: pembersihan → seleksi variabel → transformasi log1p → standardisasi")

    kiri, kanan = st.columns([2, 3])
    with kiri:
        st.subheader("Alur pembersihan data")
        f = P["funnel"].copy()
        f["Jumlah data"] = f["Jumlah"].apply(lambda v: f"{'−' if v < 0 else ''}{angka(abs(v))}")
        tabel(f[["Tahap", "Jumlah data", "Keterangan"]])
        st.info(
            f"**{FUNNEL_TESIS['galat'] if P['sudah_bersih'] else len(P['df_salah'])} data galat satuan** memiliki aset paling besar "
            "Rp5.000 dan omzet paling besar Rp9.000 — sekitar 500 kali lebih kecil daripada median kelompok terendah berikutnya. "
            "Seluruhnya berada pada blok data terakhir (nomor urut 1.976–2.057), terbanyak dari Palabuhanratu (21), "
            "Jampang Tengah (15), dan Nyalindung (9). Pola ini konsisten dengan galat satuan pencatatan (ribuan rupiah), "
            "sehingga dikeluarkan, bukan dikoreksi."
        )
        if len(P["df_salah"]):
            with st.expander(f"Lihat {len(P['df_salah'])} data galat satuan"):
                kol = [c for c in ["No", "Kecamatan", "Asset", "Omset"] if c in P["df_salah"].columns]
                ds = P["df_salah"][kol].copy()
                tabel(ds, height=300)
                tabel(ds["Kecamatan"].value_counts().rename_axis("Kecamatan").reset_index(name="Jumlah"))
        if P["n_kosong"] is not None:
            with st.expander(f"Jumlah nilai kosong pada data awal (n = {angka(N_AWAL)})"):
                tabel(P["n_kosong"].rename_axis("Atribut").reset_index(name="Nilai kosong"))
    with kanan:
        st.subheader("Mengapa transformasi log?")
        var = st.segmented_control("Variabel", ["Asset", "Omset"], default="Asset") \
            if hasattr(st, "segmented_control") else st.radio("Variabel", ["Asset", "Omset"], horizontal=True)
        var = var or "Asset"
        nama_var = "Aset" if var == "Asset" else "Omzet"
        a, b = st.columns(2)
        with a:
            fig = px.histogram(df, x=var, nbins=40, title=f"{nama_var} sebelum transformasi (Rp)",
                               color_discrete_sequence=["#2a78d6"])
            fig.update_traces(marker_line_width=1, marker_line_color="white")
            fig.update_layout(yaxis_title="Jumlah UMKM", xaxis_title=f"{nama_var} (Rp)", bargap=0.05)
            tampil(rapikan(fig, 340, legenda=False))
        with b:
            d = pd.DataFrame({f"log1p({nama_var})": np.log1p(df[var])})
            fig = px.histogram(d, x=f"log1p({nama_var})", nbins=40, title=f"{nama_var} setelah log1p",
                               color_discrete_sequence=["#1baf7a"])
            fig.update_traces(marker_line_width=1, marker_line_color="white")
            fig.update_layout(yaxis_title="Jumlah UMKM", bargap=0.05)
            tampil(rapikan(fig, 340, legenda=False))
        skew_asli, skew_log = df[var].skew(), np.log1p(df[var]).skew()
        st.caption(f"Skewness {nama_var.lower()}: **{angka(skew_asli, 2)}** (asli) → **{angka(skew_log, 2)}** (log1p). "
                   "Distribusi asli sangat menceng ke kanan; tanpa log, segelintir usaha bernilai ekstrem akan mendominasi perhitungan jarak.")
        st.caption(f"Data sangat diskret: hanya **{df['Asset'].nunique()}** nilai aset unik dan **{df['Omset'].nunique()}** nilai omzet unik "
                   f"(contoh: aset Rp10 juta muncul {(df['Asset'] == 10_000_000).sum()} kali, omzet Rp13,4 juta {(df['Omset'] == 13_400_000).sum()} kali).")

    st.subheader("Statistik deskriptif")
    t1, t2, t3 = st.tabs(["Nilai asli (Rp) — Tabel 10", "Setelah log1p — Tabel 11", "Setelah StandardScaler"])
    with t1:
        s = df[["Asset", "Omset"]]
        st10 = pd.DataFrame({
            "Statistik": ["Rata-rata", "Simpangan baku", "Minimum", "Kuartil 1 (Q1)", "Median",
                          "Kuartil 3 (Q3)", "Maksimum", "Skewness"],
            **{f"{'Aset' if c == 'Asset' else 'Omzet'} (Rp)": [
                angka(s[c].mean()), angka(s[c].std()), angka(s[c].min()), angka(s[c].quantile(.25)),
                angka(s[c].median()), angka(s[c].quantile(.75)), angka(s[c].max()), angka(s[c].skew(), 2)]
               for c in ["Asset", "Omset"]},
        })
        tabel(st10)
    with t2:
        s = pd.DataFrame(P["X_log"], columns=["Asset", "Omset"])
        st11 = pd.DataFrame({
            "Statistik": ["Rata-rata", "Simpangan baku", "Minimum", "Median", "Maksimum", "Skewness"],
            **{f"log1p({'Aset' if c == 'Asset' else 'Omzet'})": [
                angka(s[c].mean(), 2), angka(s[c].std(), 2), angka(s[c].min(), 2),
                angka(s[c].median(), 2), angka(s[c].max(), 2), angka(s[c].skew(), 2)]
               for c in ["Asset", "Omset"]},
        })
        tabel(st11)
    with t3:
        s = pd.DataFrame(X, columns=["Aset", "Omzet"]).describe().T
        tabel(s.reset_index().rename(columns={"index": "Variabel"}).round(3))

    st.subheader("Profil responden")
    a, b, c = st.columns(3)
    with a:
        if "Jenis Kelamin" in df.columns:
            g = df["Jenis Kelamin"].value_counts().reset_index()
            g.columns = ["Jenis Kelamin", "Jumlah"]
            g["Label"] = g.apply(lambda r: f"{angka(r['Jumlah'])} ({angka(r['Jumlah'] / N * 100, 1)}%)", axis=1)
            fig = px.bar(g, x="Jenis Kelamin", y="Jumlah", text="Label", title="Jenis kelamin pemilik",
                         color_discrete_sequence=["#2a78d6"])
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(xaxis_title=None, bargap=0.45)
            tampil(rapikan(fig, 360, legenda=False))
    with b:
        g = df["Kecamatan"].value_counts().head(10).reset_index()
        g.columns = ["Kecamatan", "Jumlah"]
        fig = px.bar(g, x="Jumlah", y="Kecamatan", orientation="h", text="Jumlah",
                     title="10 kecamatan dengan UMKM terbanyak", color_discrete_sequence=["#2a78d6"])
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(yaxis=dict(autorange="reversed", title=None), bargap=0.3)
        tampil(rapikan(fig, 360, legenda=False))
    with c:
        g = df["Jenis Usaha"].value_counts().reset_index()
        g.columns = ["Jenis usaha", "Jumlah"]
        fig = px.bar(g, x="Jumlah", y="Jenis usaha", orientation="h", text="Jumlah",
                     title="Jenis usaha (dari kolom Produksi)", color_discrete_sequence=["#2a78d6"])
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(yaxis=dict(autorange="reversed", title=None), bargap=0.3)
        tampil(rapikan(fig, 360, legenda=False))
    if "Pelatihan yang di ikuti" in df.columns:
        n_prog = df["Pelatihan yang di ikuti"].nunique()
        tdk = (df["Jenis Usaha"] == "Tidak terisi").mean() * 100
        st.caption(f"Atribut pelatihan terisi pada seluruh data dan berisi **{n_prog} program** berbeda — seluruh UMKM adalah peserta "
                   f"program pelatihan pemerintah daerah. Jenis usaha kosong pada **{angka(tdk, 1)}%** data analisis.")
        with st.expander("Program pelatihan paling banyak diikuti"):
            g = df["Pelatihan yang di ikuti"].value_counts().head(10).reset_index()
            g.columns = ["Program pelatihan", "Jumlah"]
            tabel(g)

# =============================================================================
# HALAMAN 3 - PENENTUAN K
# =============================================================================
elif halaman == "🔢 Penentuan Jumlah Cluster":
    st.title("Penentuan Jumlah Cluster (K)")
    st.caption("K-Means dijalankan untuk K = 2–10 dan dievaluasi dengan empat indikator (Gambar 10 & Tabel 12 tesis)")
    ek = evaluasi_k(X)

    def garis(kolom, judul, arah):
        fig = px.line(ek, x="K", y=kolom, markers=True, title=f"{judul}  ·  {arah}",
                      color_discrete_sequence=["#2a78d6"])
        fig.update_traces(line_width=2, marker_size=8)
        pilih = ek[ek["K"] == K_FINAL]
        fig.add_trace(go.Scatter(x=pilih["K"], y=pilih[kolom], mode="markers", name=f"K={K_FINAL} (terpilih)",
                                 marker=dict(size=15, color="#e87ba4", line=dict(width=2, color="white")),
                                 hovertemplate=f"K={K_FINAL}<br>{kolom}: %{{y:.4f}}<extra></extra>"))
        fig.update_layout(xaxis=dict(dtick=1, title="Jumlah cluster (K)"), yaxis_title=None)
        return rapikan(fig, 300, legenda=False)

    a, b = st.columns(2)
    with a:
        tampil(garis("WCSS", "Elbow Method (WCSS)", "cari 'siku'"))
        tampil(garis("DBI", "Davies-Bouldin Index", "lebih rendah lebih baik"))
    with b:
        tampil(garis("Silhouette", "Silhouette Score", "lebih tinggi lebih baik"))
        tampil(garis("CHI", "Calinski-Harabasz Index", "lebih tinggi lebih baik"))

    st.subheader("Skor berbobot per K")
    kiri, kanan = st.columns([3, 2])
    with kiri:
        e = ek.copy()
        e["Warna"] = np.where(e["K"] == K_FINAL, f"K={K_FINAL} (terpilih)", "Kandidat lain")
        fig = px.bar(e, x="K", y="Skor", color="Warna", text=e["Skor"].map(lambda v: angka(v, 3)),
                     color_discrete_map={f"K={K_FINAL} (terpilih)": "#e87ba4", "Kandidat lain": "#2a78d6"},
                     title="Skor berbobot (30% Sil + 30% DBI + 20% CHI + 20% Balance)")
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(xaxis=dict(dtick=1, title="K"), yaxis_title="Skor", bargap=0.35)
        tampil(rapikan(fig, 360))
    with kanan:
        k2 = ek.loc[ek.K == 2].iloc[0]
        st.markdown(
            f"""
**Mengapa K = 4?**

- **K = 2** mendapat skor tertinggi ({angka(k2['Skor'], 3)}), tetapi hanya membagi UMKM menjadi dua kelompok besar ({angka(k2['Terkecil'])} dan {angka(k2['Terbesar'])} UMKM) — terlalu umum untuk kebijakan yang terdiferensiasi.
- **K = 6** ({angka(ek.loc[ek.K == 6, 'Skor'].iloc[0], 3)}) menghasilkan segmen yang terlalu sempit untuk diinterpretasikan hanya dengan dua variabel.
- **K = 4** ({angka(ek.loc[ek.K == 4, 'Skor'].iloc[0], 3)}) menjadi titik siku Elbow, memiliki keseimbangan ukuran cluster yang baik ({angka(ek.loc[ek.K == 4, 'Balance'].iloc[0], 3)}), dan menyeimbangkan kelayakan statistik dengan kebutuhan segmentasi kebijakan.
"""
        )
    tampilan = ek[["K", "WCSS", "Silhouette", "DBI", "CHI", "Terkecil", "Terbesar", "Balance", "Skor"]].copy()
    tampilan["Cluster terkecil–terbesar"] = tampilan["Terkecil"].map(angka) + "–" + tampilan["Terbesar"].map(angka)
    tampilan = tampilan[["K", "WCSS", "Silhouette", "DBI", "CHI", "Cluster terkecil–terbesar", "Balance", "Skor"]]
    tabel(tampilan.style.format({"WCSS": lambda v: angka(v, 2), "Silhouette": lambda v: angka(v, 4),
                                 "DBI": lambda v: angka(v, 4), "CHI": lambda v: angka(v, 2),
                                 "Balance": lambda v: angka(v, 4), "Skor": lambda v: angka(v, 4)})
          .highlight_between(subset=["K"], left=K_FINAL, right=K_FINAL, color="rgba(232,123,164,0.25)"))

    st.subheader("Pratinjau hasil K-Means untuk K lain")
    k_pilih = st.slider("Pilih K", 2, 10, K_FINAL)
    lab = KMeans(n_clusters=k_pilih, init="k-means++", random_state=RANDOM_STATE, n_init=10).fit_predict(X)
    d = df[["Asset", "Omset", "Nama Usaha", "Kecamatan"]].copy()
    d["Cluster"] = (lab + 1).astype(str)
    urut = [str(i) for i in range(1, k_pilih + 1)]
    fig = px.scatter(d, x="Asset", y="Omset", color="Cluster", category_orders={"Cluster": urut},
                     hover_data={"Nama Usaha": True, "Kecamatan": True, "Asset": ":,.0f", "Omset": ":,.0f"},
                     opacity=0.75, title=f"K-Means dengan K = {k_pilih}",
                     labels={"Asset": "Aset (Rp, skala log)", "Omset": "Omzet per tahun (Rp, skala log)"},
                     color_discrete_sequence=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                                              "#008300", "#6250d6", "#e34948", "#52514e", "#8a8a85"])
    fig.update_traces(marker=dict(size=7))
    tampil(rapikan(sumbu_log(fig), 480))
    if k_pilih > 3:
        st.caption("Catatan: lebih dari 3 warna pada scatter sulit dibedakan untuk sebagian pembaca; arahkan kursor untuk melihat nomor cluster.")

# =============================================================================
# HALAMAN 4 - PERBANDINGAN ALGORITMA
# =============================================================================
elif halaman == "⚖️ Perbandingan Algoritma":
    st.title("Perbandingan Algoritma Clustering")
    st.caption(f"Semua algoritma memakai ruang fitur yang sama (log1p + StandardScaler). "
               f"Feasibility gate: cluster terkecil ≥ {int(MIN_SHARE*100)}% data (≥ {int(np.ceil(MIN_SHARE*N))} UMKM) dan noise ≤ {int(MAX_NOISE*100)}%.")

    st.subheader("Kinerja algoritma (Tabel 13)")
    t = eval_algo.copy()
    t["Rentang ukuran"] = t["Terkecil"].map(angka) + "–" + t["Terbesar"].map(angka)
    tabel(t[["Algoritma", "Parameter", "Cluster", "Noise", "Silhouette", "DBI", "CHI",
             "Rentang ukuran", "Balance"]]
          .style.format({"Silhouette": lambda v: angka(v, 4), "DBI": lambda v: angka(v, 4),
                         "CHI": lambda v: angka(v, 2), "Balance": lambda v: angka(v, 4)}))

    st.subheader("Metrik per algoritma (Gambar 14)")
    cols = st.columns(4)
    for col, (kolom, ket) in zip(cols, [("Silhouette", "↑ lebih baik"), ("DBI", "↓ lebih baik"),
                                       ("CHI", "↑ lebih baik"), ("Balance", "↑ lebih seimbang")]):
        with col:
            fig = px.bar(eval_algo, x="Algoritma", y=kolom, color="Algoritma", color_discrete_map=WARNA_ALGO,
                         text=eval_algo[kolom].map(lambda v: angka(v, 2 if kolom == "CHI" else 4)),
                         title=f"{kolom} ({ket})")
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(xaxis_title=None, yaxis_title=None, bargap=0.35)
            tampil(rapikan(fig, 320, legenda=False))

    st.subheader("Hasil feasibility gate (Tabel 14)")
    g = eval_algo[["Algoritma", "Terkecil", "Noise", "Lolos Gate"]].copy()
    g["% dari sampel"] = (g["Terkecil"] / N * 100).map(lambda v: f"{angka(v, 2)}%")
    g["Noise (%)"] = g["Noise"].map(lambda v: f"{v} ({angka(v / N * 100, 2)}%)")
    g["Lolos?"] = np.where(g["Lolos Gate"], "✅ Ya", "❌ Tidak")
    tabel(g.rename(columns={"Terkecil": "Cluster terkecil"})[["Algoritma", "Cluster terkecil", "% dari sampel", "Noise (%)", "Lolos?"]])

    st.subheader("Skor komposit berbobot (Tabel 15)")
    tabel(skor_layak[["Algoritma", "Sil_norm", "1-DBI_norm", "CHI_norm", "Balance", "Skor"]]
          .rename(columns={"Skor": "Skor komposit"})
          .style.format({c: (lambda v: angka(v, 4)) for c in ["Sil_norm", "1-DBI_norm", "CHI_norm", "Balance", "Skor komposit"]}))

    kiri, kanan = st.columns([2, 3])
    with kiri:
        with st.container(border=True):
            st.markdown("**Coba atur bobot** (otomatis dinormalisasi agar total = 1)")
            ws = st.slider("Silhouette", 0.0, 1.0, BOBOT_DEFAULT["Silhouette"], 0.05)
            wd = st.slider("DBI (dibalik)", 0.0, 1.0, BOBOT_DEFAULT["DBI"], 0.05)
            wc = st.slider("CHI", 0.0, 1.0, BOBOT_DEFAULT["CHI"], 0.05)
            wb = st.slider("Balance", 0.0, 1.0, BOBOT_DEFAULT["Balance"], 0.05)
            total = ws + wd + wc + wb or 1
            bobot = (ws / total, wd / total, wc / total, wb / total)
            pakai_gate = st.toggle("Terapkan feasibility gate", value=True,
                                   help="Matikan untuk melihat bagaimana DBSCAN 'menang' jika hanya mengandalkan metrik internal.")
    with kanan:
        kandidat = eval_algo[eval_algo["Lolos Gate"]] if pakai_gate else eval_algo
        s = skor_komposit(kandidat.reset_index(drop=True), bobot).sort_values("Skor", ascending=False)
        fig = px.bar(s, x="Skor", y="Algoritma", orientation="h", color="Algoritma",
                     color_discrete_map=WARNA_ALGO, text=s["Skor"].map(lambda v: angka(v, 4)),
                     title="Skor komposit " + ("(hanya kandidat layak)" if pakai_gate else "(tanpa gate)"))
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(xaxis=dict(range=[0, 1.1], title="Skor"), yaxis=dict(autorange="reversed", title=None),
                          bargap=0.4)
        tampil(rapikan(fig, 300, legenda=False))
        juara = s.iloc[0]
        if pakai_gate:
            st.success(f"**{juara['Algoritma']}** terbaik dengan skor **{angka(juara['Skor'], 4)}**. "
                       f"DBSCAN dikeluarkan karena cluster terkecilnya hanya {m_db['Terkecil']} UMKM "
                       f"({angka(m_db['Terkecil'] / N * 100, 2)}% data).")
        else:
            st.warning(f"Tanpa gate, **{juara['Algoritma']}** memperoleh skor tertinggi — padahal partisinya "
                       "tidak berguna untuk segmentasi. Inilah alasan feasibility gate diperlukan.")

    st.subheader("Analisis sensitivitas bobot (Tabel 16)")
    rows = []
    for nama, w in SKENARIO_BOBOT.items():
        sc = skor_komposit(layak, w).set_index("Algoritma")["Skor"]
        rows.append({"Skenario bobot (Sil/DBI/CHI/Balance)": nama, **{a: sc.get(a, np.nan) for a in layak["Algoritma"]}})
    sens = pd.DataFrame(rows)
    if {"K-Means", "Hierarchical (Ward)"}.issubset(sens.columns):
        sens["Selisih"] = sens["K-Means"] - sens["Hierarchical (Ward)"]
        sens["Terbaik"] = np.where(sens["Selisih"] > 0, "K-Means", "Hierarchical (Ward)")
    tabel(sens.style.format({c: (lambda v: angka(v, 4)) for c in ["K-Means", "Hierarchical (Ward)"] if c in sens.columns}
                            | {"Selisih": lambda v: angka(v, 3)}))
    st.caption("K-Means unggul pada seluruh skema, termasuk skema ekstrem yang hanya mempertimbangkan balance — pemilihan K-Means bersifat robust terhadap pembobotan.")

    st.subheader("Detail tiap algoritma")
    tk, th, td = st.tabs(["K-Means", "Hierarchical (Ward)", "DBSCAN"])
    with tk:
        ruang = pilih_ruang("ruang_km")
        tampil(scatter_segmen(df, f"Segmentasi akhir K-Means (K={K_FINAL})", ruang=ruang))
    with th:
        a, b = st.columns(2)
        with a:
            Z = matriks_linkage(X)
            potong = (Z[-K_FINAL, 2] + Z[-(K_FINAL - 1), 2]) / 2
            fig, ax = plt.subplots(figsize=(8, 4.6))
            dendrogram(Z, truncate_mode="lastp", p=30, leaf_rotation=90, leaf_font_size=8,
                       show_contracted=True, color_threshold=potong, ax=ax)
            ax.axhline(potong, color="#e34948", ls="--", lw=1.5,
                       label=f"Pemotongan {K_FINAL} cluster ≈ {angka(potong, 1)}")
            ax.set_title("Dendrogram Hierarchical Clustering (Ward)", fontweight="bold")
            ax.set_xlabel("Kelompok data (jumlah anggota)")
            ax.set_ylabel("Jarak Ward")
            ax.legend()
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        with b:
            d = df[["Asset", "Omset", "Nama Usaha", "Kecamatan"]].copy()
            d["Cluster HC"] = df["Cluster_HC"].astype(str)
            fig = px.scatter(d, x="Asset", y="Omset", color="Cluster HC",
                             category_orders={"Cluster HC": ["1", "2", "3", "4"]},
                             color_discrete_sequence=["#2a78d6", "#eda100", "#1baf7a", "#e87ba4"],
                             hover_data={"Nama Usaha": True, "Kecamatan": True, "Asset": ":,.0f", "Omset": ":,.0f"},
                             labels={"Asset": "Aset (Rp, skala log)", "Omset": "Omzet (Rp, skala log)"},
                             opacity=0.75, title=f"Hierarchical Ward (K={K_FINAL})")
            fig.update_traces(marker=dict(size=7))
            tampil(rapikan(sumbu_log(fig), 460))
        ukuran = df["Cluster_HC"].value_counts().sort_index()
        st.caption("Ukuran cluster Hierarchical (diurutkan dari terbesar): "
                   + ", ".join(f"C{i}: {angka(v)}" for i, v in ukuran.items())
                   + f" · rentang {angka(ukuran.min())}–{angka(ukuran.max())}")
        st.markdown("**Komposisi anggota: K-Means vs Hierarchical**")
        ct = pd.crosstab(df["Segmen"], df["Cluster_HC"]).reindex(URUTAN_LABEL)
        ct.columns = [f"HC {c}" for c in ct.columns]
        tabel(ct.reset_index().rename(columns={"Segmen": "Cluster K-Means"}))
        st.caption(f"Cluster Hierarchical terbesar ({angka(ukuran.max())} UMKM) menggabungkan sebagian besar anggota "
                   "Cluster 1 K-Means dengan sebagian anggota Cluster 2 dan Cluster 4.")
    with td:
        a, b = st.columns(2)
        with a:
            mp = st.selectbox("MinPts untuk K-distance plot", [5, 10, 15, 20], index=0)
            kd = k_distance(X, mp)
            fig = px.line(x=np.arange(len(kd)), y=kd, title=f"K-distance plot (MinPts = {mp})",
                          labels={"x": "Indeks data (terurut)", "y": f"Jarak ke tetangga ke-{mp}"},
                          color_discrete_sequence=["#2a78d6"])
            fig.add_hline(y=DB_EPS, line_dash="dash", line_color="#e34948",
                          annotation_text=f"ε = {angka(DB_EPS, 2)}", annotation_position="top left")
            tampil(rapikan(fig, 380, legenda=False))
        with b:
            d = df[["Asset", "Omset", "Nama Usaha", "Kecamatan"]].copy()
            d["Label DBSCAN"] = np.where(P["lab_db"] == -1, "Noise", "Cluster " + (P["lab_db"] + 1).astype(str))
            fig = px.scatter(d, x="Asset", y="Omset", color="Label DBSCAN",
                             color_discrete_map={"Cluster 1": "#2a78d6", "Cluster 2": "#eda100", "Noise": "#8a8a85"},
                             hover_data={"Nama Usaha": True, "Kecamatan": True, "Asset": ":,.0f", "Omset": ":,.0f"},
                             labels={"Asset": "Aset (Rp, skala log)", "Omset": "Omzet (Rp, skala log)"},
                             opacity=0.75, title=f"DBSCAN (ε={angka(DB_EPS, 2)}, MinPts={DB_MINPTS})")
            fig.update_traces(marker=dict(size=7))
            tampil(rapikan(sumbu_log(fig), 380))
        st.caption("Ukuran: " + ", ".join(f"{k}: {angka(v)}" for k, v in d["Label DBSCAN"].value_counts().items())
                   + f" · noise {angka(m_db['Noise'] / N * 100, 2)}%")
        with st.expander("Hasil grid search DBSCAN (10 konfigurasi terbaik)"):
            g, n_komb = grid_dbscan(X)
            if len(g):
                st.markdown(f"Dari **{n_komb} kombinasi**, **{len(g)} konfigurasi** memenuhi syarat 2–10 cluster dengan noise ≤ 10%.")
                g = g.sort_values("Skor", ascending=False).head(10)
                tabel(g[["eps", "MinPts", "Cluster", "Noise", "Silhouette", "DBI", "CHI", "Terkecil", "Terbesar", "Balance", "Skor"]]
                      .style.format({"eps": lambda v: angka(v, 2), "Silhouette": lambda v: angka(v, 4),
                                     "DBI": lambda v: angka(v, 4), "CHI": lambda v: angka(v, 2),
                                     "Balance": lambda v: angka(v, 4), "Skor": lambda v: angka(v, 4)}))
                st.caption("Konfigurasi terbaik (ε = 0,41; MinPts = 5) dan lima berikutnya (ε = 0,42–0,46) sama-sama menghasilkan "
                           "satu cluster dominan dan satu cluster berisi 5 UMKM — pola ini adalah sifat data, bukan akibat pemilihan parameter.")

# =============================================================================
# HALAMAN 5 - PROFIL SEGMEN
# =============================================================================
elif halaman == "🧩 Profil Segmen":
    st.title("Profil Segmen UMKM")
    kec = st.multiselect("Filter kecamatan (kosongkan = semua)", sorted(df["Kecamatan"].unique()))
    data = df[df["Kecamatan"].isin(kec)] if kec else df
    if data.empty:
        st.warning("Tidak ada data untuk filter ini.")
        st.stop()
    st.caption(f"Menampilkan **{angka(len(data))}** dari {angka(N)} UMKM")

    ruang = pilih_ruang("ruang_profil")
    tampil(scatter_segmen(data, "Sebaran aset dan omzet per segmen", centroid=not kec, ruang=ruang))

    pf = (data.groupby(["Cluster", "Segmen_Key"])
          .agg(Jumlah=("Asset", "size"), **{"Rata-rata aset": ("Asset", "mean"), "Rata-rata omzet": ("Omset", "mean"),
                                            "Median aset": ("Asset", "median"), "Median omzet": ("Omset", "median"),
                                            "Omzet minimum": ("Omset", "min"), "Omzet maksimum": ("Omset", "max")})
          .reset_index().sort_values("Cluster"))
    pf.insert(3, "Persen", pf["Jumlah"] / len(data) * 100)
    pf["Rasio omzet/aset"] = pf["Rata-rata omzet"] / pf["Rata-rata aset"]
    pf["Interpretasi"] = pf["Segmen_Key"].map(lambda k: SEGMEN[k]["nama"])
    st.subheader("Profil ekonomi (Tabel 17)")
    kolom_rp = ["Rata-rata aset", "Rata-rata omzet", "Median aset", "Median omzet", "Omzet minimum", "Omzet maksimum"]
    tabel(pf[["Cluster", "Interpretasi", "Jumlah", "Persen"] + kolom_rp + ["Rasio omzet/aset"]].style.format(
        {"Persen": lambda v: f"{angka(v, 2)}%", "Rasio omzet/aset": lambda v: angka(v, 1),
         **{c: rp_penuh for c in kolom_rp}}))

    a, b = st.columns(2)
    with a:
        p = pf.melt(id_vars=["Cluster"], value_vars=["Rata-rata aset", "Rata-rata omzet"], var_name="Variabel", value_name="Nilai")
        p["Juta"] = p["Nilai"] / 1e6
        p["Cluster"] = "Cluster " + p["Cluster"].astype(str)
        fig = px.bar(p, x="Cluster", y="Juta", color="Variabel", barmode="group",
                     text=p["Juta"].map(lambda v: angka(v, 1)),
                     color_discrete_map={"Rata-rata aset": "#2a78d6", "Rata-rata omzet": "#eb6834"},
                     title="Rata-rata aset dan omzet per cluster (juta rupiah) — Gambar 15")
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(xaxis_title=None, yaxis_title="Juta rupiah", bargap=0.3)
        tampil(rapikan(fig, 380))
    with b:
        var = st.radio("Distribusi per segmen", ["Asset", "Omset"], horizontal=True,
                       format_func=lambda v: "Aset" if v == "Asset" else "Omzet")
        fig = px.box(data, x="Segmen", y=var, color="Segmen", color_discrete_map=WARNA_LABEL,
                     category_orders={"Segmen": URUTAN_LABEL}, points=False,
                     title=f"Distribusi {'aset' if var == 'Asset' else 'omzet'} per segmen (skala log)")
        fig.update_layout(xaxis=dict(title=None, tickvals=URUTAN_LABEL,
                                     ticktext=[f"C{SEGMEN[k]['no']}" for k in URUTAN_SEGMEN]), yaxis_title=None)
        sumbu_log(fig, x=False)
        tampil(rapikan(fig, 380, legenda=False))

    st.subheader("Profil sosial (Tabel 18)")
    ps = profil_sosial(data)
    tabel(ps.style.format({"Jenis usaha terisi": lambda v: f"{angka(v, 1)}%", "Perempuan": lambda v: f"{angka(v, 1)}%"}))
    st.caption(f"Perempuan keseluruhan: **{angka((data['Jenis Kelamin'] == 'Perempuan').mean() * 100, 1)}%**. "
               "Kuliner dominan di seluruh cluster, sehingga jenis usaha bukan pembeda utama antarsegmen; perbedaan yang menonjol "
               "terlihat pada komposisi gender dan kelengkapan data jenis usaha.")

    st.subheader("Rekomendasi kebijakan per cluster (Tabel 19)")
    rk = pd.DataFrame([{"Cluster": SEGMEN[k]["no"], "Segmen": SEGMEN[k]["nama"],
                        "Prioritas kebijakan": "; ".join(SEGMEN[k]["rekomendasi"])}
                       for k in ["subsisten", "berkembang", "dagang", "mapan"]])
    tabel(rk)

    st.subheader("Detail segmen")
    pilih = st.selectbox("Pilih segmen", URUTAN_SEGMEN, format_func=lambda k: LABEL_SEGMEN[k])
    s = SEGMEN[pilih]
    sub = data[data["Segmen_Key"] == pilih]
    a, b, c = st.columns([2, 2, 3])
    with a:
        st.markdown(f"<div style='border-left:6px solid {s['warna']};padding-left:10px'>"
                    f"<b>{s['nama']}</b><br><span style='opacity:0.7'>{s['en']}</span></div>",
                    unsafe_allow_html=True)
        st.write(s["ciri"])
        st.markdown("**Rekomendasi intervensi:**")
        for r in s["rekomendasi"]:
            st.markdown(f"- {r}")
    with b:
        top = sub["Kecamatan"].value_counts().head(8).reset_index()
        top.columns = ["Kecamatan", "Jumlah"]
        fig = px.bar(top, x="Jumlah", y="Kecamatan", orientation="h", text="Jumlah",
                     color_discrete_sequence=[s["warna"]], title="Kecamatan terbanyak")
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(yaxis=dict(autorange="reversed", title=None), bargap=0.3)
        tampil(rapikan(fig, 340, legenda=False))
    with c:
        kol = [k for k in ["Nama Usaha", "Kecamatan", "Produksi", "Jenis Usaha", "Asset", "Omset"] if k in sub.columns]
        st.markdown(f"**Contoh anggota** ({angka(len(sub))} UMKM)")
        tabel(sub[kol].sort_values("Omset", ascending=False).head(50)
              .rename(columns={"Asset": "Aset", "Omset": "Omzet"})
              .style.format({"Aset": rp_penuh, "Omzet": rp_penuh}), height=300)

# =============================================================================
# HALAMAN 6 - SEBARAN KECAMATAN
# =============================================================================
elif halaman == "🗺️ Sebaran Kecamatan":
    st.title("Sebaran Segmen per Kecamatan")
    ct = pd.crosstab(df["Kecamatan"], df["Segmen"]).reindex(columns=URUTAN_LABEL, fill_value=0)
    ct["Total"] = ct.sum(axis=1)
    ct = ct.sort_values("Total", ascending=False)

    a, b, c = st.columns([2, 2, 3])
    n_top = a.slider("Jumlah kecamatan ditampilkan", 5, len(ct), min(20, len(ct)))
    mode = b.radio("Tampilan", ["Jumlah", "Proporsi (%)"], horizontal=True)
    urut = c.selectbox("Urutkan berdasarkan", ["Total UMKM"] + URUTAN_LABEL)

    d = ct.head(n_top) if urut == "Total UMKM" else \
        ct.assign(_p=ct[urut] / ct["Total"]).query("Total >= 5").sort_values("_p", ascending=False).head(n_top).drop(columns="_p")
    if urut != "Total UMKM":
        st.caption("Pengurutan berdasarkan proporsi segmen; hanya kecamatan dengan ≥ 5 UMKM.")
    long = d.drop(columns="Total").reset_index().melt(id_vars="Kecamatan", var_name="Segmen", value_name="Jumlah")
    if mode == "Proporsi (%)":
        long["Nilai"] = long["Jumlah"] / long["Kecamatan"].map(d["Total"]) * 100
    else:
        long["Nilai"] = long["Jumlah"]
    fig = px.bar(long, x="Nilai", y="Kecamatan", color="Segmen", orientation="h",
                 color_discrete_map=WARNA_LABEL, category_orders={"Segmen": URUTAN_LABEL, "Kecamatan": list(d.index)},
                 custom_data=["Segmen", "Jumlah"],
                 title=f"Komposisi segmen pada {len(d)} kecamatan")
    fig.update_traces(marker_line_width=1, marker_line_color="white",
                      hovertemplate="<b>%{y}</b><br>%{customdata[0]}<br>Jumlah: %{customdata[1]}<br>"
                                    + ("Proporsi: %{x:.1f}%" if mode != "Jumlah" else "") + "<extra></extra>")
    fig.update_layout(barmode="stack", yaxis_title=None, bargap=0.25,
                      xaxis_title="Proporsi (%)" if mode != "Jumlah" else "Jumlah UMKM")
    tampil(rapikan(fig, max(420, 26 * len(d) + 120)))

    st.subheader("Komposisi satu kecamatan")
    kc = st.selectbox("Pilih kecamatan", list(ct.index))
    baris = ct.loc[kc]
    cols = st.columns(5)
    cols[0].metric("Total UMKM", angka(baris["Total"]))
    for i, lab in enumerate(URUTAN_LABEL):
        cols[i + 1].metric(lab.split(" · ")[1], angka(baris[lab]))
        cols[i + 1].caption(f"{angka(baris[lab] / baris['Total'] * 100, 1)}% dari kecamatan ini")

    with st.expander("Tabel lengkap kecamatan × segmen"):
        tabel(ct.reset_index())
        st.download_button("⬇️ Unduh CSV", ct.to_csv().encode("utf-8"), "sebaran_kecamatan_segmen.csv", "text/csv")

# =============================================================================
# HALAMAN 7 - CEK SEGMEN UMKM BARU
# =============================================================================
elif halaman == "🎯 Cek Segmen UMKM":
    st.title("Cek Segmen UMKM")
    st.caption("Masukkan aset dan omzet tahunan sebuah UMKM; sistem memprediksi segmennya memakai model K-Means yang sama.")

    def prediksi(asset, omset):
        z = P["scaler"].transform(np.log1p(np.column_stack([asset, omset])))
        idx = P["kmeans"].predict(z)
        return [P["peta"][i] for i in idx]

    t1, t2 = st.tabs(["Satu UMKM", "Banyak UMKM (unggah Excel)"])
    with t1:
        a, b = st.columns([2, 3])
        with a:
            with st.form("form_cek"):
                nama = st.text_input("Nama usaha (opsional)")
                asset = st.number_input("Total aset (Rp)", min_value=0, value=5_000_000, step=500_000, format="%d")
                omset = st.number_input("Omzet per tahun (Rp)", min_value=0, value=24_000_000, step=1_000_000, format="%d")
                kirim = st.form_submit_button("Prediksi segmen", type="primary")
            st.caption(f"Pratinjau input: aset **{rp(asset)}**, omzet **{rp(omset)}**")
        with b:
            if kirim:
                if asset <= 0 or omset <= 0:
                    st.error("Aset dan omzet harus lebih dari 0.")
                else:
                    if asset < BATAS_GALAT and omset < BATAS_GALAT:
                        st.warning("Nilai sangat kecil — mungkin tercatat dalam ribuan rupiah. "
                                   "Data seperti ini dikeluarkan saat pelatihan model. Periksa kembali satuannya.")
                    key = prediksi([asset], [omset])[0]
                    s = SEGMEN[key]
                    st.markdown(
                        f"<div style='border-left:8px solid {s['warna']};padding:8px 14px;"
                        f"background:rgba(127,127,127,0.08);border-radius:6px'>"
                        f"<span style='opacity:0.7'>{nama or 'UMKM ini'} termasuk</span><br>"
                        f"<span style='font-size:1.5rem;font-weight:700'>Cluster {s['no']} · {s['nama']}</span><br>"
                        f"{s['ciri']}</div>", unsafe_allow_html=True)
                    st.markdown("**Rekomendasi intervensi:**")
                    for r in s["rekomendasi"]:
                        st.markdown(f"- {r}")
            else:
                st.info("Isi formulir di kiri lalu klik **Prediksi segmen**.")
        if kirim and asset > 0 and omset > 0:
            titik = go.Scatter(x=[asset], y=[omset], mode="markers+text", name="UMKM yang dicek",
                               text=[nama or "UMKM yang dicek"], textposition="top center",
                               marker=dict(symbol="star", size=24, color="#e34948", line=dict(width=2, color="white")),
                               hovertemplate="Aset: Rp %{x:,.0f}<br>Omzet: Rp %{y:,.0f}<extra></extra>")
            tampil(scatter_segmen(df, "Posisi UMKM terhadap seluruh data", ekstra=titik))

    with t2:
        st.markdown("Unggah file Excel dengan kolom **Asset** dan **Omset** (kolom lain boleh ada).")
        contoh = pd.DataFrame({"Nama Usaha": ["Contoh A", "Contoh B"], "Asset": [1_000_000, 30_000_000],
                               "Omset": [2_000_000, 150_000_000]})
        st.download_button("⬇️ Unduh templat CSV", contoh.to_csv(index=False).encode("utf-8"), "templat_cek_segmen.csv")
        f = st.file_uploader("File UMKM baru", type=["xlsx", "csv"], key="batch")
        if f is not None:
            baru = pd.read_csv(f) if f.name.endswith(".csv") else pd.read_excel(f)
            baru.columns = baru.columns.str.strip()
            if not {"Asset", "Omset"}.issubset(baru.columns):
                st.error("Kolom Asset dan Omset wajib ada.")
            else:
                baru["Asset"] = pd.to_numeric(baru["Asset"], errors="coerce")
                baru["Omset"] = pd.to_numeric(baru["Omset"], errors="coerce")
                ok = (baru["Asset"] > 0) & (baru["Omset"] > 0)
                baru["Segmen"] = "Tidak valid (Asset/Omset kosong atau 0)"
                if ok.any():
                    keys = prediksi(baru.loc[ok, "Asset"].values, baru.loc[ok, "Omset"].values)
                    baru.loc[ok, "Segmen"] = [LABEL_SEGMEN[k] for k in keys]
                tabel(baru)
                st.download_button("⬇️ Unduh hasil prediksi (CSV)", baru.to_csv(index=False).encode("utf-8"),
                                   "hasil_prediksi_segmen.csv", "text/csv")

# =============================================================================
# HALAMAN 8 - DATA & UNDUHAN
# =============================================================================
elif halaman == "📄 Data & Unduhan":
    st.title("Data Hasil Segmentasi")
    a, b = st.columns([2, 3])
    cari = a.text_input("Cari nama usaha / pemilik / produk")
    seg = b.multiselect("Segmen", URUTAN_LABEL, default=URUTAN_LABEL)
    kec = st.multiselect("Kecamatan", sorted(df["Kecamatan"].unique()))

    data = df[df["Segmen"].isin(seg)]
    if kec:
        data = data[data["Kecamatan"].isin(kec)]
    if cari:
        kolom_cari = [c for c in ["Nama Usaha", "Nama Lengkap", "Produksi"] if c in data.columns]
        m = np.zeros(len(data), dtype=bool)
        for c in kolom_cari:
            m |= data[c].astype(str).str.contains(cari, case=False, na=False).values
        data = data[m]

    # Data pribadi (No HP, alamat, tanggal lahir) tidak ditampilkan
    kolom = [c for c in ["Nama Lengkap", "Nama Usaha", "Jenis Kelamin", "Kecamatan", "Produksi", "Jenis Usaha",
                         "Pelatihan yang di ikuti", "Asset", "Omset", "Cluster", "Segmen"] if c in data.columns]
    st.caption(f"{angka(len(data))} baris · kolom No HP, alamat, dan tanggal lahir disembunyikan")
    tabel(data[kolom].style.format({"Asset": rp_penuh, "Omset": rp_penuh}), height=480)

    st.subheader("Unduh")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df[kolom + ["Cluster_HC", "Cluster_DBSCAN"]].to_excel(w, sheet_name="Hasil Cluster", index=False)
        profil.drop(columns="Segmen_Key").to_excel(w, sheet_name="Profil Ekonomi", index=False)
        profil_sosial(df).to_excel(w, sheet_name="Profil Sosial", index=False)
        eval_algo.to_excel(w, sheet_name="Perbandingan Algoritma", index=False)
        skor_layak.to_excel(w, sheet_name="Skor Komposit", index=False)
        evaluasi_k(X).drop(columns=["Sil_norm", "1-DBI_norm", "CHI_norm"]).to_excel(w, sheet_name="Evaluasi K", index=False)
        if len(P["df_salah"]):
            kol_s = [c for c in ["No", "Kecamatan", "Asset", "Omset"] if c in P["df_salah"].columns]
            P["df_salah"][kol_s].to_excel(w, sheet_name="Data Galat (57)", index=False)
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Excel lengkap", buf.getvalue(), "Hasil_Segmentasi_UMKM_Sukabumi.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    c2.download_button("⬇️ CSV data yang difilter", data[kolom].to_csv(index=False).encode("utf-8"),
                       "data_umkm_filter.csv", "text/csv")

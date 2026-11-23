# Dashboard Segmentasi UMKM Kabupaten Sukabumi

## Isi folder
- app.py                 : aplikasi dashboard Streamlit
- UMKM 2057.xlsx         : data mentah (nama file harus persis seperti ini)
- requirements.txt       : daftar library
- .streamlit/config.toml : tema warna (opsional)

## Cara menjalankan (Windows CMD)
1. Ekstrak ZIP, misalnya ke  D:\umkm_dashboard
2. Buka CMD, lalu:   cd /d D:\umkm_dashboard
3. (sekali saja)     pip install -r requirements.txt
4. Jalankan:         streamlit run app.py
5. Browser terbuka otomatis di http://localhost:8501
6. Hentikan server:  tekan Ctrl + C di CMD

Saat pertama dibuka, model dilatih sekitar 5-10 detik; setelah itu tersimpan di cache.

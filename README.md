# Prediksi Penghasilan Individu: Analisis Klasifikasi Dataset Adult Income

## Deskripsi Proyek
Proyek ini merupakan bagian dari "Big Data Challenge 4: Classification Challenge" yang berfokus pada pembangunan model klasifikasi untuk memprediksi apakah penghasilan seseorang melebihi $50.000 per tahun atau tidak, berdasarkan data sensus. Proyek ini meliputi tahapan analisis data eksplorasi (EDA), pra-pemrosesan data, pelatihan model Machine Learning, dan evaluasi performa model.

## Dataset
Dataset yang digunakan adalah **Adult Income Dataset**, yang umumnya berasal dari data sensus penduduk Amerika Serikat tahun 1994. Dataset ini berisi berbagai fitur demografi dan pekerjaan yang dapat memengaruhi tingkat penghasilan individu.

### Fitur Kunci:
* `age`: Usia individu.
* `workclass`: Tipe pekerjaan (misalnya Private, Self-emp-not-inc, State-gov).
* `education`: Tingkat pendidikan (misalnya Bachelors, HS-grad, Some-college).
* `education-num`: Representasi numerik dari tingkat pendidikan.
* `marital-status`: Status perkawinan.
* `occupation`: Jenis pekerjaan/profesi.
* `relationship`: Hubungan dalam keluarga (misalnya Husband, Not-in-family, Own-child).
* `race`: Ras individu.
* `sex`: Jenis kelamin.
* `capital-gain`: Keuntungan modal.
* `capital-loss`: Kerugian modal.
* `hours-per-week`: Jumlah jam kerja per minggu.
* `native-country`: Negara asal.

### Target Variabel:
* `income`: Penghasilan, dikategorikan menjadi `<=50K` atau `>50K`.

## Teknologi yang Digunakan
* **Python**
* **Pandas:** Untuk manipulasi dan analisis data.
* **NumPy:** Untuk komputasi numerik.
* **Scikit-learn (sklearn):** Untuk pra-pemrosesan data (scaling, train-test split), model klasifikasi (Logistic Regression, RandomForestClassifier), dan metrik evaluasi.
* **Matplotlib & Seaborn:** Untuk visualisasi data.
* **imbalanced-learn (imblearn):** Untuk menangani ketidakseimbangan kelas (menggunakan SMOTE).

## Tahapan Proyek
Proyek ini mengikuti alur standar dalam alur kerja Machine Learning:

1.  **Pemuatan & Pratinjau Dataset:** Memuat data dari file `.csv` dan melihat beberapa baris pertamanya.
2.  **Pengecekan Missing Values & Tipe Data:** Mengidentifikasi nilai yang hilang dan tipe data setiap kolom. Ditemukan bahwa dataset ini relatif bersih dari missing values.
3.  **Pra-pemrosesan Data:**
    * Mengisi nilai yang hilang (walaupun tidak ditemukan dalam dataset ini, implementasi metode pengisian disiapkan).
    * One-Hot Encoding untuk variabel kategorikal (`pd.get_dummies()`).
    * Transformasi variabel target `income` menjadi biner (0 untuk `<=50K` dan 1 untuk `>50K`).
4.  **Visualisasi Hubungan Fitur-Target:** Membuat plot (box plot, bar plot) untuk memahami hubungan antara beberapa fitur kunci (misalnya usia, jenis kelamin, pendidikan, jam kerja) dengan variabel target `income`.
5.  **Train-Test Split & Scaling:**
    * Membagi dataset menjadi data pelatihan (80%) dan data pengujian (20%).
    * Menerapkan `StandardScaler` pada kolom numerik untuk menstandarisasi skala fitur.
6.  **Penanganan Ketidakseimbangan Kelas (Balancing):**
    * Menganalisis distribusi kelas target dan mengidentifikasi ketidakseimbangan (sekitar 75% kelas `<=50K` dan 25% kelas `>50K`).
    * Menerapkan **SMOTE (Synthetic Minority Over-sampling Technique)** pada data pelatihan (`X_train` dan `y_train`) untuk menyeimbangkan jumlah sampel di kedua kelas.
7.  **Pelatihan Model Klasifikasi:**
    * Melatih dua model klasifikasi: **Logistic Regression** dan **RandomForestClassifier** menggunakan data pelatihan yang sudah diseimbangkan (`X_train_resampled`, `y_train_resampled`).
8.  **Evaluasi Model:**
    * Mengevaluasi performa kedua model pada data pengujian (`X_test`, `y_test`) menggunakan metrik: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, dan ROC Curve dengan AUC Score.

## Hasil & Temuan
Setelah melalui proses pra-pemrosesan dan pelatihan, kedua model menunjukkan performa yang menjanjikan:

* **Logistic Regression:**
    * Accuracy: `0.8194`
    * Precision (`>50K`): `0.5935`
    * Recall (`>50K`): `0.7982`
    * F1-Score (`>50K`): `0.6808`
    * AUC Score: `0.9015`

* **RandomForestClassifier:**
    * Accuracy: `0.8491`
    * Precision (`>50K`): `0.6742`
    * Recall (`>50K`): `0.7244`
    * F1-Score (`>50K`): `0.6984`
    * AUC Score: `0.9040`

**Kesimpulan Evaluasi:**
Kedua model menunjukkan kemampuan diskriminasi yang **sangat baik** antara kelas penghasilan (`AUC > 0.90`). **RandomForestClassifier sedikit mengungguli Logistic Regression** dalam akurasi keseluruhan dan F1-Score untuk kelas minoritas (`>50K`), menjadikannya pilihan model yang lebih baik dalam kasus ini. Hal ini menunjukkan bahwa penanganan imbalance menggunakan SMOTE berhasil membantu model belajar dari kelas minoritas.

## Cara Menjalankan Proyek
1.  **Clone repositori ini:**
    ```bash
    git clone [LINK_REPOSITORI_ANDA]
    ```
2.  **Instal dependensi:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn
    ```
3.  **Unduh dataset:** Pastikan file `data.csv` (atau nama file dataset yang relevan) berada di direktori yang sama dengan notebook/script Python Anda.
4.  **Jalankan notebook/script:** Buka file Jupyter Notebook atau script Python yang berisi kode proyek ini dan jalankan setiap sel secara berurutan.

## Pengembangan Lanjut (Future Work)
Proyek ini dapat ditingkatkan lebih lanjut dengan:
* **Hyperparameter Tuning:** Mengoptimalkan parameter model untuk performa yang lebih baik.
* **Feature Engineering:** Membuat fitur baru yang lebih bermakna dari fitur yang sudah ada.
* **Feature Selection:** Menganalisis dan memilih fitur paling relevan untuk mengurangi kompleksitas model.
* **Eksplorasi Model Lain:** Mencoba algoritma klasifikasi yang lebih canggih seperti XGBoost, LightGBM, atau Support Vector Machines.
* **Analisis Outlier:** Mengidentifikasi dan menangani nilai-nilai pencilan yang mungkin memengaruhi kinerja model.

## Kontributor
* **Khairunnisa Maharani**

---
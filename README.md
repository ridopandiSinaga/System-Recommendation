# Laporan Proyek Machine Learning - Ridopandi Sinaga

## Domain Proyek

Bob adalah pemilik toko buku online yang diberi nama "BukuSeru.com". Selama beberapa tahun terakhir, bisnisnya telah berjalan dengan baik, tetapi akhir-akhir ini, ia mengalami penurunan penjualan yang signifikan. Semakin maraknya persaingan dari toko online besar seperti AWS dan Netflix, Bob merasa perlu untuk mencari cara inovatif untuk mempertahankan bisnisnya.

Dalam upayanya untuk memahami mengapa penjualan menurun, Bob melakukan survei dan memberikan form keluhan dari pelanggan-pelanggannya lalu menemukan bahwa banyak dari mereka merasa kewalahan oleh jumlah besar buku yang ditawarkan di situs webnya. Ini membuat pelanggan merasa kesulitan untuk menemukan buku yang sesuai dengan preferensi mereka, dan akhirnya, mereka pun memutuskan untuk berbelanja di tempat lain.

Bob menyadari bahwa untuk memperbaiki situasi ini, ia perlu menyediakan pengalaman belanja yang lebih personal dan relevan bagi pelanggan. Salah satu cara untuk melakukannya adalah dengan mengimplementasikan sistem rekomendasi yang cerdas [[1]](https://repository.unair.ac.id/97252/5/5.%20BAB%20II%20TINJAUAN%206.%20KKC%20KK%20FST.ST.SI.10-20%20Fir%20a%20PUSTAKA.pdf).

Melalui penelitian lebih lanjut, Bob menemukan bahwa ternyata benar bahwa sistem rekomendasi dapat memiliki dampak terhadap perilaku pembelian [[2]](https://repository.unair.ac.id/97252/5/5.%20BAB%20II%20TINJAUAN%206.%20KKC%20KK%20FST.ST.SI.10-20%20Fir%20a%20PUSTAKA.pdf). Bob berencana untuk memanfaatkan teknologi ini untuk memahami preferensi pelanggan dan memberikan rekomendasi yang lebih akurat. Hal ini akan membantu pelanggan merasa lebih terhubung dengan toko BukuSeru dan meningkatkan peluang mereka untuk menemukan buku yang mereka benar-benar nikmati. Dengan cara ini, Bob berharap setidaknya dapat mengembalikan minat pelanggan, mempertahankan bisnisnya dalam persaingan yang semakin ketat.

Namun masalahnya Bob tidak mengetahui bagaimana itu sistem rekomendasi, apa aja yang dibutuhkan dan cara membuatnya. Bob hanya memiliki beberapa data dari sistem webnya, yaitu data rating, data buku, dan data user.

Dengan demikian maka di dalam proyek ini akan dibuat sebuah model _machine learning_ berupa _recommendation system_ atau sistem rekomendasi untuk menentukan rekomendasi buku yang terbaik kepada pengguna. Model ini nantinya dapat digunakan dan di-_deploy_ pada website jual buku online milik Bob.

## Business Understanding

### Problem Statements

Berdasarkan latar belakang yang telah dijelaskan di atas, maka diperoleh rumusan masalah yang akan diselesaikan pada proyek ini, yaitu:

1. Bagaimana mempersiapkan data buku, pengguna, dan _rating_ agar dapat digunakan sebagai informasi untuk membuat model _machine learning_ sistem rekomendasi?
2. Bagaimana cara membuat model _machine learning_ untuk sistem rekomendasi buku yang baik?

### Goals

Berdasarkan rumusan masalah yang telah dipaparkan di atas, maka didapatkan tujuan dari proyek ini, yaitu:

1. Memproses data supaya siap digunakan untuk melatih model.
2. Membangun sistem rekomendasi buku yang dapat memberikan rekomendasi yang relevan kepada pengguna.

### Solution Statements

![test](https://github.com/ridopandiSinaga/Price-of-Mobile-Phone-Predictive-Analytics/assets/89272004/ad512d38-14d7-472b-aa8a-97d9a652ef40)

Gambar 1. Diagram alir kerja

Pada Gambar 1, menujukkan proses bagaimana alur kerja (_workflow_) yang dilakukan dalam menyelesaikan proyek ini.

Berikut adalah beberapa solusi yang dapat dilakukan agar dapat mencapai tujuan dari proyek ini, yaitu:

1. Tahap _data preprocessing_ yaitu tahap untuk mengubah data mentah atau _raw data_ menjadi data yang bersih atau _clean data_ yang siap untuk digunakan pada proses selanjutnya. Tahap ini dapat dilakukan dengan cara, yaitu:

   - Melakukan penyesuaian dan mengubah nama kolom atau atribut sehingga memudahkan ketika proses pemanggilan _dataset_ beserta nama atribut atau kolom tertentu.
   - Menggabungkan data yang terpisah sehingga dapat digunakan pada tahap selanjutnya.

2. Tahap _data preparation_ yaitu proses transformasi data sehingga data menjadi bentuk yang cocok untuk melakukan proses pemodelan pada tahap selanjutnya. Tahap ini dilakukan dengan beberapa teknik, yaitu:

   - Melakukan pengecekan nilai data yang kosong, tidak ada, ataupun _null_ (_missing value_) dan menghapus data tersebut atau mengganti/mengisinya dengan suatu nilai tertentu.
   - Melakukan pengecekan data yang mungkin duplikat agar tidak akan mengganggu hasil dari pemodelan dan sistem yang telah dibangun.
   - Melakukan pegecekan data, apakah memang benar sesuai atau merupakan data _missmatch_ tidak sesuai.

3. Tahap _Building machine learning Model_.
   Pembuatan model akan menggunakan dua pendekatan, yaitu _content-based filtering recommendation_, dan pendekatan _collaborative filtering recommendation_.

   - **Content-based Filtering Recommendation**  
     Sistem rekomendasi yang berbasis konten (_content-based filtering_) merupakan sistem rekomendasi yang memberikan rekomendasi item yang hampir sama dengan item yang disukai oleh pengguna di masa lalu. _Content-based filtering_ akan mempelajari profil minat pengguna baru berdasarkan data dari objek yang telah dinilai oleh pengguna lain sebelumnya. Pada pendekatan menggunakan _content-based filtering_ akan menggunakan algoritma TF-IDF Vectorizer dan Cosine Similarity.

     - TF-IDF Vectorizer  
       Algoritma Term Frequency Inverse Document Frequency Vectorizer (TF-IDF Vectorizer) adalah algoritma yang dapat melakukan kalkulasi dan transformasi dari teks mentah menjadi representasi angka yang memiliki makna tertentu dalam bentuk matriks serta dapat digunakan dan dimengerti oleh model _machine learning_. [[5]](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html 'TfidfVectorizer - sci-kit Documentation')

       Kelebihan dari teknik ini adalah tidak membutuhkan data yang diperoleh dari pengguna lain karena rekomendasi yang akan diberikan akan spesifik hanya untuk pengguna tersebut. Sedangkan kekurangan dengan menggunakan teknik ini ialah hasil rekomendasi yang hanya terbatas dari pengguna itu saja dan tidak dapat memperluas data dari penilaian pengguna lain. TF-IDF dapat dihitung menggunakan rumus sebagai berikut: [[6]](https://towardsdatascience.com/tf-idf-simplified-aba19d5f5530 'TF-IDF Simplified - Towards Data Science')

       $$idf_i=log \left( \frac{n}{df_i} \right)$$

       Di mana $idf_i$ merupakan skor IDF untuk _term_ $i$; $df_i$ adalah jumlah dokumen yang mengandung _term_ $i$; dan $n$ adalah jumlah total dokumen. Semakin tinggi nilai $df$ suatu _term_, maka semakin rendah $idf$ untuk _term_ tersebut. Ketika jumlah $df$ sama dengan $n$ yang berarti istilah/_term_ tersebut muncul di semua dokumen, $idf$ akan menjadi 0, karena $log(1)=0$.

       Sedangkan nilai TF-IDF merupakan perkalian dari matriks frekuensi _term_ dengan IDF-nya.

       $$w_{i,j}=tf_{i,j} \times idf_i$$

       Di mana $w_{i,j}$ merupakan skor TF-IDF untuk _term_ $i$ pada dokumen $j$; $tf_{i,j}$ adalah frekuensi _term_ untuk _term_ $i$ pada dokumen $j$, dan $idf_i$ adalah skor $idf$ untuk _term_ $i$.

     - Cosine Similarity  
       Teknik _cosine similarity_ digunakan untuk melakukan perhitungan derajat kesamaan (_similarity degree_) antara dua sampel. [[7]](https://www.sciencedirect.com/topics/computer-science/cosine-similarity 'Cosine Similarity - ScienceDirect Topics')

       $$S_c(A,B)=cos(\theta)= \frac{A \times B}{\|A\| \|B\|} = \frac{\displaystyle\sum^{n}_{i=1} A_iB_i}{\sqrt{\displaystyle\sum^{n}_{i=1} A^{2}_{i} } \sqrt{\displaystyle\sum^{n}_{i=1} B^{2}_{i}} }$$

       Di mana $A_i$ dan $B_i$ merupakan komponen dari masing-masing vektor A dan B.

   - **Collaborative Filtering Recommendation**

     Sistem rekomendasi yang berbasis penyaringan kolaboratif (_collaborative filtering_) adalah sistem rekomendasi yang memberikan rekomendasi item yang hampir sama dengan preferensi pengguna di masa lalu berdasarkan riwayat pengguna lain yang memiliki preferensi yang sama, misalnya berdasarkan penilaian atau _rating_ yang telah diberikan pengguna di masa lalu. [[8]](https://realpython.com/build-recommendation-engine-collaborative-filtering 'Build a Recommendation Engine With Collaborative Filtering - Real Python') Namun, teknik ini memilki kekurangan yaitu, tidak dapat memberikan rekomendasi item yang tidak memiliki riwayat penilaian/_rating_ atau transaksi.

     Menggunakan pendekatan model Deeplearning teknik _collaborative filtering recommendation_ akan memerlukan proses penyandian (_encoding_) fitur-fitur yang terdapat pada _dataset_ ke dalam bentuk indeks integer, lalu memetakannya ke dalam _dataframe_ yang berkaitan. Kemudian akan dilakukan pembagian distribusi **dataset** dengan rasio tertentu untuk memisahkan data latih (_training data_) dan juga data uji (_validation data_) sebelum dilakukan tahap pemodelan. Juga nantinya model ini menggunakan Binary Crossentropy untuk menghitung loss function, Adam (Adaptive Moment Estimation) sebagai optimizer, dan root mean squared error (RMSE) sebagai metrics evaluation.

## Data Understanding

[<img src="https://github.com/ridopandiSinaga/Diagnostic-Prediction-of-Diabetes/assets/89272004/e3173cfd-4711-4c67-a5c8-265c657af43c" alt="Book Recommendation Kaggle Dataset" title="Book Recommendation Kaggle Dataset" width="100%">](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset)

Data yang digunakan dalam proyek ini adalah _dataset_ yang diambil dari [Kaggle Dataset: Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset "Build state-of-the-art models for book recommendation system")

Dalam dataset tersebut berisi tiga (3) berkas CSV yaitu `Books.csv`, `Ratings.csv`, `Users.csv`.

- **Books.csv**, memiliki atribut atau fitur sebagai berikut.

  | No. | Column              | Non-Null Count  | Dtype  |
  | --- | ------------------- | --------------- | ------ |
  | 0   | ISBN                | 271360 non-null | object |
  | 1   | Book-Title          | 271360 non-null | object |
  | 2   | Book-Author         | 271359 non-null | object |
  | 3   | Year-Of-Publication | 271360 non-null | object |
  | 4   | Publisher           | 271358 non-null | object |
  | 5   | Image-URL-S         | 271360 non-null | object |
  | 6   | Image-URL-M         | 271360 non-null | object |
  | 7   | Image-URL-L         | 271357 non-null | object |

  Tabel 1. Informasi data buku

  Berikut penjelasan dari variabel masing-masing kolom pada Tabel 1:

  - `ISBN` : _International Standard Book Number_
  - `Book-Title` : Judul buku
  - `Book-Author` : Penulis buku
  - `Year-Of-Publication` : Tahun terbit buku
  - `Publisher` : Penerbit buku
  - `Image-URL-S` : Tautan sampul buku ukuran kecil
  - `Image-URL-M` : Tautan sampul buku ukuran sedang
  - `Image-URL-L` : Tautan sampul buku ukuran besar

- **Ratings.csv**, memiliki atribut atau fitur sebagai berikut,

  | No. | Column      | Non-Null Count   | Dtype  |
  | --- | ----------- | ---------------- | ------ |
  | 0   | User-ID     | 1149780 non-null | int64  |
  | 1   | ISBN        | 1149780 non-null | object |
  | 2   | Book-Rating | 1149780 non-null | int64  |

  Tabel 2. Informasi data Ratings

  Berikut penjelasan masing-masing variabel kolom pada Tabel 2 diatas:

  - `User-ID` : Identitas unik pengguna berupa bilangan bulat atau integer
  - `ISBN` : _International Standard Book Number_
  - `Book-Rating` : _Rating_ buku yang diberikan pengguna

- **Users.csv**, memiliki atribut atau fitur sebagai berikut,

  | No. | Column   | Non-Null Count  | Dtype   |
  | --- | -------- | --------------- | ------- |
  | 0   | User-ID  | 278858 non-null | int64   |
  | 1   | Location | 278858 non-null | object  |
  | 2   | Age      | 168096 non-null | float64 |

  Tabel 3. Informasi data Users

  Berikut penjelasan singkat mengenai variabel-variabel pada Tabel 3 diatas:

  - `User-ID` : Identitas unik pengguna berupa bilangan bulat atau integer
  - `Location` : Lokasi tempat tinggal pengguna
  - `Age` : Umur pengguna

Deskripsi statistik untuk _dataset_ `ratings` pada fitur `Book-Rating` dapat dilihat pada Tabel 4 dibawah ini.

| No. | Statistik | Nilai     |
| --- | --------- | --------- |
| 1   | Count     | 1,149,780 |
| 2   | Mean      | 3         |
| 3   | Std       | 4         |
| 4   | Min       | 0         |
| 5   | 25%       | 0         |
| 6   | 50%       | 0         |
| 7   | 75%       | 7         |
| 8   | Max       | 10        |

Tabel 4. Statisik Ratings.

Dari Tabel 4, di atas dapat dilihat bahwa terdapat,

- Total jumlah data (`count`) sebanyak 1.149.780;
- Rata-rata _rating_ (`mean`) bernilai 3;
- Simpangan baku/standar deviasi _rating_ (`std`) bernilai 4;
- _Rating_ Minimal (`min`), kuartil bawah/Q1 _rating_ (`25%`), kuartil tengah/Q2/median _rating_ (`50%`) bernilai 0;
- Kuartil atas/Q3 _rating_ (`75%`) bernilai 7;
- _Rating_ maksimum (`max`) bernilai 10

Berikut adalah visualisasi grafik histogram frekuensi sebaran data _rating_ pengguna terhadap buku yang sudah pernah dibaca, mulai dari _rating_ 1 hingga _rating_ 10.

![ratings](https://github.com/ridopandiSinaga/Price-of-Mobile-Phone-Predictive-Analytics/assets/89272004/fcf9481d-e8c1-4b0c-b2e6-f42b94152f6a)

Gambar 2. Histogram sebaran data rating pengguna terhadap buku yang sudah pernah dibaca

Berdasarkan hasil visualisasi grafik pada Gambar 2, "Jumlah Rating Buku" di atas, didapat bahwa _rating_ terbanyak dari buku yang sudah pernah dibaca adalah _rating_ 0, dengan jumlah _rating_ kira-kira sebanyak lebih dari 700.000. _Rating_ 0 tersebut dapat menyebabkan bias dan mempengaruhi hasil analisis, dikarenakan rating pada nyatanya bernilai antara 1 sampai 10.

## Data Preprocessing

Pada tahap pra-pemrosesan data atau _data preprocessing_ dilakukan untuk mengubah data mentah (_raw data_) menjadi data yang bersih (_clean data_) yang siap untuk digunakan pada proses selanjutnya. Ada beberapa tahap yang dilakukan pada _data preprocessing_, yaitu:

- **Mengubah Nama Kolom/Atribut/Fitur**  
  Proses pengubahan nama kolom atau atribut atau fitur dari masing-masing _dataframe_ bertujuan untuk memudahkan proses pemanggilan _dataframe_ tersebut. Berikut hasil setelah perbaikan nama atribut terkait.

  - Books

    | No. | isbn       | book_title           | book_author          | pub_year | publisher               | image_s_url                                                        | image_m_url                                                        | image_l_url                                                        |
    | --- | ---------- | -------------------- | -------------------- | -------- | ----------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------ | ------------------------------------------------------------------ |
    | 0   | 0195153448 | Classical Mythology  | Mark P. O. Morford   | 2002     | Oxford University Press | [Image-S-URL-0](http://images.amazon.com/images/P/0195153448.0...) | [Image-M-URL-0](http://images.amazon.com/images/P/0195153448.0...) | [Image-L-URL-0](http://images.amazon.com/images/P/0195153448.0...) |
    | 1   | 0002005018 | Clara Callan         | Richard Bruce Wright | 2001     | HarperFlamingo Canada   | [Image-S-URL-1](http://images.amazon.com/images/P/0002005018.0...) | [Image-M-URL-1](http://images.amazon.com/images/P/0002005018.0...) | [Image-L-URL-1](http://images.amazon.com/images/P/0002005018.0...) |
    | 2   | 0060973129 | Decision in Normandy | Carlo D'Este         | 1991     | HarperPerennial         | [Image-S-URL-2](http://images.amazon.com/images/P/0060973129.0...) | [Image-M-URL-2](http://images.amazon.com/images/P/0060973129.0...) | [Image-L-URL-2](http://images.amazon.com/images/P/0060973129.0...) |

    Tabel 5. Tabel data books selesai perbaikan nama atribut.

  - Ratings

    | No. | user_id | isbn       | book_rating |
    | --- | ------- | ---------- | ----------- |
    | 0   | 276725  | 034545104X | 0           |
    | 1   | 276726  | 0155061224 | 5           |
    | 2   | 276727  | 0446520802 | 0           |

    Tabel 6. Tabel data ratings selesai perbaikan nama atribut.

  - Users

    | No. | user_id | location                        | age  |
    | --- | ------- | ------------------------------- | ---- |
    | 0   | 1       | nyc, new york, usa              | NaN  |
    | 1   | 2       | stockton, california, usa       | 18.0 |
    | 2   | 3       | moscow, yukon territory, russia | NaN  |

    Tabel 7. Tabel data users selesai perbaikan nama atribut.

- **Menggabungkan Data ISBN**  
  Penggabungan data ISBN buku dilakukan menggunakan fungsi `.concatenate` dengan bantuan _library_ [`numpy`](https://numpy.org). Data ISBN terdapat pada _dataframe_ buku dan _dataframe_ _rating_, sehingga dilakukan penggabungan data tersebut pada atribut atau kolom `isbn`.
- **Menggabungkan Data User**  
  Penggabungan data `user_id` buku dilakukan menggunakan fungsi `.concatenate` dengan bantuan _library_ [`numpy`](https://numpy.org). Data `user_id` terdapat pada _dataframe_ _rating_ dan _dataframe_ _user_, sehingga dilakukan penggabungan data tersebut pada atribut atau kolom `user_id`.

## Data Preparation

Pada tahap persiapan data atau _data preparation_ dilakukan proses transformasi pada data sehingga menjadi bentuk yang cocok untuk proses pemodelan nantinya. Ada beberapa tahap yang dilakukan pada _data preparation_, yaitu:

- **Pengecekan _Missing Value_**

  Proses pengecekan data yang kosong, hilang, _null_, atau _missing value_ ditemukan pada _dataframe_ `books`, Terdapat missing value pada atribut `book_author`, `publisher`, dan `image_l_url` dikarenakan jumlah data yang missing sedikit jika dibandingkan dengan total keseluruhan data, tidak masalah menggunakan fungsi `drop()` dihapus saja baris dari variabel atau kolom yang kosong tersebut.

  Kemudian pada _dataframe_ `Users`, terdapat sebanyak 110.762 _missing value_ pada fitur `age` artinya dengan jumlah data sebanyak itu yaitu hampir 40% data `Users` jika hanya dihapus saja akan menghilangkan banyak informasi. Solusinya data tersebut diganti atau diisi dengan nilai _mean_ atau nilai rata-rata data umur tersebut ,

  ```
  users['age'] = users['age'].fillna(users['age'].mean())
  ```

  , karena selain untuk mencegah kehilangan informasi, juga karena umur adalah variabel numerik, dan juga _mean_ dapat mentolerir data miring.

- **Pengecekan Ketidaksesuaian Data**

  Pada _dataframe_ `ratings` terdapat _rating_ 0 yang banyak yaitu sebanyak 716.109 data, yang artinya buku tersebut adalah buku yang belum pernah dirating. Hal tersebut dapat menyebabkan bias nantinya. Sehingga solusinya kategori _rating_ 0 tidak diikutsertakan, sehingga diperoleh hasil visualisasi grafik histogram di bawah ini.

- **Pengecekan Data Duplikat**  
  Melakukan pengecekan data duplikat atau data yang sama pada masing-masing _dataframe_. Hasilnya tidak ada data yang duplikat dari ketiga _dataframe_.
- **Data Buku dan Rating**  
  Melakukan penggabungan atau _merge_ data buku dan _rating_ menjadi sebuah _dataframe_.

- **Data Preparation** for collaborative Filtering Recomendation

  - _Data preparation_ yang dilakukan adalah dengan melakukan penyandian (_encoding_) fitur `user_id` dan `isbn` pada _dataframe_ `ratings` ke dalam bentuk indeks integer. Kemudian melakukan pemetaan fitur yang telah di-_encoding_ tersebut ke dalam masing-masing _dataframe_ yang `ratings`.

    Diperoleh jumlah _user_ sebesar 1204, jumlah buku sebesar 4565, nilai minimal _rating_ yaitu 1, dan nilai maksimum _rating_ yaitu 10.

  - Split Training Data dan Validation Data  
    Tahap ini dilakukan pengacakan _dataframe ratings_ terlebih dahulu, lalu kemudian membagi data dengan rasio 80:20, di mana 80% untuk data latih (_training data_) dan 20% sisanya adalah untuk data uji (_validation data_). Rasio tersebut sudah menjadi umum digunakan ketika membuat model machine learning dengan data ribuan dan dimana rasio data latih lebih besar dari pada rasio data uji untuk skala data ribuan. Namun lain halnya jika yang digunakan sampai jutaan data rasio 90:10 bisa menjadi pilihan.

## Modeling

Tahap selanjutnya adalah proses _modeling_ atau membuat model _machine learning_ yang dapat digunakan sebagai sistem rekomendasi untuk menentukan rekomendasi buku yang terbaik kepada pengguna dengan beberapa algoritma sistem rekomendasi tertentu.

Berdasarkan tahap pemahaman data atau [_data understanding_](#data-understanding "Data Understanding") sebelumnya, dapat dilihat bahwa data untuk masing-masing _dataframe_, yaitu data buku, _rating_, dan _users_ tergolong data yang cukup banyak, mencapai ratusan bahkan jutaan jika disatukan. Hal tersebut akan memperlambat dalam melakukan proses pemodelan _machine learning_, seperti memakan waktu yang lama dan _resource_ RAM ataupun GPU yang cukup besar. Oleh karena itu, dalam kasus ini data yang akan digunakan untuk proses pemodelan _machine learning_ data akan dibatasi hanya 10.000 baris data pertama saja, baik itu data buku maupun data rating. dengan cara metode _slicing_ ( `books[:10000]` dan
`ratings[:10000]`). Tidak ada alasan kuat kenapa angka 10.000 digunakan, namun karena datanya berpuluh ribuan, diambil 10 ribu supaya data yang digunakan tetap banyak yang juga berdampak baik pada kualitas rekomendasi.

1.  **Content-based Recommendation**

    - TF-IDF Vectorizer  
      TF-IDF Vectorizer akan mentransformasikan teks menjadi representasi angka yang memiliki makna tertentu dalam bentuk matriks. Ukuran matriks yang diperoleh adalah sebesar 10.000 data buku dan 5.575 data _author_ atau penulis buku. Hasilnya tampak pada Tabel 8 dibawah.

      | No. | book_title                                                          | levinson | garwood | gurko | hazzard | rennison | binchy | truddi | opdyke | wouk | mildred | wiersbe | sand | moix | haruf | marilynn | tsukiyama | merline | petievich | klein | urbano |
      | --- | ------------------------------------------------------------------- | -------- | ------- | ----- | ------- | -------- | ------ | ------ | ------ | ---- | ------- | ------- | ---- | ---- | ----- | -------- | --------- | ------- | --------- | ----- | ------ |
      | 1   | Go, Dog, Go (I Can Read It All by Myself Beginner Books)            | 0.0      | 0.0     | 0.0   | 0.0     | 0.0      | 0.0    | 0.0    | 0.0    | 0.0  | 0.0     | 0.0     | 0.0  | 0.0  | 0.0   | 0.0      | 0.0       | 0.0     | 0.0       | 0.0   | 0.0    |
      | 2   | The House of Death (Sweet Valley University Thriller Edition, No 4) | 0.0      | 0.0     | 0.0   | 0.0     | 0.0      | 0.0    | 0.0    | 0.0    | 0.0  | 0.0     | 0.0     | 0.0  | 0.0  | 0.0   | 0.0      | 0.0       | 0.0     | 0.0       | 0.0   | 0.0    |

      . . .

      Tabel 8. Hasil TF-IDF Vectorizer

    - _Cosine Similarity_  
      _Cosine Similarity_ akan melakukan perhitungan derajat kesamaan (_similarity degree_) antar judul buku tampak seperti pada Tabel 9. Ukuran matriks yang diperoleh adalah sebesar 10.000 data buku dan 10.000 data buku juga.

      | No. | book_title                                       | Foundation Trilogy | Homecoming | The Business Plan for the Body | Song of the Exile (Ballantine Reader's Circle) | Judith Gautier: A Biography |
      | --- | ------------------------------------------------ | ------------------ | ---------- | ------------------------------ | ---------------------------------------------- | --------------------------- |
      | 1   | No Second Chance                                 | 0.0                | 0.0        | 0.0                            | 0.0                                            | 0.0                         |
      | 2   | Bird's-Eye View                                  | 0.0                | 0.0        | 0.0                            | 0.0                                            | 0.0                         |
      | 3   | The Black Echo (Detective Harry Bosch Mysteries) | 0.0                | 0.0        | 0.0                            | 0.0                                            | 0.0                         |
      | 4   | LUCKY                                            | 0.0                | 0.0        | 0.0                            | 0.0                                            | 0.0                         |
      | 5   | At Weddings and Wakes                            | 0.0                | 0.0        | 0.0                            | 0.0                                            | 0.0                         |

      . . .

      Tabel 9. Hasil Cosine Similarity

    - Hasil _Top-N Recommendation_

      Hasil pengujian sistem rekomendasi dengan pendekatan _content-based recommendation_ adalah sebagai berikut.

      Setelah selesai akan dicari derajat kesamaan (similarity degree) antar judul buku. Judul buku yang dipakai untuk rekomendasi untuk uji coba menggunakan buku yang tertera pada Tabel 10 dibawah ini.

      | isbn | book_title | book_author | pub_year | publisher    | image_s_url                                                                                                                                  | image_m_url                                                                                                                                   | image_l_url                                                                                                                                  |
      | ---- | ---------- | ----------- | -------- | ------------ | -------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
      | 2354 | 0835902242 | Mark Twain  | 1992     | Globe Fearon | [![Small Image](http://images.amazon.com/images/P/0835902242.01.THUMBZZZ.jpg)](http://images.amazon.com/images/P/0835902242.01.THUMBZZZ.jpg) | [![Medium Image](http://images.amazon.com/images/P/0835902242.01.MZZZZZZZ.jpg)](http://images.amazon.com/images/P/0835902242.01.MZZZZZZZ.jpg) | [![Large Image](http://images.amazon.com/images/P/0835902242.01.LZZZZZZZ.jpg)](http://images.amazon.com/images/P/0835902242.01.LZZZZZZZ.jpg) |

      Tabel 10. Referensi data buku yang dijadikan patokan rekomendasi

      Tabel 10 di atas merupakan data judul buku yang dipilih oleh pengguna dan akan dipakai dalam preferensi rekomendasi. Lalu berikut hasil rekomenasinya.

      | No  | book_title                                        | book_author |
      | --- | ------------------------------------------------- | ----------- |
      | 0   | ADVENTURES OF HUCKLEBERRY FINN (ENRICHED CLASS... | Mark Twain  |
      | 1   | Treasury of Illustrated Classics: Adventures o... | Mark Twain  |
      | 2   | The Complete Short Stories of Mark Twain (Bant... | Mark Twain  |
      | 3   | A Connecticut Yankee in King Arthur's Court (D... | Mark Twain  |
      | 4   | A Connecticut Yankee in King Arthur's Court (B... | Mark Twain  |
      | 5   | The Diaries of Adam and Eve                       | Mark Twain  |
      | 6   | The Adventures of Tom Sawyer                      | Mark Twain  |

      Tabel 11. Tabel hasil rekomendasi Content-based

      Dapat dilihat pada Tabel 11, ketika pengguna tertarik dengan buku "Adventures of Huckleberry Finn" akan direkomendasikan beberapa buku terkait yang relevan terkait judul buku.

2.  **Collaborative Filtering Recommendation**

    Model based adalah metode yang digunakan pada teknik ini yaitu dengan pendekatan Deep learning atau Neural Network. Model yang dibangun akan menghitung skor kecocokan antara pengguna dan buku dengan teknik embedding. Pertama dilakukan proses embedding terhadap data user dan buku. Selanjutnya, dilakukan operasi perkalian dot product antara embedding user dan buku. Kemudian, juga dapat ditambahkan bias untuk setiap user dan buku. Skor kecocokan ditetapkan dalam skala [0,1] dengan fungsi aktivasi sigmoid. Model dengan pendekatan Deep Learning ini dibangun dengan membuat class RecommenderNet dengan keras Model class. Selanjutnya, dilakukan proses compile terhadap model. Model yang dibangun menggunakan `Binary Crossentropy` untuk menghitung _loss function_, `Adam` (Adaptive Moment Estimation) sebagai _optimizer_, dan `root mean squared error (RMSE)` sebagai _metrics evaluation_. Setelah itu dilakukan proses training terhadap model.

    Untuk mendapatkan rekomendasi buku, untuk sementara sampel user diambil secara acak dan definisikan variabel books_not_read yang merupakan daftar buku yang belum pernah dibaca oleh pengguna, daftar books_not_read inilah yang akan menjadi buku yang rekomendasikan. Variabel books_not_read diperoleh dengan menggunakan operator bitwise (~) pada variabel books_read_by_user. Sebelumnya, pengguna telah memberi rating pada beberapa buku yang telah mereka baca. Rating ini akan digunakan untuk membuat rekomendasi buku yang mungkin cocok untuk pengguna. Kemudian, untuk memperoleh rekomendasi buku, menggunakan fungsi `model.predict()` dari library `Keras`. Hasil rekomendasinya adalah seperti berikut.

    - Model Development dan Hasil  
      Berdasarkan model yang telah di-_training_, berikut adalah hasil pengujian sistem rekomendasi buku dengan pendekatan _collaborative filtering recommendation_.

           Showing recommendations for users: 5709
           ========================================
           Book with high ratings from user
           ----------------------------------------
           All-American Girl : Meg Cabot
           New York Minute : The Movie Novelization (New York Minute) : Mary-Kate &amp; Ashley Olsen
           CHOCOLATE FOR A WOMANS SOUL : 77 STORIES TO FEED YOUR SPIRIT AND WARM YOUR HEART (Chocolate) : Kay Allenbaugh
           Coraline : Neil Gaiman
           Mates, Dates, and Sleepover Secrets (Mates, Dates) : Cathy Hopkins
           ========================================
           Top 10 Books Recommendation
           ----------------------------------------
           To Kill a Mockingbird : Harper Lee
           The Red Tent (Bestselling Backlist) : Anita Diamant
           The Giver (21st Century Reference) : LOIS LOWRY
           The Grapes of Wrath: John Steinbeck Centennial Edition (1902-2002) : John Steinbeck
           Harry Potter and the Sorcerer's Stone (Harry Potter (Paperback)) : J. K. Rowling
           My Lady Pirate : Elizabeth Doyle
           The Runaway Jury : JOHN GRISHAM
           Daughter of the Blood (Black Jewels Trilogy) : Anne Bishop
           The Watsons Go to Birmingham - 1963 (Yearling Newbery) : CHRISTOPHER PAUL CURTIS
           Perfume: The Story of a Murderer (Vintage International) : Patrick Suskind

      Berdasarkan hasil di atas, untuk melakukan uji rekomendasi untuk sementara sistem akan mengambil pengguna secara acak, yaitu pengguna dengan `user_id` **5709**. Lalu akan dicari buku dengan referensi _rating_ terbaik yang pernah user berikan yaitu,

      - All-American Girl, by Meg Cabot
      - New York Minute, by The Movie Novelization (New York Minute), by Mary-Kate &amp; Ashley Olsen
      - CHOCOLATE FOR A WOMANS SOUL, by 77 STORIES TO FEED YOUR SPIRIT AND WARM YOUR HEART (Chocolate), by Kay Allenbaugh
      - Coraline, by Neil Gaiman
      - Mates, Dates, and Sleepover Secrets (Mates, Dates), by Cathy Hopkins

      Kemudian sistem akan membandingkan antara buku dengan _rating_ tertinggi dari _user_ 5709 dan juga _rating user_ lain yang sereferensi terkait rating serupa yang belum pernah dibaca user 5709 tersebut, lalu akan mengurutkan buku yang akan direkomendasikan berdasarkan nilai rekomendasi yang tertinggi dari atas kebawah yang mungkin akan disukai. Dapat dilihat terdapat 10 daftar buku yang direkomendasikan.

## Evaluation

1. **Content-based Recommendation**

   Dalam sistem rekomendasi Content-based, precision adalah presentase jumlah item rekomendasi yang relevan. Tidak bisa menghitung dengan memanggil _library scikit learn_ karena tidak ada data target/label seperti pada supervised learning.

   Sehingga tahap evaluasi untuk model sistem rekomendasi dengan pendekatan _content-based recommendation_ dapat menggunakan evaluasi secara manual dengan metrik akurasi yang diperoleh dari:

   $$Accuracy=\frac{\displaystyle\sum_{i=1}^{n} BooksWithSameAuthor_i}{\displaystyle\sum_{i=1}^{n} RecommendedBooks_i} \times 100$$

   Menggunakan data yang sama pada tahap [Modeling](#modeling "Modeling") _content-based recommendation_, pada proses Hasil _Top-N Recommendation_, dilakukan proses pencarian rekomendasi buku berdasarkan `book_title` dan `author`. Kemudian pengguna menginput buku **Adventures of Huckleberry Finn oleh Mark Twain**. Tak lupa juga dalam model sudah diatur banyak rekomendasi yang diinginkan yaitu sebesar 10. Hasil rekomendasi yang diperoleh adalah sebagai berikut:

   | No  | book_title                                        | book_author |
   | --- | ------------------------------------------------- | ----------- |
   | 0   | ADVENTURES OF HUCKLEBERRY FINN (ENRICHED CLASS... | Mark Twain  |
   | 1   | Treasury of Illustrated Classics: Adventures o... | Mark Twain  |
   | 2   | The Complete Short Stories of Mark Twain (Bant... | Mark Twain  |
   | 3   | A Connecticut Yankee in King Arthur's Court (D... | Mark Twain  |
   | 4   | A Connecticut Yankee in King Arthur's Court (B... | Mark Twain  |
   | 5   | The Diaries of Adam and Eve                       | Mark Twain  |
   | 6   | The Adventures of Tom Sawyer                      | Mark Twain  |

   Tabel 11. Tabel hasil rekomendasi Content-based

   Pada Tabel 11 menunjukkan diperoleh hanya 7 rekomendasi dan semuanya relevan baik itu judul yaitu buku tentang petualangan dan penulis bukunya juga sama.

   Jika dilakukan proses perhitungan akurasi yaitu dengan membagi banyaknya "rekomendasi jumlah buku yang bersesuaian" dengan banyaknya "rekomendasi buku yang diinginkan", kemudian dikalikan dengan 100. Sehingga diperoleh nilai **akurasi** sebesar  
   $$\frac{7}{10} \times 100\% = 70\%$$

2. **Collaborative Filtering Recommendation**  
   Berdasarkan model _machine learning_ yang sudah dibangun menggunakan _embedding layer_ dengan _Adam optimizer_ dan _binary crossentropy loss function_, metrik yang digunakan adalah _Root Mean Squared Error_ (RMSE). Perhitungan RMSE dapat dilakukan menggunakan rumus berikut,

   $$RMSE=\sqrt{\sum^{n}_{i=1} \frac{y_i - y\\_pred_i}{n}}$$

   Di mana, nilai $n$ merupakan jumlah _dataset_, nilai $y_i$ adalah nilai sebenarnya, dan $y\\_pred$ yaitu nilai prediksinya terdahap $i$ sebagai urutan data dalam _dataset_.

   Hasil nilai RMSE yang rendah menunjukkan bahwa variasi nilai yang dihasilkan dari model sistem rekomendasi mendekati variasi nilai observasinya. Artinya, semakin kecil nilai RMSE, maka akan semakin dekat nilai yang diprediksi dan diamati. Besar RSME sudah berhasil mencapai dibawah 0.1 pada epoch ke 45.

   Berikut merupakan visualisasi hasil _training_ dan _validation error_ dari metrik RMSE serta _training_ dan _validation loss_ ke dalam grafik plot.

   ![5b474072-bf7c-4021-852e-eb4e7c9d9945](https://github.com/ridopandiSinaga/Diagnostic-Prediction-of-Diabetes/assets/89272004/d4a7fe08-3ea1-4c03-92a1-9c72833e8946)

   Gambar 3. Plot visual history training dan validation RSME

   Dari Gambar 3, history plot model tidak tampak overfitting atau underfitting. Overfitting biasanya akan terjadi jika RMSE pada data pelatihan terus menurun sementara RMSE pada data validasi meningkat. Underfitting biasanya terjadi jika RMSE baik pada data pelatihan maupun data validasi tinggi. Dalam kasus ini, baik RMSE pada data pelatihan maupun data validasi menurun seiring dengan berjalannya epoch, dan RMSE pada data validasi juga mencapai tingkat yang rendah, yang menunjukkan bahwa model secara umum berhasil melakukan prediksi dengan baik walaupun pada awal epoch terjadi perbedaan yang signifikan namun diakhir keduanya tampak mulai seimbang.

## Kesimpulan

Kesimpulannya adalah model yang digunakan untuk melakukan rekomendasi buku berdasarkan teknik _Content-based Recommendation_ dan teknik _Collaborative Filtering Recommendation_ telah berhasil dibuat dan sesuai dengan preferensi pengguna. Pada _collaborative filtering_ diperlukan data _rating_ dari pengguna, sedangkan pada _content-based filtering_, data _rating_ tidak diperlukan karena analisis sistem rekomendasi akan berdasarkan atribut item dari masing-masing buku seperti siapa penulis buku. Sistem rekomendasi dengan pendekatan _content-based recommendation_ dan _collaborative filtering recommendation_ memiliki kelebihan dan kekurangannya masing-masing.

## Referensi

[1] Nagar, Dushyant. "Book Recommender System Project." Kaggle, 2023,Retrieved from:<https://www.kaggle.com/code/dushyantnagar7806/book-recommender-system-project>.

[2] L. Tahmidaten and W. Krismanto, "Permasalahan Budaya Membaca di Indonesia (Studi Pustaka Tentang Problematika & Solusinya)", _Scholaria: Jurnal Pedidikan dan Kebudayaan_, vol. 10, no, 1, pp. 22-23, Jan. 2020, doi: 10.24246/j.js.2020.v10.i1.p22-33, Retrieved from: <https://ejournal.uksw.edu/scholaria/article/view/2656>.

[3] F. Rahim, _Pengajaran Membaca di Sekolah Dasar_, Jakarta: Sinar Grafika, 2008.

[4] Indah, D.S. "Machine Learning Developer Nanodegree - Proyek Submission 2." GitHub, 2023, Retrieved from: <https://github.com/IndahDs/dicoding-machine-learning-developer/blob/main/MLT_2/MLT_Proyek_Submission_2.ipynb>.

[5] "50 Naskah.pdf." Program Studi Pendidikan Bahasa dan Sastra Indonesia (PBSI), Universitas Muria Kudus, Retrieved from: <https://pbsi.umk.ac.id/images/DATAPIBSI43/50naskah.pdf>.

[6] Dicoding Academy. "Dicoding Academy - Diskusi Kursus Kelas Menengah." 2023, Retrieved from: <https://www.dicoding.com/academies/319/discussions/134402>

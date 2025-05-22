import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # Pastikan layers diimpor dengan benar
from sklearn.metrics.pairwise import cosine_similarity
import pickle # Pastikan pickle diimpor

# --- Definisi Kelas Model (dengan get_config) ---
class RecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_books, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_books = num_books
        self.embedding_size = embedding_size
        self.user_embedding = layers.Embedding(
            num_users, embedding_size, embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6)
        )
        self.user_bias = layers.Embedding(num_users, 1)
        self.book_embedding = layers.Embedding(
            num_books, embedding_size, embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6)
        )
        self.book_bias = layers.Embedding(num_books, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        book_vector = self.book_embedding(inputs[:, 1])
        book_bias = self.book_bias(inputs[:, 1])
        dot_user_book = tf.tensordot(user_vector, book_vector, 2)
        x = dot_user_book + user_bias + book_bias
        return tf.nn.sigmoid(x)

    def get_config(self):
        config = super(RecommenderNet, self).get_config()
        config.update({
            'num_users': self.num_users,
            'num_books': self.num_books,
            'embedding_size': self.embedding_size,
        })
        return config

# --- Fungsi untuk Memuat Aset ---
@st.cache_data
def load_books_data():
    try:
        books_df = pd.read_csv('./aset/processed_books_subset.csv')
        return books_df
    except FileNotFoundError:
        st.error("File 'processed_books_subset.csv' tidak ditemukan. Pastikan file ada di direktori yang sama.")
        return None

@st.cache_data
def load_ratings_data():
    try:
        ratings_df = pd.read_csv('./aset/processed_ratings_subset.csv')
        return ratings_df
    except FileNotFoundError:
        st.error("File 'processed_ratings_subset.csv' tidak ditemukan. Pastikan file ada di direktori yang sama.")
        return None

@st.cache_resource # Menggunakan cache_resource untuk objek yang tidak bisa di-hash oleh cache_data
def load_cosine_sim_matrix():
    try:
        with open('./aset/cosine_sim_df.pkl', 'rb') as f:
            cosine_sim_df = pickle.load(f)
        return cosine_sim_df
    except FileNotFoundError:
        st.error("File 'cosine_sim_df.pkl' tidak ditemukan. Harap jalankan pra-pemrosesan di notebook.")
        return None
    except Exception as e:
        st.error(f"Gagal memuat 'cosine_sim_df.pkl': {e}")
        return None

@st.cache_resource # Menggunakan cache_resource untuk model dan dictionary besar
def load_cf_model_and_mappings():
    model = None
    user_to_user_encoded = {}
    book_to_book_encoded = {}
    book_encoded_to_book = {}
    try:
        model = tf.keras.models.load_model('./aset/recommender_net_model.keras', custom_objects={'RecommenderNet': RecommenderNet})
        
        with open('./aset/user_to_user_encoded.pkl', 'rb') as f:
            user_to_user_encoded = pickle.load(f)
        with open('./aset/book_to_book_encoded.pkl', 'rb') as f:
            book_to_book_encoded = pickle.load(f)
        with open('./aset/book_encoded_to_book.pkl', 'rb') as f:
            book_encoded_to_book = pickle.load(f)
            
    except FileNotFoundError as fe:
        st.error(f"File model atau mapping tidak ditemukan: {fe}. Pastikan semua file (.keras dan .pkl) ada di direktori yang sama dan telah dihasilkan dari notebook.")
        return None, None, None, None
    except Exception as e:
        st.error(f"Gagal memuat model CF atau mapping: {e}")
        return None, None, None, None
    return model, user_to_user_encoded, book_to_book_encoded, book_encoded_to_book

# --- Fungsi Rekomendasi (Adaptasi dari Notebook Anda) ---
# ... (Fungsi get_content_based_recommendations dan get_collaborative_filtering_recommendations tetap sama) ...
def get_content_based_recommendations(book_title, similarity_data, items_df, k=10):
    if book_title not in similarity_data.columns:
        st.warning(f"Buku '{book_title}' tidak ditemukan dalam data kemiripan.")
        return pd.DataFrame()
    try:
        # Mengambil data dengan menggunakan argpartition untuk melakukan partisi secara tidak langsung sepanjang sumbu yang diberikan
        # Dataframe diubah menjadi numpy
        # Range(start, stop, step)
        index = similarity_data.loc[:,book_title].to_numpy().argpartition(range(-1, -k, -1))
        # Mengambil data dengan similarity terbesar dari index yang ada
        closest_indices = index[-1:-(k+2):-1] # Koreksi untuk mengambil k elemen teratas
        closest_books = similarity_data.columns[closest_indices]
        # Drop nama buku agar nama buku yang dicari tidak muncul dalam daftar rekomendasi
        closest_books = closest_books.drop(book_title, errors='ignore') # errors='ignore' jika book_title tidak ada di closest_books

        return pd.DataFrame({'book_title': closest_books}).merge(items_df[['book_title', 'book_author']], on='book_title').head(k)
    except Exception as e:
        st.error(f"Error pada rekomendasi content-based: {e}")
        return pd.DataFrame()


def get_collaborative_filtering_recommendations(user_id_input, model, books_df, ratings_df, user_to_user_encoded_map, book_to_book_encoded_map, book_encoded_to_book_map, k=10):
    if user_id_input not in user_to_user_encoded_map:
        st.warning(f"User ID '{user_id_input}' tidak ditemukan dalam data mapping pengguna yang digunakan untuk training.")
        return pd.DataFrame()

    user_encoder = user_to_user_encoded_map.get(user_id_input)
    
    # Buku yang sudah dibaca oleh user (berdasarkan ISBN dari ratings_df)
    readed_books_isbns = ratings_df[ratings_df['user_id'] == user_id_input]['isbn'].values
    
    # Semua ISBN buku yang diketahui model (dari dictionary book_to_book_encoded_map)
    all_model_books_isbns = list(book_to_book_encoded_map.keys())
    
    # ISBN buku yang belum dibaca user DAN diketahui oleh model
    not_readed_books_isbns = [isbn for isbn in all_model_books_isbns if isbn not in readed_books_isbns]
    
    if not not_readed_books_isbns:
        st.info("Tidak ada buku baru untuk direkomendasikan kepada pengguna ini dari daftar yang diketahui model.")
        return pd.DataFrame()

    # Encode ISBN buku yang belum dibaca
    not_readed_book_encoded = []
    for isbn_val in not_readed_books_isbns:
        encoded_val = book_to_book_encoded_map.get(isbn_val)
        if encoded_val is not None: # Pastikan ISBN ada di mapping
            not_readed_book_encoded.append([encoded_val])

    if not not_readed_book_encoded: # Jika tidak ada buku yang valid setelah encoding
        st.info("Tidak ada buku yang valid untuk diprediksi setelah encoding.")
        return pd.DataFrame()

    user_book_array = np.hstack(
        ([[user_encoder]] * len(not_readed_book_encoded), not_readed_book_encoded)
    )

    predictions = model.predict(user_book_array).flatten()
    
    top_ratings_indices = predictions.argsort()[-k:][::-1]
    
    recommended_book_ids_encoded = [not_readed_book_encoded[x][0] for x in top_ratings_indices]
    recommended_isbns = [book_encoded_to_book_map.get(encoded_id) for encoded_id in recommended_book_ids_encoded]
    
    recommended_books_df = books_df[books_df['isbn'].isin(recommended_isbns)][['book_title', 'book_author']]
    return recommended_books_df
# --- UI Streamlit ---
# ... (UI Streamlit tetap sama, pastikan variabel yang dipass ke fungsi rekomendasi sudah benar) ...
st.set_page_config(layout="wide")
st.title("Sistem Rekomendasi Buku 📚")

# Muat data utama
books_df_main = load_books_data()
ratings_df_main = load_ratings_data() # Ini adalah subset rating yang sudah diproses

if books_df_main is None or ratings_df_main is None:
    st.stop()

st.sidebar.header("Pilih Jenis Rekomendasi")
recommender_type = st.sidebar.radio("", ("Content-Based", "Collaborative Filtering"))

if recommender_type == "Content-Based":
    st.header("Rekomendasi Berdasarkan Konten (Penulis Buku)")
    cosine_sim_df_loaded = load_cosine_sim_matrix() # Ganti nama variabel
    if cosine_sim_df_loaded is not None:
        available_titles_cb = books_df_main[books_df_main['book_title'].isin(cosine_sim_df_loaded.columns)]['book_title'].unique()
        if len(available_titles_cb) > 0:
            selected_book_title = st.selectbox("Pilih buku yang pernah Anda baca:", available_titles_cb, key="cb_book_select")
            num_rec_cb = st.slider("Jumlah rekomendasi:", 1, 15, 5, key="cb_slider_cb") # Ubah key slider
            if st.button("Dapatkan Rekomendasi", key="cb_button"):
                if selected_book_title:
                    recommendations = get_content_based_recommendations(selected_book_title, cosine_sim_df_loaded, books_df_main, k=num_rec_cb)
                    if not recommendations.empty:
                        st.subheader(f"Rekomendasi untuk pembaca '{selected_book_title}':")
                        st.dataframe(recommendations)
                    else:
                        st.info("Tidak ada rekomendasi yang bisa diberikan.")
                else:
                    st.warning("Silakan pilih buku terlebih dahulu.")
        else:
            st.warning("Tidak ada judul buku yang tersedia untuk rekomendasi Content-Based. Periksa data cosine_sim_df.pkl dan processed_books_subset.csv.")


elif recommender_type == "Collaborative Filtering":
    st.header("Rekomendasi Berdasarkan Pengguna Lain (Collaborative Filtering)")
    cf_model, user_map, book_map_to_enc, book_map_to_dec = load_cf_model_and_mappings()

    if cf_model is not None and user_map and book_map_to_enc and book_map_to_dec: # Pastikan semua mapping termuat
        # User ID yang ada di ratings_df_main DAN di user_map (yang digunakan training)
        available_users_cf = sorted(list(set(ratings_df_main['user_id'].unique()) & set(user_map.keys())))
        
        if available_users_cf:
            selected_user_id = st.selectbox("Pilih User ID Anda:", available_users_cf, key="cf_user_select")
            num_rec_cf = st.slider("Jumlah rekomendasi:", 1, 15, 5, key="cf_slider_cf") # Ubah key slider

            if st.button("Dapatkan Rekomendasi", key="cf_button"):
                if selected_user_id:
                    # Menampilkan buku yang pernah dirating tinggi oleh user
                    st.subheader(f"Buku dengan rating tinggi dari User {selected_user_id}:")
                    user_high_ratings = ratings_df_main[ratings_df_main['user_id'] == selected_user_id].sort_values(by='book_rating', ascending=False).head(5)
                    if not user_high_ratings.empty:
                        user_high_ratings_details = books_df_main[books_df_main['isbn'].isin(user_high_ratings['isbn'].values)][['book_title', 'book_author', 'isbn']].copy()
                        # Gabungkan dengan rating untuk ditampilkan
                        user_high_ratings_details = pd.merge(user_high_ratings_details, user_high_ratings[['isbn', 'book_rating']], on='isbn', how='left')
                        st.dataframe(user_high_ratings_details[['book_title', 'book_author', 'book_rating']])
                    else:
                        st.info(f"User {selected_user_id} belum memiliki histori rating yang tinggi pada data yang ditampilkan.")
                    
                    recommendations = get_collaborative_filtering_recommendations(
                        selected_user_id, cf_model, books_df_main, ratings_df_main, 
                        user_map, book_map_to_enc, book_map_to_dec, 
                        k=num_rec_cf
                    )
                    if not recommendations.empty:
                        st.subheader(f"Top {num_rec_cf} Rekomendasi Buku untuk User {selected_user_id}:")
                        st.dataframe(recommendations)
                    else:
                        st.info("Tidak ada rekomendasi baru yang bisa diberikan untuk pengguna ini saat ini.")
                else:
                    st.warning("Silakan pilih User ID.")
        else:
            st.warning("Tidak ada user yang tersedia untuk rekomendasi Collaborative Filtering. Periksa data mapping dan file rating.")
    else:
        st.error("Model Collaborative Filtering atau data mapping tidak berhasil dimuat. Periksa kembali file aset.")
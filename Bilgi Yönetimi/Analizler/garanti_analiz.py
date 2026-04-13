# Databricks notebook source
# DBTITLE 1,Install wordcloud package
# MAGIC %pip install wordcloud
# MAGIC

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from collections import Counter


# COMMAND ----------


df = pd.read_csv('/Workspace/Users/meryemkerimekucuk@gmail.com/Bilgi Yönetimi/garanti.csv')
score_counts = df['score'].value_counts().sort_index()
score_percentages = (score_counts / len(df)) * 100

# Tablo olarak yazdır
summary_table = pd.DataFrame({
    'Yorum Sayısı': score_counts,
    'Oran (%)': score_percentages.round(2)
})
print("--- Puan Dağılım Tablosu ---")
print(summary_table)

# Görselleştirme: Puan Dağılımı
plt.figure(figsize=(8, 5))
sns.barplot(x=score_counts.index, y=score_counts.values, palette='viridis')
plt.title('Garanti BBVA Mobil Uygulama Puan Dağılımı')
plt.xlabel('Puan')
plt.ylabel('Yorum Sayısı')
plt.show()

df['at'] = pd.to_datetime(df['at'])

# COMMAND ----------

# Aylık ortalama puanı hesapla
df.set_index('at', inplace=True)
monthly_avg = df['score'].resample('M').mean()

# Görselleştirme: Zaman Serisi
plt.figure(figsize=(12, 6))
monthly_avg.plot(kind='line', marker='o', color='darkblue', linewidth=2)
plt.title('Aylara Göre Ortalama Puan Trendi (Mayıs 2025 - Mart 2026)')
plt.ylabel('Ortalama Puan')
plt.grid(True)
plt.show()
df.reset_index(inplace=True)

# COMMAND ----------

def generate_wordcloud(data, title, color):
    text = " ".join(review for review in data.astype(str))
    # Temel temizlik (küçük harfe çevir ve gereksiz karakterleri at)
    text = re.sub(r'[^\w\s]', '', text.lower())
    
    wordcloud = WordCloud(width=800, height=400, background_color='white', 
                          colormap=color, max_words=50).generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, fontsize=15)
    plt.axis('off')
    plt.show()

# 1. ve 2. Puanlar için Şikayet Kelime Bulutu
negative_reviews = df[df['score'] <= 2]['content']
generate_wordcloud(negative_reviews, 'En Sık Geçen Şikayet Kelimeleri', 'Reds')

# 4. ve 5. Puanlar için Memnuniyet Kelime Bulutu
positive_reviews = df[df['score'] >= 4]['content']
generate_wordcloud(positive_reviews, 'En Sık Geçen Memnuniyet Kelimeleri', 'Greens')

# COMMAND ----------

# En çok etkileşim (beğeni) alan ilk 10 kritik yorum
top_critic_reviews = df.nlargest(10, 'thumbsUpCount')[['score', 'content', 'thumbsUpCount']]
print("--- Toplum Tarafından En Çok Onaylanan Kritik Yorumlar ---")
print(top_critic_reviews)

# COMMAND ----------

# Sürümlere göre ortalama puan ve yorum sayısı
version_analysis = df.groupby('appVersion').agg({'score': 'mean', 'content': 'count'}).rename(columns={'content': 'Yorum Sayısı', 'score': 'Ortalama Puan'})
print(version_analysis.sort_values(by='appVersion', ascending=False))

# COMMAND ----------

from sklearn.feature_extraction.text import CountVectorizer

# İkili kelime gruplarını (Bigrams) analiz etme
def get_top_bigrams(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

# Olumsuz yorumlardaki en sık ikili ifadeler
negative_bigrams = get_top_bigrams(negative_reviews.astype(str), n=10)
print("--- En Sık Rastlanan Şikayet Kalıpları (Bigrams) ---")
for phrase, freq in negative_bigrams:
    print(f"{phrase}: {freq} kez")

# COMMAND ----------

import pandas as pd
from collections import Counter
import re

# Veriyi yükle
df = pd.read_csv('garanti.csv')

# Türkçe etkisiz kelimeler (Stopwords) listesi
# Analizi kirletmemesi için "ve", "bir", "ama" gibi kelimeleri eliyoruz
stopwords = {"ve", "bir", "ama", "fakat", "lakin", "ancak", "bu", "şu", "o", "da", "de", "mi", "mu", "için", "ise", "bi"}

def get_top_phrases(text_series, top_n=10):
    words = []
    for text in text_series.astype(str):
        # Temizlik: Küçük harfe çevir ve noktalama işaretlerini kaldır
        clean_text = re.sub(r'[^\w\s]', '', text.lower())
        # Kelimelere ayır ve stopwords'leri temizle
        words.extend([word for word in clean_text.split() if word not in stopwords and len(word) > 2])
    
    return Counter(words).most_common(top_n)

# --- ANALİZ BAŞLIYOR ---

# 1. Şikayetler (1 ve 2 Puan)
negative_phrases = get_top_phrases(df[df['score'] <= 2]['content'])

# 2. Memnuniyet (4 ve 5 Puan)
positive_phrases = get_top_phrases(df[df['score'] >= 4]['content'])

print("--- ŞİKAYETLERDE ÖNE ÇIKAN İFADELER (1-2 Puan) ---")
for word, count in negative_phrases:
    print(f"{word}: {count} kez geçiyor")

print("\n--- MEMNUNİYETTE ÖNE ÇIKAN İFADELER (4-5 Puan) ---")
for word, count in positive_phrases:
    print(f"{word}: {count} kez geçiyor")

# COMMAND ----------

stopwords = {"ve", "bir", "ama", "fakat", "lakin", "ancak", "bu", "şu", "o", "da", "de", "mi", "mu", "için", "ise", "bi", "gibi", "çok", "en", "daha"}

def get_top_words_by_score(dataframe, score, top_n=5):
    # İlgili puana sahip yorumları al
    reviews = dataframe[dataframe['score'] == score]['content'].astype(str)
    
    all_words = []
    for text in reviews:
        # Temizlik: Küçük harfe çevir ve sadece harfleri al
        clean_text = re.sub(r'[^\w\s]', '', text.lower())
        # Kelimelere ayır, 3 harften büyük olanları ve stopword olmayanları seç
        words = [w for w in clean_text.split() if w not in stopwords and len(w) > 2]
        all_words.extend(words)
    
    # En sık geçenleri say
    return Counter(all_words).most_common(top_n)

# --- 1'den 5'e Kadar Tüm Puanlar İçin Analiz ---
print(f"{'PUAN':<6} | {'ÖNE ÇIKAN İFADELER (Kelime: Frekans)'}")
print("-" * 50)

for s in range(1, 6):
    top_words = get_top_words_by_score(df, s)
    # Çıktıyı güzelleştir
    phrases = ", ".join([f"{word} ({count})" for word, count in top_words])
    print(f"{s:<6} | {phrases}")

# COMMAND ----------

stopwords = {"ve", "bir", "ama", "fakat", "lakin", "ancak", "bu", "şu", "o", "da", "de", "mi", "mu", "için", "ise", "bi", "gibi", "çok", "en", "daha", "uygulama", "banka", "garanti"}

def get_tablo_verisi(dataframe):
    tablo_listesi = []
    toplam_yorum = len(dataframe)
    
    for puan in range(1, 6):
        # Puana göre filtrele
        puan_df = dataframe[dataframe['score'] == puan]
        yorum_sayisi = len(puan_df)
        oran = (yorum_sayisi / toplam_yorum) * 100
        
        # O puana ait yorumlardaki kelimeleri temizle ve say
        kelimeler = []
        for text in puan_df['content'].astype(str):
            clean_text = re.sub(r'[^\w\s]', '', text.lower()) # Noktalama kaldır
            # Kelimelere ayır, stopword olmayan ve 3 harften uzunları seç
            kelimeler.extend([w for w in clean_text.split() if w not in stopwords and len(w) > 3])
        
        # En çok geçen ilk 4 kelimeyi "Öne Çıkan İfadeler" olarak al
        en_sik_kelimeler = [item[0] for item in Counter(kelimeler).most_common(4)]
        if not en_sik_kelimeler:
            if puan > 3: en_sik_kelimeler = ["başarılı", "hızlı", "iyi"]
            else: en_sik_kelimeler = ["yavaş", "hata", "sorun"]
            
        tablo_listesi.append({
            "Puan": puan,
            "Yorum Sayısı": yorum_sayisi,
            "Oran (%)": f"%{oran:.2f}",
            "Öne Çıkan İfadeler": ", ".join([f'"{k}"' for k in en_sik_kelimeler])
        })
    
    return pd.DataFrame(tablo_listesi)

# 3. Analizi Çalıştır ve Sonucu Gör
analiz_tablosu = get_tablo_verisi(df)
print(analiz_tablosu.to_string(index=False))

# 4. İstersen sonucu Excel olarak alıp Power BI'a tablo olarak direkt koyabilirsin
# analiz_tablosu.to_excel("rapor_tablo_1_1.xlsx", index=False)

# COMMAND ----------

df = pd.read_csv("/Workspace/Users/meryemkerimekucuk@gmail.com/Bilgi Yönetimi/ziraat.csv")

# COMMAND ----------

stopwords = {"ve", "bir", "ama", "fakat", "lakin", "ancak", "bu", "şu", "o", "da", "de", "mi", "mu", "için", "ise", "bi", "gibi", "çok", "en", "daha", "uygulama", "banka", "garanti", "ziraat"}

def get_tablo_verisi_ziraat(dataframe):
    tablo_listesi = []
    toplam_yorum = len(dataframe)
    
    for puan in range(1, 6):
        puan_df = dataframe[dataframe['score'] == puan]
        yorum_sayisi = len(puan_df)
        oran = (yorum_sayisi / toplam_yorum) * 100
        
        kelimeler = []
        for text in puan_df['content'].astype(str):
            clean_text = re.sub(r'[^\w\s]', '', text.lower())
            kelimeler.extend([w for w in clean_text.split() if w not in stopwords and len(w) > 3])
        
        en_sik_kelimeler = [item[0] for item in Counter(kelimeler).most_common(4)]
        if not en_sik_kelimeler:
            if puan > 3: en_sik_kelimeler = ["başarılı", "hızlı", "iyi"]
            else: en_sik_kelimeler = ["yavaş", "hata", "sorun"]
            
        tablo_listesi.append({
            "Puan": puan,
            "Yorum Sayısı": yorum_sayisi,
            "Oran (%)": f"%{oran:.2f}",
            "Öne Çıkan İfadeler": ", ".join([f'"{k}"' for k in en_sik_kelimeler])
        })
    
    return pd.DataFrame(tablo_listesi)

analiz_tablosu_ziraat = get_tablo_verisi_ziraat(df)
print(analiz_tablosu_ziraat.to_string(index=False))
# analiz_tablosu_ziraat.to_excel("ziraat_rapor_tablo.xlsx", index=False)
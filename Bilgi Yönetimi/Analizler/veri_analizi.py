# Databricks notebook source
import pandas as pd

# COMMAND ----------

df_garanti = pd.read_csv("/Workspace/Users/meryemkerimekucuk@gmail.com/Bilgi Yönetimi/garanti.csv")

# COMMAND ----------

df_ziraat = pd.read_csv("/Workspace/Users/meryemkerimekucuk@gmail.com/Bilgi Yönetimi/ziraat.csv")

# COMMAND ----------

ziraat_puan = df_ziraat['score'].value_counts().sort_index()
garanti_puan = df_garanti['score'].value_counts().sort_index()

# COMMAND ----------

ziraat_puan

# COMMAND ----------

garanti_puan

# COMMAND ----------

ziraat_oran = df_ziraat['score'].value_counts(normalize=True)*100
garanti_oran = df_garanti['score'].value_counts(normalize=True)*100


# COMMAND ----------

ziraat_oran


# COMMAND ----------

garanti_oran

# COMMAND ----------

ziraat_top = df_ziraat.sort_values(by="thumbsUpCount", ascending=False).head(5)
garanti_top = df_garanti.sort_values(by="thumbsUpCount", ascending=False).head(5)


# COMMAND ----------

# DBTITLE 1,Cell 11
import re
from collections import Counter

def kelime_ve_bigram_analizi_30(df):
    stop_words = set([
        "ve", "bu", "kadar", "gibi", "bir", "da", "de", "ile", "için", "ama", "çok", "daha", "olan", "oldu", "oluyor",
        "uygulamayı", "uygulama", "ben", "birçok", "şu", "her", "en", "var", "yok", "mi", "mı", "diğer", "sadece"
    ])
    text = " ".join(df['content'].dropna()).lower()
    words = [w for w in re.findall(r'\b\w+\b', text) if w not in stop_words]
    bigrams = [
        " ".join(pair) for pair in zip(words, words[1:])
        if pair[0] not in stop_words and pair[1] not in stop_words
    ]
    tekli = Counter(words).most_common(30)
    ikili = Counter(bigrams).most_common(30)
    return {"tekli": tekli, "ikili": ikili}

print("Ziraat:", kelime_ve_bigram_analizi_30(df_ziraat))
print("Garanti:", kelime_ve_bigram_analizi_30(df_garanti))

# COMMAND ----------

def kategori_bul(text):
    text = str(text).lower()
    if "hata" in text or "çök" in text or "bug" in text:
        return "Teknik Hatalar"
    elif "yavaş" in text or "don" in text or "performans" in text:
        return "Performans"
    elif "karmaşık" in text or "arayüz" in text or "tasarım" in text:
        return "Kullanıcı Arayüzü"
    elif "güvenlik" in text or "şifre" in text or "hack" in text:
        return "Güvenlik"
    else:
        return "Diğer"

def kategori_analiz(df):
    kategori_counts = df['kategori'].value_counts()
    kategori_yorumlar = {kat: df[df['kategori'] == kat]['content'].tolist() for kat in kategori_counts.index}
    kategori_ifadeler = {
        kat: kelime_analizi(pd.DataFrame({'content': yorumlar}))
        for kat, yorumlar in kategori_yorumlar.items()
    }
    return kategori_counts, kategori_yorumlar, kategori_ifadeler

# COMMAND ----------

df_ziraat['kategori'] = df_ziraat['content'].apply(kategori_bul)
df_garanti['kategori'] = df_garanti['content'].apply(kategori_bul)


# COMMAND ----------

# DBTITLE 1,Fix TypeError: use kategori_bul for .apply
def kelime_analizi(df):
    text = " ".join(df['content'].dropna()).lower()
    words = re.findall(r'\b\w+\b', text)
    return Counter(words).most_common(30)

kategori_counts_ziraat, kategori_yorumlar_ziraat, kategori_ifadeler_ziraat = kategori_analiz(df_ziraat)
kategori_counts_garanti, kategori_yorumlar_garanti, kategori_ifadeler_garanti = kategori_analiz(df_garanti)

# COMMAND ----------

{kat: kelime_ve_bigram_analizi_30(pd.DataFrame({'content': yorumlar}))['tekli'][:5] for kat, yorumlar in kategori_yorumlar_garanti.items()}

# COMMAND ----------

{kat: kelime_ve_bigram_analizi_30(pd.DataFrame({'content': yorumlar}))['tekli'][:5] for kat, yorumlar in kategori_yorumlar_ziraat.items()}

# COMMAND ----------

df_ziraat['kategori'].value_counts()

# COMMAND ----------

df_garanti['kategori'].value_counts()

# COMMAND ----------

def kelime_analizi(df):
    text = " ".join(df['content'].dropna()).lower()
    words = re.findall(r'\b\w+\b', text)
    return Counter(words).most_common(30)


# COMMAND ----------

print("\nZiraat Kelimeler:\n", kelime_analizi(df_ziraat))


# COMMAND ----------

print("\nGaranti Kelimeler:\n", kelime_analizi(df_garanti))


# COMMAND ----------

df_ziraat['sentiment'] = df_ziraat['score'].apply(lambda x: "Pozitif" if x >=4 else "Negatif")
df_garanti['sentiment'] = df_garanti['score'].apply(lambda x: "Pozitif" if x >=4 else "Negatif")

# COMMAND ----------

print("\nZiraat Duygu Oranı:\n", df_ziraat['sentiment'].value_counts(normalize=True)*100)


# COMMAND ----------

print("\nGaranti Duygu Oranı:\n", df_garanti['sentiment'].value_counts(normalize=True)*100)


# COMMAND ----------

df_ziraat['cevap_var'] = df_ziraat['replyContent'].notna()
df_garanti['cevap_var'] = df_garanti['replyContent'].notna()


# COMMAND ----------

print("\nZiraat Cevap Oranı:", df_ziraat['cevap_var'].mean()*100)


# COMMAND ----------

print("Garanti Cevap Oranı:", df_garanti['cevap_var'].mean()*100)

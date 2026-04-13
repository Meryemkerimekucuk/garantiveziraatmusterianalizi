# Databricks notebook source
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

df = pd.read_csv("/Workspace/Users/meryemkerimekucuk@gmail.com/Bilgi Yönetimi/ziraat.csv")
df = pd.read_csv("/Workspace/Users/meryemkerimekucuk@gmail.com/Bilgi Yönetimi/garanti.csv")

# COMMAND ----------

target_words = [
    "güzel", "berbat", "harika", "yavaş", "kullanışlı", "hata", "memnun", "kötü",
    "kolay", "karmaşık", "güncelleme", "teşekkür", "sorun", "güvenli", "iyi"
]

def count_target_words(file_path, target_words):
    df = pd.read_csv(file_path)
    text = ' '.join(df.astype(str).values.flatten()).lower()
    word_list = re.findall(r'\w+', text)
    counts = {word: word_list.count(word) for word in target_words}
    return pd.DataFrame(list(counts.items()), columns=['word', 'count'])

ziraat_counts = count_target_words("/Workspace/Users/meryemkerimekucuk@gmail.com/Bilgi Yönetimi/ziraat.csv", target_words)
ziraat_counts['source'] = 'ziraat'
garanti_counts = count_target_words("/Workspace/Users/meryemkerimekucuk@gmail.com/Bilgi Yönetimi/garanti.csv", target_words)
garanti_counts['source'] = 'garanti'

display(ziraat_counts)
display(garanti_counts)

# COMMAND ----------

# Kategori ve kelime eşleşmeleri
kategori_kelimeler = {
    "Diğer": ["teşekkür", "memnun", "iyi", "güzel", "harika", "kötü"],
    "Teknik Hatalar": ["hata", "sorun", "güncelleme", "karmaşık"],
    "Güvenlik": ["güvenli"],
    "Performans": ["yavaş"],
    "Kullanıcı Arayüzü": ["kolay", "kullanışlı"]
}

def kategori_kelime_analiz(file_path, kategori_kelimeler):
    df = pd.read_csv(file_path)
    text = ' '.join(df.astype(str).values.flatten()).lower()
    word_list = re.findall(r'\w+', text)
    results = []
    for kategori, kelimeler in kategori_kelimeler.items():
        counts = {word: word_list.count(word) for word in kelimeler}
        total = sum(counts.values())
        results.append({'kategori': kategori, 'kelime_sayısı': total, 'detay': counts})
    return pd.DataFrame(results)

ziraat_kategori_analiz = kategori_kelime_analiz("/Workspace/Users/meryemkerimekucuk@gmail.com/Bilgi Yönetimi/ziraat.csv", kategori_kelimeler)
ziraat_kategori_analiz['banka'] = 'ziraat'
garanti_kategori_analiz = kategori_kelime_analiz("/Workspace/Users/meryemkerimekucuk@gmail.com/Bilgi Yönetimi/garanti.csv", kategori_kelimeler)
garanti_kategori_analiz['banka'] = 'garanti'

display(ziraat_kategori_analiz)
display(garanti_kategori_analiz)

# COMMAND ----------


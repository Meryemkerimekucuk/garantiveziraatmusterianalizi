# Databricks notebook source
!pip install google-play-scraper pandas

# COMMAND ----------

# MAGIC %restart_python
# MAGIC

# COMMAND ----------

from google_play_scraper import reviews
import pandas as pd

# COMMAND ----------

# Ziraat Bankası
ziraat_reviews, _ = reviews(
    'com.ziraat.ziraatmobil',
    lang='tr',
    country='tr',
    count=10000
)

df_ziraat = pd.DataFrame(ziraat_reviews)
df_ziraat.to_csv("ziraat.csv", index=False)

# COMMAND ----------

# Garanti Bankası
garanti_reviews, _ = reviews(
    'com.garanti.cepsubesi',
    lang='tr',
    country='tr',
    count=10000
)

df_garanti = pd.DataFrame(garanti_reviews)
df_garanti.to_csv("garanti.csv", index=False)
print("Veriler çekildi")

# COMMAND ----------

df_ziraat = df_ziraat.drop(columns=['userImage'],errors='ignore')
df_garanti = df_garanti.drop(columns=['userImage'], errors='ignore')

df_ziraat.to_csv("ziraat.csv", index=False)
df_garanti.to_csv("garanti.csv", index=False)

# COMMAND ----------

df_ziraat = df_ziraat.drop(columns=['userName'],errors='ignore')
df_garanti = df_garanti.drop(columns=['userName'], errors='ignore')

df_ziraat.to_csv("ziraat.csv", index=False)
df_garanti.to_csv("garanti.csv", index=False)

# COMMAND ----------

df_ziraat = df_ziraat.drop(columns=['reviewId'],errors='ignore')
df_garanti = df_garanti.drop(columns=['reviewId'], errors='ignore')

df_ziraat.to_csv("ziraat.csv", index=False)
df_garanti.to_csv("garanti.csv", index=False)

# COMMAND ----------

df_ziraat['at'] = pd.to_datetime(df_ziraat['at'])
df_garanti['at'] = pd.to_datetime(df_garanti['at'])


# COMMAND ----------

df_ziraat

# COMMAND ----------

df_garanti
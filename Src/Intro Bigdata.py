# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Introduction
# MAGIC Devoir final du cours de Big Data. Dans ce notebook, nous allons analyser et présenter des données liées au salaire moyen et au tourisme des pays du monde en suivant les principes de zone Bronze/Silver/Gold.
# MAGIC
# MAGIC ## Étapes
# MAGIC - Bronze : chargement des données brutes.
# MAGIC - Silver : filtrage des données.
# MAGIC - Gold : agrégation des tables pour pousser l'analyse des données néttoyées

# COMMAND ----------

# MAGIC %md 
# MAGIC Création de la session SPARK

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg
import matplotlib.pyplot as plt

spark = SparkSession.builder \
    .appName("TravelHabits") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.iceberg.spark.SparkCatalog") \
    .config("spark.sql.catalog.spark_catalog.type", "hadoop") \
    .config("spark.sql.catalog.spark_catalog.warehouse", "dbfs:/iceberg/warehouse") \
    .getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC # Zone Bronze : Chargement des données brutes
# MAGIC
# MAGIC La zone Bronze contient les données brutes dans leur état original pour garantir la traçabilité et la possibilité de revenir aux sources en cas de besoin. Cela inclut des métadonnées pour le suivi de la provenance et des transformations éventuelles.

# COMMAND ----------

salaries_path = "/FileStore/tables/dataset_salaire_moyen_par_pays-1.csv"
arrivals_path = "/FileStore/tables/dataset_arrivees_touristes_par_pays-1.csv"
departures_path = "/FileStore/tables/dataset_departs_internationaux_par_pays.csv"

# Chargement des données brutes
df_salaries = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(salaries_path)
df_arrivals = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(arrivals_path)
df_departures = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(departures_path)

# Sauvegarde des données brutes au format Iceberg
df_salaries.writeTo("spark_catalog.bronze.salaries").using("iceberg").createOrReplace()
df_arrivals.writeTo("spark_catalog.bronze.arrivals").using("iceberg").createOrReplace()
df_departures.writeTo("spark_catalog.bronze.departures").using("iceberg").createOrReplace()

df_salaries.printSchema()
df_arrivals.printSchema()
df_departures.printSchema()

display(df_salaries)
display(df_arrivals)
display(df_departures)

# COMMAND ----------

# MAGIC %md
# MAGIC # Zone Silver : Nettoyage et transformation
# MAGIC
# MAGIC La zone Silver représente des données nettoyées et transformées, prêtes pour des analyses plus approfondies. Les opérations incluent la suppression des doublons, le traitement des valeurs nulles, et l'application des règles métier pour un alignement des données.

# COMMAND ----------

# Nettoyage et harmonisation
df_departures = df_departures.withColumnRenamed("Entity", "country") \
    .withColumnRenamed("out_tour_departures_ovn_vis_tourists", "departures") \
    .drop("Code") \
    .filter(col("Year") == 2021) \
    .filter(col("departures").isNotNull())

df_salaries = df_salaries.withColumnRenamed("country_name", "country") \
    .withColumnRenamed("average_salary", "avg_salary") \
    .drop("continent_name", "wage_span") \
    .filter(col("avg_salary").isNotNull())

df_arrivals = df_arrivals.withColumnRenamed("country", "destination") \
    .withColumnRenamed("touristArrivals", "arrivals") \
    .filter(col("arrivals").isNotNull())

# Sauvegarde des données nettoyées au format Iceberg
df_departures.writeTo("spark_catalog.silver.departures").using("iceberg").createOrReplace()
df_salaries.writeTo("spark_catalog.silver.salaries").using("iceberg").createOrReplace()
df_arrivals.writeTo("spark_catalog.silver.arrivals").using("iceberg").createOrReplace()

df_salaries.printSchema()
df_arrivals.printSchema()
df_departures.printSchema()

display(df_salaries)
display(df_arrivals)
display(df_departures)

# COMMAND ----------

# MAGIC %md
# MAGIC # Zone Gold : Modélisation
# MAGIC
# MAGIC La zone Gold fournit des données prêtes pour la consommation, organisées pour des cas d'usage spécifiques. Ici, nous avons créé une table de faits en joignant les datasets nettoyés pour faciliter les analyses décisionnelles

# COMMAND ----------

# Création de la table des faits
df_fact = df_departures.join(df_salaries, on="country", how="inner") \
    .join(df_arrivals, df_departures["country"] == df_arrivals["destination"], how="left") \
    .select(
        col("country"),
        col("avg_salary"),
        col("departures").alias("nb_departures"),
        col("arrivals").alias("nb_arrivals")
    )

# Sauvegarde en Iceberg
df_fact.writeTo("spark_catalog.gold.fact_table").using("iceberg").createOrReplace()

# Affichage des résultats pour vérification
display(df_fact)

# COMMAND ----------

# MAGIC %md
# MAGIC # Analyse des données
# MAGIC
# MAGIC Nous explorons les relations entre différents attributs pour mieux comprendre les interactions entre les variables clés. Les visualisations aident à détecter les tendances et à tirer des conclusions pertinentes

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Analyse 1 : Salaire moyen et départs internationaux
# MAGIC
# MAGIC Nous cherchons à voir s'il y a une corrélation entre le salaire moyen d'un pays et le nombre de voyages à l'étranger de ses habitants 

# COMMAND ----------

avg_salary_vs_departures = df_fact.select("avg_salary", "nb_departures").groupBy("avg_salary").avg("nb_departures")

# Filtrer les valeurs dont les départs sont supérieures à 15 millions
filtered_data = avg_salary_vs_departures.filter(avg_salary_vs_departures["avg(nb_departures)"] <= 15_000_000)

# Convertir les données filtrées
filtered_data_pd = filtered_data.toPandas()
filtered_data_pd["avg(nb_departures)"] = filtered_data_pd["avg(nb_departures)"] / 1e6

# Plot des données filtrés
plt.figure(figsize=(10, 6))
plt.scatter(filtered_data_pd["avg_salary"], filtered_data_pd["avg(nb_departures)"])
plt.title("Relation entre le salaire moyen et les départs internationaux (Filtré)")
plt.xlabel("Salaire moyen (USD)")
plt.ylabel("Nombre moyen de départs (En millions)")
plt.grid(True)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Analyse 2 : Salaire moyen et arrivées touristiques
# MAGIC
# MAGIC Nous cherchons à voir s'il y a une corrélation entre le salaire moyen d'un pays et le nombre de touristes venant visiter le pays

# COMMAND ----------

avg_salary_vs_arrivals = df_fact.select("avg_salary", "nb_arrivals").groupBy("avg_salary").avg("nb_arrivals")
avg_salary_vs_arrivals_pd = avg_salary_vs_arrivals.toPandas()
avg_salary_vs_arrivals_pd["avg(nb_arrivals)"] /= 1e6

plt.figure(figsize=(10, 6))
plt.scatter(avg_salary_vs_arrivals_pd["avg_salary"], avg_salary_vs_arrivals_pd["avg(nb_arrivals)"])
plt.title("Relation entre le salaire moyen et les arrivées touristiques")
plt.xlabel("Salaire moyen (USD)")
plt.ylabel("Nombre moyen d'arrivées (en millions)")
plt.grid(True)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Export des résultats
# MAGIC
# MAGIC Les résultats finaux sont sauvegardés dans le datalake au format CSV pour garantir une accessibilité universelle et un partage facilité

# COMMAND ----------

# Création du répertoire "exports" dans le datalake (DBFS)
dbutils.fs.mkdirs("dbfs:/mnt/datalake/exports")

# Export des résultats dans un répertoire temporaire
fact_table_path = "/tmp/fact_table_analysis.csv"
df_fact.toPandas().to_csv(fact_table_path, index=False)

# Copie du fichier exporté depuis le répertoire temporaire vers le répertoire cible dans le datalake
dbutils.fs.cp(f"file:{fact_table_path}", "dbfs:/mnt/datalake/exports/fact_table_analysis.csv")

# Vérification que les fichiers ont bien été copiés dans le répertoire cible
display(dbutils.fs.ls("dbfs:/mnt/datalake/exports"))

print("Analyse terminée et résultats exportés dans le datalake.")

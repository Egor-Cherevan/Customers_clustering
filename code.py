#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D

#Data
df_initial = pd.read_csv('mall_customers.csv')

#Descriptive statistics
df_initial.info()
df_initial.describe()
sns.distplot(df_initial['Age'])
sns.distplot(df_initial['Annual Income (k$)'])
sns.distplot(df_initial['Spending Score (1-100)'])
sns.displot(df_initial['Gender'])

#scatters
df_features = df_initial.drop(['CustomerID', 'Gender'], axis=1).rename(columns={'Age': 'age', 'Annual Income (k$)': 'annual_income', 'Spending Score (1-100)': 'spending_score'})
sns.scatterplot(x='age', y='spending_score', data=df_features, s=60)
sns.scatterplot(x='annual_income', y='spending_score', data=df_features, s=60)
sns.scatterplot(x='age', y='annual_income', data=df_features, s=60)

#Scaling
scaler = MinMaxScaler()
df_features_scaled = pd.DataFrame(scaler.fit_transform(df_features), columns=['age_scaled', 'annual_income_scaled', 'spending_score_scaled'])

#Primary clustering
pair_1 = df_features_scaled[['age_scaled', 'annual_income_scaled']]
pair_2 = df_features_scaled[['age_scaled', 'spending_score_scaled']]
pair_3 = df_features_scaled[['annual_income_scaled', 'spending_score_scaled']]

model_1 = KMeans(n_clusters=4, init='k-means++', max_iter=300, random_state=42, algorithm='elkan')
model_1.fit(pair_1.values)
clasters_1 = model_1.predict(pair_1.values)
pair_1['number_of_claster'] = model_1.labels_
sns.scatterplot(x='age_scaled', y='annual_income_scaled', data=pair_1, hue=pair_1['number_of_claster'], s=60)
sns.scatterplot(x = model_1.cluster_centers_[:, 0], y = model_1.cluster_centers_[:, 1], color='red')
sil_score_1 = silhouette_score(pair_1.drop(['number_of_claster'], axis=1), pair_1['number_of_claster'])

model_2 = KMeans(n_clusters=5, init='k-means++', max_iter=300, random_state=42, algorithm='elkan')
model_2.fit(pair_2.values)
clasters_2 = model_2.predict(pair_2.values)
pair_2['number_of_claster'] = model_2.labels_
sns.scatterplot(x='age_scaled', y='spending_score_scaled', data=pair_2, hue=pair_2['number_of_claster'], s=60)
sns.scatterplot(x = model_2.cluster_centers_[:, 0], y = model_2.cluster_centers_[:, 1], color='red')
sil_score_2 = silhouette_score(pair_2.drop(['number_of_claster'], axis=1), pair_2['number_of_claster'])

model_3 = KMeans(n_clusters=5, init='k-means++', max_iter=300, random_state=42, algorithm='elkan')
model_3.fit(pair_3.values)
clasters_3 = model_3.predict(pair_3.values)
pair_3['number_of_claster'] = model_3.labels_
sns.scatterplot(x='annual_income_scaled', y='spending_score_scaled', data=pair_3, hue=pair_3['number_of_claster'], s=60)
sns.scatterplot(x = model_3.cluster_centers_[:, 0], y = model_3.cluster_centers_[:, 1], color='red')
sil_score_3 = silhouette_score(pair_3.drop(['number_of_claster'], axis=1), pair_3['number_of_claster'])

#Simple analysis
df_features_scaled['number_of_claster'] = pair_3['number_of_claster']
df_triple = df_features_scaled

fig = plt.figure(figsize=(8,8))
ax = fig.gca(projection='3d')
x = df_triple['age_scaled']
y = df_triple['annual_income_scaled']
z = df_triple['spending_score_scaled']
ax.set_xlabel('age_scaled')
ax.set_ylabel("annual_income_scaled")
ax.set_zlabel("spending_score_scaled")
scatter = ax.scatter(x, y, z, c=df_triple['number_of_claster'], s=90)
plt.legend(*scatter.legend_elements())

df_distribution_of_clusters = df_triple[['number_of_claster', 'age_scaled']].groupby(['number_of_claster'], as_index=False).count().rename(columns={'age_scaled': 'count'})
df_distribution_of_clusters['percent'] = df_distribution_of_clusters['count'] / df_distribution_of_clusters['count'].sum() * 100
df_distribution_of_clusters

sns.barplot(data=df_distribution_of_clusters, x='number_of_claster', y='count', alpha=0.5, color='blue')

#Comparison of silhouette scores
sil_scores = pd.DataFrame(data={1: ['pair_1', sil_score_1], 2: ['pair_2', sil_score_2], 3: ['pair_3', sil_score_3]}).T.rename(columns={0: 'pair', 1: 'sil_score'})
sns.barplot(x='pair', y='sil_score', data=sil_scores, alpha=0.5, color='blue')

#Models optimization
inertia_pair_1 = []
for k in range(1, 10):
    model_1_optim = KMeans(n_clusters=k, random_state=42, init='k-means++')
    model_1_optim.fit(pair_1)
    inertia_pair_1.append(model_1_optim.inertia_)
plt.plot(range(1, 10), inertia_pair_1, marker='o')
plt.xlabel('k')
plt.ylabel('inertia')
df_inertia_1 = pd.DataFrame(inertia_pair_1, range(1, 10), columns=['inertia'])

inertia_pair_2 = []
for k in range(1, 10):
    model_2_optim = KMeans(n_clusters=k, random_state=42, init='k-means++')
    model_2_optim.fit(pair_2)
    inertia_pair_2.append(model_2_optim.inertia_)
plt.plot(range(1, 10), inertia_pair_2, marker='o')
plt.xlabel('k')
plt.ylabel('inertia')
df_inertia_2 = pd.DataFrame(inertia_pair_2, range(1, 10), columns=['inertia'])

inertia_pair_3 = []
for k in range(1, 10):
    model_3_optim = KMeans(n_clusters=k, random_state=42, init='k-means++')
    model_3_optim.fit(pair_3)
    inertia_pair_3.append(model_3_optim.inertia_)
plt.plot(range(1, 10), inertia_pair_3, marker='o')
plt.xlabel('k')
plt.ylabel('inertia')
df_inertia_3 = pd.DataFrame(inertia_pair_3, range(1, 10), columns=['inertia'])

#Viewing the data without scaling
df_for_viewing = df_initial.drop(['CustomerID', 'Gender'], axis=1).rename(columns={'Age':'age', 'Annual Income (k$)': 'annual_income', 'Spending Score (1-100)': 'spending_score'})
df_for_viewing['number_of_claster'] = pair_3['number_of_claster']

fig = plt.figure(figsize=(8,8))
ax = fig.gca(projection='3d')
x = df_for_viewing['age']
y = df_for_viewing['annual_income']
z = df_for_viewing['spending_score']
ax.set_xlabel('age')
ax.set_ylabel("annual")
ax.set_zlabel("spending")
scatter = ax.scatter(x, y, z, c=df_for_viewing['number_of_claster'], s=90)
plt.legend(*scatter.legend_elements())

#Descriptive statistics for clusters
df_descriptive_statistics = df_for_viewing.groupby(['number_of_claster'], as_index=False).agg({'age': 'mean', 'annual_income': 'mean', 'spending_score': 'mean'}).rename(columns={'age': 'age_mean', 'annual_income': 'annual_income_mean', 'spending_score': 'spending_score_mean'}).sort_values(['spending_score_mean', 'annual_income_mean'], ascending=False)
sns.scatterplot(x='annual_income', y='spending_score', data=df_for_viewing, hue=df_for_viewing['number_of_claster'], s=60, palette='Spectral')

groups_3_4 = df_distribution_of_clusters.query('number_of_claster == 3 or number_of_claster == 4')
sns.barplot(data=groups_3_4, x='number_of_claster', y='count', color='blue', alpha=0.5)

#Definition the most useful group of customers for mall
groups_3_4['annual_income_mean'] = df_descriptive_statistics['annual_income_mean']
groups_3_4['potential_profit'] = groups_3_4['count'] * groups_3_4['annual_income_mean'] * 0.4 * 1000
sns.barplot(data=groups_3_4, x='number_of_claster', y='potential_profit', color='green', alpha=0.7)


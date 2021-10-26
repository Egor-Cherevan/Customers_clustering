# Customers clustering
In this case i've tried to show the use of unsupervised learning in businesses for potential improving KPI's.
## Data
I've used the [Mall Customers dataset](https://github.com/Egor-Cherevan/Customers_clustering/blob/main/mall_customers.csv) as a train dataset. It consist of 5 features:
1.  CustomerID - It is the unique ID assigned to the customer
2.  Gender - Customers gender
3.  Age - Customers age(years)
4.  Annual Income(k$) - Customers annual income in k$
5.  Spending Score (1-100) - Customers score assigned by the mall based on the customer behavior
## Analysis
First of all I looked at the distribution of values. Here we can notice the apparent patterns between **Annual Income** and **Spending Score**. Thus we can assume with help of these values the data will be most clearly divided into groups. But I was also interested in making segmentation by **Age** and **Spending Score**.\
![](Pictures/Matrix.png)\
I was also interested in looking at the distribution of three quantities at once.
![](Pictures/Values_distribution.png)
## Scaling
I've scaled my data with help of **MinMaxScaler** to place all values in the range [0,1].\
<img src="Pictures/minmaxscaler.png" width="300">\
You can see the results of scaling below.\
<img src="Pictures/Scaled_age.png" width="250">
<img src="Pictures/Scaled_income.png" width="250">
<img src="Pictures/Scaled_score.png" width="250">
## Clustering
I've used a **KMeans** algorithm for segmenting the data.\
In order to find the optimal number of clusters I used a loop and did a segmentation for each value of the number of clusters. Based on the results of comparing the results it seems that the most optimal number of groups is 5. Reults of segmentations are presented below. \
<img src="Pictures/Different_n.png" width="1000">\
Also i've checked the inertia and silhouette score. And i can conclude that **5** is the most optimal number of clusters in my case.\
<img src="Pictures/inertia.png" width="750">
<img src="Pictures/Silhouette.png" width="750">

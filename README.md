# WDM-Ex.No-5-CUSTOMER-SEGMENTATION-ANALYSIS-USING-CLUSTERING-
**Aim:** 
To perform an unsupervised clustering of data on the customer's records from a groceries firm's database.  
**Description:** 
Customer segmentation is the practice of separating customers into groups that reflect similarities among customers in 
each cluster. I will divide customers into segments to optimize the significance of each customer to the business. To 
modify products according to distinct needs and behaviors of the customers. It also helps the business to cater to the 
concerns of different types of customers. <br>
**Program:** <br>
#Importing the Libraries 
```import numpy as np 
import pandas as pd 
import datetime 
import matplotlib 
import matplotlib.pyplot as plt 
from matplotlib import colors 
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA 
from yellowbrick.cluster import KElbowVisualizer 
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt, numpy as np 
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.cluster import AgglomerativeClustering 
from matplotlib.colors import ListedColormap 
from sklearn import metrics 
import warnings 
import sys 
if not sys.warnoptions: 

    warnings.simplefilter("ignore") 
np.random.seed(42)
```

**Loading the dataset**
data = pd.read_csv("../input/customer-personality-analysis/marketing_campaign.csv", sep="\t") 
print("Number of datapoints:", len(data)) 
data.head() 
**Information on features**  
data.info() 
**To remove the NA values** 
data = data.dropna() 
print("The total number of data-points after removing the rows with missing values are:", len(data)) 
data["Dt_Customer"] = pd.to_datetime(data["Dt_Customer"]) 
dates = [] 
for i in data["Dt_Customer"]: 
    i = i.date() 
    dates.append(i)   
**Dates of the newest and oldest recorded customer** 
print("The newest customer's enrolment date in therecords:",max(dates)) 
print("The oldest customer's enrolment date in the records:",min(dates)) 
**Created a feature "Customer_For"** 
days = [] 
d1 = max(dates) #taking it to be the newest customer 
for i in dates: 
    delta = d1 - i 
    days.append(delta) 
data["Customer_For"] = days 
data["Customer_For"] = pd.to_numeric(data["Customer_For"], errors="coerce") 
print("Total categories in the feature Marital_Status:\n", data["Marital_Status"].value_counts(), "\n") 
print("Total categories in the feature Education:\n", data["Education"].value_counts()) 
 

Performing the following steps to engineer some new features: 
 Extract the "Age" of a customer by the "Year_Birth" indicating the birth year of the respective person. 
 Create another feature "Spent" indicating the total amount spent by the customer in various categories over the 
span of two years. 
 Create another feature "Living_With" out of "Marital_Status" to extract the living situation of couples. 
 Create a feature "Children" to indicate total children in a household that is, kids and teenagers. 
 To get further clarity of household, creating feature indicating "Family_Size" 
 Create a feature "Is_Parent" to indicate parenthood status. 
 Lastly, create three categories in the "Education" by simplifying its value counts. 
 Dropping some of the redundant features 
 
#Feature Engineering 
#Age of customer today  
data["Age"] = 2021-data["Year_Birth"] 
#Total spendings on various items 
data["Spent"] = data["MntWines"]+ data["MntFruits"]+ data["MntMeatProducts"]+ data["MntFishProducts"]+ 
data["MntSweetProducts"]+ data["MntGoldProds"] 
#Deriving living situation by marital status"Alone" 
data["Living_With"]=data["Marital_Status"].replace({"Married":"Partner", "Together":"Partner", "Absurd":"Alone", 
"Widow":"Alone", "YOLO":"Alone", "Divorced":"Alone", "Single":"Alone",}) 
#Feature indicating total children living in the household 
data["Children"]=data["Kidhome"]+data["Teenhome"] 
#Feature for total members in the householde 
data["Family_Size"] = data["Living_With"].replace({"Alone": 1, "Partner":2})+ data["Children"] 
#Feature pertaining parenthood 
data["Is_Parent"] = np.where(data.Children> 0, 1, 0) 
#Segmenting education levels in three groups 
data["Education"]=data["Education"].replace({"Basic":"Undergraduate","2n Cycle":"Undergraduate", 
"Graduation":"Graduate", "Master":"Postgraduate", "PhD":"Postgraduate"}) 
#For clarity 
data=data.rename(columns={"MntWines":"Wines","MntFruits":"Fruits","MntMeatProducts":"Meat","MntFishProduc
ts":"Fish","MntSweetProducts":"Sweets","MntGoldProds":"Gold"}) 
#Dropping some of the redundant features 
to_drop = ["Marital_Status", "Dt_Customer", "Z_CostContact", "Z_Revenue", "Year_Birth", "ID"] 
data = data.drop(to_drop, axis=1) 
data.describe() 

DATA PREPROCESSING 
The following steps are applied to preprocess the data: 
 Label encoding the categorical features. 
 Scaling the features using the standard scaler 
 Creating a subset data frame for dimensionality reduction 
#Get list of categorical variables 
s = (data.dtypes == 'object') 
object_cols = list(s[s].index) 
print("Categorical variables in the dataset:", object_cols) 
#Label Encoding the object dtypes. 
LE=LabelEncoder() 
for i in object_cols: 
    data[i]=data[[i]].apply(LE.fit_transform) 
print("All features are now numerical") 
#Creating a copy of data 
ds = data.copy() 
# creating a subset of dataframe by dropping the features on deals accepted and promotions 
cols_del = ['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1','AcceptedCmp2', 'Complain', 
'Response'] 
ds = ds.drop(cols_del, axis=1) 
#Scaling 
scaler = StandardScaler() 
scaler.fit(ds) 
scaled_ds = pd.DataFrame(scaler.transform(ds),columns= ds.columns ) 
print("All features are now scaled") 
#Scaled data to be used for reducing the dimensionality 
print("Dataframe to be used for further modelling:") 
scaled_ds.head() 
 
 
 
 

DIMENTIONALITY REDUCTION 
Dimensionality reduction is the process of reducing the number of random variables under consideration, by obtaining 
a set of principal variables. 
Principal component analysis (PCA) is a technique for reducing the dimensionality of such datasets, increasing 
interpretability but at the same time minimizing information loss. 
Steps in this section: 
 Dimensionality reduction with PCA 
 Plotting the reduced dataframe 
#Initiating PCA to reduce dimentions aka features to 3 
pca = PCA(n_components=3) 
pca.fit(scaled_ds) 
PCA_ds = pd.DataFrame(pca.transform(scaled_ds), columns=(["col1","col2", "col3"])) 
PCA_ds.describe().T 
#A 3D Projection Of Data In The Reduced Dimension 
x =PCA_ds["col1"] 
y =PCA_ds["col2"] 
z =PCA_ds["col3"] 
#To plot 
fig = plt.figure(figsize=(10,8)) 
ax = fig.add_subplot(111, projection="3d") 
ax.scatter(x,y,z, c="maroon", marker="o" ) 
ax.set_title("A 3D Projection Of Data In The Reduced Dimension") 
plt.show() 
 
 
 
 
 
 
 
 
 
 

CLUSTERING 
Agglomerative clustering is a hierarchical clustering method. It involves merging examples until the desired number 
of clusters is achieved. 
Steps involved in the Clustering: 
 Elbow Method to determine the number of clusters to be formed. 
 Clustering via Agglomerative Clustering 
 Examining the clusters formed via scatter plot 
# Quick examination of elbow method to find numbers of clusters to make. 
print('Elbow Method to determine the number of clusters to be formed:') 
Elbow_M = KElbowVisualizer(KMeans(), k=10) 
Elbow_M.fit(PCA_ds) 
Elbow_M.show() 
#Initiating the Agglomerative Clustering model  
AC = AgglomerativeClustering(n_clusters=4) 
# fit model and predict clusters 
yhat_AC = AC.fit_predict(PCA_ds) 
PCA_ds["Clusters"] = yhat_AC 
#Adding the Clusters feature to the orignal dataframe. 
data["Clusters"]= yhat_AC 
#Plotting the clusters 
fig = plt.figure(figsize=(10,8)) 
ax = plt.subplot(111, projection='3d', label="bla") 
ax.scatter(x, y, z, s=40, c=PCA_ds["Clusters"], marker='o', cmap = cmap ) 
ax.set_title("The Plot Of The Clusters") 
plt.show() 
 
 
 
 
 
 
 

EVALUATING MODELS 
#Plotting countplot of clusters 
pal = ["#682F2F","#B9C0C9", "#9F8A78","#F3AB60"] 
pl = sns.countplot(x=data["Clusters"], palette= pal) 
pl.set_title("Distribution Of The Clusters") 
plt.show() 
pl = sns.scatterplot(data = data,x=data["Spent"], y=data["Income"],hue=data["Clusters"], palette= pal) 
pl.set_title("Cluster's Profile Based On Income And Spending") 
plt.legend() 
plt.show() 
Income vs spending plot shows the clusters pattern 
 group 0: high spending & average income 
 group 1: high spending & high income 
 group 2: low spending & low income 
 group 3: high spending & low income 
plt.figure() 
pl=sns.swarmplot(x=data["Clusters"], y=data["Spent"], color= "#CBEDDD", alpha=0.5 ) 
pl=sns.boxenplot(x=data["Clusters"], y=data["Spent"], palette=pal) 
plt.show() 
PROFILING 
Personal = [ "Kidhome","Teenhome","Customer_For", "Age", "Children", "Family_Size", "Is_Parent", 
"Education","Living_With"] 
for i in Personal: 
    plt.figure() 
    sns.jointplot(x=data[i], y=data["Spent"], hue =data["Clusters"], kind="kde", palette=pal) 
    plt.show() 
 
 
 ```
 
 
 
 

Output: 
 
 
 <img width="932" height="448" alt="image" src="https://github.com/user-attachments/assets/851491fc-9478-4ef2-b7a2-e7c03d6fc3a0" />

<img width="932" height="463" alt="image" src="https://github.com/user-attachments/assets/094ad7a1-275a-4427-9562-51d2ab0e5fd1" />



 
 
 
 
 
 
 
 
 
 
Result: 
Thus, dimensionality reduction and agglomerative clustering have been performed for profiling customers in 
clusters according to their family structures and income/spending. 

# WDM – Ex. No. 5: Customer Segmentation Analysis Using Clustering

## Aim

To perform unsupervised clustering of customer records from a grocery firm's database.

## Description

Customer segmentation is the process of dividing customers into groups based on similarities in their characteristics and behavior. It helps businesses optimize customer value, customize products according to customer needs, and address the concerns of different customer groups.

## Program

### 1. Importing the Libraries

```python
import numpy as np
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

### 2. Loading the Dataset

```python
data = pd.read_csv(
    "../input/customer-personality-analysis/marketing_campaign.csv",
    sep="\t"
)

print("Number of datapoints:", len(data))
data.head()
```

### 3. Information on Features

```python
data.info()
```

### 4. Removing Missing Values

```python
data = data.dropna()

print(
    "The total number of data-points after removing the rows with missing values are:",
    len(data)
)

data["Dt_Customer"] = pd.to_datetime(data["Dt_Customer"])

dates = []

for i in data["Dt_Customer"]:
    dates.append(i.date())
```

### 5. Dates of the Newest and Oldest Recorded Customers

```python
print(
    "The newest customer's enrolment date in the records:",
    max(dates)
)

print(
    "The oldest customer's enrolment date in the records:",
    min(dates)
)
```

### 6. Creating the Customer_For Feature

```python
days = []
d1 = max(dates)

for i in dates:
    delta = d1 - i
    days.append(delta)

data["Customer_For"] = days

data["Customer_For"] = pd.to_numeric(
    data["Customer_For"],
    errors="coerce"
)

print(
    "Total categories in the feature Marital_Status:\n",
    data["Marital_Status"].value_counts(),
    "\n"
)

print(
    "Total categories in the feature Education:\n",
    data["Education"].value_counts()
)
```

### 7. Feature Engineering

Feature engineering creates new features to better understand customer behavior.

#### Age of Customer

```python
data["Age"] = 2021 - data["Year_Birth"]
```

#### Total Spending on Various Items

```python
data["Spent"] = (
    data["MntWines"]
    + data["MntFruits"]
    + data["MntMeatProducts"]
    + data["MntFishProducts"]
    + data["MntSweetProducts"]
    + data["MntGoldProds"]
)
```

#### Living Situation

```python
data["Living_With"] = data["Marital_Status"].replace({
    "Married": "Partner",
    "Together": "Partner",
    "Absurd": "Alone",
    "Widow": "Alone",
    "YOLO": "Alone",
    "Divorced": "Alone",
    "Single": "Alone"
})
```

#### Total Children in the Household

```python
data["Children"] = data["Kidhome"] + data["Teenhome"]
```

#### Family Size

```python
data["Family_Size"] = (
    data["Living_With"].replace({
        "Alone": 1,
        "Partner": 2
    })
    + data["Children"]
)
```

#### Parenthood Status

```python
data["Is_Parent"] = np.where(data["Children"] > 0, 1, 0)
```

#### Simplifying Education Levels

```python
data["Education"] = data["Education"].replace({
    "Basic": "Undergraduate",
    "2n Cycle": "Undergraduate",
    "Graduation": "Graduate",
    "Master": "Postgraduate",
    "PhD": "Postgraduate"
})
```

#### Renaming Columns

```python
data = data.rename(columns={
    "MntWines": "Wines",
    "MntFruits": "Fruits",
    "MntMeatProducts": "Meat",
    "MntFishProducts": "Fish",
    "MntSweetProducts": "Sweets",
    "MntGoldProds": "Gold"
})
```

#### Dropping Redundant Features

```python
to_drop = [
    "Marital_Status",
    "Dt_Customer",
    "Z_CostContact",
    "Z_Revenue",
    "Year_Birth",
    "ID"
]

data = data.drop(to_drop, axis=1)

data.describe()
```

### 8. Data Preprocessing

The following steps are performed:

- Label encoding categorical features.
- Scaling features using StandardScaler.
- Creating a subset for dimensionality reduction.

#### Identifying Categorical Variables

```python
s = data.dtypes == "object"
object_cols = list(s[s].index)

print("Categorical variables in the dataset:", object_cols)
```

#### Label Encoding

```python
LE = LabelEncoder()

for i in object_cols:
    data[i] = LE.fit_transform(data[i])

print("All features are now numerical")
```

#### Creating a Copy of the Dataset

```python
ds = data.copy()

cols_del = [
    "AcceptedCmp3",
    "AcceptedCmp4",
    "AcceptedCmp5",
    "AcceptedCmp1",
    "AcceptedCmp2",
    "Complain",
    "Response"
]

ds = ds.drop(cols_del, axis=1)
```

#### Feature Scaling

```python
scaler = StandardScaler()

scaler.fit(ds)

scaled_ds = pd.DataFrame(
    scaler.transform(ds),
    columns=ds.columns
)

print("All features are now scaled")
print("Dataframe to be used for further modelling:")

scaled_ds.head()
```

### 9. Dimensionality Reduction Using PCA

Principal Component Analysis (PCA) reduces the number of features while retaining as much useful information as possible.

#### Applying PCA

```python
pca = PCA(n_components=3)

pca.fit(scaled_ds)

PCA_ds = pd.DataFrame(
    pca.transform(scaled_ds),
    columns=["col1", "col2", "col3"]
)

PCA_ds.describe().T
```

#### 3D Projection of Reduced Dimensions

```python
x = PCA_ds["col1"]
y = PCA_ds["col2"]
z = PCA_ds["col3"]

fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(111, projection="3d")

ax.scatter(x, y, z, c="maroon", marker="o")

ax.set_title("A 3D Projection Of Data In The Reduced Dimension")

plt.show()
```

### 10. Clustering

Agglomerative clustering is a hierarchical clustering technique that groups similar data points by progressively merging clusters.

#### Determining the Number of Clusters Using the Elbow Method

```python
print("Elbow Method to determine the number of clusters to be formed:")

Elbow_M = KElbowVisualizer(KMeans(), k=10)

Elbow_M.fit(PCA_ds)

Elbow_M.show()
```

#### Applying Agglomerative Clustering

```python
AC = AgglomerativeClustering(n_clusters=4)

yhat_AC = AC.fit_predict(PCA_ds)

PCA_ds["Clusters"] = yhat_AC

data["Clusters"] = yhat_AC
```

#### Plotting the Clusters

```python
fig = plt.figure(figsize=(10, 8))

ax = plt.subplot(111, projection="3d", label="bla")

ax.scatter(
    x, y, z,
    s=40,
    c=PCA_ds["Clusters"],
    marker="o",
    cmap="viridis"
)

ax.set_title("The Plot Of The Clusters")

plt.show()
```

### 11. Evaluating the Clusters

#### Distribution of Clusters

```python
pal = ["#682F2F", "#B9C0C9", "#9F8A78", "#F3AB60"]

pl = sns.countplot(x=data["Clusters"], palette=pal)

pl.set_title("Distribution Of The Clusters")

plt.show()
```

#### Income vs. Spending

```python
pl = sns.scatterplot(
    data=data,
    x="Spent",
    y="Income",
    hue="Clusters",
    palette=pal
)

pl.set_title("Cluster's Profile Based On Income And Spending")

plt.legend()

plt.show()
```

The income-versus-spending plot helps identify customer groups with different income and spending patterns.

#### Spending Distribution by Cluster

```python
plt.figure()

pl = sns.swarmplot(
    x=data["Clusters"],
    y=data["Spent"],
    color="#CBEDDD",
    alpha=0.5
)

pl = sns.boxenplot(
    x=data["Clusters"],
    y=data["Spent"],
    palette=pal
)

plt.show()
```

### 12. Customer Profiling

The following features are used to analyze customer profiles:

```python
Personal = [
    "Kidhome",
    "Teenhome",
    "Customer_For",
    "Age",
    "Children",
    "Family_Size",
    "Is_Parent",
    "Education",
    "Living_With"
]

for i in Personal:
    plt.figure()

    sns.jointplot(
        x=data[i],
        y=data["Spent"],
        hue=data["Clusters"],
        kind="kde",
        palette=pal
    )

    plt.show()
```

## Output

The output includes the following visualizations:

1. A 3D projection of the reduced dataset.
2. Distribution of customer clusters.
3. Customer segmentation based on income and spending.
4. Spending distribution across clusters.
5. Customer profiles based on demographic and household features.

### Output Screenshots

Add the generated output screenshots below.

![Customer Segmentation Output 1](https://github.com/user-attachments/assets/851491fc-9478-4ef2-b7a2-e7c03d6fc3a0)

![Customer Segmentation Output 2](https://github.com/user-attachments/assets/094ad7a1-275a-4427-9562-51d2ab0e5fd1)

## Result

Thus, dimensionality reduction using PCA and agglomerative clustering was performed to segment customers according to their family structure, income, spending behavior, and demographic characteristics.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')



df = pd.read_csv("/kaggle/input/social-media-sentiments-analysis-dataset/sentimentdataset.csv")

df

df.head()

df.tail()

df.shape

df.columns

df.info()

df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Day'] = df['Timestamp'].dt.day
df['Month'] = df['Timestamp'].dt.month
df['Year'] = df['Timestamp'].dt.year

df['Text'] = df['Text'].str.strip()
df['Sentiment'] = df['Sentiment'].str.strip()
df['User'] = df['User'].str.strip()
df['Platform'] = df['Platform'].str.strip()
df['Hashtags'] = df['Hashtags'].str.strip()
df['Country'] = df['Country'].str.strip()

df["Retweets"] = df["Retweets"].astype(int)
df["Likes"] = df["Likes"].astype(int)

df.drop(columns = ["Unnamed: 0.1","Unnamed: 0"],axis = 1, inplace = True)

df.info()

df.isnull().sum().sum()

df.duplicated().value_counts()

df.describe()

k = df.describe(include="object")
k

# for i in k.columns:
#     print(i)
#     print(df[i].unique())
#     print("*"*50)

df

"""# **Analysis**"""

df.columns

sten = df["Sentiment"].value_counts().head(10).reset_index()
sen = pd.DataFrame(sten)
sen

"""# **Platform wise Distribution**"""

df['Platform'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Percentages of Platforms')
plt.legend()
plt.show()

"""**Whole Sentiments**"""

plt.figure(figsize=(10,8))
s = plt.pie(sen["count"], labels = sen["Sentiment"],autopct='%1.1f%%')
plt.show()

"""**Whole Hashtags**"""

k = df["Hashtags"].value_counts().head(10).reset_index()
Hash = pd.DataFrame(k)
Hash

plt.figure(figsize = (10,8))
k = sns.barplot(y = "Hashtags", x = "count", data = Hash, palette = "gist_ncar")
for bars in k.containers:
    k.bar_label(bars)
plt.show()

"""**top 10 hashtag retweeted**"""

k = df.groupby("Hashtags")["Retweets"].max().nlargest(10).sort_values(ascending = False).reset_index()
b = pd.DataFrame(k)
b

plt.figure(figsize = (8,6))
sns.barplot(x = "Retweets", y = "Hashtags", data = b)
plt.show()

"""Top 10 liked Hashtags"""

k = df.groupby("Hashtags")["Likes"].max().nlargest(10).reset_index()
b = pd.DataFrame(k)
b

plt.figure(figsize = (8,6))
sns.barplot(x = "Likes", y = "Hashtags", data = b)
plt.show()

"""**Whole likes year wise**"""

k = df.groupby('Year')['Likes'].sum().reset_index()
ly = pd.DataFrame(k)
ly

plt.figure(figsize = (8,6))
sns.lineplot(x = "Year", y = "Likes", data = ly, marker = "o")
plt.show()

"""**likes wise top 10 Sentiments**"""

k = df.groupby("Sentiment")["Likes"].sum().nlargest(10).reset_index()
k = pd.DataFrame(k)
k

plt.figure(figsize=(10,10))
sns.barplot(x = "Sentiment",y = "Likes",data= k ,palette="gist_ncar",)
plt.show()

"""**Likes by the Hours**"""

plt.figure(figsize=(8,6))
sns.scatterplot(data = df, x = 'Hour', y = 'Likes',hue = "Platform")
plt.show()

"""**Country wise Distribution of user**"""

k = df["Country"].value_counts().nlargest(10).reset_index()
b = pd.DataFrame(k)
b

plt.figure(figsize=(12,8))
s = plt.pie(b["count"], labels = b["Country"],autopct='%1.1f%%')
plt.show()

sns.histplot(df.Retweets,kde = True)
plt.show()

sns.displot(df.Likes,kde = True)
plt.show()

"""**Top 10 retweeted text**"""

k = df.groupby("Text")["Retweets"].sum().nlargest(10)
tophash = pd.DataFrame(k)
tophash

"""**Top 10 liked text**"""

k = df.groupby("Text")["Likes"].sum().nlargest(10)
toplike = pd.DataFrame(k)
toplike

"""# **Twitter**

**Sentiment**
"""

k = df[df["Platform"] == "Twitter"]
sentw = k["Sentiment"].value_counts().head(10).reset_index()
sentwitter = pd.DataFrame(sentw)
sentwitter

plt.figure(figsize = (8,6))
a = sns.barplot(x = "Sentiment", y = "count", data = sentwitter, palette = "gist_ncar")
for bars in a.containers:
    a.bar_label(bars)

"""**Top 10 Hashtags Twitter wise**"""

k = df[df["Platform"] == "Twitter"]
b = k["Hashtags"].value_counts().head(10).reset_index()
hashtw = pd.DataFrame(b)
hashtw

plt.figure(figsize = (10,8))
sns.barplot(y= "Hashtags", x ="count", data = hashtw, palette = "gist_ncar")
plt.show()

"""**top 10 hashtag retwitted in Twitter**"""

k = df[df["Platform"] == "Twitter"]
b = k.groupby("Hashtags")["Retweets"].max().nlargest(10).sort_values(ascending = False).reset_index()
hashret = pd.DataFrame(b)
hashret

plt.figure(figsize=(8,6))
sns.barplot(x = "Retweets", y = "Hashtags", data = hashret)
plt.show()

"""**top 10 User Liked in Twitter**"""

k = df[df["Platform"] == "Twitter"]
b = k.groupby("User")["Likes"].max().nlargest(10).sort_values(ascending = False).reset_index()
hlt = pd.DataFrame(b)
hlt

plt.figure(figsize = (8,6))
sns.barplot(x = "Likes", y = "User", data = hlt)
plt.show()

"""**likes years wise in Twitter**"""

k = df[df["Platform"] == "Twitter"]
b = k.groupby("Year")["Likes"].sum().reset_index()
lyt = pd.DataFrame(b)
lyt

plt.figure(figsize = (8,6))
sns.lineplot(x = "Year", y = "Likes", data = lyt,marker='*')
plt.show()

"""# **Instagram**

**Sentiment**
"""

k = df[df["Platform"] == "Instagram"]
seni = k["Sentiment"].value_counts().head(10).reset_index()
seninsta = pd.DataFrame(seni)
seninsta

plt.figure(figsize = (10,8))
k = sns.barplot(x = "Sentiment", y = "count", data = seninsta, palette = "gist_ncar")
for bars in k.containers:
    k.bar_label(bars)
plt.show()

k = df[df["Platform"] == "Instagram"]
b = k["Hashtags"].value_counts().head(10).reset_index()
hashin = pd.DataFrame(b)
hashin

plt.figure(figsize = (10,8))
sns.barplot(y = "Hashtags", x = "count", data = hashin, palette = "gist_ncar")
plt.show()

k = df[df["Platform"] == "Instagram"]
b = k.groupby("Hashtags")["Retweets"].max().nlargest(10).sort_values(ascending = False).reset_index()
hashin = pd.DataFrame(b)
hashin

plt.figure(figsize=(8,6))
sns.barplot(x = "Retweets", y = "Hashtags", data = hashin)
plt.show()

"""**top 10 User Liked in Instagram**"""

k = df[df["Platform"] == "Instagram"]
b = k.groupby("User")["Likes"].max(10).nlargest(10).sort_values(ascending = False).reset_index()
hli = pd.DataFrame(b)
hli

plt.figure(figsize = (8,6))
sns.barplot(x = "Likes", y = "User",data = hli)
plt.show()

"""**likes years wise in Instagram**"""

k = df[df["Platform"] == "Instagram"]
b = k.groupby("Year")["Likes"].sum().reset_index()
lyi = pd.DataFrame(b)
lyi

plt.figure(figsize = (8,6))
sns.lineplot(x = "Year", y = "Likes", data = lyi,marker='*')
plt.show()

"""# **Facebook**

**Sentiment**
"""

k = df[df["Platform"] == "Facebook"]
seni = k["Sentiment"].value_counts().head(10).reset_index()
senface = pd.DataFrame(seni)
senface

plt.figure(figsize = (10,8))
k = sns.barplot(x = "Sentiment", y = "count", data = senface, palette = "gist_ncar")
for bars in k.containers:
    k.bar_label(bars)
plt.show()

"""**Hashtags**"""

k = df[df["Platform"]=="Facebook"]
b = k["Hashtags"].value_counts().head(10).reset_index()
hashfa = pd.DataFrame(b)
hashfa

plt.figure(figsize = (10,8))
sns.barplot(y = "Hashtags", x = "count", data = hashfa, palette = "gist_ncar")
plt.show()



"""**Top 5 Hashtags retwitted**"""

k = df[df["Platform"] == "Facebook"]
b = k.groupby("Hashtags")["Retweets"].max().nlargest(10).sort_values(ascending = False).reset_index()
hashfa = pd.DataFrame(b)
hashfa

plt.figure(figsize=(8,6))
sns.barplot(x = "Retweets", y = "Hashtags", data = hashfa)
plt.show()

"""**top 10 hashtag Liked in Facebook**"""

k = df[df["Platform"] == "Facebook"]
b = k.groupby("User")["Likes"].max(10).nlargest(10).sort_values(ascending = False).reset_index()
hlf = pd.DataFrame(b)
hlf

plt.figure(figsize = (8,6))
sns.barplot(x = "Likes", y = "User",data = hlf)
plt.show()

"""**likes years wise in Facebook**"""

k = df[df["Platform"] == "Facebook"]
b = k.groupby("Year")["Likes"].sum().reset_index()
lyf = pd.DataFrame(b)
lyf

plt.figure(figsize = (8,6))
sns.lineplot(x = "Year", y = "Likes", data = lyf,marker='*')
plt.show()

"""# **Thank you 😍**"""

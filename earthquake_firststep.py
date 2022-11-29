#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Kütüphaneler yüklenir.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import folium

df = pd.read_csv("C:\\Users\\evata\\datascience\\earthquake.csv")
df.head()
#Süreye Bağlı Büyüklük (Md)+++

#Yerel (Lokal) Büyüklük (Ml)

#Yüzey Dalgası Büyüklüğü (Ms)+++

#Cisim Dalgası Büyüklüğü (Mb)+++

#Moment Büyüklüğü (Mw)

#deprem şiddeti (xm) 0.0 değerler için hesaplanmadı demek


# In[2]:


df["country"].unique()


# In[3]:


#Türkiye bağlı enlem boylam sınırlandırıldı
df = df[df["country"] == "turkey"]

df=df[df["lat"]>=36]
df=df[df["lat"]<=42]
df=df[df["long"]<=45]
df=df[df["long"]>=26]


# In[4]:


#toplam bos değer
df.isna().sum()


# In[5]:


df.shape


# In[6]:


df.describe()


# In[7]:


#missingno kayıp verileri bulmak için kurulur
import missingno as mno
mno.matrix(df, figsize = (20, 6))


# In[8]:


#mw sutununda cok fazla bos satır bulunduğu için sutun silinir.
df = df.drop(columns= "mw")


# In[9]:


df.tail()


# In[10]:


#son 1697 değer bos olduğu için dikkate alınmadı
df=df[:9949]
mno.matrix(df, figsize = (20, 6))


# In[11]:


df.isnull().sum()


# In[12]:


#area sutununda bos kısımlar için bilinmiyor eklendi.
df = df.fillna("bilinmiyor")


# In[13]:


data = df.copy()


# In[14]:


data['time'] = pd.to_datetime(data['time'])
data['time'] = pd.to_numeric(pd.to_datetime(data['time']))


# In[15]:


# date sütunu gün yıl ay a ayrılır
data[["year", "month", "day"]] = data["date"].str.split(".", expand = True)


# In[16]:


data["year"]= data["year"].astype(int)
data["month"]= data["month"].astype(int)
data["day"]= data["day"].astype(int)


# In[17]:


# şiddetine göre haritalanması 
data.plot(x="long", y="lat", kind="scatter", c="xm", colormap="hot",figsize=(25, 10))


# In[18]:


data.info()


# In[19]:


import geopandas
from folium import GeoJson
from folium import plugins


# In[20]:


lat = np.asarray(data.lat)
lon = np.asarray(data.long)
mag = np.asarray(data.xm)
year = np.asarray(data.year)

mean_lat = data.lat.mean()
mean_lon = data.long.mean()


# In[21]:


location = [lat, lon]


# In[22]:


m= folium.Map(location=[mean_lat, mean_lon], zoom_start=7)
m


# In[23]:


data["lat"]= data["lat"].astype(float)
data["long"]= data["long"].astype(float)
data["xm"]= data["xm"].astype(float)
data["year"]= data["year"].astype(int)


# In[24]:


for idx in range(len(lat)):
    marker = folium.CircleMarker(
        location=[lat[idx],lon[idx]],
        radius=mag[idx]**5/2500,
        color='Red',
        popup=f'lat: {lat[idx]}, long: {lon[idx]} , year: {year[idx]}',
        tooltip=f'Magnitude: {mag[idx]}')
    marker.add_to(m)
# deprem şiddetine göre çap büyüyen haritalandırma


# In[25]:


import fiona
import pandas as pd
import geopandas


# In[27]:


data_map_geo = geopandas.GeoDataFrame(
    data, geometry=geopandas.points_from_xy(data.long, data.lat))


# In[28]:


data_map_geo


# In[29]:


# türkiye sınırları için geojson dosyası 
countries = geopandas.read_file("C:\\Users\\evata\datascience\\Turkey_population_with_geopandas-main\\Turkey_population_with_geopandas-main\\countries.geojson")


# In[30]:


countries


# In[31]:


countries.plot()


# In[32]:


turkey = countries[countries["ADMIN"] == "Turkey"]
turkey.plot()


# In[45]:


turkey


# In[33]:


# türkiye sınırları için shp dosyası 
turkey_gf = geopandas.read_file("C:\\Users\\evata\datascience\\Turkey_population_with_geopandas-main\\Turkey_population_with_geopandas-main\\TUR_adm\\TUR_adm1.shp")


# In[34]:


turkey_geo = turkey_gf.copy()


# In[35]:


turkey_geo = turkey_geo[["NAME_1","geometry"]]


# In[36]:


turkey_geo.rename(columns={"NAME_1":"provience"}, inplace=True)


# In[37]:


turkey_geo.plot(figsize=(15,15),edgecolor="b", facecolor="none")


# In[38]:


turkey_geo.crs #crs EPSG:4326> tipinde


# In[39]:


data_map_geo = gdf.to_crs("EPSG:4326")
# data_map_geo nun da crs EPSG:4326 olarak ayarlanır


# In[40]:


data_map_geo.crs


# In[41]:


data_map_geo


# In[42]:


from shapely.geometry import Polygon, LineString, Point


# In[43]:


data = data.loc[~np.isnan(data["lat"])]
data_merge_turkey = gpd.GeoDataFrame(
    data, geometry=gpd.points_from_xy(data.long, data.lat))


# In[44]:


fig, ax = plt.subplots(figsize=(40, 10))
turkey_geo.plot(ax=ax, alpha=0.2, color="grey")
data_merge_turkey.plot(column="xm", ax=ax, legend=True)
plt.title("Earthquake")


# In[ ]:





# In[ ]:





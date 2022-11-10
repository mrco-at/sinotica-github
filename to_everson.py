#!/usr/bin/env python
# coding: utf-8

# In[1]:


#%%
# chamando as bibliotecas
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from metpy.plots import add_metpy_logo, USCOUNTIES
##matplotlib inline
import metpy.calc as mpcalc
from metpy.units import units
import numpy as np
import xarray as xr
import pandas as pd
import cartopy.mpl.ticker as cticker
from cartopy.util import add_cyclic_point


# In[2]:


#%%
# abrindo os dados do netcdf 24062021.nc chamando de ds. Contem dados de temperatura, 
#v wind e u wind para hemisferio sul no dia 24 de 06 de 2021 (09z). 
ds = xr.open_dataset('24062021.nc')#.metpy.parse_cf()
#%% visualizando informacoes do ds
print("INFORMACOES DO DS:")
ds


# In[4]:


#%%
#selecionda um dia para tornas dataset em 2D apenas
dssel=ds.sel(time='2021-06-24T09:00:00')
#%%
#visualizando dataset selecionado
dssel


# In[5]:


#%%
# separando as variaveis
dsu=ds['u']
dsv=ds['v']
dst=ds['t']


# In[6]:


#%%
# Compute the Q-vector components
qvector = mpcalc.q_vector(dsu*units('m/s'),dsv*units('m/s'),dst*units('K'),850*units('hPa'),11.1*units.km,11.1*units.km)
#%%
qvector
#%%
q_vector_u=qvector[0]
q_vector_v=qvector[1]
#%%
q_vector_u
#%%
q_vector_v


# In[7]:


#%% obtendo o crs
dat=ds.metpy.parse_cf('t')
#%% obtendo a projecao 
proj=dat.metpy.cartopy_crs


# In[8]:


#%% selecionando um unico dia para tornar array do vetor q em 2D
q_vector_u_sel=q_vector_u.sel(time='2021-06-24T09:00:00')
q_vector_u_sel


# In[9]:


#%% selecionando um unico dia para tornar array em 2D
q_vector_v_sel=q_vector_v.sel(time='2021-06-24T09:00:00')
q_vector_v_sel


# In[10]:


#%% mapa do campo de temperatura (teste)
plt.figure(figsize=(12,9))
ax = plt.axes(projection=proj)
ax.coastlines()
ax.set_extent([-90,-30,0,-90])            
cf=ax.contourf(dssel['longitude'],dssel['latitude'],dssel['t'],cmap='coolwarm')
# Define the xticks for longitude
ax.set_xticks(np.arange(-90,-30,10), crs=proj)
lon_formatter = cticker.LongitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)

# Define the yticks for latitude
ax.set_yticks(np.arange(-90,0,10), crs=proj)
lat_formatter = cticker.LatitudeFormatter()
ax.yaxis.set_major_formatter(lat_formatter)
cb=plt.colorbar(cf)#, shrink=0.5)
cb.ax.set_title('K')#°C
#cb.ax.set_title('mmol m$^{-3}$')
ax.set_title('Temperatura em 2021-06-24T09:00:00')
plt.show() 


# In[11]:


#%% contourplot das componentes do vetor q
plt.contour(q_vector_v_sel)
plt.colorbar()
plt.show()  
plt.contour(q_vector_u_sel)
plt.colorbar()
plt.show() 


# In[12]:


#%% vector q plot america do sul
plt.figure(figsize=(12,9))
ax = plt.axes(projection=proj)
ax.coastlines()
ax.set_extent([-90,-30,0,-90]) 
cf=ax.quiver(dssel['longitude'],dssel['latitude'],q_vector_u_sel,q_vector_v_sel,scale=40e-10)           
#cf=ax.contourf(dssel['longitude'],dssel['latitude'],dssel['t'],cmap='coolwarm')

# Define the xticks for longitude
ax.set_xticks(np.arange(-90,-30,10), crs=proj)
lon_formatter = cticker.LongitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)

# Define the yticks for latitude
ax.set_yticks(np.arange(-90,0,10), crs=proj)
lat_formatter = cticker.LatitudeFormatter()
ax.yaxis.set_major_formatter(lat_formatter)

plt.show() 


# In[13]:


#%% vector q plot RS
plt.figure(figsize=(12,9))
ax = plt.axes(projection=proj)
ax.coastlines()
ax.set_extent([-70,-50,-20,-40]) 
cf=ax.quiver(dssel['longitude'],dssel['latitude'],q_vector_u_sel,q_vector_v_sel,scale=40e-10)           
#cf=ax.contourf(dssel['longitude'],dssel['latitude'],dssel['t'],cmap='coolwarm')

# Define the xticks for longitude
ax.set_xticks(np.arange(-70,-50,5), crs=proj)
lon_formatter = cticker.LongitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)

# Define the yticks for latitude
ax.set_yticks(np.arange(-40,-20,5), crs=proj)
lat_formatter = cticker.LatitudeFormatter()
ax.yaxis.set_major_formatter(lat_formatter)

plt.show()  


# In[ ]:





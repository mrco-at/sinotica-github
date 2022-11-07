#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 13:46:10 2022

@author: tet
"""

#%%
# chamando as bibliotecas
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from metpy.plots import add_metpy_logo, USCOUNTIES
%matplotlib inline
import metpy.calc as mpcalc
from metpy.units import units
import numpy as np
import xarray as xr
import pandas as pd
#%%
# abrindo os dados do netcdf 24062021.nc chamando de ds. Contem dados de temperatura, v wind e u wind para hemisferio sul no dia 24 de 11 de 2021 (09z). 
ds = xr.open_dataset('24062021.nc')#.metpy.parse_cf()
ds
#%%
# abrindo os dados do netcdf 30062020.nc chamando de ds1. Contem dados de temperatura, u wind e v wind para o dia 30 de junho de 2020 (12z)
ds1 = xr.open_dataset('30062020.nc')#.metpy.parse_cf()
ds1
#%%
# abrindo os dados do netcdf 2022.nc chamando de ds2. Contem dados de temperatura, u wind e v wind para os dias 01/05, 03/05, 15/05, 01/07, 03/07, 15/07 as 00, 03  e 09 z.
ds2 = xr.open_dataset('2022.nc')#.metpy.parse_cf()
ds2
#%%
# separando as variaveis
dsu=ds['u']
dsv=ds['v']
#dsz=ds['z']
dst=ds['t']
#%%
#visualizando temperatura
dst
#%%
#transformando em um nparray
dst2=np.array(dst)
#%%
dst2.shape
#%%
dst3=np.squeeze(dst2).shape
#%%
dst3=dst2[0,:,:]
print(dst3.shape)
#%%
np.info(dst2)
#%%
# Compute grid spacings for data
dx, dy = mpcalc.lat_lon_grid_deltas(ds['longitude'], ds['latitude'])



# Compute the divergence of the Q-vectors calculated above
#q_div = -2*mpcalc.divergence(uqvect, vqvect, dx, dy, dim_order='yx')
#%%
# Compute the Q-vector components
qvector = mpcalc.q_vector(dsu*units('m/s'),dsv*units('m/s'),dst*units('kelvin'),850*units('hPa'),11.1*units.km,11.1*units.km)
#%%
qvector
#%%
m='meter ** 2 / kilogram / second'
#%%
q_vector_u=qvector[0]
q_vector_v=qvector[1]
#%%
#rs=ccrs.LambertConformal(central_longitude=-53,central_latitude=45)
crs=ccrs.LambertCylindrical()
data_crs=ccrs.LambertCylindrical()
#%%
lons, lats = np.meshgrid(ds['longitude'], ds['latitude'])
#%%
lons
#%%
plt.figure(figsize=(12,9))
ax = plt.axes(projection=ccrs.LambertCylindrical())
ax.set_extent([-90,-30,10,-89])
ax.coastlines(resolution='110m')
ax.add_feature(cfeature.BORDERS)
c =  ax.contourf(lons,lats,dst3,cmap='coolwarm')
ax.quiver(dsu,dsv,q_vector_u,q_vector_v)
ax.clabel(c,inline=True,fontsize=10)
plt.colorbar(c,shrink=0.6)

#%%
fig=plt.figure(figsize=(12,9))
add_metpy_logo(fig,20,20,size='small')

ax=fig.add_subplot(1,1,1,projection=crs)
#x.set_extent(crs=data_crs)
#.add_feature(cfeature.STATES, linewidth=0.75)
#x.add_feature(USCOUNTIES.with_scale('5m'),linewidth=0.25)

         
c1=ax.contourf(dx,dy,dst,transform=data_crs,colors='k',linewidths=1)

ax.quiver(dx,dy,qvector)
ax.clabel(c1, inline=True,fontsize=10)
plt.colobar(c,shrink=0.6)
#%%
#%%

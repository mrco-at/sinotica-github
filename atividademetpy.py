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
##matplotlib inline
import metpy.calc as mpcalc
from metpy.units import units
import numpy as np
import xarray as xr
import pandas as pd
#%%
# abrindo os dados do netcdf 24062021.nc chamando de ds. Contem dados de temperatura, v wind e u wind para hemisferio sul no dia 24 de 11 de 2021 (09z). 
ds = xr.open_dataset('24062021.nc')#.metpy.parse_cf()
#%%
print("INFORMACOES DO DS:")
ds
#%%
dssel=ds.sel(time='2021-06-24T09:00:00')
#%%
dssel
#%%
# abrindo os dados do netcdf 30062020.nc chamando de ds1. Contem dados de temperatura, u wind e v wind para o dia 30 de junho de 2020 (12z)
ds1 = xr.open_dataset('30062020.nc')#.metpy.parse_cf()
#%%
print("INFORMACOES DO DS1:")
ds1
#%%
# abrindo os dados do netcdf 2022.nc chamando de ds2. Contem dados de temperatura, u wind e v wind para os dias 01/05, 03/05, 15/05, 01/07, 03/07, 15/07 as 00, 03  e 09 z.
ds2 = xr.open_dataset('2022.nc')#.metpy.parse_cf()
#%%
print("INFORMACOES DO DS2:")
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
temp = getvar(dst, "t")
#%%
print(dst.shape)
print(dsu.shape)
print(dsv.shape)
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
dx1, dy1 = mpcalc.lat_lon_grid_deltas(ds1['longitude'], ds1['latitude'])
dx2, dy2 = mpcalc.lat_lon_grid_deltas(ds2['longitude'], ds2['latitude'])

# Compute the divergence of the Q-vectors calculated above
#q_div = -2*mpcalc.divergence(uqvect, vqvect, dx, dy, dim_order='yx')
#%%
# Compute the Q-vector components
qvector = mpcalc.q_vector(dsu*units('m/s'),dsv*units('m/s'),dst*units('K'),850*units('hPa'),11.1*units.km,11.1*units.km)
#%%
qvector
#%%
m='meter ** 2 / kilogram / second'
#%%
q_vector_u=qvector[0]
q_vector_v=qvector[1]
#%%
q_vector_u
#%%
q_vector_u_dset=q_vector_u.to_dataset(name="vector-q-u")
#%%
q_vector_u_dset
#%%
qu_mean=q_vector_u_dset['vector-q-u'].mean(('time','longitude','latitude'))
qu_mean
#%%
#%%
q_vector_v_dset=q_vector_v.to_dataset(name="vector-q-v")
qv_mean=q_vector_v_dset['vector-q-v'].mean(('time','longitude','latitude'))
qv_mean
#%%
q_vector_v

#%%
dat=ds.metpy.parse_cf('t')
#%%
proj=dat.metpy.cartopy_crs
#%%

#%%
#rs=ccrs.LambertConformal(central_longitude=-53,central_latitude=45)
crs=ccrs.LambertCylindrical()
data_crs=ccrs.LambertCylindrical()
#%%
#lons, lats = np.meshgrid(ds['longitude'], ds['latitude'])
lons=dst['longitude'] 
lats=dst['latitude']
#%%
q_vector_u_sel=q_vector_u.sel(time='2021-06-24T09:00:00')
q_vector_u
#%%
q_vector_v_sel=q_vector_v.sel(time='2021-06-24T09:00:00')
q_vector_v_sel
#%%
plt.contourf(dssel['t'],cmap='coolwarm')
plt.colorbar()
#%%
plt.figure(figsize=(12,9))
ax = plt.axes(projection=proj)
ax.coastlines()
            
#ax.set_extent([-90,-30,10,-89])
#ax.coastlines #(resolution='110m')
#ax.add_feature(cfeature.BORDERS)
ax.contourf(dssel['longitude'],dssel['latitude'],dssel['t'],cmap='coolwarm')

#ax.quiver(dsu,dsv,q_vector_u,q_vector_v)
#ax.clabel(c,inline=True,fontsize=10)
#plt.colorbar(c,shrink=0.6)
plt.show()  
#%%
plt.contour(q_vector_u_sel)
plt.colorbar()
#%%
plt.figure(figsize=(12,9))
ax = plt.axes(projection=proj)
ax.coastlines()
            
#ax.set_extent([-90,-30,10,-89])
#ax.coastlines #(resolution='110m')
#ax.add_feature(cfeature.BORDERS)
#ax.contourf(dx,dy,dst,transfo,colors='k',linewidths=1)
ax.quiver(q_vector_u_sel['longitude'],q_vector_u_sel['latitude'],q_vector_u_sel,q_vector_v_sel,scale=1e-13)
#ax.clabel(c,inline=True,fontsize=10)
#plt.colorbar(c,shrink=0.6)
plt.show()  
#%%
fig=plt.figure(figsize=(12,9))
add_metpy_logo(fig,20,20,size='small')

ax=fig.add_subplot(1,1,1,projection=crs)
#x.set_extent(crs=data_crs)
#.add_feature(cfeature.STATES, linewidth=0.75)
#x.add_feature(USCOUNTIES.with_scale('5m'),linewidth=0.25)

         
c1=
ax.quiver(dx,dy,qvector)
ax.clabel(c1, inline=True,fontsize=10)
plt.colobar(c,shrink=0.6)
#%%
#%%

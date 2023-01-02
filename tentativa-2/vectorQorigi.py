#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 10:13:53 2022
Script desenvolvido para criação de mapa de vetor na disciplina e meteorologia sinótica em 2022 - versão compartilhada 
por joyce SEM minhas alterações
@author: inpe
"""
#%%
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from metpy.units import units
import metpy.calc as mpcalc
import numpy as np
import xarray as xr
from mpl_toolkits.axes_grid1 import make_axes_locatable
#%%
ds = xr.open_dataset("/home/tet/Documents/sinotica/atividade-metpy/sinotica-github/tentativa-2/30_06_2020.nc")
#%%
ds
#%%
#renomeando latitude e longitude
ds=ds.rename({"latitude":"lat","longitude":"lon"})
ds

lon_slice = slice( -120, 0)
lat_slice = slice(10, -60)

lats = ds.lat.sel(lat=lat_slice).values
lons = ds.lon.sel(lon=lon_slice).values

level = ds.level[1] * units.hPa

###criando arquivo para conversão
hgt_850 = mpcalc.smooth_n_point(ds.z.metpy.sel(vertical=level, lat=lat_slice, lon=lon_slice).squeeze(), 9, 50)

width = 9.81 * units("m^2/s^2")
hgt_850_1 = (units.Quantity(hgt_850[:].data))/width
hgt_850_2= (hgt_850_1 * units.dam)/10

tmpk_850 = mpcalc.smooth_n_point(ds.t.metpy.sel(vertical=level, lat=lat_slice, lon=lon_slice).squeeze(), 9, 25)

###convertendo pra celsius

tmpk_1 =  (units.Quantity(tmpk_850[:].data,"degC"))

uwnd_850 = mpcalc.smooth_n_point(ds.u.metpy.sel(vertical=level, lat=lat_slice, lon=lon_slice).squeeze(), 9, 50)
vwnd_850 = mpcalc.smooth_n_point(ds.v.metpy.sel(vertical=level, lat=lat_slice, lon=lon_slice).squeeze(), 9, 50)

vtime = ds.time.data[0].astype('datetime64[ms]').astype('O')
#%%
# Compute grid spacings for data
dx, dy = mpcalc.lat_lon_grid_deltas(lons, lats)

# Compute the Q-vector components
uqvect, vqvect = mpcalc.q_vector(uwnd_850, vwnd_850, tmpk_1, level, dx, dy)

# Compute the divergence of the Q-vectors calculated above
q_div = 2*mpcalc.divergence(uqvect, vqvect, dx=dx, dy=dy, x_dim=-1, y_dim=-2)



#### criando o mapa
# Set the map projection (how the data will be displayed)
mapcrs = ccrs.Mercator()
# Set the data project (GFS is lat/lon format)
datacrs = ccrs.PlateCarree()

# PLOT
fig = plt.figure(1, figsize=(16, 14))
ax = plt.subplot(111, projection=mapcrs)
ax.set_extent([-120, 0, 10, -60], ccrs.PlateCarree())

# Add map features to plot coastlines and state boundaries
ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
ax.add_feature(cfeature.STATES.with_scale('50m'))

# Plot 850-hPa Q-Vector Divergence and scale
clevs_850_tmpc = np.arange(-40, 41, 2)
clevs_qdiv = list(range(-30, -4, 5))+list(range(5, 31, 5))
cf = ax.contourf(lons, lats, q_div*1e18, clevs_qdiv, cmap=plt.cm.bwr,
                 extend='both', transform=datacrs)
#cb = plt.colorbar(cf, orientation='horizontal', pad=0, aspect=20, extendrect=True,
#                  ticks=clevs_qdiv)
cb = plt.colorbar(cf, orientation='horizontal', pad=0.04, fraction=0.046,
                  ticks=clevs_qdiv)
cb.set_label('Q-Vector Div. (*10$^{18}$ m s$^{-1}$ kg$^{-1}$)')

# Plot 850-hPa Temperatures
csf = ax.contour(lons, lats, tmpk_1, clevs_850_tmpc, colors='grey',
                 linestyles='dashed', transform=datacrs)
plt.clabel(csf, fmt='%d')

# Plot 850-hPa Geopotential Heights
clevs_850_hght = np.arange(0, 200, 3)
cs = ax.contour(lons, lats, hgt_850_2, clevs_850_hght, colors='black', transform=datacrs)
plt.clabel(cs, fmt='%d')

# Plot 850-hPa Q-vectors, scale to get nice sized arrows
wind_slice = (slice(None, None, 5), slice(None, None, 5))
ax.quiver(lons[wind_slice[0]], lats[wind_slice[1]],
          uqvect[:].data[wind_slice],
          vqvect[:].data[wind_slice],
          pivot='mid', color='black',
          scale=1e-11, scale_units='inches',
          transform=datacrs)

# Add some titles
plt.title('850-hPa GFS Geo. Heights (m), Temp (C), and Q-Vectors (m$^2$ kg$^{-1}$ s$^{-1}$)\n', loc='left')
plt.title('{}'.format(vtime), loc='right')

plt.show()
#%%
plt.savefig('30-06-20-zoom.png')
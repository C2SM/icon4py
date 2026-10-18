import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import netCDF4 as nc4

plt.close()


grid_filename = (
    "/scratch/mch/cong/debug/icon4py/testdata/grids/r02b04_global/icon_grid_0013_R02B04_R.nc"
)
f = nc4.Dataset("/scratch/mch/cong/investigation_JW_noise/icon4py/data/r2b5/remap/", "r")

ax = plt.axes(projection=ccrs.PlateCarree())

plt.contourf(lons, lats, sst, 60,
             transform=ccrs.PlateCarree())

ax.coastlines()

plt.show()
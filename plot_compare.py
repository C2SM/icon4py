import os
from icon_uxarray import icon_grid_2_ugrid, remove_torus_boundaries
import uxarray as ux
import holoviews as hv
import panel as pn

grid_fname = './icon-exclaim-data/ux_Torus_Triangles_50000m_x_5000m_res500m.nc'
if not os.path.isfile(grid_fname):
    grid_fname = icon_grid_2_ugrid(
        './icon-exclaim-data/Torus_Triangles_50000m_x_5000m_res500m.nc',
    )

uxds0 = ux.open_dataset(grid_fname, './icon-exclaim-data/torus_exclaim_almost_default/torus_exclaim_insta_DOM01_ML_0001.nc')
uxds0 = remove_torus_boundaries(uxds0)

uxds1 = ux.open_dataset(grid_fname, './gauss3d_output/data_output_0.nc')
uxds2 = ux.open_dataset(grid_fname, './gauss3d_output/data_output_1.nc')
def preprocessing(ds):
    return ds.expand_dims(dim='t')
uxds1 = ux.open_mfdataset(grid_fname, './gauss3d_output/data_output_*.nc')
uxds1 = remove_torus_boundaries(uxds1)

# interactive plot
def sliders_plot(itime, iheight):
    return uxds0['temp'].isel(time=itime).isel(height=iheight).plot()
    #sub0 = uxds0['temp'].isel(time=itime).isel(height=iheight).plot()
    #sub1 = uxds1['temp'].isel(time=itime).isel(height=iheight).plot()

torus = hv.DynamicMap(sliders_plot, kdims=['time', 'height'])
hvplot = torus.redim.range(time=(0, len(uxds0.time)), height=(0, len(uxds0.height)))
server = pn.panel(hvplot).show()

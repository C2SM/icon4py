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

#uxds_ft = ux.open_dataset(grid_fname, './icon-exclaim-data/torus_exclaim_almost_default/torus_exclaim_insta_DOM01_ML_0001.nc')
uxds_ft = ux.open_dataset(grid_fname, './icon-exclaim-data/torus_exclaim_const_velocity/torus_exclaim_insta_DOM01_ML_0001.nc')
uxds_ft = remove_torus_boundaries(uxds_ft)

uxds_py = ux.open_dataset(grid_fname, './gauss3d_output/data_output_0.nc')
uxds_py = remove_torus_boundaries(uxds_py)

# comparison plot
def sliders_plot(itime, iheight):
    sub0 = uxds_ft['u'].isel(time=itime).isel(height=iheight).plot()
    sub1 = uxds_py['u'].isel(time=itime).isel(height_2=iheight).plot()
    return hv.Layout(sub0 + sub1).cols(1)
torus = hv.DynamicMap(sliders_plot, kdims=['time', 'height'])
hvplot = torus.redim.range(time=(0, len(uxds_ft.time)), height=(0, len(uxds_ft.height)))
server = pn.panel(hvplot).show()

# single plot
def sliders_plot(itime, iheight):
    return uxds_ft['u'].isel(time=itime).isel(height=iheight).plot()
torus = hv.DynamicMap(sliders_plot, kdims=['time', 'height'])
hvplot = torus.redim.range(time=(0, len(uxds_ft.time)), height=(0, len(uxds_ft.height)))
server = pn.panel(hvplot).show()

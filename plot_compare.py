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

uxds_ft = ux.open_dataset(grid_fname, './icon-exclaim-data/torus_exclaim_almost_default/torus_exclaim_insta_DOM01_ML_0001.nc')
#uxds_ft = ux.open_dataset(grid_fname, './icon-exclaim-data/torus_exclaim_const_velocity/torus_exclaim_insta_DOM01_ML_0001.nc')
uxds_ft = remove_torus_boundaries(uxds_ft)

uxds_py = ux.open_dataset(grid_fname, './gauss3d_output/data_output_0.nc')
uxds_py = remove_torus_boundaries(uxds_py)

# comparison plot
def sliders_plot(itime, iheight):
    vname='u'
    clim_ft = (
        float(uxds_ft[vname].isel(time=itime).isel(height=iheight).min()),
        float(uxds_ft[vname].isel(time=itime).isel(height=iheight).max()),
        )
    clim_py = (
        float(uxds_py[vname].isel(time=itime).isel(height=iheight).min()),
        float(uxds_py[vname].isel(time=itime).isel(height=iheight).max()),
        )
    sub0 = uxds_ft[vname].isel(time=itime).isel(height=iheight).plot(title='ftn', clim=clim_ft)
    sub1 = uxds_py[vname].isel(time=itime).isel(height=iheight).plot(title='py',  clim=clim_py)
    #return hv.Layout(sub0 + sub1).cols(1)
    data_dl = (uxds_ft[vname].isel(time=itime).isel(height=iheight) - uxds_py[vname].isel(time=itime).isel(height=iheight))
    clim_delta = ( float(data_dl.min()), float(data_dl.max()),)
    sub2 = data_dl.plot(title='delta', clim=clim_delta)
    return hv.Layout(sub0 + sub1 + sub2).cols(1)
torus = hv.DynamicMap(sliders_plot, kdims=['time', 'height'])
hvplot = torus.redim.range(time=(0, len(uxds_ft.time)), height=(0, len(uxds_ft.height)))
server = pn.panel(hvplot).show()

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
vname='u'
fig=plt.figure(1); plt.clf(); plt.show(block=False)
fig, axs = plt.subplots(1, 3, num=fig.number)
caxs = [make_axes_locatable(ax).append_axes('bottom', size='5%', pad=0.05) for ax in axs]
itime=0
iheight=0
tri = mpl.tri.Triangulation(uxds_ft.uxgrid.node_lon, uxds_ft.uxgrid.node_lat, triangles=uxds_ft.uxgrid.face_node_connectivity)
while itime>=0:
    data_ft = uxds_ft[vname].isel(time=itime).isel(height=iheight).values
    data_py = uxds_py[vname].isel(time=itime).isel(height=iheight).values
    data_dl = data_ft - data_py
    [ax.cla() for ax in axs]
    im0 = axs[0].tripcolor(tri, data_ft, shading='flat', vmin= data_ft.min(), vmax= data_ft.max())
    im1 = axs[1].tripcolor(tri, data_py, shading='flat', vmin= data_py.min(), vmax= data_py.max())
    im2 = axs[2].tripcolor(tri, data_dl, shading='flat', vmin= data_dl.min(), vmax= data_dl.max())
    fig.colorbar(im0, cax=caxs[0], orientation='horizontal')
    fig.colorbar(im1, cax=caxs[1], orientation='horizontal')
    fig.colorbar(im2, cax=caxs[2], orientation='horizontal')
    plt.draw()
    itime, iheight = map(int, input('itime, iheight: ').split())
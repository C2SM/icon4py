import os
from icon_uxarray import icon_grid_2_ugrid, remove_torus_boundaries
import uxarray as ux
import numpy as np
import matplotlib as mpl
mpl.use('QtAgg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

if os.uname()[1][0:3] == 'tsa':
    base_dir='/scratch/jcanton/icon4py/'
elif os.uname()[1][0:3] == 'mac':
    base_dir='/Users/jcanton/Downloads/icon4py/'

enr = 1
if enr == 0:
    ename = 'mountain_default'
elif enr == 1:
    ename = 'seven_velocity'
elif enr == 2:
    ename = 'zero_velocity'

grid_fname = base_dir + 'icon-exclaim-data/ux_Torus_Triangles_50000m_x_5000m_res500m.nc'
if not os.path.isfile(grid_fname):
    grid_fname = icon_grid_2_ugrid(
        './icon-exclaim-data/Torus_Triangles_50000m_x_5000m_res500m.nc',
    )

uxds_ft = ux.open_dataset(grid_fname, base_dir + '/icon-exclaim-data/torus_exclaim_'+ename+'/torus_exclaim_insta_DOM01_ML_0001.nc')
uxds_ft = remove_torus_boundaries(uxds_ft)

uxds_py = ux.open_dataset(grid_fname, base_dir + '/gauss3d_output.'+ename+'/data_output_0.nc')
uxds_py = remove_torus_boundaries(uxds_py)

#===============================================================================

tri = mpl.tri.Triangulation(uxds_ft.uxgrid.node_lon, uxds_ft.uxgrid.node_lat, triangles=uxds_ft.uxgrid.face_node_connectivity)

vname='w'
fig=plt.figure(2); plt.clf(); plt.show(block=False)
fig, axs = plt.subplots(1, 3, num=fig.number, sharex=True, sharey=True)
caxs = [make_axes_locatable(ax).append_axes('bottom', size='5%', pad=0.05) for ax in axs]
itime=0
iheight=0
while itime>=0:
    data_ft = uxds_ft[vname].isel(time=itime).isel(height_2=iheight).values
    data_py = uxds_py[vname].isel(time=itime).isel(height_2=iheight).values
    data_dl = data_ft - data_py
    [ax.cla() for ax in axs]
    im0 = axs[0].tripcolor(tri, data_ft, edgecolor='none', shading='flat', cmap='viridis', norm=colors.Normalize(vmin=data_ft.min(), vmax=max(data_ft.min()+1e-9, data_ft.max())))
    im1 = axs[1].tripcolor(tri, data_py, edgecolor='none', shading='flat', cmap='viridis', norm=colors.Normalize(vmin=data_py.min(), vmax=max(data_py.min()+1e-9, data_py.max())))
    im2 = axs[2].tripcolor(tri, data_dl, edgecolor='none', shading='flat', cmap='seismic', norm=colors.TwoSlopeNorm(vmin=min(-1e-9, data_dl.min()), vcenter=0, vmax=max(1e-9, data_dl.max())))
    cbar = fig.colorbar(im0, cax=caxs[0], orientation='horizontal', label=vname); cbar.set_ticks(np.linspace(cbar.vmin, cbar.vmax, 5))
    cbar = fig.colorbar(im1, cax=caxs[1], orientation='horizontal', label=vname); cbar.set_ticks(np.linspace(cbar.vmin, cbar.vmax, 5))
    cbar = fig.colorbar(im2, cax=caxs[2], orientation='horizontal', label=vname); cbar.set_ticks(np.r_[np.linspace(cbar.vmin, 0, 3), np.linspace(0, cbar.vmax, 3)[1:]])
    axs[0].set_title('icon-exclaim')
    axs[1].set_title('icon4py')
    axs[2].set_title('delta')
    plt.draw()
    itime, iheight = map(int, input('itime, iheight: ').split())
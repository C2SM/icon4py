import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from icon_uxarray import icon_grid_2_ugrid, remove_torus_boundaries
import uxarray as ux

enr = 0
if   enr == 0: ename = 'mountain_default'
elif enr == 1: ename = 'four_velocity'
elif enr == 2: ename = 'zero_everything'

uname = os.uname()[1][0:3]
if uname == 'tsa':
    base_dir='/scratch/jcanton/icon4py/'
elif uname in ['mac', 'wor']:
    base_dir='/Users/jcanton/sshfsVolumes/tsa/icon4py/'

grid_fname = base_dir + 'icon-exclaim-data/ux_Torus_Triangles_50000m_x_5000m_res500m.nc'
if not os.path.isfile(grid_fname):
    grid_fname = icon_grid_2_ugrid(
        base_dir + 'icon-exclaim-data/Torus_Triangles_50000m_x_5000m_res500m.nc',
    )

uxds_ft = ux.open_dataset(grid_fname, base_dir + '/icon-exclaim-data/torus_exclaim.'+ename+'/torus_exclaim_insta_DOM01_ML_0001.nc')
uxds_ft = remove_torus_boundaries(uxds_ft)

uxds_py = ux.open_dataset(grid_fname, base_dir + '/gauss3d_output.'+ename+'/data_output_0.nc')
uxds_py = remove_torus_boundaries(uxds_py)

tri = mpl.tri.Triangulation(uxds_ft.uxgrid.node_lon, uxds_ft.uxgrid.node_lat, triangles=uxds_ft.uxgrid.face_node_connectivity)
probe_cell = 428

#===============================================================================

vname='u'
fig=plt.figure(1); plt.clf(); plt.show(block=False)
fig, axs = plt.subplots(1, 3, num=fig.number, sharex=True, sharey=True)
caxs = [make_axes_locatable(ax).append_axes('bottom', size='5%', pad=0.05) for ax in axs]
itime=0
iheight=0
while itime>=0:
    fig.suptitle('itime {:} iheight {:}'.format(itime,iheight))
    data_ft = uxds_ft[vname].values[itime,iheight,:]
    data_py = uxds_py[vname].values[itime,iheight,:]
    data_dl = data_ft - data_py
    [ax.cla() for ax in axs]
    im0 = axs[0].tripcolor(tri, data_ft, edgecolor='none', shading='flat', cmap='viridis') #, norm=colors.Normalize(vmin=data_ft.min(), vmax=max(data_ft.min()+1e-9, data_ft.max())))
    im1 = axs[1].tripcolor(tri, data_py, edgecolor='none', shading='flat', cmap='viridis') #, norm=colors.Normalize(vmin=data_py.min(), vmax=max(data_py.min()+1e-9, data_py.max())))
    im2 = axs[2].tripcolor(tri, data_dl, edgecolor='none', shading='flat', cmap='seismic', norm=colors.TwoSlopeNorm(vmin=min(-1e-15, data_dl.min()), vcenter=0, vmax=max(1e-15, data_dl.max())))
    cbar = fig.colorbar(im0, cax=caxs[0], orientation='horizontal', label=vname); cbar.set_ticks(np.linspace(cbar.vmin, cbar.vmax, 5))
    cbar = fig.colorbar(im1, cax=caxs[1], orientation='horizontal', label=vname); cbar.set_ticks(np.linspace(cbar.vmin, cbar.vmax, 5))
    cbar = fig.colorbar(im2, cax=caxs[2], orientation='horizontal', label=vname); cbar.set_ticks(np.r_[np.linspace(cbar.vmin, 0, 3), np.linspace(0, cbar.vmax, 3)[1:]])
    axs[0].set_title('icon-exclaim {:.18e} {:.18e}'.format(data_ft.min(), data_ft.max()))
    axs[1].set_title('icon4py {:.18e} {:.18e}'.format(data_py.min(), data_py.max()))
    axs[2].set_title('delta')
    plt.draw()
    itime, iheight = map(int, input('itime, iheight: ').split())
    if itime == 313:
        vname = input('vname: ')
        itime =0; iheight=0

vname='z_ifc'
fig=plt.figure(2); fig.clf(); plt.show(block=False)
fig, ax = plt.subplots(1, 1, num=fig.number, sharex=True, sharey=True)
cax = make_axes_locatable(ax).append_axes('bottom', size='5%', pad=0.05)
iheight=0
while iheight>=0:
    data_ft = uxds_ft['z_ifc'].values[iheight,:]
    ax.cla()
    im0 = ax.tripcolor(tri, data_ft, edgecolor='none', shading='flat', cmap='viridis') #, norm=colors.Normalize(vmin=data_ft.min(), vmax=max(data_ft.min()+1e-9, data_ft.max())))
    cbar = fig.colorbar(im0, cax=cax, orientation='horizontal', label=vname); cbar.set_ticks(np.linspace(cbar.vmin, cbar.vmax, 5))
    ax.set_title('icon-exclaim')
    fig.suptitle('z_ifc | min {:.1f} max {:.1f}'.format(data_ft.min(),data_ft.max()))
    plt.draw()
    iheight = int(input('iheight: '))

# u/w sanity checks
plt.figure(3); plt.clf(); plt.show(block=False)
for iheight in range(35):
    w_ft = uxds_ft['u'].values[:,iheight,probe_cell]
    w_py = uxds_py['u'].values[:,iheight,probe_cell]
    itime=3
    print('{:.18e}, {:.18e}, {:.18e}'.format(w_ft[itime], w_py[itime], w_ft[itime]-w_py[itime]))
    plt.plot(w_ft, '-')
    plt.plot(w_py, '--')
plt.draw()

print('\n\n')

# rho sanity checks
plt.figure(4); plt.clf(); plt.show(block=False)
for iheight in range(35):
    w_ft = uxds_ft['rho'].values[:,iheight,probe_cell]
    w_py = uxds_py['rho'].values[:,iheight,probe_cell]
    itime=1
    print('{:.18e}, {:.18e}, {:.18e}'.format(w_ft[itime], w_py[itime], w_ft[itime]-w_py[itime]))
    plt.plot(w_ft, '-')
    plt.plot(w_py, '--')
plt.draw()

# rho sanity 2
for itime in range(15):
    print(itime)
    for iheight in range(35):
        w_ft = uxds_ft['rho'].values[:,iheight,probe_cell]
        w_py = uxds_py['rho'].values[:,iheight,probe_cell]
        print('{:.18e}, {:.18e}, {:.18e}'.format(w_ft[itime], w_py[itime], w_ft[itime]-w_py[itime]))

import cortex
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np

def plot_flat_lateral(
    data,
    subj,
    save_name,
    cbar_title='Correlation',
    cmap='RdYlBu_r',
    vmin=-0.2,
    vmax=0.2,
    
):

    # gather data Volume
    cmap_name = cmap
    volume = cortex.Volume(data, f'UTS0{subj}', f'UTS0{subj}_auto', cmap=cmap_name, vmin=vmin, vmax=vmax)
    params = cortex.export.params_flatmap_lateral_medial
    cortex.export.plot_panels(volume, **params)
    fig = plt.gcf()
    cmap = cm.get_cmap(cmap_name)

    # Create a normalization based on vmin and vmax
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  
    cbar = fig.colorbar(sm, ax=fig.axes, orientation='horizontal', shrink=0.35, location='top', pad=0.01)
    cbar.set_ticks([vmin, vmax])  
    cbar.ax.tick_params(labelsize=20)  
    # Move the ticks to the bottom of the colorbar
    cbar.ax.xaxis.set_ticks_position('bottom')  # Set ticks position to bottom
    cbar.ax.xaxis.set_label_position('top')     # Keep label on top
    cbar.set_label(cbar_title, fontsize=26, labelpad=10)

    ## Save figure
    plt.savefig(f'./s{subj}_lateral_{save_name}_{cmap_name}.pdf',
                format='pdf', bbox_inches='tight')

    plt.close()
    
if __name__ == "__main__":
    subj = 3
    da = np.random.rand(95556) #nvoxels for subj 3 
    plot_flat_lateral(
        data=da,
        subj=subj,
        save_name='test_plot',
        cbar_title='Test Colorbar',
        cmap='RdYlBu_r',
        vmin=-0.2,
        vmax=0.2,
    )
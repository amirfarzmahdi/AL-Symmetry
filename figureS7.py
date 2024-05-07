# plot code for figureS7
# useful packages:
import matplotlib.gridspec as gridspec
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy
import scipy.io as sio
import warnings
warnings.filterwarnings("ignore")
import pickle
from PIL import Image
from scipy.stats import pearsonr
import scipy.stats as stats
import statsmodels.stats.multitest as multi
import seaborn as sns

# start settings:
figname = 'figureS7'
# conditions of the network
conds = ['3D_objects_trained',
        'ImageNet_trained',
        'white_noise_trained',
        '3D_objects_untrained',
        'ImageNet_untrained',
        'white_noise_untrained',
        ]
panels = ['A','C','E','B','D','F']

num_imgs = 2025

# main setting
xtlbs = ['conv1','conv2','conv3','conv4','conv5'] # ['conv1','conv2','conv3','conv4','conv5','fc6','fc7']
n = len(xtlbs) # number of layers: 7 or 5
lstyl = '--' # '--', '-'
panel = 'F'
cond = 'r_r90imgs_rfmap' # 1:r_himgs, 2:r_vimgs, 3:r_r90imgs, 4:r_himgs_hfmap, 5:r_vimgs_vfmap, 6:r_r90imgs_rfmap, 
color = [0,0.298,0.5608] # 1:[0.9294,0.1373,0.1647] ,2:[0.1059,0.7373,0.6078], 3: [0,0.298,0.5608], 
annots = False # annotations

# figure 
msz = 3
lw = 0.2
y_pos = [-0.5,0,0.5,1]
x_pos = np.array(np.arange(n),dtype='int64')

# save figure flag
savefig = 1


# font
mpl.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 4 # 6
plt.rcParams['axes.linewidth'] = 0.3 # 0.8



# Initialize grid with 2 rows and 3 columns
ncols = 3
nrows = 2
grid = gridspec.GridSpec(nrows, ncols,
            left=0.05, bottom=0.1, right=0.95, top=0.91, wspace=0.23, hspace=0.3)
fig = plt.figure(figsize=(3,1),constrained_layout=True)
fig.clf()
# end settings

# load the correlation files
for i, ax in enumerate(grid):
    with open('figure3'+panels[i]+'_corr_invaraince_euqivaraince_zscored_'+conds[i]+'_'+str(num_imgs)+'.csv','rb') as fp:
        data_dict = pickle.load(fp)

    # Add axes which can span multiple grid boxes
    ax_ = fig.add_subplot(ax)

    data = data_dict[cond][:,0:n] # r_himgs,
    parts = ax_.violinplot(data,x_pos[0:n],showmeans=False, showmedians=False, showextrema=False)
    
    for pc in parts['bodies']:
        pc.set_facecolor(color)
        pc.set_edgecolor(color)
        pc.set_alpha(0.3)
        pc.set_linestyles(lstyl)
        pc.set_linewidths(0.5)
        pc.set_clip_on(False)

    quartile1, medians, quartile3 = np.percentile(np.transpose(data),[25,50,75],axis=1)
    ax_.scatter(x_pos, medians, marker='o', color=color, s=0.5, zorder=3, clip_on=False)
    ax_.vlines(x_pos, quartile1, quartile3, color=color, linestyle='-', lw=1, alpha= 0.5, clip_on=False)
    ax_.hlines(0,-0.5,6.5, color=[0.7,0.7,0.7], linestyle='--', lw=0.2, clip_on=True)

    # set tick params
    plt.tick_params(length = 0.5, width = 0.1)
    
    # set x axis
    ax_.set_xticks(x_pos)
    ax_.set_xlim([-0.5,n-1.0])
    if i < 3: # xticklabels
        ax_.set_xticklabels([])
    else:
        ax_.set_xticklabels(xtlbs,rotation = 45, )

    # set y axis
    ax_.set_yticks(y_pos)
    ax_.set_ylim((-0.5,1))
    if i == 0 or i == 3: # yticklabels
        ax_.set_yticks(y_pos)
        ax_.set_yticklabels(y_pos)
    else:
        ax_.set_yticks(y_pos)
        ax_.set_yticklabels([])

    # set ylabel
    if i == 3:
        ax_.set_ylabel("Pearson's r",labelpad=0.8)

    # set panels name
    if annots == True:
        if i == 0:
            ax_.set_title('3D Objects',pad = 12,
                        fontdict = {'fontsize': 6,
                            'fontweight': 'bold',
                            'color': [0.6,0.6,0.6],
                            'verticalalignment': 'center',
                            'horizontalalignment': 'center'})
            
            ax_.annotate('Trained', xy=(0, 0.5), xytext=(-3, 0),
                    xycoords=ax_.yaxis.label, textcoords='offset points',
                        ha='right', va='center',rotation=90,c=[0.6,0.6,0.6],
                        fontsize = 6,fontweight='bold')
        elif i == 1:
            ax_.set_title('Natural images',pad = 12,
                        fontdict = {'fontsize': 6,
                            'fontweight': 'bold',
                            'color': [0.6,0.6,0.6],
                            'verticalalignment': 'center',
                            'horizontalalignment': 'center'})
        elif i == 2:
            ax_.set_title('Random noise images',pad = 12,
                        fontdict = {'fontsize': 6,
                            'fontweight': 'bold',
                            'color': [0.6,0.6,0.6],
                            'verticalalignment': 'center',
                            'horizontalalignment': 'center'})
        elif i == 3:
                    ax_.annotate('Untrained', xy=(0, 0.5), xytext=(-3, 0),
                    xycoords=ax_.yaxis.label, textcoords='offset points',
                    ha='right', va='center',rotation=90,c=[0.6,0.6,0.6],
                    fontsize = 6,fontweight='bold')
        
    # remove top and right lines
    ax_.spines['top'].set_visible(False)
    ax_.spines['right'].set_visible(False)
    
    
# Save the figure and show
if savefig == 1:
    plt.tight_layout()
    plt.savefig(figname + '_' + panel + '.pdf',
                dpi=300,bbox_inches='tight',facecolor='w',pad_inches=0.01)
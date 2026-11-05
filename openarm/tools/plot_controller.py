"""Plot fixed source-controller comparisons; reads archived results only."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p=argparse.ArgumentParser();p.add_argument('folder');args=p.parse_args();folder=Path(args.folder)
summary=json.loads((folder/'summary.json').read_text());modes=['position_pd_bias','source_mit_zero','source_mit_gravity']
labels=['PD + bias (runtime)','Source MIT: zero state','Source MIT: gravity proxy']
colors=['#af7659','#608497','#8c9471'];plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,3,figsize=(13.3,6.3),layout='constrained')
for ep in range(3):
 top,bottom=axes[:,ep]
 for i,(mode,label,color) in enumerate(zip(modes,labels,colors)):
  row=json.loads((folder/f'{mode}-ep{ep}.json').read_text())
  top.bar(np.arange(2)+i*.23,[row['metrics'][s]['rmse_rad']*1000 for s in ('left','right')],width=.23,label=label,color=color)
  ts=row['recorded_times_s'];pred=np.array(row['prediction']);target=np.array(row['target'])
  bottom.plot(ts,pred[1,:,2],color=color,label=label,lw=1.6)
  if i==0:bottom.plot(ts,target[1,:,2],color='#363a35',label='Recorded encoder',lw=1.6)
 top.set_xticks(np.arange(2)+.23,['Left arm','Right arm']);top.set_ylabel('Joint RMSE / mrad');top.set_title(f'Development episode {ep}')
 bottom.set_xlabel('Record logger time / s');bottom.set_ylabel('Right joint 3 / rad');bottom.grid(alpha=.17)
axes[0,0].legend(frameon=False,fontsize=8);axes[1,0].legend(frameon=False,fontsize=8)
fig.suptitle('Controller state matters: fixed proposals, no fitting or runtime application',fontsize=14)
fig.savefig(folder/'comparison.png',dpi=170);fig.savefig(folder/'comparison.svg');plt.close(fig)

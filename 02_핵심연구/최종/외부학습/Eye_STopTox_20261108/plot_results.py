"""Render scientific plots from the completed numerical results."""
from common import HERE, MODELS
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def render():
    paired=pd.read_csv(HERE/'paired_delta_by_seed.csv')
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':10})
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharex=True)
    colors={42:'#2a6fbb',202:'#d57923',777:'#138e7d'}
    for ax,label in zip(axes,['A','B']):
        data=paired[paired.label.eq(label)]
        for j,model in enumerate(MODELS):
            subset=data[data.model.eq(model)].sort_values('seed')
            vals=subset.delta_MCC.to_numpy()
            ax.plot([vals.min(),vals.max()],[j,j],color='#bbbbbb',lw=2)
            for seed,delta in zip(subset.seed,vals):
                ax.scatter(delta,j,s=55,color=colors[seed],label=str(seed) if j==0 else None,zorder=3)
            ax.scatter(vals.mean(),j,marker='|',s=200,color='black',zorder=4)
        ax.axvline(0,color='#555555',ls='--',lw=1)
        ax.set_yticks(range(len(MODELS)),[m.upper() for m in MODELS]);ax.set_title(f'EPA {label}')
        ax.set_xlabel('ΔMCC = STopTox 보조학습 - 대조군');ax.grid(axis='x',alpha=.2)
    axes[0].legend(title='분할 seed',loc='best',fontsize=8)
    fig.suptitle('기존 606행의 동일 grouped OOF 비교 · 점: 각 seed · 검은 선: 평균')
    fig.tight_layout();fig.savefig(HERE/'paired_delta_mcc.png',dpi=180);fig.savefig(HERE/'paired_delta_mcc.pdf');plt.close(fig)

if __name__=='__main__':render()

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
PILOT=ROOT/'data/figure_inputs/figure3_calibration_coordinates.csv'
DIST=ROOT/'data/distance'
OUT=ROOT/'outputs/publication_figures'
BLUE,RED,GRAY='#4C78A8','#E15759','#B8B0AD'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':8.5,'axes.labelsize':8,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'legend.fontsize':7.5,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
def check_and_save(fig,name):
    OUT.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','png','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=300,facecolor='white')
    plt.close(fig)
def figure3():
    pilot=pd.read_csv(PILOT,encoding='utf-8-sig')
    pilot=pilot.loc[pilot.distance_observed].copy()
    assert len(pilot)==249
    annual=pd.read_csv(DIST/'rq1_year_distance_descriptive.csv')
    adjusted=pd.read_csv(DIST/'rq1_year_adjusted_mean_distance.csv')
    fig=plt.figure(figsize=(4.91,4.15))
    a=fig.add_axes([.14,.34,.32,.50]);b=fig.add_axes([.64,.34,.34,.50])
    closer=pilot.own_portfolio_closer_than_permuted_median
    for flag,color,label,alpha in [(False,GRAY,'Own not closer (26.5%)',.62),(True,BLUE,'Own closer (73.5%)',.68)]:
        d=pilot.loc[closer.eq(flag)]
        a.scatter(d.specter2_title_portfolio_distance,d.permuted_other_portfolio_median_distance,
                  s=8,alpha=alpha,color=color,edgecolor='none',label=label)
    limits=[float(min(pilot.specter2_title_portfolio_distance.min(),pilot.permuted_other_portfolio_median_distance.min())-.008),
            float(max(pilot.specter2_title_portfolio_distance.max(),pilot.permuted_other_portfolio_median_distance.max())+.008)]
    a.plot(limits,limits,'--',color='#455A64',lw=.8)
    a.set(xlim=limits,ylim=limits,xlabel='Own prior-portfolio\ndistance',ylabel='Median randomized distance')
    a.set_xticks([0,.1,.2]);a.set_yticks([0,.05,.10,.15,.20])
    a.set_title('A. Random-portfolio\n    calibration',loc='left',weight='bold',pad=6)
    a.text(.02,.97,'Median margin\n= 0.026',transform=a.transAxes,ha='left',va='top',fontsize=7.5)
    a.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.33),fontsize=7.5,handlelength=1,handletextpad=.25)
    x=annual.index_year.to_numpy()
    b.fill_between(x,annual.q25_distance,annual.q75_distance,color=BLUE,alpha=.11,label='Q1–Q3 band')
    b.errorbar(x,annual.median_distance,yerr=[annual.median_distance-annual.median_ci_low,
                annual.median_ci_high-annual.median_distance],fmt='o-',color=BLUE,lw=1.2,capsize=2,markersize=3,
                label='Observed median\n(95% interval)')
    b.plot(x,adjusted.adjusted_mean_distance,'s-',color=RED,lw=1.2,markersize=3,label='Adjusted mean')
    b.fill_between(x,adjusted.ci_low,adjusted.ci_high,color=RED,alpha=.12)
    b.set_xticks([2018,2020,2022,2024]);b.set(xlabel='Entry year',ylabel='Title-portfolio distance')
    b.set_title('B. Prior-to-entry\n    distance by year',loc='left',weight='bold',pad=6)
    b.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.25),fontsize=7.5,handlelength=1.3,handletextpad=.3)
    check_and_save(fig,'fig3')

if __name__=='__main__':figure3()

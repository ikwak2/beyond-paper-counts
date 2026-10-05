"""Render typography changes from frozen inputs; never refit a model."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT/'outputs'
FIG.mkdir(exist_ok=True)
DIST = ROOT/'data/distance'
PILOT = ROOT/'data/figure_inputs/figure3_calibration_coordinates.csv'
BLUE, RED, GRAY = '#4C78A8', '#E15759', '#B8B0AD'
PATHS = ['no_recurrence','coauthor_continuity_only','at_least_one_no_index_coauthor']
LABELS = ['No record','Entry-coauthor papers only','≥1 paper without entry coauthors']
STYLE = {'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':8.5,
         'axes.labelsize':8,'xtick.labelsize':7.5,'ytick.labelsize':7.5,
         'legend.fontsize':7.5,'pdf.fonttype':42,'svg.fonttype':'none',
         'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':None}


def check_and_save(fig, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in fig.findobj(matplotlib.text.Text):
        if not text.get_visible() or not text.get_text():
            continue
        assert text.get_fontsize() >= 7.5, (name,text.get_text(),text.get_fontsize())
        b = text.get_window_extent(renderer)
        assert fig.bbox.contains(b.x0,b.y0) and fig.bbox.contains(b.x1,b.y1), (name,text.get_text(),tuple(b.bounds))
    for ext in ['pdf','png','svg']:
        fig.savefig(FIG/f'{name}.{ext}',dpi=300,facecolor='white')
    plt.close(fig)


def figure1():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.5,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(4.91,2.70));ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    ink='#29333D';gray='#7C8792';cols=['#A56C24','#48765D','#74628F'];xs=[.012,.348,.684];w=.304
    texts=[]
    def label(x,y,t,size=7.5,bold=False):
     obj=ax.text(x,y,t,ha='center',va='center',fontsize=size,fontweight='bold' if bold else 'normal',color=ink,linespacing=1.24);texts.append(obj);return obj
    checks=[]
    def box(x,y,h,t,c,size=7.5,bold=False):
     patch=Rectangle((x,y),w,h,facecolor='white',edgecolor='#C8CDD2',linewidth=.65);ax.add_patch(patch)
     ax.plot([x,x+w],[y+h,y+h],color=c,lw=1.4)
     obj=label(x+w/2,y+h/2,t,size,bold);checks.append((patch,obj))
    def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=6,lw=.65,color=gray,shrinkA=0,shrinkB=0))
    ax.add_patch(Rectangle((.19,.845),.62,.14,facecolor='#F2F4F6',edgecolor='#C8CDD2',lw=.65))
    label(.5,.945,'ICML and NeurIPS, 2018–2024',8.5,True)
    label(.5,.885,'26,872 accepted papers')
    for x,t,c in zip(xs,['Representation','Entry','Persistence'],cols):label(x+w/2,.777,t,8.5,True)
    ax.plot([.5,.5],[.845,.825],color=gray,lw=.65)
    ax.plot([xs[0]+w/2,.5],[.825,.825],color=gray,lw=.65)
    for x in xs[:2]:arrow((x+w/2,.825),(x+w/2,.800))
    box(xs[0],.46,.27,'Accepted-paper shares\nvs observed AI output\n\nPaper–country credits',cols[0])
    box(xs[1],.46,.27,'Observed new entrants\n12,094 researchers',cols[1])
    box(xs[2],.46,.27,'Complete +2 follow-up\n2018–2022 entry years\n6,704 researchers',cols[2])
    arrow((xs[1]+w,.595),(xs[2],.595))
    box(xs[0],.027,.365,'44 specifications\n\nProduction-adjusted\nrepresentation\nRQ1',cols[0])
    box(xs[1],.027,.365,'Prior-title histories\n9,639 (79.7%)\n\nPrior-to-entry\ntitle-portfolio distance\nRQ2, Diagnostic D1',cols[1])
    box(xs[2],.027,.365,'Primary analytic sample\n5,122\n\nReappearance at +2\nRQ3',cols[2])
    for x in xs:arrow((x+w/2,.46),(x+w/2,.402))
    fig.canvas.draw();r=fig.canvas.get_renderer()
    for t in texts:
     b=t.get_window_extent(r);assert t.get_fontsize()>=7.5;assert fig.bbox.contains(b.x0,b.y0) and fig.bbox.contains(b.x1,b.y1),t.get_text()
    for p,t in checks:
     b=p.get_window_extent(r);q=t.get_window_extent(r);assert b.contains(q.x0,q.y0) and b.contains(q.x1,q.y1),t.get_text()
    check_and_save(fig,'fig1')
    plt.rcParams.update(STYLE)


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


def figure4():
    prob=pd.read_csv(DIST/'rq3_exact_plus2_probabilities.csv').query("specification == 'semantic_primary'")
    con=pd.read_csv(DIST/'rq3_exact_plus2_contrasts.csv').query("specification == 'semantic_primary' and contrast_family == 'experienced_minus_no_experienced'")
    fig=plt.figure(figsize=(4.91,4.9))
    a=fig.add_axes([.14,.625,.83,.28]);b=fig.add_axes([.34,.21,.63,.235])
    scenarios=[(q,e) for q in ['Q25','Q75'] for e in [False,True]]
    bottom=np.zeros(4)
    for pathway,label,color in zip(PATHS,LABELS,[GRAY,RED,BLUE]):
        vals=np.array([prob.loc[prob.distance_level.eq(q)&prob.experienced_coauthor.eq(e)&prob.pathway.eq(pathway),'adjusted_probability'].item() for q,e in scenarios])
        a.bar(range(4),vals,bottom=bottom,color=color,width=.72,label=label)
        bottom+=vals
    np.testing.assert_allclose(bottom,1,atol=1e-10)
    a.set(ylim=(0,1),ylabel='Standardized pathway probability')
    a.set_xticks(range(4),[f"{q}\n{'With' if e else 'No'}\nrecent-history\ncoauthor" for q,e in scenarios],fontsize=7.5)
    a.set_title('A. Exact +2 pathway probabilities',loc='left',weight='bold',pad=7)
    for q,shift,color,marker in [('Q25',.12,BLUE,'o'),('Q75',-.12,'#E17C05','s')]:
        d=con.loc[con.distance_level.eq(q)].set_index('pathway').loc[PATHS]
        est=100*d.probability_difference.to_numpy()
        b.errorbar(est,np.arange(3)[::-1]+shift,xerr=[est-100*d.ci_low,100*d.ci_high-est],
                   fmt=marker,color=color,capsize=3,markersize=4,label=q)
    b.axvline(0,color='#455A64',ls='--',lw=.8)
    b.set_yticks([2,1,0],['No record','Entry-coauthor\npapers only','≥1 paper without\nentry coauthors'],fontsize=7.5)
    b.set(xlim=(-19,12),ylim=(-.55,2.55),xlabel='With − without recent-history coauthor\n(percentage points)')
    b.set_xticks([-15,-10,-5,0,5,10])
    b.legend(frameon=False,loc='upper right',fontsize=7.5)
    fig.text(.055,.475,'B. Recent-history coauthor contrasts',weight='bold',fontsize=8.5)
    handles,labels=a.get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.53,.004),frameon=False,fontsize=7.5,
               labelspacing=.25,handlelength=1.3)
    check_and_save(fig,'fig4')


def main():
    import subprocess
    if not (FIG/'pri_annual.csv').exists() or not (FIG/'openalex_capacity.csv').exists():
        subprocess.run([sys.executable,str(ROOT/'scripts/build_pri.py')],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/build_figure2.py')],cwd=ROOT,check=True)
    plt.rcParams.update(STYLE)
    figure1();figure3();figure4()
    print('PASS: final Figure 1–4 rendered from preserved public inputs.')
if __name__=='__main__':main()

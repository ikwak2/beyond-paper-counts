from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
OUT=Path(__file__).resolve().parent
def make_figure():
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none'})
    fig, ax = plt.subplots(figsize=(13.6,8.1))
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
    boxes=[]
    def box(x,y,w,h,text,edge,face,size=13):
        patch=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=0.017',
                            linewidth=1.8,edgecolor=edge,facecolor=face)
        ax.add_patch(patch)
        label=ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,color='#263238',linespacing=1.30)
        boxes.append((patch,label))
    def arrow(a,b):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=16,color='#607d8b',lw=1.8))
    ax.text(.5,.984,'Accepted-program measurement architecture',ha='center',va='top',fontsize=20,weight='bold')
    box(.20,.81,.60,.115,'ICML and NeurIPS accepted papers, 2018–2024\n26,872 papers','#37688f','#e7eef5',15)
    box(.025,.585,.30,.155,'First-listed-author\naffiliation-country evidence\nAccepted paper–country credits','#df8500','#fff5e4')
    box(.35,.585,.30,.155,'Observed new entrants\nFirst-listed in entry year\n12,094 people','#388e3c','#eaf5ea')
    box(.675,.585,.30,.155,'Complete exact +2 follow-up\n2018–2022 cohorts\n6,704 people','#7e57a0','#f1e8f6')
    box(.025,.10,.30,.405,'RQ1: production-adjusted representation\n\nPrimary: country-level API aggregates\nAPI query-target count: 864,215\n\nSeparate work-level sensitivities\nand denominator missingness bounds\nWork-level data used: 863,752\n\n12 aggregate + 32 work-level specs','#df8500','#fff0d6',11.1)
    box(.35,.335,.30,.155,'Prior-title portfolio observed\n9,639 people (79.7%)','#388e3c','#eaf5ea')
    box(.35,.10,.30,.155,'RQ2: title-portfolio distance\nDiagnostic D1: mean team size ≥ 2\n9,521 people','#388e3c','#d4edce',12)
    box(.675,.335,.30,.155,'Distance observed: 5,207\nMean team size ≥ 2\nPrimary sample: 5,122','#7e57a0','#f1e8f6')
    box(.675,.10,.30,.155,'RQ3: mutually exclusive\nexact +2 pathways','#7e57a0','#e2c9ed')
    arrow((.38,.802),(.175,.75)); arrow((.50,.802),(.50,.75))
    arrow((.656,.663),(.667,.663))
    for x,ends in [(.175,[(.574,.516)]),(.50,[(.574,.501),(.324,.266)]),(.825,[(.574,.501),(.324,.266)])]:
        for a,b in ends: arrow((x,a),(x,b))
    ax.text(.5,.041,'API query-target and work-level counts differ from country-credit share denominators.',
            ha='center',va='center',fontsize=10,color='#455a64')
    ax.text(.5,.016,'Arrows show data construction and sample restriction, not causal pathways.',
            ha='center',va='center',fontsize=10,color='#455a64')
    fig.subplots_adjust(left=.015,right=.985,bottom=.02,top=.98)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for patch,label in boxes:
        p=patch.get_window_extent(renderer); t=label.get_window_extent(renderer)
        assert t.x0>=p.x0 and t.x1<=p.x1 and t.y0>=p.y0 and t.y1<=p.y1, label.get_text()
    for ext in ['png','pdf','svg']:
        fig.savefig(OUT/f'fig1.{ext}',dpi=200,facecolor='white')
    plt.close(fig)
if __name__=='__main__': make_figure()

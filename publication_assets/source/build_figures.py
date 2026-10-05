"""Render typography changes from frozen inputs; never refit a model."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import sys

import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
PROJECT = ROOT/'ai_geographic_entry'
FIG = OUT/'figures'
FIG.mkdir(exist_ok=True)
DIST = PROJECT/'outputs/epj_distance_based_v10/tables'
PILOT = PROJECT/'outputs/epj_origin_feasibility_v01/private/epj_specter2_title_distance_pilot_private.csv'
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
    fig,ax=plt.subplots(figsize=(4.91,3.45))
    ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    def box(x,y,w,h,color,title,lines):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.008',
                                   ec=color,fc=matplotlib.colors.to_rgba(color,.12),lw=.7))
        ax.text(x+w/2,y+h-.045,title,ha='center',va='top',fontsize=8.5,weight='bold')
        ax.text(x+w/2,y+h-.13,lines,ha='center',va='top',fontsize=7.5,linespacing=1.32)
    box(.16,.815,.68,.155,BLUE,'ICML and NeurIPS, 2018–2024','26,872 accepted papers')
    box(.015,.045,.29,.59,'#DF8500','Representation',
        'Accepted-paper shares\nvs observed AI output\n\nPaper–country credits\n44 specifications\n\nProduction-adjusted\nrepresentation\nRQ1')
    box(.355,.045,.29,.59,'#388E3C','Entry',
        'Observed new entrants\n12,094 researchers\n\nPrior-title histories\n9,639 (79.7%)\n\nPrior-to-entry\ntitle-portfolio distance\nRQ2, Diagnostic D1')
    box(.695,.045,.29,.59,'#7E57A0','Persistence',
        'Complete +2 follow-up\n2018–2022 entry years\n6,704 researchers\n\nPrimary analytic sample\n5,122\n\nReappearance at +2\nRQ3')
    ax.plot([.5,.5],[.815,.72],color='#8B8B8B',lw=.65)
    ax.plot([.16,.84],[.72,.72],color='#8B8B8B',lw=.65)
    for x in [.16,.5,.84]:
        ax.add_patch(FancyArrowPatch((x,.72),(x,.64),arrowstyle='->',mutation_scale=7,color='#8B8B8B',lw=.65))
    ax.add_patch(FancyArrowPatch((.655,.695),(.687,.695),arrowstyle='->',mutation_scale=7,color='#8B8B8B',lw=.65))
    ax.text(.665,.755,'Follow-up subset',ha='center',fontsize=7.5,color='#555555')
    fig.subplots_adjust(left=.015,right=.985,bottom=.01,top=.99)
    check_and_save(fig,'fig1')


def figure2():
    path=PROJECT/'EPJDS_v06_assets/build_figure2_v06.py'
    spec=importlib.util.spec_from_file_location('frozen_figure2',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    mod.OUT=FIG
    source=path.read_text()
    node=next(x for x in ast.parse(source).body if isinstance(x,ast.FunctionDef) and x.name=='main')
    code=ast.get_source_segment(source,node)
    code=code[:code.index('    fig, axes = plt.subplots')]
    start=code.index('    note = fig.text(.5, .010,')
    stop=code.index('    assert_within_canvas',start)
    code=code[:start]+code[stop:]
    code=code.replace('[legA, legB, legC, note,','[legA, legB, legC,')
    ns=mod.__dict__;exec(code,ns);ns['main']()
    doc=fitz.open(FIG/'fig2.pdf');page=doc[0]
    rectangles=[fitz.Rect(x[:4]) for x in page.get_text('blocks')]
    rectangles += [x['rect'] for x in page.get_drawings()
                   if not (x['rect'].width>page.rect.width-1 and x['rect'].height>page.rect.height-1)]
    bottom=max(r.y1 for r in rectangles)+5
    clipped=fitz.open();p=clipped.new_page(width=page.rect.width,height=bottom)
    p.show_pdf_page(p.rect,doc,0,clip=fitz.Rect(0,0,page.rect.width,bottom))
    clipped.save(FIG/'fig2_crop.pdf',garbage=4,deflate=True)
    doc.close();clipped.close();(FIG/'fig2_crop.pdf').replace(FIG/'fig2.pdf')
    with fitz.open(FIG/'fig2.pdf') as d:
        assert 'PRI = 1 denotes' not in d[0].get_text()
        d[0].get_pixmap(matrix=fitz.Matrix(300/72,300/72)).save(FIG/'fig2.png')


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
    plt.rcParams.update(STYLE)
    inputs=[PILOT]+[DIST/x for x in ['rq1_year_distance_descriptive.csv','rq1_year_adjusted_mean_distance.csv',
                                    'rq3_exact_plus2_probabilities.csv','rq3_exact_plus2_contrasts.csv']]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    figure1();figure2();plt.rcParams.update(STYLE);figure3();figure4()
    assert hashes=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    report={'source_pdf':'Beyond_Paper_Count (6).pdf','models_refit':False,'original_inputs_unchanged':True,
            'figure2_footer_removed':True,'figures':{},'frozen_input_hashes':hashes}
    print_width=372*72/72.27*.95
    for n in range(1,5):
        with fitz.open(FIG/f'fig{n}.pdf') as d:
            page=d[0]
            spans=[s for b in page.get_text('dict')['blocks'] for l in b.get('lines',[]) for s in l['spans'] if s['text'].strip()]
            smallest=min(s['size'] for s in spans)*print_width/page.rect.width
            assert smallest>=7,(n,smallest)
            assert all(page.rect.contains(fitz.Rect(s['bbox'])) for s in spans)
            assert page.rect.height*print_width/page.rect.width<225*72/25.4
            report['figures'][str(n)]={'file':f'fig{n}.pdf','minimum_font_pt_at_0.95_textwidth':round(smallest,3),
                                      'pdf_width_pt':page.rect.width,'pdf_height_pt':page.rect.height}
    (OUT/'figure_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='frozen_input_hashes'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()

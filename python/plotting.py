import matplotlib
import matplotlib.pyplot as plt
import pylab as plt
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from xgboost import plot_importance
import numpy as np
import seaborn as sns
import ROOT
from array import array
import pandas as pd

def DrawROCCurve(act,pred,wt,output="roc_curve"):

  fpr, tpr, threshold = metrics.roc_curve(act, pred,sample_weight=wt)
  roc_auc = roc_auc_score(act, pred, sample_weight=wt)
  
  plt.title('ROC')
  plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
  print "AUC Score:", roc_auc
  plt.legend(loc = 'lower right')
  plt.plot([0, 1], [0, 1],'r--')
  plt.xlim([0, 1])
  plt.ylim([0, 1])
  plt.ylabel('True Positive Rate')
  plt.xlabel('False Positive Rate')
  plt.draw()
  plt.savefig("plots/"+output+".pdf")
  print "plots/"+output+".pdf created"
  plt.close()


def DrawDistributions(signal_dataframe,bkg_dataframe,variable,xrange,nbins,channel="channel",location="plots/Distributions/"):
    x_lower_bound = xrange[0]
    x_upper_bound = xrange[1]
    plt.figure()
    sns.distplot(signal_dataframe['{}'.format(variable)],  kde=True, hist_kws={"color": "k","range": [x_lower_bound,x_upper_bound]},kde_kws={"color": "k","clip":[x_lower_bound,x_upper_bound]}, label='Signal',bins = nbins)
    sns.distplot(bkg_dataframe['{}'.format(variable)],  kde=True,hist_kws={"color": "g","range": [x_lower_bound,x_upper_bound]},kde_kws={"color": "g","clip":[x_lower_bound,x_upper_bound]}, label='bkg',bins = nbins)
    plt.legend(prop={'size': 12})
    plt.title('{}_{}'.format(channel,variable))
    plt.xlabel('{}'.format(variable))
    plt.ylabel('Density')
    plt.xlim(x_lower_bound,x_upper_bound)
    plt.savefig("{}{}_{}.png".format(location,channel,variable))
    plt.close()


def DrawMultipleROCCurves(act,pred,wt,output="roc_curve",name="ROC"):

  for key,val in act.iteritems():
    fpr, tpr, threshold = metrics.roc_curve(act[key], pred[key], sample_weight=wt[key])
    roc_auc = roc_auc_score(act[key], pred[key], sample_weight=wt[key])
    #print key,"AUC Score:", roc_auc
    plt.plot(fpr, tpr, label = r"{}".format(key) +', AUC = %0.4f' % roc_auc)
  plt.title(name)
  plt.legend(loc = 'lower right',fontsize='x-small')
  plt.plot([0, 1], [0, 1],'r--')
  plt.xlim([0, 1])
  plt.ylim([0, 1])
  plt.ylabel('True Positive Rate')
  plt.xlabel('False Positive Rate')
  plt.draw()
  plt.savefig("plots/"+output+".pdf")
  print "plots/"+output+".pdf created"
  plt.close()

def DrawMultipleROCCurvesOnOnePage(act_dict,pred_dict,wt_dict,output="combined_roc_curves",len_x=3,len_y=3):
  plt.rcParams.update({'font.size': 3})
  fig, axs = plt.subplots(len_x,len_y)
  plt.subplots_adjust(left=0.05, 
                      bottom=0.05, 
                      right=0.95, 
                      top=0.95, 
                      wspace=0.2, 
                      hspace=0.3)
  i = 0
  j = 0
  for name, act in act_dict.iteritems():
    ind = 0
    max_roc_auc = 0
    max_roc_auc_ind = 0
    for key,val in act.iteritems():
      fpr, tpr, threshold = metrics.roc_curve(act_dict[name][key], pred_dict[name][key], sample_weight=wt_dict[name][key])
      roc_auc = roc_auc_score(act_dict[name][key], pred_dict[name][key], sample_weight=wt_dict[name][key])
      axs[i,j].plot(fpr, tpr, label = r"${}$".format(key) +', AUC = %0.4f' % roc_auc)
      if roc_auc > max_roc_auc: 
        max_roc_auc = 1*roc_auc
        max_roc_auc_ind = 1*ind
      ind += 1
    axs[i,j].set_title(r"{}".format(name))
    leg = axs[i,j].legend(loc = 'lower right')
    axs[i,j].plot([0, 1], [0, 1],'r--')
    axs[i,j].set_xlim([0, 1])
    axs[i,j].set_ylim([0, 1])
    axs[i,j].set_ylabel('True Positive Rate')
    axs[i,j].set_xlabel('False Positive Rate')
    for ind_leg, text in enumerate(leg.get_texts()):
      if max_roc_auc_ind == ind_leg:
        plt.setp(text, color = 'r')
    i += 1
    if i % len_x == 0:
      i = 0
      j += 1
  fig.savefig("plots/"+output+".pdf")
  print "plots/"+output+".pdf created"

def DrawBDTScoreDistributions(pred_dict,output="bdt_score",bins=100):

  for key, val in pred_dict.items():
    _, bins, _ = plt.hist(val["preds"], weights=val["weights"] ,bins=bins, histtype='step', label=key)
  plt.xlabel("BDT Score")
  plt.xlim(bins[0], bins[-1])
  plt.legend(loc='best')
  plt.draw()
  plt.savefig("plots/"+output+".pdf")
  plt.close()
  print "plots/"+output+".pdf created"
  
def DrawVarDistribution(df,n_classes,variable,xlim_low,xlim_high,output="var_dist",bin_edges=100):
  if n_classes == 2:
    c0 = df.loc[(df.loc[:,"y"]==0)]
    c1 = df.loc[(df.loc[:,"y"]==1)]
    
    wt0 = c0.loc[:,"weights"]
    wt1 = c1.loc[:,"weights"]
    
    val0 = c0.loc[:,variable]
    val1 = c1.loc[:,variable]
 
    dict_ = {"cat1":{"preds":val0,"weights":wt0},"cat2":{"preds":val1,"weights":wt1}}

  else:
    c0 = df.loc[(df.loc[:,"y"]==0)]
    c1 = df.loc[(df.loc[:,"y"]==1)]
    c2 = df.loc[(df.loc[:,"y"]==2)]
    
    wt0 = c0.loc[:,"weights"]
    wt1 = c1.loc[:,"weights"]
    wt2 = c2.loc[:,"weights"]
    
    val0 = c0.loc[:,variable]
    val1 = c1.loc[:,variable]
    val2 = c2.loc[:,variable]
    
    dict_ = {"cat1":{"preds":val0,"weights":wt0},"cat2":{"preds":val1,"weights":wt1},"cat3":{"preds":val2,"weights":wt2}}
    
  for key, val in dict_.items():
    _, bins, _ = plt.hist(val["preds"], weights=val["weights"] ,bins=bin_edges, histtype='step', label=key)
    
  plt.xlabel(variable)
  plt.xlim(xlim_low,xlim_high)
  plt.legend(loc='best')
  plt.draw()
  plt.savefig("plots/"+output+".pdf")
  plt.close()
  print "plots/"+output+".pdf created"

def DrawFeatureImportance(model,imp_type,output="feature_importance"):
  ax = plot_importance(model, importance_type = imp_type, xlabel = imp_type)
  ax.tick_params(axis='y', labelsize=5)
  ax.figure.savefig("plots/"+output+".pdf")
  print "plots/"+output+".pdf created"

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def DrawConfusionMatrix(y_test,preds,wt_test,label,output="confusion_matrix"):
  cm = confusion_matrix(y_test,preds,sample_weight=wt_test)
  new_cm = []
  for ind, i in enumerate(cm):
    norm = sum(i)
    new_cm.append([])
    for j in i: new_cm[ind].append(j/norm)

  import seaborn as sns
  
  ax = sns.heatmap(new_cm, annot=True, cmap='Blues')
  
  ax.set_title('');
  ax.set_xlabel('Predicted Label')
  ax.set_ylabel('True Label');
  
  ## Ticket labels - List must be in alphabetical order
  ax.xaxis.set_ticklabels(label)
  ax.yaxis.set_ticklabels(label)
  
  plt.savefig("plots/"+output+".pdf")
  print "plots/"+output+".pdf created"
  plt.close()


def DrawTitle(pad, text, align, scale=1):
    pad_backup = ROOT.gPad
    pad.cd()
    t = pad.GetTopMargin()
    l = pad.GetLeftMargin()
    r = pad.GetRightMargin()

    pad_ratio = (float(pad.GetWh()) * pad.GetAbsHNDC()) / \
        (float(pad.GetWw()) * pad.GetAbsWNDC())
    if pad_ratio < 1.:
        pad_ratio = 1.

    textSize = 0.6
    textOffset = 0.2

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextAngle(0)
    latex.SetTextColor(ROOT.kBlack)
    latex.SetTextFont(42)
    latex.SetTextSize(textSize * t * pad_ratio * scale)

    y_off = 1 - t + textOffset * t
    if align == 1:
        latex.SetTextAlign(11)
    if align == 1:
        latex.DrawLatex(l, y_off, text)
    if align == 2:
        latex.SetTextAlign(21)
    if align == 2:
        latex.DrawLatex(l + (1 - l - r) * 0.5, y_off, text)
    if align == 3:
        latex.SetTextAlign(31)
    if align == 3:
        latex.DrawLatex(1 - r, y_off, text)
    pad_backup.cd()


def PlotDistributionComparison(x_label,y_label,dist_0,dist_0_name,dist_1,dist_1_name,output_folder,save_name,logx=False,title_left="",title_right="",fillcolour=38, extra_dist=None):
  ROOT.gROOT.SetBatch(ROOT.kTRUE)
  dist_0_divide = dist_0.Clone()
  for i in range(0,dist_0_divide.GetNbinsX()+2): dist_0_divide.SetBinError(i,0)

  if extra_dist != None:
    dist_0_alt = extra_dist.Clone()
    dist_0_alt_divide = dist_0_alt.Clone()
    for i in range(0,dist_0_alt_divide.GetNbinsX()+2): dist_0_alt_divide.SetBinError(i,0)

  dist_1_ratio = dist_1.Clone()
  dist_1_ratio.Divide(dist_0_divide)

  dist_0_ratio = dist_0.Clone()
  dist_0_ratio.Divide(dist_0_divide)

  if extra_dist != None:
    dist_1_alt_ratio = dist_1.Clone()
    dist_1_alt_ratio.Divide(dist_0_alt_divide)

  c = ROOT.TCanvas('c','c',600,600)

  pad1 = ROOT.TPad("pad1","pad1",0,0.37,1,1)
  pad1.SetBottomMargin(0.02)
  pad1.SetLeftMargin(0.12)
  if logx: pad1.SetLogx()
  pad1.Draw()
  pad1.cd()

  dist_0.Draw("hist SAME")
  dist_0.SetLineColor(1)
  dist_0.SetFillColor(fillcolour)
  dist_0.SetStats(0)
  dist_0.GetXaxis().SetTitle(x_label)
  dist_0.GetYaxis().SetTitle(y_label)
  dist_0.GetYaxis().SetTitleOffset(1)
  dist_0.GetYaxis().SetTitleSize(0.06)
  dist_0.GetYaxis().SetLabelSize(0.05)
  dist_0.GetXaxis().SetLabelSize(0)

  dist_0_uncert = dist_0.Clone()
  dist_0_uncert.SetMarkerSize(0)
  dist_0_uncert.SetFillColorAlpha(12,0.5)
  dist_0_uncert.SetLineWidth(0)
  dist_0_uncert.Draw("e2 same")

  dist_1.Draw("E SAME")
  dist_1.SetMarkerColor(1)
  dist_1.SetLineColor(1)
  dist_1.SetMarkerStyle(19)

  if extra_dist != None:
    dist_0_alt.Draw("SAME")
    #dist_0_alt.SetMarkerColor(2)
    dist_0_alt.SetLineColor(2)
    dist_0_alt.SetLineWidth(2)
    #dist_0_alt.SetMarkerStyle(19)

  l = ROOT.TLegend(0.6,0.65,0.88,0.85);
  l.SetBorderSize(0)
  l.AddEntry(dist_0,dist_0_name,"f")
  l.AddEntry(dist_1,dist_1_name,"lep")
  l.AddEntry(dist_0_alt,"norm x fail","lep")
  l.Draw()

  c.cd()
  pad2 = ROOT.TPad("pad2","pad2",0,0.05,1,0.35)
  pad2.SetLeftMargin(0.12)
  pad2.SetTopMargin(0.03)
  pad2.SetBottomMargin(0.3)
  if logx: pad2.SetLogx()
  pad2.Draw()
  pad2.cd()

  ratio_line = ROOT.TLine(dist_0.GetBinLowEdge(1),1,dist_0.GetBinLowEdge(dist_0.GetNbinsX()+1),1)
  dist_0_ratio.SetMarkerSize(0)
  dist_0_ratio.SetFillColorAlpha(12,0.5)
  dist_0_ratio.SetLineWidth(0)
  dist_0_ratio.SetAxisRange(0.5,1.5,'Y')
  #dist_0_ratio.SetAxisRange(0.8,1.2,'Y')
  dist_0_ratio.GetYaxis().SetNdivisions(4)
  dist_0_ratio.SetStats(0)
  dist_0_ratio.GetXaxis().SetLabelSize(0.1)
  dist_0_ratio.GetYaxis().SetLabelSize(0.1)
  dist_0_ratio.GetXaxis().SetTitle(x_label)
  dist_0_ratio.GetYaxis().SetTitle("Ratio")
  dist_0_ratio.GetYaxis().SetTitleColor(1)
  dist_0_ratio.GetYaxis().SetTitleSize(0.12)
  dist_0_ratio.GetYaxis().SetTitleOffset(0.4)
  dist_0_ratio.GetXaxis().SetTitleSize(0.12)
  dist_0_ratio.GetXaxis().SetTitleOffset(1.2)
  if logx:
    dist_0_ratio.GetXaxis().SetMoreLogLabels()
    dist_0_ratio.GetXaxis().SetNoExponent()

  dist_1_ratio.SetMarkerColor(1)
  dist_1_ratio.SetLineColor(1)
  dist_1_ratio.SetMarkerStyle(19)
  if extra_dist != None:
    dist_1_alt_ratio.SetLineColor(2)
    dist_1_alt_ratio.SetMarkerColor(2)  
    #dist_1_alt_ratio.SetMarkerStyle(19)

  ratio_line_up = ROOT.TLine(dist_0.GetBinLowEdge(1),1.5,dist_0.GetBinLowEdge(dist_0.GetNbinsX()+1),1.5)
  ratio_line_down = ROOT.TLine(dist_0.GetBinLowEdge(1),0.5,dist_0.GetBinLowEdge(dist_0.GetNbinsX()+1),0.5)
  ratio_line.SetLineStyle(3)
  ratio_line_up.SetLineStyle(3)
  ratio_line_down.SetLineStyle(3)
  
  dist_0_ratio.Draw("e2")
  if extra_dist != None: dist_1_alt_ratio.Draw("e same")
  ratio_line.Draw("l same")
  ratio_line_up.Draw("l same")
  ratio_line_down.Draw("l same")
  dist_1_ratio.Draw("E same")

  DrawTitle(pad1, title_left, 1, scale=1)
  DrawTitle(pad1, title_right, 3, scale=1)

  c.Update()
  name = '%(output_folder)s/%(save_name)s.pdf' % vars()
  c.SaveAs(name)
  c.Close()

def ReplaceName(name):
  replace_dict = {
    "tttt":"#tau#tau#tau#tau",
    "ttt":"#tau#tau#tau",
    "mttt":"#mu#tau#tau#tau",
    "ettt":"e#tau#tau#tau",
    "emtt":"e#mu#tau#tau",
    "mmtt":" #mu#mu#tau#tau",
    "eett":"ee#tau#tau",
    "tt":"#tau_{h}#tau_{h}",
    "mt":"#mu#tau_{h}",
    "et":"e#tau_{h}",
    "2016_preVFP":"2016-preVFP: 19.5 fb^{-1} (13 TeV)",
    "2016_postVFP":"2016-postVFP: 16.8 fb^{-1} (13 TeV)",
    "2017":"2017: 41.5 fb^{-1} (13 TeV)",
    "2018":"2018: 59.7 fb^{-1} (13 TeV)",
    "all_years":"138 fb^{-1} (13 TeV)",
#    "m_vis": "m_{vis} (GeV)",
#    "met": "E_{T}^{miss} (GeV)",
#    "n_jets": "N_{jets}",
#    "n_deepbjets": "N_{b-jets}",
#    "tau_decay_mode_1": "#tau_{1} DM",
#    "tau_decay_mode_2": "#tau_{2} DM",
#    "mvis_12": "m_{vis}^{12} (GeV)",
#    "mvis_13": "m_{vis}^{13} (GeV)",
#    "mvis_14": "m_{vis}^{14} (GeV)",
#    "mvis_23": "m_{vis}^{23} (GeV)",
#    "mvis_24": "m_{vis}^{24} (GeV)",
#    "mvis_34": "m_{vis}^{34} (GeV)",
#    "pt_1": "p_{T}^{1} (GeV)",
#    "pt_2": "p_{T}^{2} (GeV)",
#    "pt_3": "p_{T}^{3} (GeV)",
#    "pt_4": "p_{T}^{4} (GeV)",
#    "svfit_mass": "m_{#tau#tau} (GeV)",
#    "mt_tot": "m_{T}^{tot} (GeV)",
#    "mva_dm_1": "#tau_{1} MVA DM",
#    "mva_dm_2": "#tau_{2} MVA DM",
#    "eta_1": "#eta_{1}",
#    "eta_2": "#eta_{2}",
#    "dN/dm_vis": "dN/dm_{vis} (1/GeV)",
#    "dN/dmet": "dN/dE_{T}^{miss} (1/GeV)",
#    "dN/dn_jets": "dN/dN_{jets}",
#    "dN/dn_deepbjets": "dN/dN_{b-jets}",
#    "dN/dtau_decay_mode_1": "dN/d#tau_{1} DM",
#    "dN/dtau_decay_mode_2": "dN/d#tau_{2} DM",
#    "dN/dmvis_12": "dN/dm_{vis}^{12} (1/GeV)",
#    "dN/dmvis_13": "dN/dm_{vis}^{13} (1/GeV)",
#    "dN/dmvis_14": "dN/dm_{vis}^{14} (1/GeV)",
#    "dN/dmvis_23": "dN/dm_{vis}^{23} (1/GeV)",
#    "dN/dmvis_24": "dN/dm_{vis}^{24} (1/GeV)",
#    "dN/dmvis_34": "dN/dm_{vis}^{34} (1/GeV)",
#    "dN/dpt_1": "dN/dp_{T}^{1} (1/GeV)",
#    "dN/dpt_2": "dN/dp_{T}^{2} (1/GeV)",
#    "dN/dpt_3": "dN/dp_{T}^{3} (1/GeV)",
#    "dN/dpt_4": "dN/dp_{T}^{4} (1/GeV)",
#    "dN/dsvfit_mass": "dN/dm_{#tau#tau} (1/GeV)",
#    "dN/dmt_tot": "dN/dm_{T}^{tot} (1/GeV)",
#    "dN/dmva_dm_1": "dN/d#tau_{1} MVA DM",
#    "dN/dmva_dm_2": "dN/d#tau_{2} MVA DM",
#    "dN/deta_1": "dN/d#eta_{1}",
#    "dN/deta_2": "dN/d#eta_{2}",
    "tau_decay_mode_1": "HPS Decay Mode",
    "pt_1": "p_{T} (GeV)",
    "deepTauVsJets_iso_2": "Alt #tau_{1} DeepTauVsJets Score",
    "deepTauVsJets_iso_3": "Alt #tau_{2} DeepTauVsJets Score",
    "deepTauVsJets_iso_4": "Alt #tau_{3} DeepTauVsJets Score",
    "eta_1": "#eta",
    "jet_pt_1_divide_pt_1": "Tau-p_{T} / jet-p_{T}",
    "mt_1": "m_{T} (GeV)",
    "n_jets": "N_{jets}",
    "n_bjets": "N_{b-jets}",
    "pt_1_times_cosh_eta_1": "Absolute Momentum (GeV)",
    "q_1": "Charge",
    "q_sum": "Total Charge",
    "rho": "#rho",
    }

  for k, v in replace_dict.items():
    if "GeV" in k:
      if " " in v.replace(" (GeV)",""):
        replace_dict["dN/d"+k] = "dN/d("+v.replace(" (GeV)","")+") (1/GeV)"
      else:
        replace_dict["dN/d"+k] = "dN/d"+v.replace(" (GeV)","")+" (1/GeV)"
    else:
      if " " in v.replace(" (GeV)",""):
        replace_dict["dN/d"+k] = "dN/d("+v+")"
      else:
        replace_dict["dN/d"+k] = "dN/d"+v

  if name in replace_dict.keys():
    return replace_dict[name]
  else:
    return name 

def FindRebinning(hist,BinThreshold=100,BinUncertFraction=0.5):

  # getting binning
  binning = []
  for i in range(1,hist.GetNbinsX()+2):
    binning.append(hist.GetBinLowEdge(i))

  # left to right
  finished = False
  k = 0
  while finished == False and k < 1000:
    k += 1
    for i in range(1,hist.GetNbinsX()):
      if hist.GetBinContent(i) != 0: uncert_frac = hist.GetBinError(i)/hist.GetBinContent(i)
      else: uncert_frac = BinUncertFraction+1
      if uncert_frac > BinUncertFraction and hist.GetBinContent(i) < BinThreshold:
        #binning.remove(hist.GetBinLowEdge(i+1))
        binning.remove(min(binning, key=lambda x:abs(x-hist.GetBinLowEdge(i+1))))
        hist = RebinHist(hist,binning)
        break
      elif i+1 == hist.GetNbinsX():
        finished = True

  # right to left
  finished = False
  k = 0
  while finished == False and k < 1000:
    k+= 1
    for i in reversed(range(2,hist.GetNbinsX()+1)):
      if hist.GetBinContent(i) != 0: uncert_frac = hist.GetBinError(i)/hist.GetBinContent(i)
      else: uncert_frac = BinUncertFraction+1
      if uncert_frac > BinUncertFraction and hist.GetBinContent(i) < BinThreshold:
        #binning.remove(hist.GetBinLowEdge(i))
        binning.remove(min(binning, key=lambda x:abs(x-hist.GetBinLowEdge(i))))
        hist = RebinHist(hist,binning)
        break
      elif i == 2:
        finished = True

  return binning

def RebinHist(hist,binning):
  ROOT.TH1.AddDirectory(ROOT.kFALSE)
  # getting initial binning
  initial_binning = []
  for i in range(1,hist.GetNbinsX()+2):
    initial_binning.append(hist.GetBinLowEdge(i))

  new_binning = array('f', map(float,binning))
  hout = ROOT.TH1D(hist.GetName(),'',len(new_binning)-1, new_binning)
  hout.SetDirectory(0)
  for i in range(1,hout.GetNbinsX()+1):
    for j in range(1,hist.GetNbinsX()+1):
      if hist.GetBinCenter(j) > hout.GetBinLowEdge(i) and hist.GetBinCenter(j) < hout.GetBinLowEdge(i+1):
        new_val = hout.GetBinContent(i)+hist.GetBinContent(j)
        new_err = (hout.GetBinError(i)**2+hist.GetBinError(j)**2)**0.5
        hout.SetBinContent(i,new_val)
        hout.SetBinError(i,new_err)
  return hout

def DrawReweightPlots(df_x,df_y, x_label, y_label, plot_name="reweight_plot", title_left="", title_right="",ignore_quantile=0.99):
  ROOT.gROOT.SetBatch(ROOT.kTRUE)
  c = ROOT.TCanvas('c','c',600,600)
  g = ROOT.TGraph()
  x_name = df_x.name
  y_name = df_y.name
  df_x.reset_index(drop=True, inplace=True) 
  df_y.reset_index(drop=True, inplace=True)
  df = pd.concat([df_x,df_y],axis=1,names=[x_name,y_name])
  x_quant_up = df_x.quantile(ignore_quantile)
  y_quant_up = df_y.quantile(ignore_quantile)
  x_quant_down = df_x.quantile(1-ignore_quantile)
  y_quant_down = df_y.quantile(1-ignore_quantile)
  for index, row in df.iterrows():
    if row[df_x.name] > x_quant_down and row[df_y.name] > y_quant_down and row[df_x.name] < x_quant_up and row[df_y.name] < y_quant_up:
      g.SetPoint(g.GetN(),row[df_x.name],row[df_y.name])
  pad = ROOT.TPad("pad","pad",0,0,1,1)
  pad.Draw()
  pad.cd()
  pad.SetBottomMargin(0.12)
  pad.SetLeftMargin(0.14)

  g.Draw("ap")
  g.SetMarkerColorAlpha(4,1.0)
  g.GetXaxis().SetTitle(ReplaceName(x_label))
  g.GetXaxis().SetTitleSize(0.04)
  g.GetXaxis().SetTitleOffset(1.3)
  g.GetYaxis().SetTitle(ReplaceName(y_label))
  g.GetYaxis().SetTitleSize(0.04)

  DrawTitle(pad, ReplaceName(title_left), 1, scale=0.7)
  DrawTitle(pad, ReplaceName(title_right), 3, scale=0.7)

  c.Update()
  name = 'plots/%(plot_name)s.pdf' % vars()
  c.SaveAs(name)
  c.Close()

def DrawColzReweightPlots(df_x, df_y, wt, x_label, y_label, plot_name="reweight_colz_plot", title_left="", title_right="",ignore_quantile=0.99, num_of_bins=50,logy=True):
  ROOT.gROOT.SetBatch(ROOT.kTRUE)
  c = ROOT.TCanvas('c','c',600,600)
  df_x.reset_index(drop=True, inplace=True)
  df_y.reset_index(drop=True, inplace=True)
  wt.reset_index(drop=True, inplace=True)
  df = pd.concat([wt,df_x,df_y],axis=1,names=[wt.name,df_x.name,df_y.name])
  x_quant_up = df_x.quantile(ignore_quantile)
  y_quant_up = df_y.quantile(ignore_quantile)
  x_quant_down = df_x.quantile(1-ignore_quantile)
  y_quant_down = df_y.quantile(1-ignore_quantile)
  x_bins = array('f', map(float,np.linspace(x_quant_down,x_quant_up,num=num_of_bins).tolist()))
  if not logy:
    y_bins = array('f', map(float,np.linspace(y_quant_down,y_quant_up,num=num_of_bins).tolist()))
  else:
    y_bins = array('f', map(float,np.logspace(np.log10(y_quant_down),np.log10(y_quant_up),num=num_of_bins).tolist()))

  hout = ROOT.TH2D('hout','',len(x_bins)-1, x_bins,len(y_bins)-1, y_bins)
  for index, row in df.iterrows():
    if row[df_x.name] > x_quant_down and row[df_y.name] > y_quant_down and row[df_x.name] < x_quant_up and row[df_y.name] < y_quant_up:
      hout.Fill(row[df_x.name],row[df_y.name],row[wt.name])
  pad = ROOT.TPad("pad","pad",0,0,1,1)
  pad.Draw()
  pad.cd()
  pad.SetBottomMargin(0.12)
  pad.SetLeftMargin(0.14)
  pad.SetRightMargin(0.2)
  if logy: pad.SetLogy()

  hout.SetStats(0)
  hout.Draw("colz")
  hout.GetXaxis().SetTitle(ReplaceName(x_label))
  hout.GetXaxis().SetTitleSize(0.04)
  hout.GetXaxis().SetTitleOffset(1.3)
  hout.GetYaxis().SetTitle(ReplaceName(y_label))
  hout.GetYaxis().SetTitleSize(0.04)
  hout.GetZaxis().SetTitle("Events")
  hout.GetZaxis().SetTitleOffset(1.5)
  hout.GetZaxis().SetTitleSize(0.04)

  DrawTitle(pad, ReplaceName(title_left), 1, scale=0.7)
  DrawTitle(pad, ReplaceName(title_right), 3, scale=0.7)

  c.Update()
  name = 'plots/%(plot_name)s.pdf' % vars()
  c.SaveAs(name)
  c.Close()

def DrawClosurePlots(df1, df2, df1_name, df2_name, var_name, var_binning, plot_name="closure_plot", title_left="", title_right="", fillcolour=38, df1_norm=None):
  if isinstance(var_binning,tuple):
    bins = array('f', map(float,[(float(var_binning[2]-var_binning[1])/var_binning[0])*i + float(var_binning[1]) for i in range(0,var_binning[0])]))
  elif isinstance(var_binning,list):
    bins = array('f', map(float,var_binning))
  hout = ROOT.TH1D('hout','',len(bins)-1, bins)
  for i in range(0,hout.GetNbinsX()+2): hout.SetBinError(i,0)
  hist1 = hout.Clone()
  hist2 = hout.Clone()
  if df1_norm != None: 
    hist3 = hout.Clone()
  else:
    hist3 = None

  # Fill not the right thing to do!
  for index, row in df1.iterrows():
    hist1.Fill(row[var_name], row["weights"])
    if df1_norm != None: hist3.Fill(row[var_name],df1_norm)
  for index, row in df2.iterrows():
    hist2.Fill(row[var_name], row["weights"])


  # rebin to find good binning
  #if hist1.GetEntries() < hist2.GetEntries():
  #  binning = FindRebinning(hist1,BinThreshold=100,BinUncertFraction=0.5)
  #else:
  #  binning = FindRebinning(hist2,BinThreshold=100,BinUncertFraction=0.5)

  #hist1 = RebinHist(hist1,binning)
  #hist2 = RebinHist(hist2,binning)

  if df1.loc[:,var_name].nunique() > 20:
    binning = FindRebinning(hist1,BinThreshold=100,BinUncertFraction=0.2)
    hist2 = RebinHist(hist2,binning)
    binning = FindRebinning(hist2,BinThreshold=100,BinUncertFraction=0.2)
    hist1 = RebinHist(hist1,binning)
    hist2 = RebinHist(hist2,binning)
    if df1_norm != None: hist3 = RebinHist(hist3,binning)

  hist1.Chi2Test(hist2, "p")

  hist1.Scale(1.0,"width")
  hist2.Scale(1.0,"width")
  if df1_norm != None: hist3.Scale(1.0,"width")

  PlotDistributionComparison(ReplaceName(var_name), ReplaceName("dN/d{}".format(var_name)),hist1, df1_name, hist2, df2_name, "plots", plot_name, title_left=ReplaceName(title_left), title_right=ReplaceName(title_right),fillcolour=fillcolour,extra_dist=hist3)

def DrawAverageReweightPlots(df1, var_name, var_binning, name, plot_name="reweight_ave_plot", title_left="", title_right="",logx=False, logy=False, compare_other_ff=None):
  ROOT.gROOT.SetBatch(ROOT.kTRUE)
  c = ROOT.TCanvas('c','c',600,600)
  if isinstance(var_binning,tuple):
    bins = array('f', map(float,[(float(var_binning[2]-var_binning[1])/var_binning[0])*i + float(var_binning[1]) for i in range(0,var_binning[0])]))
  elif isinstance(var_binning,list):
    bins = array('f', map(float,var_binning))
  hout = ROOT.TH1D('hout','',len(bins)-1, bins)
  for i in range(0,hout.GetNbinsX()+2): hout.SetBinError(i,0)
  hist1 = hout.Clone()
  hist2 = hout.Clone() 
  if compare_other_ff != None: hist3 = hout.Clone()

  for index, row in df1.iterrows():
    hist1.Fill(row[var_name], row["weights"]*row["reweights"])
    hist2.Fill(row[var_name], row["weights"])
    if compare_other_ff != None: hist3.Fill(row[var_name], row["weights"]*row[compare_other_ff])

  if df1.loc[:,var_name].nunique() > 20:
    binning = FindRebinning(hist1,BinThreshold=100,BinUncertFraction=0.2)
    hist2 = RebinHist(hist2,binning)
    binning = FindRebinning(hist2,BinThreshold=100,BinUncertFraction=0.2)
    hist1 = RebinHist(hist1,binning)
    hist2 = RebinHist(hist2,binning)
    if compare_other_ff != None: hist3 = RebinHist(hist3,binning)

  for i in range(0,hist2.GetNbinsX()+2): hist2.SetBinError(i,0)
  hist1.Divide(hist2)
  if compare_other_ff != None: hist3.Divide(hist2)

  pad = ROOT.TPad("pad","pad",0,0,1,1)
  pad.Draw()
  pad.cd()
  pad.SetBottomMargin(0.12)
  pad.SetLeftMargin(0.14)
  pad.SetRightMargin(0.2)
  if logy: pad.SetLogy()

  hist1.SetStats(0)
  hist1.Draw("E1")
  hist1.GetXaxis().SetTitle(ReplaceName(var_name))
  hist1.GetXaxis().SetTitleSize(0.04)
  hist1.GetXaxis().SetTitleOffset(1.3)
  hist1.GetYaxis().SetTitle("Average Reweight")
  hist1.GetYaxis().SetTitleSize(0.04)
  hist1.SetMarkerStyle(19) 
  hist1.SetMinimum(0.1) 

  if compare_other_ff != None:
    hist3.Draw("E1 same")
    hist3.SetMarkerStyle(3)
    hist3.SetMarkerColor(4)
    hist3.SetLineColor(4)

  l = ROOT.TLegend(0.18,0.65,0.38,0.85);
  l.SetBorderSize(0)
  l.AddEntry(hist1,"BDT F_{F}","lep")
  if compare_other_ff != None: l.AddEntry(hist3,compare_other_ff,"lep")
  l.Draw()

  DrawTitle(pad, ReplaceName(title_left), 1, scale=0.7)
  DrawTitle(pad, ReplaceName(title_right), 3, scale=0.7)

  c.Update()
  name = 'plots/%(plot_name)s.pdf' % vars()
  c.SaveAs(name)
  c.Close()

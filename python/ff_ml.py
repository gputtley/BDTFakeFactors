from UserCode.sig_vs_bkg_discriminator import reweighter
from UserCode.sig_vs_bkg_discriminator.plotting import DrawClosurePlots, DrawColzReweightPlots

class ff_ml():

  def __init__(self):
    self.raw_df1 = None
    self.raw_df2 = None
    self.raw_model = None
    self.alt_df1 = None
    self.alt_df2 = None
    self.alt_model = None
    self.corr_df1 = None
    self.corr_df2 = None
    self.corr_model = None
 
  def PlotReweights(model_name, sel):

  def PlotClosure(model_name, sel):


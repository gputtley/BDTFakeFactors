from hep_ml.reweight import GBReweighter
from hep_ml.metrics_utils import ks_2samp_weighted, compute_cdf
import copy
import itertools
import numpy as np
import os
import pickle as pkl
import json

class reweighter(GBReweighter):
  
  def __init__(self,
               n_estimators=40,
               learning_rate=0.2,
               max_depth=3,
               min_samples_leaf=200,
               loss_regularization=5.,
               gb_args=None):

    self.learning_rate = learning_rate
    self.n_estimators = n_estimators
    self.max_depth = max_depth
    self.min_samples_leaf = min_samples_leaf
    self.gb_args = gb_args
    self.loss_regularization = loss_regularization
    self.normalization = 1.0
    self.initial_normalization = 1.0
    self.column_names = []
    self.KS_value = None
    self.wt_function = []
    self.wt_function_var = []


  def norm_and_fit(self, original, target, original_weight, target_weight, cap_at=None):

    print "Original Weights Sum:", original_weight.sum()
    print "Target Weights Sum:", target_weight.sum()

    initial_original_weight = copy.deepcopy(original_weight)
    initial_target_weight = copy.deepcopy(target_weight)
    if self.wt_function != [] and self.wt_function_var != []:
      for ind, wt_function in enumerate(self.wt_function):
        print "Using weight scaling function for variable {}:".format(self.wt_function_var[ind])
        print self.wt_function[ind]
        original_weight *= original.loc[:,self.wt_function_var[ind]].apply(lambda x: eval(self.wt_function[ind]))
        target_weight *= target.loc[:,self.wt_function_var[ind]].apply(lambda x: eval(self.wt_function[ind]))

    self.column_names = original.keys()
    self.normalization = target_weight.sum()/original_weight.sum()
    self.initial_normalization = target_weight.sum()/original_weight.sum()
    self.fit(original, target, original_weight=self.normalization*original_weight, target_weight=target_weight)

    # renormalise after fit
    total_original_weights = np.multiply(self.predict_reweights(original,add_norm=False,cap_at=cap_at),initial_original_weight)
    self.normalization = initial_target_weight.sum()/total_original_weights.sum()
    print self.normalization

    #for i in range(0,10):
    #  total_original_weights = np.multiply(self.predict_reweights(original,add_norm=True,cap_at=cap_at),original_weight)
    #  self.normalization = self.normalization*target_weight.sum()/total_original_weights.sum()

    pred = self.predict_reweights(original,add_norm=True,cap_at=cap_at)

    print np.multiply(self.predict_reweights(original,add_norm=True,cap_at=cap_at),initial_original_weight).sum(), initial_target_weight.sum()

    print "Average Weight:", pred.sum()/len(pred)
 

  def predict_reweights(self, original, add_norm=True,cap_at=None):
    wts = self.predict_weights(original)
    if add_norm:
      new_wts = self.normalization*wts
    else:
      new_wts = wts
    if cap_at != None:
      new_wts = np.clip(new_wts,-cap_at,cap_at)
    return new_wts

  def dump_hyperparameters(self):
    if self.gb_args != None:
      hp = copy.deepcopy(self.gb_args) 
    else:
      hp = {}
    hp["learning_rate"] = self.learning_rate
    hp["n_estimators"] = self.n_estimators
    hp["max_depth"] = self.max_depth
    hp["min_samples_leaf"] = self.min_samples_leaf
    hp["loss_regularization"] = self.loss_regularization
    return hp

  def draw_distributions(self, original, target, original_weights, target_weights, columns=['pt_1']):
    plt.figure(figsize=[15, 12])
    for id, column in enumerate(columns, 1):
      xlim = np.percentile(np.hstack([target[column]]), [0.01, 99.99])
      plt.subplot(3, 3, id)
      plt.hist(original[column], weights=original_weights, range=xlim, bins=50, alpha=0.8, color="red", label="original")
      plt.hist(target[column], weights=target_weights, range=xlim, bins=50, alpha=0.8, color="blue", label="target")
      plt.title(column)
      plt.legend()
    plt.show()
  
  def ks_2samp_weighted_unnormalised(self, data1, data2, weights1, weights2):
    x = np.unique(np.concatenate([data1, data2]))
    inds1 = np.searchsorted(x, data1)
    inds2 = np.searchsorted(x, data2)
    w1 = np.bincount(inds1, weights=weights1, minlength=len(x))
    w2 = np.bincount(inds2, weights=weights2, minlength=len(x))
    F1 = compute_cdf(w1)
    F2 = compute_cdf(w2)
    return np.max(np.abs(F1 - F2))

  def KS(self, original, target, original_weights, target_weights, columns=['pt_1']):
    ks_dict = {}
    ks_total = 0
    for id, column in enumerate(columns, 1):
      ks_dict[column] = round(self.ks_2samp_weighted_unnormalised(original[column], target[column], weights1=original_weights, weights2=target_weights),6)
      ks_total += ks_dict[column]
    return round(float(ks_total)/len(columns),6), ks_dict

  def set_params(self,val):
    if "learning_rate" in val.keys():
      self.learning_rate = val["learning_rate"]
      del val["learning_rate"]
    if "n_estimators" in val.keys():
      self.n_estimators = val["n_estimators"]
      del val["n_estimators"]
    if "max_depth" in val.keys():
      self.max_depth = val["max_depth"]
      del val["max_depth"]
    if "min_samples_leaf" in val.keys():
      self.min_samples_leaf = val["min_samples_leaf"]
      del val["min_samples_leaf"]
    if "loss_regularization" in val.keys():
      self.loss_regularization = val["loss_regularization"]
      del val["loss_regularization"]
    self.gb_args = copy.deepcopy(val)

  def grid_search(self, original_train, target_train, original_train_weight, target_train_weight, original_test, target_test, original_test_weight, target_test_weight, param_grid={}, scoring_variables=["pt_1"],cap_at=None):
    keys, values = zip(*param_grid.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    lowest_KS = 9999999
    for ind, val in enumerate(permutations_dicts):
      unchanged_val = copy.deepcopy(val)
      self.set_params(val)
      self.norm_and_fit(original_train, target_train, original_train_weight, target_train_weight, cap_at=cap_at)
      test_reweights = self.predict_reweights(original_test,cap_at=cap_at)
      score = self.KS(original_test, target_test, np.multiply(original_test_weight,test_reweights), target_test_weight, columns=scoring_variables)
      print "Parameter grid:", unchanged_val
      print "KS score:", score
      print ""
      if score[0] < lowest_KS:
        lowest_KS = score[0]*1.0
        best_model = copy.deepcopy(self)
        best_hp = copy.deepcopy(unchanged_val)
    print "Best hyperparameters:", best_hp
    print "Lowest KS score:", lowest_KS
    self.KS_value = lowest_KS
    self = best_model 
    
  def grid_search_batch(self, name, original_train, target_train, original_train_weight, target_train_weight, original_test, target_test, original_test_weight, target_test_weight, param_grid={}, scoring_variables=["pt_1"], cap_at=None):
    from UserCode.BDTFakeFactors.batch import CreateJob,CreateBatchJob,SubmitBatchJob
    if not os.path.isdir("scan_batch"): os.system("mkdir scan_batch")
    if not os.path.isdir("scan_batch/dataframes"): os.system("mkdir scan_batch/dataframes")
    if not os.path.isdir("scan_batch/models"): os.system("mkdir scan_batch/models")
    if not os.path.isdir("scan_batch/jobs"): os.system("mkdir scan_batch/jobs")
    if not os.path.isdir("scan_batch/outputs"): os.system("mkdir scan_batch/outputs")
    if not os.path.isdir("scan_batch/hyperparameters"): os.system("mkdir scan_batch/hyperparameters")
    if not os.path.isdir("scan_batch/scores"): os.system("mkdir scan_batch/scores")

    # dump the dataframes to dataframes file
    original_train.to_pickle("scan_batch/dataframes/{}_original_train_dataframe.pkl".format(name))
    target_train.to_pickle("scan_batch/dataframes/{}_target_train_dataframe.pkl".format(name))
    original_train_weight.to_pickle("scan_batch/dataframes/{}_original_train_weight_dataframe.pkl".format(name))
    target_train_weight.to_pickle("scan_batch/dataframes/{}_target_train_weight_dataframe.pkl".format(name))
    original_test.to_pickle("scan_batch/dataframes/{}_original_test_dataframe.pkl".format(name))
    target_test.to_pickle("scan_batch/dataframes/{}_target_test_dataframe.pkl".format(name))
    original_test_weight.to_pickle("scan_batch/dataframes/{}_original_test_weight_dataframe.pkl".format(name))
    target_test_weight.to_pickle("scan_batch/dataframes/{}_target_test_weight_dataframe.pkl".format(name))

    # loop through hyperparameters
    keys, values = zip(*param_grid.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    for ind, val in enumerate(permutations_dicts):
      for k,v in val.items(): val[k] = [v]
      
      p_cmds = [
                "import pickle as pkl",
                "import pandas as pd",
                "import os",
                "import json",
                "from UserCode.BDTFakeFactors import reweighter",
                "original_train = pd.read_pickle('scan_batch/dataframes/{}_original_train_dataframe.pkl')".format(name),
                "target_train = pd.read_pickle('scan_batch/dataframes/{}_target_train_dataframe.pkl')".format(name),
                "original_train_weight = pd.read_pickle('scan_batch/dataframes/{}_original_train_weight_dataframe.pkl')".format(name),
                "target_train_weight = pd.read_pickle('scan_batch/dataframes/{}_target_train_weight_dataframe.pkl')".format(name),
                "original_test = pd.read_pickle('scan_batch/dataframes/{}_original_test_dataframe.pkl')".format(name),
                "target_test = pd.read_pickle('scan_batch/dataframes/{}_target_test_dataframe.pkl')".format(name),
                "original_test_weight = pd.read_pickle('scan_batch/dataframes/{}_original_test_weight_dataframe.pkl')".format(name),
                "target_test_weight = pd.read_pickle('scan_batch/dataframes/{}_target_test_weight_dataframe.pkl')".format(name),
                "rwter = reweighter.reweighter()",
                "rwter.grid_search(original_train, target_train, original_train_weight, target_train_weight, original_test, target_test, original_test_weight, target_test_weight, param_grid={}, scoring_variables={},cap_at={})".format(str(val),str(scoring_variables),cap_at),
                "pkl.dump(rwter,open('scan_batch/models/{}_{}.pkl', 'wb'))".format(name,ind),
                "with open('scan_batch/hyperparameters/{}_{}.json', 'w') as outfile: json.dump(rwter.dump_hyperparameters(), outfile)".format(name,ind),
                "score = str(rwter.KS_value)",
                "os.system('echo %(score)s &> scan_batch/scores/{}_{}.txt' % vars())".format(name,ind),
                ]
      CreateJob("scan_batch/jobs/"+name+"_"+str(ind)+".py",p_cmds)
      cmssw_base = os.getcwd().replace('src/UserCode/BDTFakeFactors','')
      CreateBatchJob("scan_batch/jobs/"+name+"_"+str(ind)+".sh",cmssw_base,["python scan_batch/jobs/{}".format(name+"_"+str(ind)+".py")])
      SubmitBatchJob("scan_batch/jobs/"+name+"_"+str(ind)+".sh")

  def collect_grid_search_batch(self, name):
    ind = 0
    for file in os.listdir("scan_batch/scores/"):
      if name+"_" in file:
        f = open("scan_batch/scores/"+file, "r")
        l = float(f.readline().rstrip())
        if ind == 0:
          lowest_score_ind = file.split("_")[-1].split(".txt")[0]
          lowest_score = 1.0*l
        elif l < lowest_score:
          lowest_score_ind = file.split("_")[-1].split(".txt")[0]
          lowest_score = 1.0*l
        ind+=1 

    #lowest_score_ind = 128
    with open('scan_batch/hyperparameters/{}_{}.json'.format(name,lowest_score_ind)) as json_file: params = json.load(json_file)
    print "Best parameter file {}".format(lowest_score_ind)
    print "Best parameters {}".format(params)
    print "Lowest score {}".format(lowest_score)
    new_self = pkl.load(open('scan_batch/models/{}_{}.pkl'.format(name,lowest_score_ind), "rb"))
    new_self.set_params(copy.deepcopy(params))
    return new_self

import pandas as pd
import xgboost as xgb
from UserCode.BDTFakeFactors.Dataframe import Dataframe
from UserCode.BDTFakeFactors.plotting import DrawROCCurve, DrawBDTScoreDistributions, DrawFeatureImportance, DrawVarDistribution

#df = pd.read_pickle("dataframes/ditau_rereco/tt_dTvJ_medium_1_eq_1_and_dTvJ_medium_2_eq_1_and_os_eq_1_mc_DY_dataframe.pkl")
df = pd.read_pickle("dataframes/ditau_rereco/tt_dTvJ_medium_2_eq_1_and_dTvJ_vvvloose_1_eq_1_and_dTvJ_medium_1_eq_0_and_os_eq_0_mc_DY_dataframe.pkl")

df = df.drop(["wt_ff_mssm_1","wt_ff_1","wt_ff_dmbins_1","q_2","mt_2","eta_2","ip_mag_2","ip_sig_2","pt_2","jpt_2","met_dphi_2","jet_pt_2","jet_pt_2/pt_2","tau_decay_mode_2","mva_dm_2","yr_2018","yr_2017","yr_2016"],axis=1)

df = df[(df.loc[:,"q_1"]==-1)]

print df

df_1 = Dataframe()
df_2 = Dataframe()

df_1.dataframe = df[((df.loc[:,"jet_flav_1"]==1) | (df.loc[:,"jet_flav_1"]==3))]
df_2.dataframe = df[((df.loc[:,"jet_flav_1"]==2) | (df.loc[:,"jet_flav_1"]==4))]

df_1.dataframe = df_1.dataframe.drop(["jet_flav_1","jet_flav_2"],axis=1)
df_2.dataframe = df_2.dataframe.drop(["jet_flav_1","jet_flav_2"],axis=1)

df_1.dataframe.loc[:,"y"] = 0
df_2.dataframe.loc[:,"y"] = 1

total_df = Dataframe()
total_df.dataframe = pd.concat([df_1.dataframe,df_2.dataframe], ignore_index=True, sort=False) 

total_df.NormaliseWeightsInCategory("y", [0,1], column="weights")

DrawVarDistribution(total_df.dataframe,2,"met_dphi_1",0,3.5,output="met_dphi_1",bin_edges=20)
DrawVarDistribution(total_df.dataframe,2,"tau_decay_mode_1",0,12,output="tau_decay_mode_1",bin_edges=12)
DrawVarDistribution(total_df.dataframe,2,"svfit_mass",0,500,output="svfit_mass",bin_edges=50)
DrawVarDistribution(total_df.dataframe,2,"dphi",-3.5,3.5,output="dphi",bin_edges=20)

X, y, wt = total_df.SplitXyWts()

xgb_model = xgb.XGBClassifier()
xgb_model.fit(X, y, sample_weight=wt)
probs = xgb_model.predict_proba(X)
preds = probs[:,1]

DrawROCCurve(y,preds,wt,"roc_curve")

wt0 = total_df.dataframe[(total_df.dataframe.loc[:,"y"]==0)].loc[:,"weights"]
wt1 = total_df.dataframe[(total_df.dataframe.loc[:,"y"]==1)].loc[:,"weights"]
xt0 = total_df.dataframe[(total_df.dataframe.loc[:,"y"]==0)].drop(["y","weights"],axis=1)
xt1 = total_df.dataframe[(total_df.dataframe.loc[:,"y"]==1)].drop(["y","weights"],axis=1)
probs0 = xgb_model.predict_proba(xt0)
probs1 = xgb_model.predict_proba(xt1)
preds0 = probs0[:,1]
preds1 = probs1[:,1]
DrawBDTScoreDistributions({"down type":{"preds":preds0,"weights":wt0},"up type":{"preds":preds1,"weights":wt1}}, bins=20)

DrawFeatureImportance(xgb_model,"gain","bc_feature_importance_gain")
DrawFeatureImportance(xgb_model,"weight","bc_feature_importance_weight")

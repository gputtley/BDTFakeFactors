from UserCode.BDTFakeFactors import reweighter
import pickle as pkl
import pandas as pd

### Open dataframes ###

original_df = pd.read_pickle("basic_example/mmtt_dTvJ_iso_3_geq_0_and_dTvJ_loose_3_eq_0_and_dTvJ_loose_4_eq_1_and_q_sum_neq_0_dataframe.pkl")
target_df = pd.read_pickle("basic_example/mmtt_dTvJ_loose_3_eq_1_and_dTvJ_loose_4_eq_1_and_q_sum_neq_0_dataframe.pkl")


### Print datafames ###

print "Original dataframe"
print original_df.head(10)
print "Length =", len(original_df)
print "Columns =", list(original_df.columns)
print ""
print "Target dataframe"
print target_df.head(10)
print "Length =", len(target_df)
print "Columns =", list(target_df.columns)




### Split out weights ###

original_weights = original_df.loc[:,"weights"]
target_weights = target_df.loc[:,"weights"]

original_df = original_df.drop(["weights"],axis=1)
target_df = target_df.drop(["weights"],axis=1)

### Select variables ###

variables = ["pt_3","tau_decay_mode_3","n_jets"]
original_df = original_df.loc[:,variables]
target_df = target_df.loc[:,variables]

### Train model ###

print ""
print "Training model"

rwter = reweighter.reweighter(
            n_estimators = 400,
            learning_rate = 0.06,
            min_samples_leaf = 200, 
            max_depth = 3,
            loss_regularization = 5.0                  
            )

rwter.norm_and_fit(original_df, target_df, original_weights ,target_weights)

### Predict weights ###

print ""
print "Predicting weights"

original_df.loc[:,"reweights"] = rwter.predict_reweights(original_df)

print ""
print original_df.head(10)
print "Length =", len(original_df)
print "Columns =", list(original_df.columns)

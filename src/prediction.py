#%%
import cleaning_data_pipeline_class as clean_data
import pandas as pd

predict=clean_data.football('https://aicore-files.s3.amazonaws.com/Data-Science/To_Predict.zip')

predict.features_df=predict.new_raw_data_df

predict.add_new_ELO()


predict.features_df
# %%
df=pd.read_csv('/home/irinakw/AiCore/FooballMatchPrediction/data/processed/cumm_values.csv')
df
# %%

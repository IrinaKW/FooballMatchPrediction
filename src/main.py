#Link used as an attribute of the class contains the new matches results that were not yet cleaned before.
#The cleaning data pipline process/clean/ results, engineer new data points, create features useful for the models
#Two previously trained models: AdaBoost and SVM are applied after the data cleaning process to compare the results.

#%%
import cleaning_data_pipeline_class as clean_data
import config
import pandas as pd

new_test=clean_data.Football()
new_test_data_link='https://aicore-files.s3.amazonaws.com/Data-Science/Results.zip'
prediction_link='https://aicore-files.s3.amazonaws.com/Data-Science/To_Predict.zip'

new_test(new_test_data_link, prediction_link)

#%%
#Preparing Data from provided new Results data set
try:
    new_test.features_df=pd.read_csv(config.FEATURES_FILE)
    new_test.cummulative_pivot_df=pd.read_csv(config.CUMMULATIVE_VALUES_FILE)
    print('prepared Features file exists')
    new_test.apply_model()      
 
except:
    try:
        new_test.features_df=pd.read_csv(config.SCRAPED_PENALTY_DATA_FILE)
        print('Penalty cards data has been scraped before')

    except:
        new_test.add_stadium_info()
        new_test.split_result()
        new_test.set_up()
        new_test.cookies()
        new_test.subscribe()
        no_result_list=new_test.empty_result_link(new_test.features_df)
        new_test.df=new_test.features_df
        new_test_features_df=new_test.scraping_missing_results(no_result_list)
        new_test.features_df=new_test.scraping_penalty_cards()
        new_test.tearDown()

    new_test.convert_categorical()
    new_test.create_outcome()
    new_test.create_win_draw_loss_data()
    new_test.total_game_cards()
    new_test.add_new_ELO()
    new_test.create_cummulative_pivot_df()

    new_test.features_df=new_test.select_features(new_test.features_df,new_test.cummulative_pivot_df)
    print('Your features dataset is ready. It is uploaded as processed/features.csv file')
    new_test.features_df
    new_test.apply_model()


#%%
#Applied Data preparation process on "to be predicted" Data

try:
    new_test.cummulative_pivot_df=pd.read_csv(config.CUMMULATIVE_VALUES_FILE)
    print('Cummulative data pivot data file exists.')
except:
    new_test.cummulative_pivot_df=new_test.create_cummulative_pivot_df()

print(new_test.predict())



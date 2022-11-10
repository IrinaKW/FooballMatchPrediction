
#XPATHS for scrapers: match results and ELO values
SUBSCRIBE_XPATH='/html/body/div[5]//div/div/div[1]/div/div[2]/div[3]'

COOKIE_XPATH='//*[@id="qc-cmp2-ui"]/div[2]/div/button[2]/span'

ANALYSIS_XPATH='//*[@id="match"]/main/section[1]/div[1]/div/div/a[6]'

HOME_YELLOW_XPATH='//*[@id="status_attr"]/div[1]/div/span[1]'

HOME_RED_XPATH='//*[@id="status_attr"]/div[1]/div/span[2]'

AWAY_YELLOW_XPATH='"]/div[3]/div[2]/div/span[1]'

AWAY_RED_XPATH='"]/div[3]/div[2]/div/span[2]'

POP_UP_ELEMENT = "return document.querySelector('#home > div.grv-dialog-host').shadowRoot.querySelector('div > div > div.buttons-wrapper > button.sub-dialog-btn.block_btn')"

HOME_ELO_XPATH='//*[@id="mod_team_analysis"]/div/div/div/table/tbody/tr[2]/td[1]/span'
            
AWAY_ELO_XPATH='//*[@id="mod_team_analysis"]/div/div/div/table/tbody/tr[2]/td[3]/span'


#PATHS for data files raw / interim/ processed
UPDATED_ELO_SCORES_FILE='/home/irinakw/AiCore/FooballMatchPrediction/data/interim/updated_ELO_scores.csv'

STADIUM_DATA_FILE='/home/irinakw/AiCore/FooballMatchPrediction/data/raw/stadium_info.csv'

HISTORICAL_ELO_FILE='/home/irinakw/AiCore/FooballMatchPrediction/data/raw/historical_ELO.csv'

HISTORICAL_MATCH_RESULTS_FILE='/home/irinakw/AiCore/FooballMatchPrediction/data/raw/historical_match_results.csv'

NEW_ELO_FILE='/home/irinakw/AiCore/FooballMatchPrediction/data/raw/new_ELO.csv'

NEW_MATCH_RESULTS_FILE='/home/irinakw/AiCore/FooballMatchPrediction/data/raw/new_match_results.csv'

MODEL_JOBLIB = '/home/irinakw/AiCore/FooballMatchPrediction/models/AdaBoost_DT.joblib'

MODEL_SVM_JOBLIB='/home/irinakw/AiCore/FooballMatchPrediction/models/SVM.joblib'


NEW_DATA_RESULTS_UPDATED_FILE='/home/irinakw/AiCore/FooballMatchPrediction/data/interim/results_updated.csv'

SCRAPED_PENALTY_DATA_FILE='/home/irinakw/AiCore/FooballMatchPrediction/data/interim/penalty_data_updated.csv'

FEATURES_FILE='/home/irinakw/AiCore/FooballMatchPrediction/data/processed/features.csv'

CUMMULATIVE_VALUES_FILE='/home/irinakw/AiCore/FooballMatchPrediction/data/processed/cumm_values.csv'

#LINKS to external files with raw data
HISTORICAL_PENALTY_CARDS_LINK= 'https://aicore-files.s3.amazonaws.com/Data-Science/Match_Info.csv'

HISTORICAL_ELO_LINK='https://aicore-files.s3.amazonaws.com/Data-Science/elo_dict.pkl'

STADIUM_DATA_LINK='https://aicore-files.s3.amazonaws.com/Data-Science/Team_Info.csv'

HISTORICAL_MATCH_RESULTS_LINK="https://aicore-files.s3.amazonaws.com/Data-Science/Football.zip"

NEW_MATCH_LINK='https://aicore-files.s3.amazonaws.com/Data-Science/Results.zip'

CLEAN_DATA_PATH='/home/irinakw/AiCore/FooballMatchPrediction/data/processed/clean_data_by_league'
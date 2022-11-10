#%%
from distutils.command.config import config
import pandas as pd
import numpy as np
import re
import upload_data_sets
import config
import joblib
import scraper_class
from sklearn.preprocessing import StandardScaler


class Football(scraper_class.Football_scraper):
    '''
    Takes data on past and future football matches:
    - process past data into cleaned_data set/ features for model training
    - applied model, that has been trained on historical data
    - retrain model if new match data with results is avaliable
    
    Return:
    - updated set of features
    - model accuracy
    - model confusion matrix
    - predicted results
    '''

    def __init__(self):
        self.cummulative_pivot_df=pd.DataFrame()
        self.features_df=pd.DataFrame()
        scraper_class.Football_scraper.__init__(self, self.features_df)
        
    def __call__(self, new_data_link, prediction_link=''):
        """
        Upload data sets and other required attributes: 
        - old training data
        - trained models: AdaBoost and SVM
        - new data with outcomes and related ELO values
        - staidum data from the existing file if exists or from the modul that uses external link
        """
        pd.set_option('display.max_columns', None)
        self.historical_data_clean_df=upload_data_sets.multiple_csv_to_df(config.CLEAN_DATA_PATH)
        self.model_adaboost=joblib.load(config.MODEL_JOBLIB)
        self.model_svm=joblib.load(config.MODEL_SVM_JOBLIB)
        self.new_raw_data_df=upload_data_sets.upload_zip_csvfiles_to_df(new_data_link)
        self.new_ELO_df=upload_data_sets.upload_new_ELO(new_data_link)
        
        try:
            self.stadium_df=pd.read_csv(config.STADIUM_DATA_FILE, index_col=0)
        except:
            self.stadium_df=upload_data_sets.upload_stadium_data(config.STADIUM_DATA_LINK)

        try:
            self.prediction_df=upload_data_sets.upload_zip_csvfiles_to_df(prediction_link)
            self.prediction_df.rename(columns={'Home Team':'Home_Team','Away Team':'Away_Team'}, inplace=True)
            self.prediction_ELO_df=upload_data_sets.upload_new_ELO(prediction_link)
        except:
            pass


    def longest_win(self, streak):
        """Module generates the value of the longest winning streak 

        Args:
            streak (str): the team resultant consists of letters W L or D
        Returns:
            int: the count of the longest wins in the row in the streak_
        """
        try: 
            return max(len(i) for i in re.findall("W+", streak))
        except:
            return 0

    def longest_loss(self, streak):
        """Module generates the value of the longest loosing streak 

        Args:
            streak (str): the team resultant consists of letters W L or D
        Returns:
            int: the count of the longest losses in the row in the streak_
        """
        try: 
            return max(len(i) for i in re.findall("L+", streak))
        except:
            return 0

    def add_stadium_info(self):
        """Function merges home stadium data to the existing dataset"""
        self.features_df=self.features_df.set_index('Home_Team')
        self.features_df[['Capacity','Pitch','Country']]=''
        self.features_df.update(self.stadium_df)
        self.features_df.reset_index(inplace=True)

    def split_result(self):
        """Function splits initial RESULT column to create number of goals per team"""
        self.features_df[['Home_Score','Away_Score']]=self.features_df['Result'].str.split('-', expand=True)
        self.features_df[['Home_Score','Away_Score']] = pd.to_numeric(self.features_df[['Home_Score','Away_Score']], errors='coerce')
        cols = ['Season', 'Round', 'Home_Score','Away_Score']
        self.features_df[cols]=self.features_df[cols].astype('Int64')
        
    def create_outcome(self):
        """Function generates Outcome column as values of -1. 0. 1 besed on L, D or W respectively"""
        self.features_df['Outcome']=0
        self.features_df['Outcome'] = np.where(self.features_df['Home_Score'] > self.features_df['Away_Score'], 1, self.features_df['Outcome'])
        self.features_df['Outcome'] = np.where(self.features_df['Home_Score'] < self.features_df['Away_Score'], -1, self.features_df['Outcome'])
        
    def empty_result_link(self, results_df):
        """Function checks for empty results in scores column
        Args:
            results_df (df): data frame to be checked for empty values
        Returns:
            list: list of links to the games which results were not recorded _
        """
        no_result_list=[]
        for i in results_df[results_df['Away_Score'].isna()]['Link']:
            no_result_list.append(i)
        return no_result_list
    
    
    def add_new_ELO(self):
        """
        MERGE: Match_df and ELO_df 
        - merge match dataframe and ELO dataframe into one.
        Args: 
            match_df and ELO_df: two dataframes containg data on matches and teams
        Returns: 
            merged: dataframe that has combined two dataframes
        """
        self.features_df.reset_index(drop=False, inplace=True)
        self.features_df['HT_link'] = self.features_df['Link'].map(lambda x: x.split("/")[-3])
        self.features_df['AT_Link'] = self.features_df['Link'].map(lambda x: x.split("/")[-2])
        self.features_df = pd.merge(self.features_df,self.new_ELO_df, how ='left',on=['HT_link', 'AT_Link','Season'])
        self.features_df.drop(['index','Link_y'], axis=1, inplace=True)
        self.features_df.rename(columns={'Link_x':'Link'}, inplace=True)   
      
    def create_win_draw_loss_data(self):
        """Create Wins/Draws/Losses columns for home and away teams and fill them up with 0 or 1 respective of the outcome"""
        self.features_df[['Home_Wins','Home_Losses','Home_Draws','Away_Wins','Away_Losses','Away_Draws']]=0 
        self.features_df['Home_Wins'] = np.where(self.features_df['Home_Score'] > self.features_df['Away_Score'], 1, self.features_df['Home_Wins'])
        self.features_df['Home_Losses'] = np.where(self.features_df['Home_Score'] < self.features_df['Away_Score'], 1, self.features_df['Home_Losses'])
        self.features_df['Home_Draws'] = np.where(self.features_df['Home_Score'] == self.features_df['Away_Score'], 1, self.features_df['Home_Draws'])
        self.features_df['Away_Wins']=np.where(self.features_df['Home_Score'] < self.features_df['Away_Score'], 1, self.features_df['Away_Wins'])
        self.features_df['Away_Draws']=self.features_df['Home_Draws']
        self.features_df['Away_Losses']= np.where(self.features_df['Home_Score'] > self.features_df['Away_Score'], 1, self.features_df['Away_Losses'])
        self.features_df.reset_index(inplace=True) 

    def total_game_cards(self):
        """Function adds yellow and red cards values per team"""
        self.features_df['Home_Game_Penalty_Cards']=self.features_df['Home_Yellow']+self.features_df['Home_Red']
        self.features_df['Away_Game_Penalty_Cards']=self.features_df['Away_Yellow']+self.features_df['Away_Red']
        

    def convert_categorical(self):    
        """Convert Pitch and Country categorical type variables into numeric values:
        There are 3 types of pitches: grass(1), artificial(2) and hybrid(3) """
        self.features_df['Pitch']=self.features_df['Pitch'].str.lower()
        self.features_df['Pitch'] = self.features_df['Pitch'].fillna('')
        self.features_df['Country']=self.features_df['Country'].str.lower()
        self.features_df['Pitch'].replace(['','natural', 'cesped real', 'césped artificial', 'airfibr ', \
        'césped natural', 'grass', 'césped', 'cesped natural'], [0,1,1,2,3,1,1,1,1], inplace=True)
        self.features_df['Country'].replace(['','germany', 'netherlands', 'france', 'england', 'portugal',\
        'spain', 'italy'], [0,1,2,3,4,5,6,7], inplace=True)
        
    def create_cummulative_pivot_df(self):
        """#Create cummulative scores, cards and resultant streak per team, per season

        Returns:
            df: dataframe is a pivot table aranged per team, per round, per season.
            Values are singular and cummulative, reflective of team standing as of the beginning of the game,
            thus zero values in the Round 1.
        """
        
        print('WARNING! It can take couple of minutes')
        
        cumm_df = pd.DataFrame()
        for team in self.features_df['Home_Team'].unique():
            temp_df=self.features_df[(self.features_df['Home_Team']==team)|(self.features_df['Away_Team']==team)]

            for index, row in temp_df.iterrows():
                cummulate_dict={}
                cummulate_dict['Team'] = team
                cummulate_dict['Season']=row['Season']
                cummulate_dict['Round']= row['Round'] 
                if row['Home_Team']==team: 
                    team_type='Home'
                elif row['Away_Team']==team:
                    team_type='Away'
                cummulate_dict['Team_Type']=team_type
                cummulate_dict['Score']=row[team_type+'_Score']
                cummulate_dict['Cards']=row[team_type+'_Game_Penalty_Cards']
                cummulate_dict['Wins']=row[team_type+'_Wins']
                cummulate_dict['Draws']=row[team_type+'_Draws']
                cummulate_dict['Losses']=row[team_type+'_Losses']
                if (team_type=='Home' and row['Outcome']==1) or (team_type=='Away' and row['Outcome']==-1) :
                    cummulate_dict['Streak']='W'
                elif row['Outcome']==0:
                    cummulate_dict['Streak']='D'
                else: 
                    cummulate_dict['Streak']='L'
                             
                temp_df = pd.DataFrame([cummulate_dict])
                cumm_df = pd.concat([cumm_df, temp_df], ignore_index=True)
            print(f"done with {team}")  
        cumm_df=cumm_df.drop_duplicates(
            subset=['Team', 'Season', 'Round','Team_Type'],
            keep='last').reset_index(drop=True)

        cumm_df=cumm_df.sort_values(by=['Season','Team','Round'])
        cumm_df=cumm_df.reset_index(drop=True) 

        cumsum_col=['Cards','Score','Wins','Draws','Losses','Streak']
        for col in cumsum_col:
            col_name='Cum_'+col 
            cumm_df[col_name] = (cumm_df.groupby(['Season','Team'])[col].transform(\
                lambda x: x.cumsum()))
            #or shifted by 1
            #cumm_df[col_name] = (cumm_df.groupby(['Season','Team'])[col].transform(\
            #lambda x: x.cumsum().shift()).fillna(0, downcast='infer'))
        cumm_df['Cum_Streak']=cumm_df['Cum_Streak'].astype('string')
        cumm_df['Longest_Win_Streak']=cumm_df['Cum_Streak'].apply(self.longest_win)
        cumm_df['Longest_Loss_Streak']=cumm_df['Cum_Streak'].apply(self.longest_loss)
        cumm_df['Round']=cumm_df['Round']+1
        cumm_df.to_csv(config.CUMMULATIVE_VALUES_FILE, index=False) 
        return cumm_df 
        

    def select_features(self, final_df, cumm_df):
        """Final step: update initial dataset after merging with the pivot table.
        Select only numerical data points for model application.
        Args:
            final_df (df): df that had all the required data points and is inclusive of stadium data, etc
            cumm_df (df): the earlier generated pivot table of cummulative values
        Returns:
            df: Data frame that consists of the all required datapoints for model application.
        """
        
        final_df=pd.merge(final_df, cumm_df, how='left', left_on=['Home_Team','Season','Round'], right_on=['Team','Season','Round'])

        final_df=final_df[['Outcome','Home_Team', 'Away_Team', 'Season', 'Round',\
            'Capacity', 'Pitch', 'Country', 'Elo_home', 'Elo_away',\
            'Cum_Cards', 'Cum_Score', 'Cum_Wins', 'Cum_Draws', 'Cum_Losses', 'Cum_Streak',\
            'Longest_Win_Streak', 'Longest_Loss_Streak']]

        final_df.rename(columns = {'Cum_Cards':'HT_Cum_Cards','Cum_Score':'HT_Cum_Score',\
            'Cum_Wins':'HT_Cum_Wins','Cum_Draws':'HT_Cum_Draws',\
            'Cum_Losses':'HT_Cum_Losses','Cum_Streak':'HT_Cum_Streak',\
            'Longest_Win_Streak': 'HT_Longest_Win_Streak', 'Longest_Loss_Streak':'HT_Longest_Loss_Streak'}, inplace = True)

        final_df=pd.merge(final_df, cumm_df, how='left', left_on=['Away_Team','Season','Round'], right_on=['Team','Season','Round'])

        final_df=final_df[['Outcome','Home_Team', 'Away_Team', 'Season', 'Round',\
            'Capacity', 'Pitch', 'Country', 'Elo_home', 'Elo_away',\
            'HT_Cum_Cards', 'HT_Cum_Score', 'HT_Cum_Wins', 'HT_Cum_Draws', 'HT_Cum_Losses', 'HT_Cum_Streak',\
            'HT_Longest_Win_Streak', 'HT_Longest_Loss_Streak',\
            'Cum_Cards', 'Cum_Score', 'Cum_Wins', 'Cum_Draws', 'Cum_Losses', 'Cum_Streak',\
            'Longest_Win_Streak', 'Longest_Loss_Streak']]

        final_df.rename(columns = {'Cum_Cards':'AT_Cum_Cards','Cum_Score':'AT_Cum_Score',\
            'Cum_Wins':'AT_Cum_Wins','Cum_Draws':'AT_Cum_Draws',\
            'Cum_Losses':'AT_Cum_Losses','Cum_Streak':'AT_Cum_Streak',\
            'Longest_Win_Streak': 'AT_Longest_Win_Streak', 'Longest_Loss_Streak':'AT_Longest_Loss_Streak'}, inplace = True)
        
        final_df=final_df.fillna(0)
        final_df['Capacity']=final_df['Capacity'].astype('str')
        final_df.Capacity = final_df.Capacity.replace('^\s*$','70000', regex=True)
        final_df.Capacity = final_df.Capacity.apply(lambda x : x.replace(',',''))
        final_df.Capacity = final_df.Capacity.apply(lambda x : x.replace('.0',''))
        final_df['Capacity']=final_df['Capacity'].astype('int')
        
        return final_df

    def apply_model(self):
        """Function splits the df into features and labels, after dropping sring data points.
        Applies two previously trained model: AdaBoost and SVM.
        Prints accuracy values on the test set"""
        X_test=self.features_df.drop(['Outcome', 'Season','HT_Cum_Streak','AT_Cum_Streak','Home_Team','Away_Team'],axis=1)
        y_test=self.features_df.Outcome
        scaler = StandardScaler()
        result = self.model_adaboost.score(scaler.fit_transform(X_test), y_test)
        print(f'Accuracy results after AdaBoost model application on new data set {result}')
        result = self.model_svm.score(scaler.fit_transform(X_test), y_test)
        print(f'Accuracy results after SVM model application on new data set {result}')
        
    def predict(self):
        """Functon generates features based on the existing values in the cummulative pivot table"""
        try:
            self.features_df=self.prediction_df
            self.add_stadium_info()
            self.new_ELO_df=self.prediction_ELO_df
            self.add_new_ELO()
            self.convert_categorical()
            self.features_df['Outcome']=0
            self.features_df=self.select_features(self.features_df,self.cummulative_pivot_df)
            X_test=self.features_df.drop(['Outcome', 'Season','HT_Cum_Streak','AT_Cum_Streak','Home_Team','Away_Team'],axis=1)
            scaler = StandardScaler()
            pred_ada=self.model_adaboost.predict(scaler.fit_transform(X_test))
            self.prediction_df['AdaBoost']=pred_ada

            pred_svm=self.model_svm.predict(scaler.fit_transform(X_test))
            self.prediction_df['SVM']=pred_svm

            labels_dict={1:'Win', 0:'Draw', -1:'Loss'}
            self.prediction_df=self.prediction_df.replace({"AdaBoost": labels_dict})
            self.prediction_df=self.prediction_df.replace({"SVM": labels_dict})
            predicted_df=self.prediction_df[['Home_Team','Away_Team','Season','Round', 'AdaBoost','SVM']]
            return predicted_df

        except:
            print('No link to prediction data was provided!')
            pass
        

# %%

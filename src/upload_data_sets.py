import requests
import zipfile
import os
import pandas as pd
import glob
import shutil
import pickle
import urllib
import config

def upload_csv(filename, df):
    os.makedirs('csv_files', exist_ok=True)
    df.to_csv(f'csv_files/{filename}.csv', index=False) 


def upload_old_ELO():
    pickle_link='https://aicore-files.s3.amazonaws.com/Data-Science/elo_dict.pkl'
    myfile = pickle.load(urllib.request.urlopen(pickle_link))
    old_ELO_df=(pd.DataFrame(myfile)).T
    upload_csv('old_ELO', old_ELO_df)

def upload_new_ELO(link):
    """
    Upload ELO values per match per team
    Continue Cleaning/Processing Data:
    - dataframe index is a link that includes participating teams and season (first 4 digits)
    Return: ELO_df - dataframe
    """
    req=requests.get(link)
    filename=link.split('/')[-1]
    with open(filename,'wb') as output_file:
        output_file.write(req.content)

    with zipfile.ZipFile(filename,"r") as zip_ref:     
        zip_ref.extractall()

    path =  "./"+link.split('/')[-1].split('.')[-2]
    pkl_files = glob.glob(path + "/**/*.pkl", recursive = True)
    ELO_df=pd.DataFrame()
    for f in pkl_files:
        myfile=pd.read_pickle(f)
        temp=pd.DataFrame(myfile).T
        temp.reset_index(inplace=True)
        ELO_df= pd.concat([ELO_df,temp], ignore_index=True)
    ELO_df.rename(columns={'index':'Link'}, inplace=True)
    os.remove(filename)
    shutil.rmtree(path)
    ELO_df['extra'],ELO_df['HT_link'],ELO_df['AT_Link'],ELO_df['Season']=ELO_df['Link'].str.\
        rsplit('/',3).str
    ELO_df.drop('extra', axis=1, inplace=True)
    ELO_df['Season']=ELO_df['Season'].str[:4]
    ELO_df=ELO_df.dropna()
    ELO_df[['Season','Elo_home','Elo_away']]=ELO_df[['Season','Elo_home','Elo_away']].astype('int')
    ELO_df = ELO_df[ELO_df['Season']>2012]
    return ELO_df
    
def upload_stadium_data(link):
    """
    Upload data on stadium the game played
    Return: stadium_df - dataframe
    """
    stadium_df = pd.read_csv(link)
    stadium_df['Team'].replace(['Gimnàstic','Mönchengladbach','Eintracht','Würzburger','Fortuna',\
        'Evian Thonon Gail.','Olympique','Queens Park Range.','Brighton Hove Alb.','Paços Ferreira',\
        'Sheffield', 'West Bromwich Alb.'],\
        ['Gimnàstic Tarragona', 'B. Mönchengladbach', 'Eintracht Frankfurt', 'Würzburger Kickers', 'Fortuna Düsseldorf',\
        'Evian Thonon Gaillard','Olympique Marseille','Queens Park Rangers', 'Brighton & Hove Albion','Paços Ferreira',\
        'Sheffield Wednesday', 'West Bromwich Albion'], inplace=True)
        
    #Add new stadium info for new teams
    stadium_add_list=[['Peterborough United', 'Peterborough', 'England', 'Weston Homes', 15314, 'Natural'],\
    ['Quevilly-Rouen', 'Le Petit-Quevilly', 'France', 'Robert-Diochon', 12018, 'Natural'],\
    ['CF Estrela', 'Amadora', 'Portugal', 'José Gomes',9288, 'Natural'],\
    ['SC Covilha', 'Covilha', 'Portugal', 'Municipal Santos Pinto', 2055, 'Natural'],\
    ['Vitória Guimarães', 'Guimaraes', 'Portugal', 'Dom Afonso Henriq', 30029, 'Natural'],\
    ['Paços de Ferreira', 'Pacos de Ferreira', 'Portugal', 'Capital do Móvel', 9077, 'Natural'],\
    ['Belenenses SAD', 'Lisbon', 'Portugal','Estádio de Honra', 37593, 'Natural'],\
    ['Pisa SC', 'Pisa', 'Italy', 'Arena Garibaldi', 25000, 'Natural'],\
    ['Vicenza', 'Vicenza', 'Italy', 'Romeo Menti', 12000, 'Natural'],\
    ['US Alessandria', 'Alessandria', 'Italy', 'Giuseppe Moccagatta', 5926, 'Natural'],\
    ['R. Sociedad B', 'Zubieta', 'Spain', 'Zubieta Facilities', 2500, 'Grass'],\
    ['UD Ibiza', 'Ibiza', 'Spain', 'Municipal de Can Misses', 4500, 'Grass'],\
    ['SD Amorebieta', 'Amorebieta', 'Spain', 'Campo Municipal de Urritxe', 3000,'Natural']]
    
    for item in stadium_add_list:
        stadium_df.loc[len(stadium_df.index)] = item
    stadium_df.to_csv(config.STADIUM_DATA_FILE, index=False) 
    return stadium_df

def upload_old_match_data():
    """
    Upload match game info, cards, referees, etc. values per match per team
    Cleaning Data:  MATCH DATAFRAME
    1. Strip the Referee data to the referees names only and make it a category (set number of referees only).
    2. Split data columns to only year values to turn into Season.
    3. Dropped unnecessary columns: the temp ones and Link.
    4. Changed data type for the yellow and red cards into integers and dropped NW values (see notes).
    Return: match_df - dataframe
    """
    link2= 'https://aicore-files.s3.amazonaws.com/Data-Science/Match_Info.csv'
    match_df = pd.read_csv(link2)
    cols=['Home_Yellow', 'Away_Yellow', 'Home_Red', 'Away_Red']
    match_df[cols] = match_df[cols].astype('Int64')
    match_df['extra'],match_df['HT_link'],match_df['AT_Link'],match_df['Season']=\
        match_df['Link'].str.rsplit('/',3).str
    match_df['HT_Game_Penalty_Cards']=match_df['Home_Yellow']+match_df['Home_Red']
    match_df['AT_Game_Penalty_Cards']=match_df['Away_Yellow']+match_df['Away_Red']
    cols=['extra','Date_New','Link','Referee','Home_Yellow', 'Away_Yellow', 'Home_Red', 'Away_Red']
    match_df.drop(cols,inplace=True, axis=1)
    match_df['Season']=match_df['Season'].astype('int')
    match_df = match_df[match_df['Season']>2012]
    upload_csv('match', match_df)

def upload_zip_csvfiles_to_df(link):
    req=requests.get(link)
    filename=link.split('/')[-1]
    with open(filename,'wb') as output_file:
        output_file.write(req.content)

    with zipfile.ZipFile(filename,"r") as zip_ref:     
        zip_ref.extractall()

    path = "./"+link.split('/')[-1].split('.')[-2]
    os.remove(filename)
    csv_files = glob.glob(path + "/**/*.csv", recursive = True)
    df = [pd.read_csv(f) for f in csv_files]
    df  = pd.concat(df, ignore_index=True)
    df.drop('Unnamed: 0', axis=1, inplace=True)
    return df

def multiple_csv_to_df(path):
    csv_files = glob.glob(path + "/**/*.csv", recursive = True)
    df = [pd.read_csv(f) for f in csv_files]
    df  = pd.concat(df, ignore_index=True)
    #df.drop('Unnamed: 0', axis=1, inplace=True)
    return df




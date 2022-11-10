
import pandas as pd 
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import logging
import os
import sys
import time
import config

class Football_scraper:
    '''
    Extract data per match for two teams: ELO, cards, match results
    
    Input: dataframe with links
    
    Output:
        files: ELO values are saved as separate csv file in the data/raw folder;
               number of given penalty cards recorded as csv file and added to the provided dataframe 
        dataframe: df returned updated with the number of penalty cards given during each match

    '''

    def __init__(self, df=pd.DataFrame()):
        self.df=df
        
    def set_up(self):
        if not sys.warnoptions:
            import warnings
            warnings.simplefilter("ignore")
        logging.getLogger('WDM').setLevel(logging.NOTSET)
        os.environ['WDM_LOG'] = "false"
        
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        self.driver = webdriver.Chrome(ChromeDriverManager().install(), options=chrome_options)
        link1="https://www.besoccer.com/"
        self.driver.get(link1)
        time.sleep(2)

    def tearDown(self):
        self.driver.quit()

    
    def cookies(self):
        """Open URL and identifies the cookies button on the page and click on it.
        Attr:
            driver (interface); the chrome webdriver used by selenium to control the webpage remotely 
        Raises:
            pass if there is no cookies button but has the URL open
         """

        try: 
            accept_cookies_button = self.driver.find_element(By.XPATH, config.COOKIE_XPATH)
            accept_cookies_button.click()
            time.sleep(2)
        except:
            pass # If there is no cookies button, we won't find it, so we can pass


    def subscribe(self):
        """Open URL and identifies the "not to subscribe" button on the page and click on it.
        Attr:
            link (str): the global variable, link to the match
            driver (interface); the chrome webdriver used by selenium to control the webpage remotely 
        Raises:
            pass if there is no "no subscribe" button but has the URL open
         """

        try: 
            popup_element = self.driver.execute_script(config.POP_UP_ELEMENT)
            popup_element.click()
            time.sleep(2)
        except:
            pass # If there is no subcribe pop-up, we won't find it, so we can pass

    def scraping_ELO(self):
        """Return ELO values per team for each match as csv file"""
        self.historical_ELO='https://www.besoccer.com'+self.df['Link']
        self.historical_ELO['Home_ELO']=''
        self.historical_ELO['Away_ELO']=''
        for i in range(len(self.historical_ELO.index)):
            link=str(self.historical_ELO.loc[i, 'Link'])
            print(link)
            self.driver.get(link)
            time.sleep(5)
            self.driver.find_element(By.XPATH, config.ANALYSIS_XPATH).click()
            time.sleep(2)
            home_elo=self.driver.find_element(By.XPATH, config.HOME_ELO_XPATH).text
            away_elo=self.driver.find_element(By.XPATH, config.AWAY_ELO_XPATH).text
            self.historical_ELO.loc[i, ['Home_ELO']]=home_elo
            self.historical_ELO.loc[i, ['Away_ELO']]=away_elo
            time.sleep(2)
        self.historical_ELO.to_csv('/home/irinakw/AiCore/FooballMatchPrediction/data/raw/historical_ELO.csv')
            
    def scraping_penalty_cards(self):
        """Scrape number of penalty cards yellow and red recieved during the match per team.
        Records scraped data in the interim data file"""
        self.df['Home_Yellow']=0
        self.df['Away_Yellow']=0
        self.df['Home_Red']=0
        self.df['Away_Red']=0
        for i in range(self.df.shape[0]):
            link=str(self.df.loc[i, 'Link'])
            self.driver.get(link)
            time.sleep(5)

            try:
                home_yellow=self.driver.find_element(By.XPATH, config.HOME_YELLOW_XPATH).text
                self.df.loc[i, ['Home_Yellow']]=home_yellow
            except:
                pass

            try:
                home_red=self.driver.find_element(By.XPATH, config.HOME_RED_XPATH).text
                self.df.loc[i, ['Home_Red']]=home_red
            except:
                pass
            
            #because XPATH is dinamic and vary for every link, @id is equal to last part of the link
            XPATH_part='//*[@id="match-'+link.rsplit('/',1)[1]
            try:    
                away_yellow=self.driver.find_element(By.XPATH, XPATH_part+config.AWAY_YELLOW_XPATH).text
                self.df.loc[i, ['Away_Yellow']]=away_yellow
            except:
                pass    
            
            try:
                away_red=self.driver.find_element(By.XPATH, XPATH_part+config.AWAY_RED_XPATH).text
                self.df.loc[i, ['Away_Red']]=away_red
            except:
                pass

            time.sleep(2)
        self.df.to_csv(config.SCRAPED_PENALTY_DATA_FILE)
        return self.df

    
    def scraping_missing_results(self, no_result_list):
        """if game result is NaN the scraper will update it from the provided link
        if results are not provided, the data on this match is deleted
        Scraped data is recorded into the interim data file.
        """
        for i in no_result_list:
            self.driver.get(i)
            time.sleep(2)

            try:
                home_result=self.driver.find_element(By.XPATH, '/html/body/main/section[1]/div[1]/section/div[2]/div[2]/div/span[1]').text
            except:
                pass

            try:
                away_result=self.driver.find_element(By.XPATH, '/html/body/main/section[1]/div[1]/section/div[2]/div[2]/div/span[2]').text
            except:
                pass

            time.sleep(2)
            self.df.loc[self.df['Link'] == i, 'Home_Score'] = int(home_result)
            self.df.loc[self.df['Link'] == i, 'Away_Score'] = int(away_result)
        self.df.to_csv(config.NEW_DATA_RESULTS_UPDATED_FILE, index=False) 
        return self.df


    


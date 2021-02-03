# Made by Amos Tai
# Date: 03/02/2021

# The goal for this project is to determine if the modern NBA starting from 2010-2011 season 
# until 2019-2020 season is truely positionless and if it isn't, can a player be classified 
# on his position based on their stats. The dataset is taken from Basketball Reference from 
# the 2010-2011 season until 2019-2020 season, which is a decade worth of data.



import streamlit as st

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline




def main():

    st.title('NBA Position Classifier')
    
    file_list = ['2010-2011.txt', '2011-2012.txt', '2012-2013.txt', '2013-2014.txt', '2014-2015.txt',
             '2015-2016.txt', '2016-2017.txt', '2017-2018.txt', '2018-2019.txt', '2019-2020.txt']

    df = read_multi(file_list)

    df.fillna(0.0, inplace=True)
    df['Pos'] = df['Pos'].apply(lambda x: x.split('-')[0] if '-' in x else x)
    
    
    
    # Sidebar
    st.sidebar.title('Option')
    side = st.sidebar.radio('Select', ('EDA','ML Model'))
    
    if side == 'EDA':
        run_eda(df)
        
        if st.sidebar.checkbox('Show Glossary'):
            st.sidebar.text("""PG: Point Guard
                            \nSG: Shooting Guard   
                            \nSF: Small Forward
                            \nPF: Power Forward
                            \nC: Center
                            \n------------------------------------------
                            \nAge : Player's age
                            \nG: Games
                            \nMP: Minutes Played Per Game
                            \nFG: Field Goals Per Game
                            \nFGA: Field Goal Attempts Per Game
                            \nFG%: Field Goal Percentage
                            \n3P: 3-Point Field Goals Per Game
                            \n3PA: 3-Point Field Goal Attempts Per Game
                            \n3P%: 3-Point Field Goal Percentage
                            \n2P: 2-Point Field Goals Per Game
                            \n2PA: 2-Point Field Goal Attempts Per Game
                            \n2P%: 2-Point Field Goal Percentage
                            \neFG%: Effective Field Goal Percentage
                            \nFT: Free Throws Per Game
                            \nFTA: Free Throw Attempts Per Game
                            \nFT%: Free Throw Percentage
                            \nORB: Offensive Rebounds Per Game
                            \nDRB: Defensive Rebounds Per Game
                            \nTRB: Total Rebounds Per Game
                            \nAST: Assists Per Game
                            \nSTL: Steals Per Game
                            \nBLK: Blocks Per Game
                            \nTOV: Turnovers Per Game
                            \nPF: Personal Fouls Per Game
                            \nPTS: Points Per Game
                            """)
        
    else:
        st.write('This model has an overall accuracy of 70% in predicting the position of an NBA player.') 

        st.info('''INFO: The numbers entered are the stats per game. It should reflect a player who has play on average of more than 40 games and more than 10 minutes per game in a single season.
            \nFor more details about this model, click [here](https://github.com/leftyamos/amost.github.io/blob/master/NBA.ipynb).''')
        
        run_model(df)
     
    

@st.cache(allow_output_mutation=True)        
def read_multi(file_list):
    '''
    take a list of files concat it together.
    
    '''
    
    list_df = []
    
    for file in (file_list):
        df = pd.read_csv(file, sep=',', header=0)
        
        df['Sea'] = file.split('.')[0]
        
        list_df.append(df)
    
    total_df = pd.concat(list_df, axis=0)
    
    return total_df



def run_eda(df):
    '''
    Visualize data
    '''
    years = ['2010-2011', '2011-2012', '2012-2013', '2013-2014', '2014-2015',
             '2015-2016', '2016-2017', '2017-2018', '2018-2019', '2019-2020']
    
    seasons = st.selectbox('Select season', years)
    
    fig = plt.figure(figsize=(10,5))
    filter_season = df[df['Sea']==seasons]['Pos'].value_counts().sort_index().loc[['PG','SG','SF','PF','C']].plot(kind='bar')
    plt.ylim(0,140)
    plt.xlabel('Positions')
    plt.ylabel('Players')
    plt.title(f'Total no. of players during {seasons} season')
    st.write(fig)
    
    row1_1, row1_2 = st.beta_columns((1,1))
    with row1_1:
        features = st.selectbox('Select feature', df.drop(['Rk','Pos','Player','Tm','GS','Sea'], axis=1).columns)
        
    with row1_2:
        positions = st.selectbox('Select position', ['PG','SG','SF','PF','C'])
    
    fig = plt.figure(figsize=(3,3))
    filter_fea_pos = sns.histplot(data=df[(df['Sea']==seasons) & (df['Pos']==positions)], x=features)
    plt.title(f'Histogram \n{seasons}: {positions}')
    st.write(fig)



@st.cache
def fit_model(df):
    # SVM Model
    X3 = df[['2P','2PA','3P','3PA','DRB','ORB','AST','STL','BLK']]
    y = df['Pos']

    svm_pipe = make_pipeline(MinMaxScaler(),SVC(C=1.0, gamma=10.0))
    
    svm_pipe.fit(X3, y)
    
    return svm_pipe
    
    

def run_model(df):
    
    svm_pipe = fit_model(df)


    # Inputs for model
    row1_1, row1_2 = st.beta_columns((1,1))
    with row1_1:
        two = st.number_input('2P')

    with row1_2:
        twoa = st.number_input('2PA')

    row2_1, row2_2 = st.beta_columns((1,1))
    with row2_1:
        three = st.number_input('3P')

    with row2_2:
        threea = st.number_input('3PA')

    row3_1, row3_2 = st.beta_columns((1,1))
    with row3_1:
        drb = st.number_input('DRB')

    with row3_2:
        orb = st.number_input('ORB')

    row4_1, row4_2, row4_3 = st.beta_columns((1,1,1))
    with row4_1:
        ast = st.number_input('AST')

    with row4_2:
        stl = st.number_input('STL')

    with row4_3:
        blk = st.number_input('BLK')



    # Model prediction
    player = [two,twoa,three,threea,drb,orb,ast,stl,blk]
    pred = svm_pipe.predict([player])



    # Exception
    if (two <=0) and (twoa <=0) and (three <=0) and (threea <=0) and (drb <=0) and (orb <=0) and (ast <=0) and (stl <=0) and (blk <= 0):
        pred = 0



    # Button to execute model
    if st.button('Predict Position'):
        if pred == ['PG']:
            st.success('Point Guard')
        elif pred == ['SG']:
            st.success('Shooting Guard')
        elif pred == ['SF']:
            st.success('Small Forward')
        elif pred == ['PF']:
            st.success('Power Forward')
        elif pred == ['C']:
            st.success('Center')
        elif pred == 0:
            st.error('Please enter some values')


            
if __name__ == "__main__":
    main()
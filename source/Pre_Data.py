import pandas as pd
import numpy as np
import os


"""
Combine all data from different categories into one file 
Add `category` column to record data's category into dataframe
- Input: a directory name(a dir which contains: information of products in 7 files with 7 categories)
- Output: return DataFrame contains all information products and write it down into a `All.csv` file
"""

file_data_name = 'ALL.csv'
file_rank_freq = 'ALL_rank_freq.csv'
file_full_data = 'ALL_MATCHED.csv'

file_cate_names = ['Bath-Bathing-Accessories.csv','Gift-Sets.csv','Hair-Care-Products.csv',\
    'Makeup.csv','Perfumes-Fragrances.csv','Skin-Care-Products.csv','Tools-Accessories.csv']
df_top_table = pd.concat([pd.DataFrame([['Top 10']*10 + ['Top 20'] *10 + ['Top 30'] *10 + ['Top 40'] *10 + ['Top 50'] *10 +\
    ['Top 60'] *10+ ['Top 70'] *10+ ['Top 80'] *10+ ['Top 90'] *10+ ['Top 100'] *10] \
    ,columns = range(1,11) + range(11,21) + range(21,31) + range(31,41) + range(41,51) +\
    range(51,61) + range(61,71) + range(71,81) + range(81,91) + range(91,101))], axis=1)

def combine_data(dir_name):    
    data_frames = []
    for file_name in file_cate_names:
        df = pd.DataFrame.from_csv(dir_name + file_name)
        df['category'] = file_name[:-4]
        data_frames.append(df)

    full_df = pd.concat(data_frames,ignore_index=True)
    
    num_top_col = 10
    temp = np.zeros(shape=(len(full_df),num_top_col))
    zero_df = pd.DataFrame(temp,columns=['Top 10','Top 20','Top 30','Top 40','Top 50','Top 60','Top 70','Top 80','Top 90','Top 100'])
    full_df = pd.concat([full_df,pd.get_dummies(full_df['category']),zero_df],axis=1)
    full_df.drop('category',axis = 1,inplace=True)
    
    full_df.to_csv(dir_name + file_data_name,sep=',', encoding='utf-8', index_label=None)
    return full_df

def count_rank_freq(data_idx): 
    i = data_idx
    print '***************** count_rank_freq Folder {0}******************'.format(i)
    cur_df = pd.DataFrame.from_csv(root_dir.format(i) + file_data_name)
    for j in range(1,i):
        pre_df = pd.DataFrame.from_csv(root_dir.format(j) + file_data_name)
        for asin in cur_df.asin.unique():
            if asin in pre_df.asin.unique():
                if pre_df.loc[pre_df['asin'] == asin,'rank'].values[0] == 101:
                    continue
                rank = pre_df.loc[pre_df['asin'] == asin,'rank'].values[0]
                top_rank = df_top_table[rank].values[0]
                cur_df.loc[cur_df['asin'] == asin,top_rank] += 1
    cur_df.to_csv(root_dir.format(i) + file_rank_freq)
    return cur_df

"""
Fill previous ranks of products into the current data frame
-Input: pre_df: previous dataframe, cur_df: current dataframe
-Output: return current dataframe filled the previous rank
"""
def fill_previous_rank(pre_df,cur_df):
    for asin in cur_df.asin:
        # Still remain best seller rank
        if asin in pre_df.asin.unique():
            cur_df.loc[cur_df['asin'] == asin,'previous rank'] = pre_df.loc[pre_df['asin'] == asin,'rank'].values[0]
        else: 
            cur_df.loc[cur_df['asin'] == asin,'previous rank'] = 101
    return cur_df



# Combine file ALL and ALL_COUNT_RANK into ALL_MATCHED
root_dir = '../DATA/data{0}/'
start_fd = 2
last_fd = 42 

combine_data(root_dir.format(1))
count_rank_freq(1)
for i in range(start_fd,last_fd):
    print '*****************Folder {0}******************'.format(i)
    
    combine_data(root_dir.format(i))
    count_rank_freq(i)
    
    cur_df = pd.DataFrame.from_csv(root_dir.format(i) + file_rank_freq)
    pre_df = pd.DataFrame.from_csv(root_dir.format(i-1) + file_rank_freq)
    cur_df = fill_previous_rank(pre_df,cur_df)
    cur_df.to_csv(root_dir.format(i) + file_full_data,encoding='utf-8', index_label=None)

#Only use folder 16 - 40 to gain the best performance
start_id = 16
last_id = 41 
root_file = '../DATA/data{0}/ALL_MATCHED.csv'
dataframes = []
for i in range(start_id,last_id):
    folder_name = root_file.format(i)
    df = pd.DataFrame.from_csv(folder_name)
    dataframes.append(df)
full_df =  pd.concat(dataframes)
    
full_df.to_csv('../DATA.csv')

# root_dir = '../DATA/data{0}/'
# start_fd = 1
# last_fd = 42 

# for i in range(start_fd,last_fd):
#     if os.path.isfile(root_dir.format(i) + 'ALL_MATCHED.csv'):
#         os.remove(root_dir.format(i) + 'ALL_MATCHED.csv')
#     if os.path.isfile(root_dir.format(i) + 'ALL_rank_freq.csv'):
#         os.remove(root_dir.format(i) + 'ALL_rank_freq.csv')
#     if os.path.isfile(root_dir.format(i) + 'ALL.csv'):
#         os.remove(root_dir.format(i) + 'ALL.csv')
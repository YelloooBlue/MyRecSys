import pickle
import pandas as pd

# 将行JSON文件转换为DataFrame
def to_df(file_path):
  with open(file_path, 'r') as fin:
    df = {}
    i = 0
    for line in fin:
      df[i] = eval(line) # eval 等同于 json.loads(line)
      i += 1
    df = pd.DataFrame.from_dict(df, orient='index')
    return df
  

raw_data_path = 'DIN.2017/raw_data/' # 修改官方代码方便自定义数据集位置

# 处理Reviews数据
reviews_df = to_df(f'{raw_data_path}/reviews_Electronics_5.json')
with open(f'{raw_data_path}/reviews.pkl', 'wb') as f:
  pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL)

# 处理Meta数据
meta_df = to_df(f'{raw_data_path}/meta_Electronics.json')
meta_df = meta_df[meta_df['asin'].isin(reviews_df['asin'].unique())] # 只保留reviews中存在的asin
meta_df = meta_df.reset_index(drop=True)
with open(f'{raw_data_path}/meta.pkl', 'wb') as f:
  pickle.dump(meta_df, f, pickle.HIGHEST_PROTOCOL)

'''
  mkdir DIN.2017/raw_data
  cd DIN.2017/raw_data/
  wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz
  gzip -d reviews_Electronics_5.json.gz
  wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Electronics.json.gz
  gzip -d meta_Electronics.json.gz
'''
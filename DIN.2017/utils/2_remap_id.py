import random
import pickle
import numpy as np

random.seed(1234)

raw_data_path = 'DIN.2017/raw_data/'

# 打开上一步处理好的DataFrame
with open(raw_data_path + 'reviews.pkl', 'rb') as f:
  reviews_df = pickle.load(f)
  reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]
with open(raw_data_path + 'meta.pkl', 'rb') as f:
  meta_df = pickle.load(f)
  meta_df = meta_df[['asin', 'categories']]
  meta_df['categories'] = meta_df['categories'].map(lambda x: x[-1][-1]) # 只保留最后一级分类

"""
       reviewerID        asin  unixReviewTime
0   AO94DHGC771SJ  0528881469      1370131200
1   AMO214LNFCEI4  0528881469      1290643200
2  A3N7T0DY83Y4IG  0528881469      1283990400
3  A1H8PY3QHMQQA0  0528881469      1290556800
4  A24EV6RXELQZ63  0528881469      1317254400
         asin                   categories
0  0528881469                 Trucking GPS
1  0594451647          Chargers & Adapters
2  0594481813               Power Adapters
3  0972683275     TV Ceiling & Wall Mounts
4  1400532620  eBook Readers & Accessories
"""

# 对df的col_name列进行remap-id（即将id映射到0, 1, 2, ...）
def build_map(df, col_name):
  key = sorted(df[col_name].unique().tolist())    # 取出所有出现的记录
  m = dict(zip(key, range(len(key))))             # 建立{key: id}的映射
  df[col_name] = df[col_name].map(lambda x: m[x]) # 将col_name列的值映射到id
  return m, key

asin_map, asin_key = build_map(meta_df, 'asin')           # asin_map为{asin: id}的映射，asin_key为所有asin的列表
cate_map, cate_key = build_map(meta_df, 'categories')
revi_map, revi_key = build_map(reviews_df, 'reviewerID')

"""
   reviewerID        asin  unixReviewTime
0      176008  0528881469      1370131200
1      173739  0528881469      1290643200
2      134504  0528881469      1283990400
3       24476  0528881469      1290556800
4       57419  0528881469      1317254400
   asin  categories
0     0         738
1     1         157
2     2         571
3     3         707
4     7         799
"""

# 统计用户数、物品数、分类数和样本数
user_count, item_count, cate_count, example_count =\
    len(revi_map), len(asin_map), len(cate_map), reviews_df.shape[0]

print('user_count: %d\titem_count: %d\tcate_count: %d\texample_count: %d' %
      (user_count, item_count, cate_count, example_count))


# 整理Meta
meta_df = meta_df.sort_values('asin')     # 此时asin已是从0开始的id
meta_df = meta_df.reset_index(drop=True)

# 将reviews_df中的asin映射到id
reviews_df['asin'] = reviews_df['asin'].map(lambda x: asin_map[x])

# 整理Reviews
reviews_df = reviews_df.sort_values(['reviewerID', 'unixReviewTime'])
reviews_df = reviews_df.reset_index(drop=True)
reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]

# iid到cid的映射（前面已经对asin（iid）进行了排序，所以行号就是iid）
cate_list = [meta_df['categories'][i] for i in range(len(asin_map))]
cate_list = np.array(cate_list, dtype=np.int32)

with open(raw_data_path + 'remap.pkl', 'wb') as f:
  pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL) # 用户uid, 物品iid, 时间戳
  pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL) # 列表，index为 物品iid 对应的类别cid
  pickle.dump((user_count, item_count, cate_count, example_count), # 统计信息
              f, pickle.HIGHEST_PROTOCOL)
  pickle.dump((asin_key, cate_key, revi_key), f, pickle.HIGHEST_PROTOCOL) 
    # 原始的asin, categories, reviewerID列表
    # 虽然是列表，index可以被视为id，也就是从id到asin原始值的映射
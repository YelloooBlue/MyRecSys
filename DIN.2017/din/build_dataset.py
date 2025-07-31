import random
import pickle

random.seed(1234)

raw_data_path = 'DIN.2017/raw_data/'
output_data_path = 'DIN.2017/'

# 从remap好的数据集中读取数据
with open(raw_data_path + 'remap.pkl', 'rb') as f:
  reviews_df = pickle.load(f)
  cate_list = pickle.load(f)
  user_count, item_count, cate_count, example_count = pickle.load(f)

"""
reviews_df:
   reviewerID   asin  unixReviewTime
0           0  13179      1400457600
1           0  17993      1400457600
2           0  28326      1400457600
3           0  29247      1400457600
4           0  62275      1400457600
"""

train_set = []
test_set = []

"""
group by reviewerID:
         reviewerID   asin  unixReviewTime
0                 0  13179      1400457600
1                 0  17993      1400457600
2                 0  28326      1400457600
3                 0  29247      1400457600
4                 0  62275      1400457600
...             ...    ...             ...
1689156      192402  24851      1262044800
1689157      192402  29488      1282262400
1689158      192402  36004      1357776000
1689159      192402  37977      1357776000
1689160      192402  39411      1357776000

"""

for reviewerID, hist in reviews_df.groupby('reviewerID'):

    # 取出用户历史交互序列
    pos_list = hist['asin'].tolist()    # 将一个用户对应的所有记录取出，并将asin列转换为列表
  
    # 生成负样本函数
    def gen_neg():
        neg = pos_list[0] # 选取一个正样本以进入循环

        # 确保不在正样本列表中
        while neg in pos_list:
            neg = random.randint(0, item_count-1)
        return neg
  
    # 生成与正样本数量相同的负样本列表
    neg_list = [gen_neg() for i in range(len(pos_list))]

    """
        此时有
        pos_list: [13179, 17993, 28326, 29247, 62275]
        neg_list: [50997, 28883, 7657, 490, 5940]
    """

    # 生成长度不同的训练集和测试集
    for i in range(1, len(pos_list)):

        # 根据 第0-i-1 个正样本构建历史
        hist = pos_list[:i]

        # 第i个分别作为正样本和负样本，加上标签
        if i != len(pos_list) - 1:
            train_set.append((reviewerID, hist, pos_list[i], 1))
            train_set.append((reviewerID, hist, neg_list[i], 0))
        else:
            label = (pos_list[i], neg_list[i])
            test_set.append((reviewerID, hist, label))

    """
        划分训练集和测试集，假设序列长度都为5
        其中训练集包含长度1-3的hist，一个正样本和一个负样本，
        train_set_0: (0, [13179], 17993, 1) # 正样本
        train_set_1: (0, [13179], 28883, 0) # 负样本

        测试集则利用最大序列长度-1的hist和（正，负）样本对
        test_set_0: (0, [13179, 17993, 28326, 29247], (62275, 5940))
    """

# 打乱训练集和测试集
random.shuffle(train_set)
random.shuffle(test_set)

# 确保测试集的用户数与用户总数一致
assert len(test_set) == user_count
# assert(len(test_set) + len(train_set) // 2 == reviews_df.shape[0])

# 打包数据集
with open(output_data_path + 'dataset.pkl', 'wb') as f:
  pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump((user_count, item_count, cate_count), f, pickle.HIGHEST_PROTOCOL)
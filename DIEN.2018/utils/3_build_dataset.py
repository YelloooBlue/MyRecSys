import pickle
import random
from pathlib import Path

random.seed(1234)

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = ROOT_DIR / "raw_data"
OUTPUT_DATA_DIR = ROOT_DIR

with open(RAW_DATA_DIR / "remap.pkl", "rb") as f:
    reviews_df = pickle.load(f)
    cate_list = pickle.load(f)
    user_count, item_count, cate_count, _ = pickle.load(f)


train_set = []
test_set = []

for reviewer_id, hist_df in reviews_df.groupby("reviewerID"):
    pos_list = hist_df["asin"].tolist()
    if len(pos_list) < 2:
        continue

    pos_set = set(pos_list)

    def gen_neg_item() -> int:
        neg_item = pos_list[0]
        while neg_item in pos_set:
            neg_item = random.randint(0, item_count - 1)
        return neg_item

    # One negative item per timestep, used by DIEN auxiliary loss.
    neg_list = [gen_neg_item() for _ in range(len(pos_list))]

    for i in range(1, len(pos_list)):
        hist_items = pos_list[:i]
        neg_hist_items = neg_list[:i]

        if i != len(pos_list) - 1:
            train_set.append((reviewer_id, hist_items, neg_hist_items, pos_list[i], 1))
            train_set.append((reviewer_id, hist_items, neg_hist_items, neg_list[i], 0))
        else:
            test_set.append((reviewer_id, hist_items, neg_hist_items, (pos_list[i], neg_list[i])))

random.shuffle(train_set)
random.shuffle(test_set)

assert len(test_set) == user_count

with open(OUTPUT_DATA_DIR / "dataset.pkl", "wb") as f:
    pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((user_count, item_count, cate_count), f, pickle.HIGHEST_PROTOCOL)

print(f"Wrote: {OUTPUT_DATA_DIR / 'dataset.pkl'}")

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = 'data/kddcup.data.corrected'
TEST_PATH = 'data/corrected'
FEATURE_NAMES = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
                 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
                 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
                 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
                 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
                 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']
CAT_FEATURES = ['protocol_type', 'service', 'flag']
CAT_ENCODERS = {}


def load_kdd_train_set():
    df = pd.read_csv(DATA_PATH, names=FEATURE_NAMES + ['label'])
    return preprocess_dataset(df, is_train=True)


def load_kdd_test_set():
    df_test = pd.read_csv(TEST_PATH, names=FEATURE_NAMES + ['label'])
    return preprocess_dataset(df_test, is_train=False)


def preprocess_dataset(df, is_train=True):
    new_data = df

    if not is_train and len(CAT_ENCODERS) == 0:
        raise Exception('Need to load training first for LabelEncoders init')

    y = (df['label'] == 'normal.').astype(int)
    y[np.where(df['label'] != 'normal.')[0]] = -1

    for cat_feat in CAT_FEATURES:
        if is_train:
            CAT_ENCODERS[cat_feat] = OneHotEncoder(handle_unknown='ignore', sparse=False).fit(
                df[cat_feat].values.reshape(-1, 1))
        new_feats_df = pd.DataFrame(CAT_ENCODERS[cat_feat].transform(df[cat_feat].values.reshape(-1, 1)),
                                    columns=CAT_ENCODERS[cat_feat].categories_[0])
        new_data = pd.concat([new_data.drop(columns=cat_feat), new_feats_df], axis=1)

    X = new_data.drop(columns=['label']).values

    return X, y

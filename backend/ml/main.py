# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 100 Sports Image Classification

# %% [markdown]
# ## データのインポート

# %%
# 初回のみ実行

# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("gpiosenka/sports-classification")

# print("Path to dataset files:", path)

# %% [markdown]
# ## データの表示

# %%
import pandas as pd

data_path = './kagglehub_cache/datasets/gpiosenka/sports-classification/versions/9/'
csv_path = data_path + 'sports.csv'
df = pd.read_csv(csv_path)
df.head()

# %% [markdown]
# ### カテゴリ列のユニーク値チェック

# %%
# categorical_columns = ["labels",'data set']
categorical_columns = ['data set']
for col in categorical_columns:
    if col in df.columns:
        unique_values = df[col].value_counts()
        print(f"{col} 列のユニーク値 ({len(unique_values)} 個):")
        for value, count in unique_values.items():
            print(f"  {value}: {count} 件")

# %% [markdown]
# ## データの分離

# %%
df_train = df[df['data set'] == 'train']
df_test = df[df['data set'] == 'test']
df_valid = df[df['data set'] == 'valid']

display(df_train.head())
# display(df_test.head())
# display(df_valid.head())

# %% [markdown]
# ## CNN

# %%
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# %% [markdown]
# ### データの前処理

# %% [markdown]
# #### 訓練用データセットの前処理

# %%
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
train_set = train_datagen.flow_from_directory(data_path + 'train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')
# 2分類の場合は class_mode = 'binary' を指定
# 多分類の場合は class_mode = 'categorical' を指定

# %% [markdown]
# #### 検証用データセットの前処理

# %%
valid_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
valid_set = valid_datagen.flow_from_directory(data_path + 'valid',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

# %% [markdown]
# #### テストデータセットの前処理

# %%
test_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_set = test_datagen.flow_from_directory(data_path + 'test',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

# %% [markdown]
# ### CNNの構築

# %% [markdown]
# #### イニシャライズ

# %%
cnn = tf.keras.models.Sequential()

# %% [markdown]
# #### 畳み込みandプーリング

# %%
# 一層目
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
cnn.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))

# 二層目
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))

# %% [markdown]
# #### Flattening

# %%
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# %% [markdown]
# #### 出力層の追加

# %%
cnn.add(tf.keras.layers.Dense(units=100, activation='softmax'))
# 2分類の場合は units=1, activation='sigmoid' を指定
# 多分類の場合は units=[number], activation='softmax' を指定

# %% [markdown]
# ### モデル学習

# %% [markdown]
# #### モデルのコンパイルと訓練

# %%
cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# 2分類の場合は loss = 'binary_crossentropy' を指定
# 多分類の場合は loss = 'categorical_crossentropy' を指定

# %% [markdown]
# #### モデルの訓練

# %%
cnn.fit(x = train_set, validation_data = valid_set, epochs = 25)

# %% [markdown]
# ## 結果の出力

# %%
import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from tqdm import tqdm

# クラス名のマッピング（index → label）
class_indices = train_set.class_indices
class_labels = {v: k for k, v in class_indices.items()}

# テスト画像のルート
test_root = os.path.join(data_path, 'test')

# 結果格納用リスト
results = []

image_id = 1

for true_class in sorted(os.listdir(test_root)):
    class_dir = os.path.join(test_root, true_class)
    if not os.path.isdir(class_dir):
        continue  # ディレクトリでないものをスキップ

    for fname in sorted(os.listdir(class_dir)):
        fpath = os.path.join(class_dir, fname)
        if not fpath.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        # 画像読み込みと前処理
        img = image.load_img(fpath, target_size=(64, 64))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # 予測実行
        preds = cnn.predict(img_array, verbose=0)[0]
        top_indices = preds.argsort()[-3:][::-1]  # 上位3件（降順）

        top_labels = [class_labels[i] for i in top_indices]

        # 結果記録
        results.append([
            image_id,
            true_class,
            top_labels[0],
            top_labels[1],
            top_labels[2]
        ])

        image_id += 1

# DataFrame化して表示
df_results = pd.DataFrame(results, columns=["画像ID", "正解", "Top1予測", "Top2予測", "Top3予測"])
# print(df_results.head())  # 一部表示
pd.set_option('display.max_rows', None) # 全行表示
display(df_results)

df_results.to_csv("prediction_results.csv", index=False)
print("CSVファイルとして 'prediction_results.csv' を出力しました。")


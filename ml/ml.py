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
# ## 設定値

# %%
EPOCHS = 50 # エポック数
IMG_SIZE = 64 # 画像のサイズ
BATCH_SIZE = 32 # バッチサイズ
FILTERS_SIZE = 32 # フィルタサイズ
OUTPUT_LAYER_SIZE = 100 # 出力層のサイズ

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
                                                 target_size = (IMG_SIZE, IMG_SIZE),
                                                 batch_size = BATCH_SIZE,
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
                                                 target_size = (IMG_SIZE, IMG_SIZE),
                                                 batch_size = BATCH_SIZE,
                                                 class_mode = 'categorical')

# %% [markdown]
# #### テストデータセットの前処理

# %%
test_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_set = test_datagen.flow_from_directory(data_path + 'test',
                                                 target_size = (IMG_SIZE, IMG_SIZE),
                                                 batch_size = BATCH_SIZE,
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
cnn.add(tf.keras.layers.Conv2D(filters=FILTERS_SIZE, kernel_size=3, activation='relu', input_shape=[IMG_SIZE, IMG_SIZE, 3]))
cnn.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))

# 二層目
cnn.add(tf.keras.layers.Conv2D(filters=FILTERS_SIZE, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))

# %% [markdown]
# #### Flattening

# %%
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# %% [markdown]
# #### 出力層の追加

# %%
cnn.add(tf.keras.layers.Dense(units=OUTPUT_LAYER_SIZE, activation='softmax'))
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
cnn.fit(x = train_set, validation_data = valid_set, epochs = EPOCHS)

# %% [markdown]
# ## 結果の出力

# %%
import os
import json

# クラス名のマッピング（index → label）
class_indices = train_set.class_indices

# 保存用ディレクトリ
model_path = './ml'
os.makedirs(model_path, exist_ok=True)

# class_indices を JSON に保存（例: { 'air hockey': 0, 'archery': 1, ... }）
with open(os.path.join(model_path, "class_indices.json"), "w") as f:
    json.dump(class_indices, f, indent=2, ensure_ascii=False)
print("クラスインデックス（class_indices）を JSON 出力しました。")

# モデル保存
model_filename = f'model_{EPOCHS}epochs.h5'
cnn.save(os.path.join(model_path, model_filename))
print(f"モデルを保存しました: {model_filename}")

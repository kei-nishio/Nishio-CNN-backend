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

# %% [markdown]
# # 100 Sports Image Classification

# %% [markdown]
# ## データのインポート

# %%
# import kagglehub
# path = kagglehub.dataset_download("gpiosenka/sports-classification")
# print("Path to dataset files:", path)

# %% [markdown]
# ## 設定値

# %%
EPOCHS = 25  # エポック数を増加して学習を安定化
IMG_SIZE = 128  # 画像のサイズ（224は転移学習向けの標準。ConvNet自作では128程度がバランス良）
BATCH_SIZE = 8  # バッチサイズを小さくして検証データの少なさに対応

# %% [markdown]
# ## データの表示

import datetime
import json
import os

# %%
import pandas as pd

# data_path = './kagglehub_cache/datasets/gpiosenka/sports-classification/versions/9/'
data_path = "./test_data/"
csv_path = data_path + "sports.csv"
df = pd.read_csv(csv_path)
df.head()

# %% [markdown]
# ### カテゴリ列のユニーク値チェック

# %%
categorical_columns = ["data set"]
for col in categorical_columns:
    if col in df.columns:
        unique_values = df[col].value_counts()
        print(f"{col} 列のユニーク値 ({len(unique_values)} 個):")
        for value, count in unique_values.items():
            print(f"  {value}: {count} 件")

# %% [markdown]
# ## データの分離

# %%
df_train = df[df["data set"] == "train"]
df_test = df[df["data set"] == "test"]
df_valid = df[df["data set"] == "valid"]

# %% [markdown]
# ## CNN

# %%
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Kerasのインポートを明示的に行う

# %% [markdown]
# ### データの前処理

# %%
# 過学習を抑制するため、データ拡張を抑制
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.1,  # 減少
    zoom_range=0.1,  # 減少
    horizontal_flip=True,
    rotation_range=10,
)  # 追加
train_set = train_datagen.flow_from_directory(
    data_path + "train",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
)

valid_datagen = ImageDataGenerator(rescale=1.0 / 255)
valid_set = valid_datagen.flow_from_directory(
    data_path + "valid",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)  # 検証データはシャッフルしない

test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_set = test_datagen.flow_from_directory(
    data_path + "test",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)  # テストデータはシャッフルしない

# %%
OUTPUT_LAYER_SIZE = len(train_set.class_indices)
print(f"検出されたクラス数: {OUTPUT_LAYER_SIZE}")

# %% [markdown]
# ### CNNの構築

# %%
# 過学習対策を強化したCNNモデル
cnn = tf.keras.models.Sequential(
    [
        tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        # 最初の畳み込みブロック - フィルタ数を減らす
        tf.keras.layers.Conv2D(
            filters=16, kernel_size=3, activation="relu", padding="same"
        ),
        tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.25),  # Dropout追加
        # 2番目の畳み込みブロック
        tf.keras.layers.Conv2D(
            filters=32, kernel_size=3, activation="relu", padding="same"
        ),
        tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.25),  # Dropout追加
        # 3番目の畳み込みブロック
        tf.keras.layers.Conv2D(
            filters=64, kernel_size=3, activation="relu", padding="same"
        ),
        tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.25),  # Dropout追加
        # 全結合層
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(
            units=128, activation="relu"
        ),  # ユニット数を減らす
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(units=OUTPUT_LAYER_SIZE, activation="softmax"),
    ]
)

# %%
cnn.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  # 学習率を下げる
    loss="categorical_crossentropy",
    metrics=[
        tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
        tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc"),
        tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_acc"),
    ],
)

# %%
# Early Stoppingコールバックを追加
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

history = cnn.fit(
    x=train_set,
    validation_data=valid_set,
    epochs=EPOCHS,
    callbacks=[early_stopping],
)

# %% 評価と保存
eval_results = cnn.evaluate(valid_set)
print("\n===== 評価結果 (evaluate) =====")
metrics_report = dict(zip(cnn.metrics_names, eval_results))
for name, value in metrics_report.items():
    print(f"{name}: {value:.4f}")

print("\n===== 最終バリデーションスコア（historyベース）=====")
for metric in ["val_accuracy", "val_top3_acc", "val_top5_acc"]:
    val_list = history.history.get(metric)
    if val_list:
        print(f"{metric} : {val_list[-1]:.4f}")
    else:
        print(f"{metric} : 指標が見つかりません")

# %% 保存
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = os.path.join("./ml/model", f"run_{timestamp}")
os.makedirs(save_dir, exist_ok=True)

with open(os.path.join(save_dir, "class_indices.json"), "w") as f:
    json.dump(train_set.class_indices, f, indent=2, ensure_ascii=False)

cnn.save(os.path.join(save_dir, "model.keras"))
cnn.save(os.path.join(save_dir, "model.h5"))

config = {
    "epochs": EPOCHS,
    "img_size": IMG_SIZE,
    "batch_size": BATCH_SIZE,
    "output_layer_size": OUTPUT_LAYER_SIZE,
    "save_format": ["keras", "h5"],
    "metrics": metrics_report,
}
with open(os.path.join(save_dir, "config.json"), "w") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print("\nモデル・設定を保存しました。保存先:", save_dir)

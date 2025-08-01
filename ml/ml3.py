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
# ## Google Drive mount for Google Colab
# %%
# from google.colab import drive
# drive.mount('/content/drive')

# %% [markdown]
# ## for Google Colab
# %%
# # !pwd
# # !pip install tensorflow
# # !nvidia-smi

# %% [markdown]
# ## データのインポート
# %%
# import kagglehub
# path = kagglehub.dataset_download("gpiosenka/sports-classification")
# print("Path to dataset files:", path)

# %% [markdown]
# ## Config
# %%
EPOCHS = 100
IMG_SIZE = 224
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# %% [markdown]
# ## Import
# %%
import datetime
import json
import os
import pandas as pd

# %% [markdown]
# ## データの表示

# %%
# data_path = '/content/drive/MyDrive/python/_test_cnn/archive/' # for google colab
# data_path = '/content/drive/MyDrive/python/_test_cnn/archive_mini/' # for google colab
data_path = './kagglehub_cache/datasets/gpiosenka/sports-classification/versions/9/'
# data_path = "./test_data/"
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

# %% [markdown]
# ### データの前処理
# %%
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    validation_split=0.2,  # 20%をvalidationに分割
)

# トレーニングデータ（80%）
train_set = train_datagen.flow_from_directory(
    data_path + "train",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",  # トレーニング用のサブセット
)

# バリデーションデータ（20%）- データ拡張なし
valid_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)
valid_set = valid_datagen.flow_from_directory(
    data_path + "train",  # 同じtrainフォルダから分割
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",  # バリデーション用のサブセット
    shuffle=False,
)

# テストデータ（元のtestフォルダを使用）
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_set = test_datagen.flow_from_directory(
    data_path + "test",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)

# %% [markdown]
# ### クラス数の検出
# %%
OUTPUT_LAYER_SIZE = len(train_set.class_indices)
print(f"検出されたクラス数: {OUTPUT_LAYER_SIZE}")

# %% [markdown]
# ### データの概要確認
# %%
print("\n===== クラスインデックス確認 =====")
print("train:", train_set.class_indices)
print("valid:", valid_set.class_indices)
print("test :", test_set.class_indices)

print("\n===== データセットサイズ確認 =====")
print(f"train samples: {train_set.samples} (trainフォルダの80%)")
print(f"valid samples: {valid_set.samples} (trainフォルダの20%)")
print(f"test samples: {test_set.samples} (元のtestフォルダ)")
print(f"batch size: {BATCH_SIZE}")
print(f"valid batches per epoch: {len(valid_set)}")
print(f"train batches per epoch: {len(train_set)}")

# validationセットサイズの確認
if valid_set.samples >= 200:
    print(f"\n✅ Validationデータが充分あります（{valid_set.samples}枚）")
    print("   - 安定した評価が期待できます")
elif valid_set.samples >= 100:
    print(f"\n🟡 Validationデータは最低限あります（{valid_set.samples}枚）")
    print("   - 評価はある程度安定します")
else:
    print(
        f"\n⚠️  WARNING: Validationデータがまだ少ないです（{valid_set.samples}枚）"
    )
    print("   - より多くのtrainデータが必要かもしれません")

# %% [markdown]
# ### CNNの構築
# %%
cnn = tf.keras.models.Sequential(
    [
        # input layer
        tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        # Block 1
        tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.BatchNormalization(),

        # Block 2
        tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.25),

        # Block 3
        tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
        tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.BatchNormalization(),

        # Block 4
        tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
        tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.25),

        # Dense layers
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1024, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        # Output layer
        tf.keras.layers.Dense(units=OUTPUT_LAYER_SIZE, activation="softmax"),
    ]
)

# %% [markdown]
# ### モデルのコンパイル
# %%
cnn.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="categorical_crossentropy",
    metrics=[
        tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
        tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc"),
        tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_acc"),
    ],
)

# %% [markdown]
# ### Early Stopping
# %%
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
)


# %% [markdown]
# ### 学習率スケジューラー
# %%
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=7, min_lr=1e-7, verbose=1
)

# %% [markdown]
# ### モデル構造の概要
# %%
print("\n===== モデル構造 =====")
cnn.summary()

print("\n===== 学習設定 =====")
print(f"学習率: {LEARNING_RATE}")
print(f"バッチサイズ: {BATCH_SIZE}")
print(f"画像サイズ: {IMG_SIZE}x{IMG_SIZE}")
print(f"最大エポック: {EPOCHS}")

# %% [markdown]
# ### 学習の実行
# %%
history = cnn.fit(
    x=train_set,
    validation_data=valid_set,
    epochs=EPOCHS,
    callbacks=[early_stopping, reduce_lr],
    verbose=1,
)

# %% [markdown]
# ### 評価と保存（＋分類レポートと混同行列）
# %%
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %%
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

# 正解ラベルと予測ラベル
y_true = valid_set.classes
y_pred_prob = cnn.predict(valid_set, verbose=0)
y_pred = np.argmax(y_pred_prob, axis=1)

# 分類レポート
print("\n===== Classification Report =====")
print(classification_report(y_true, y_pred, target_names=list(valid_set.class_indices.keys())))

# 混同行列
print("\n===== Confusion Matrix =====")
cm = confusion_matrix(y_true, y_pred)
print(cm)

# 可視化（オプション）
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=False, cmap="Blues", fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### データの保存
# %%
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
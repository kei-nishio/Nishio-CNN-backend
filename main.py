import json
from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from tensorflow import keras

# パス設定
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / 'ml' / 'model.h5'
CLASS_INDICES_PATH = BASE_DIR / 'ml' / 'class_indices.json'

# 設定値
IMG_SIZE = (64, 64)  # ml.ipynbで学習したモデルのサイズに合わせる
CONFIDENCE_THRESHOLD = 0.3  # この値未満は「不明」とする

app = FastAPI(title='Sports Classification API', version='1.0.0')

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],  # 本番環境では特定のオリジンのみを許可すること
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# グローバル変数でモデルとクラス名を保持
model = None
class_names = {}


@app.on_event('startup')
async def load_model():
    """アプリ起動時にモデルとクラス情報を読み込み"""
    global model, class_names

    try:
        # モデル読み込み
        model = keras.models.load_model(str(MODEL_PATH))

        # クラス名読み込み
        with open(CLASS_INDICES_PATH, 'r', encoding='utf-8') as f:
            class_indices = json.load(f)

        # インデックス -> クラス名の辞書を作成
        class_names = {v: k for k, v in class_indices.items()}

        print(
            f'モデルとクラス情報を正常に読み込みました。クラス数: {len(class_names)}'
        )

    except Exception as e:
        print(f'モデル読み込みエラー: {e}')
        raise


def preprocess_image(image_data: bytes) -> np.ndarray:
    """画像データを前処理してモデル入力形式に変換"""
    try:
        # PILで画像を開く
        image = Image.open(BytesIO(image_data))

        # RGBに変換（RGBA等の場合）
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # リサイズ
        image = image.resize(IMG_SIZE)

        # numpy配列に変換
        img_array = np.array(image)

        # バッチ次元を追加し、正規化
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array.astype('float32') / 255.0

        return img_array

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f'画像の前処理でエラーが発生しました: {str(e)}',
        )


@app.get('/health')
async def health_check():
    """ヘルスチェックエンドポイント"""
    return {'status': 'healthy', 'model_loaded': model is not None}


@app.post('/predict')
async def predict_sport(file: UploadFile = File(...)):
    """画像からスポーツを分類する"""

    if model is None:
        raise HTTPException(
            status_code=500, detail='モデルが読み込まれていません'
        )

    # ファイル形式チェック
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, detail='画像ファイルをアップロードしてください'
        )

    try:
        # 画像データ読み込み
        image_data = await file.read()

        # 前処理
        processed_image = preprocess_image(image_data)

        # 予測実行
        predictions = model.predict(processed_image)

        # 最も確率の高いクラスを取得
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])

        # 確信度が低い場合は「不明」とする
        if confidence < CONFIDENCE_THRESHOLD:
            predicted_sport = '不明'
            confidence = 0.0
        else:
            predicted_sport = class_names.get(predicted_class_idx, '不明')

        # 上位3つの結果も取得
        top_3_indices = np.argsort(predictions[0])[::-1][:3]
        top_3_results = []

        for idx in top_3_indices:
            sport_name = class_names.get(idx, '不明')
            score = float(predictions[0][idx])
            if score >= CONFIDENCE_THRESHOLD:
                top_3_results.append(
                    {'sport': sport_name, 'confidence': round(score, 4)}
                )

        return JSONResponse(
            {
                'predicted_sport': predicted_sport,
                'confidence': round(confidence, 4),
                'top_3_predictions': top_3_results,
                'filename': file.filename,
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f'予測処理でエラーが発生しました: {str(e)}'
        )


@app.get('/classes')
async def get_classes():
    """利用可能なスポーツクラス一覧を取得"""
    return {
        'total_classes': len(class_names),
        'classes': list(class_names.values()),
    }


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=8000)

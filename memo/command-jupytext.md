慧さん、了解です！
ここでは、`jupytext` を使って **`.ipynb ⇄ .py` の双方向変換（相互変換）をコードで実行する方法**を、以下にまとめます。

---

## ✅ 1. `jupytext` をインストール（まだの場合）

```bash
pip install jupytext
```

---

## ✅ 2. 相互変換の基本コード

以下のように Python スクリプト内からも `jupytext` を使って変換できます。

### 📄 `ipynb → py` に変換（Jupyter Notebook → Python スクリプト）

```python
import jupytext

# Notebookを読み込み
nb = jupytext.read("your_notebook.ipynb")

# Pythonスクリプト形式（# %% セル付き）で保存
jupytext.write(nb, "your_script.py", fmt="py:percent")
```

---

### 📄 `py → ipynb` に変換（Pythonスクリプト → Jupyter Notebook）

```python
import jupytext

# スクリプトを読み込み（# %% セルあり）
nb = jupytext.read("your_script.py")

# Notebook形式で保存
jupytext.write(nb, "your_notebook.ipynb")
```

---

## ✅ 3. コマンドラインでの相互変換

### ipynb → py（セル形式つき）

```bash
jupytext --to py:percent your_notebook.ipynb
```

### py → ipynb

```bash
jupytext --to notebook your_script.py
```

---

## ✅ 4. ペア運用（同期）する場合

```bash
# ノートブックを .py:percent とペア化して、保存時に両方更新されるようにする
jupytext --set-formats ipynb,py:percent your_notebook.ipynb
```

この設定にすると：

* `your_notebook.ipynb`
* `your_notebook.py`

の両方が **リンクされて常に同期される状態**になります。

---

## ✅ 5. 注意点

| 注意点                                  | 内容        |
| ------------------------------------ | --------- |
| `py:percent` を指定することで `# %%` セル形式になる | VSCode対応可 |
| `py:light` は jupytext 独自の簡略形式        | Git管理向き   |
| `.ipynb` を Git で扱うなら `.py` ペア化が便利    | 差分が見やすい   |

---

## 🔚 おすすめ運用

```bash
# 1. 一度だけ変換
jupytext --to py:percent your_notebook.ipynb

# 2. その後ペア運用（常時同期）
jupytext --set-formats ipynb,py:percent your_notebook.ipynb
```

これで `.ipynb` を編集・保存するたびに `.py` も更新されて、Git管理にも最適です！

---

必要であれば、**変換を自動化するMakefileやPythonスクリプト**もご用意できますよ！
「日次で全部変換」みたいな運用もOKです。

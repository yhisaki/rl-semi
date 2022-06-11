# 強化学習 新人ゼミ

[小野研究室](http://www.ic.dis.titech.ac.jp/main/doku.php)での新人ゼミで用いた資料．
テーブル型強化学習~DDPGまでの内容を扱っています．

## ドキュメントのビルド

ドキュメントの作成は[jupyter-book](https://jupyterbook.org/en/stable/intro.html)を用いて行っていきます．

- venvの作成~Activate(optional)
  ```
  python3 -m venv .venv
  source .venv/bin/activate
  ```

- 必要なパッケージのインストール
  ```
  pip -r requirements.txt
  ```

- ドキュメントのビルド
  ```
  jb build .
  ```

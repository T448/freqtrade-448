# Requirements Document

## Project Description (Input)
- freqtradeで実装する
- ルールベースの1次モデル(指値注文)と機械学習による2次モデルの組み合わせで最適な取引を目指す
- richmanbtcのチュートリアルを参考にする
    - https://github.com/richmanbtc/mlbot_tutorial/blob/master/work/tutorial.ipynb
    - チュートリアルではバックテストの手法やコードも紹介されているが、今回のタスクではfreqtradeのバックテスト機能を使用するため、1次モデル+2次モデルの構成や概念だけ理解する
- richmanbtcのチュートリアルでは、直近14日間のATR*0.5だけcloseから上下に離したところに指値を置くというもの
- チュートリアルからの発展として、entry_lengthとentry_pointを用意しoptunaで最適化する
- 特徴量をいくつか用意し、2次モデルでLightGBMの学習をする(ここではfraqtradeでの実装例を作りたいだけなので、移動平均やRSIなど有名なテクニカル指標を10個程度使用する)
- LightGBMのパラメータもoptunaで最適化する
- 最適化対象はシャープレシオ(大きいほうが良い),最終損益(大きいほうが良い),最大ドローダウン(小さいほうがよい)からそれぞれ個別で選べるようにする
- 1次モデル(ATR戦略)によるreturnの正負を2次モデルの分類問題のラベルとする
- 2次モデルありでバックテストする際は、出力が1のときのみ指値注文を出す(0の時は出さない、もしくは難しければ極端に離したところに指値注文を置くようにする)

## Requirements
<!-- Will be generated in /kiro:spec-requirements phase -->
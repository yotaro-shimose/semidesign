# semidesign

シリコンの製造工程を深さ方向（1次元）でシミュレートする小さなツールです。
工程（レシピ）を TOML ファイルに書くと、各不純物の濃度分布と接合の深さを計算し、CSV とグラフを出力します。

## 扱える工程

| 工程 | モデル | 備考 |
|---|---|---|
| イオン注入 `implant` | ガウス分布 | 飛程は B / P / As / Sb の近似表から補間。`rp_nm` / `drp_nm` で直接指定も可。酸化膜越しの注入に対応（酸化膜はSiと同じ止まり方と仮定） |
| 熱処理 `anneal` | 拡散方程式（Crank–Nicolson法） | 不純物ごとの拡散係数は低濃度での教科書値（Fair） |
| 熱酸化 `oxidize` | Deal–Grove モデル | dry / wet、面方位 100 / 111。酸化で消費されたSiの分だけ表面が下がる |
| 酸化膜除去 `strip_oxide` | — | |

## 使い方

```bash
pip install -e .
semidesign run examples/nmos_source_drain.toml -o out
```

出力: `out/<レシピ名>.csv`（深さごとの濃度）と `out/<レシピ名>.png`（濃度分布のグラフ）

### レシピの書き方

```toml
name = "例"

[wafer]
background_dopant = "B"   # 基板の不純物
background_conc = 1e16    # cm^-3
depth_nm = 600            # 計算する深さ
dx_nm = 0.5               # 格子の間隔

[[step]]
type = "implant"
dopant = "As"
dose = 5e15               # cm^-2
energy_kev = 50

[[step]]
type = "anneal"
temp_c = 1000
time_min = 10
```

## 現バージョンで入っていないこと（結果を見るときの注意）

- 高濃度で拡散が速くなる効果、注入ダメージや酸化による拡散の加速
- 酸化時に不純物が酸化膜側へ移る／Si側にたまる現象（消費されたSi中の不純物は単に捨てている）
- 注入プロファイルの非対称性やチャネリング
- 薄い（数十nm以下の）dry酸化の初期成長（Deal–Groveは過小評価しがち）

教育・当たりづけ用の精度です。実プロセスの合わせ込みには校正が必要です。

## テスト

```bash
pip install -e '.[dev]'
pytest
```

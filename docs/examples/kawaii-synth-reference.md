# Kawaii Synthesizer モジュール (kawaii.py)

## 概要

Kawaii Synthesizerモジュールは、Future Bass等の現代音楽制作に適した3種類の特徴的なシンセサイザーを実装しています。各シンセサイザーは`ModularSynth`を基底クラスとして、独自の音色設計を行っています。

## シンセサイザー実装

### SuperSawSynth

Future Bass向けの厚みのある温かいパッド系サウンドを生成するシンセサイザー。

#### 音色設計
1. オシレーター構成
   - 基本音のノコギリ波
   - オクターブ上のノコギリ波（音の厚み向上）

2. エフェクト構成
   - ユニゾン（7ボイス、デチューン0.002）：基本音用
   - ユニゾン（5ボイス、デチューン0.004）：オクターブ上用
   - コーラス（rate=0.6Hz, depth=0.002s）：温かみの付加

3. フィルター設定
   - ローパスフィルター（カットオフ2000Hz）
   - レゾナンス：0.8

4. エンベロープ設定
```python
attack = 0.1  # なめらかな立ち上がり
decay = 0.3
sustain = 0.6
release = 0.4
```

### KawaiiPadSynth

ホワンとした特徴的なパッドサウンドを生成するシンセサイザー。フィルターエンベロープによる音色変化が特徴。

#### 音色設計
1. オシレーター構成
   - メインのノコギリ波
   - サブのパルス波（pulse_width=0.3）

2. フィルター設定
   - ローパスフィルター（初期カットオフ500Hz）
   - レゾナンス：1.2（特徴的な音色用）
   - フィルターエンベロープによる動的制御

3. エフェクト構成
   - コーラス（rate=0.7Hz, depth=0.003s）：音の広がり用
   - リバーブ（room_size=0.6）：空間系効果

4. エンベロープ設定
```python
# アンプエンベロープ
attack = 0.15
decay = 0.3
sustain = 0.5
release = 0.8

# フィルターエンベロープ
filter_attack = 0.2   # ゆっくり開く
filter_decay = 0.4
filter_sustain = 0.3  # やや閉じた状態を維持
filter_release = 0.6
```

### KawaiiSynthBell

FMシンセシスを用いた金属的なベル系サウンドを生成するシンセサイザー。

#### 音色設計
1. オシレーター構成
   - キャリア（サイン波）：基本音
   - モジュレータ（サイン波）：8倍の周波数
   - 輝き用サイン波：4倍の周波数

2. フィルター設定
   - ハイパス的な使用（カットオフ1000Hz）
   - レゾナンス：0.9

3. エフェクト構成
   - コーラス（rate=1.2Hz, depth=0.001s）：キラキラ感
   - ディレイ（time=0.25s, feedback=0.3）：空間的な広がり
   - リバーブ（room_size=0.8, damping=0.2）：余韻

4. エンベロープ設定
```python
attack = 0.02   # 素早い立ち上がり
decay = 0.3    # 適度な減衰
sustain = 0.2  # 控えめなサステイン
release = 1.5  # 長めの余韻
```

5. FM変調特性
   - 初期変調指数：2.0
   - 時間経過による指数関数的な減衰
   - モジュレータ周波数は基本周波数の8倍

## 使用例

### SuperSawSynthの使用
```python
# Future Bass的なコード進行
super_saw = SuperSawSynth(44100)
saw_sequence = [
    Note("C4", 1.0),  # ドミソシの進行
    Note("E4", 1.0),
    Note("G4", 1.0),
    Note("B4", 1.0),
]
```

### KawaiiPadSynthの使用
```python
# ゆったりとした進行
kawaii_pad = KawaiiPadSynth(44100)
pad_sequence = [
    Note("C5", 2.0),  # 2拍ずつのゆっくりした動き
    Note("G4", 2.0),
    Note("A4", 2.0),
]
```

### KawaiiSynthBellの使用
```python
# キラキラしたアルペジオ
bell_synth = KawaiiSynthBell(44100)
bell_sequence = [
    Note("C6", 0.5),  # 8分音符での早めの動き
    Note("G5", 0.5),
    Note("E6", 0.5),
    Note("C6", 0.5),
]
```

## サウンドデザインのポイント

1. SuperSawSynth
   - ユニゾンの重ね合わせによる厚みの制御
   - オクターブ上の音声の混合比率の調整
   - コーラスによる適度な揺らぎの付加

2. KawaiiPadSynth
   - フィルターエンベロープによる音色の時間変化
   - パルス波の混合による倍音の強化
   - リバーブによる空間的な広がり

3. KawaiiSynthBell
   - FM変調による金属的な倍音の生成
   - 時間変化する変調指数による音色の変化
   - 複数のエフェクトによる立体的な空間表現

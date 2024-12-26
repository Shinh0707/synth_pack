# エフェクトモジュール (effect.py)

## 概要

エフェクトモジュールは、オーディオ信号に対する様々な音響効果を提供します。すべてのエフェクトは抽象基底クラス `Effect` を継承しており、ステレオ信号の処理に対応しています。

## 基底クラス

### Effect (ABC)

すべてのエフェクトの基底となる抽象クラス。

#### コンストラクタ
```python
def __init__(
    self,
    mix: float = 0.5,
    offset_phase_time: float = 0.0,
    sample_rate: int = 44100
)
```

#### パラメータ
- `mix`: ドライ/ウェット比（0.0 - 1.0）
- `offset_phase_time`: 初期位相オフセット（秒）
- `sample_rate`: サンプリングレート（Hz）

#### メソッド
```python
def set_mix(self, mix: float)
```
- ドライ/ウェット比の設定
- **引数**: `mix`: 0.0（ドライ信号のみ）から1.0（ウェット信号のみ）

```python
def reset_phase(self, offset_phase_time: float = 0.0)
```
- 位相のリセット
- **引数**: `offset_phase_time`: 新しい位相オフセット（秒）

## タイムベースエフェクト

### Delay

基本的なディレイエフェクト。

#### コンストラクタ
```python
def __init__(
    self,
    delay_time: float = 0.5,
    feedback: float = 0.5,
    mix: float = 0.5,
    sample_rate: int = 44100
)
```

#### パラメータ
- `delay_time`: ディレイ時間（秒）
- `feedback`: フィードバック量（0.0 - 0.99）
- `mix`: ドライ/ウェット比
- `sample_rate`: サンプリングレート

### Echo

ディレイを基にした、よりシンプルなエコーエフェクト。

#### コンストラクタ
```python
def __init__(
    self,
    delay_time: float = 1.0,
    feedback: float = 0.3,
    mix: float = 0.5,
    sample_rate: int = 44100
)
```

### Reverb

マルチタップディレイを使用したリバーブエフェクト。

#### コンストラクタ
```python
def __init__(
    self,
    room_size: float = 0.8,
    damping: float = 0.5,
    mix: float = 0.5,
    sample_rate: int = 44100
)
```

#### パラメータ
- `room_size`: 空間サイズ（0.0 - 1.0）
- `damping`: 減衰特性（0.0 - 1.0）

## モジュレーションエフェクト

### Chorus

コーラスエフェクト。複数の遅延信号をLFOで変調。

#### コンストラクタ
```python
def __init__(
    self,
    rate: float = 1.0,
    depth: float = 0.002,
    mix: float = 0.5,
    sample_rate: int = 44100
)
```

#### パラメータ
- `rate`: 変調周波数（Hz）
- `depth`: 変調の深さ（秒）

### Flanger

フランジャーエフェクト。短い遅延時間とフィードバックを使用。

#### コンストラクタ
```python
def __init__(
    self,
    rate: float = 0.5,
    depth: float = 0.002,
    feedback: float = 0.5,
    mix: float = 0.5,
    sample_rate: int = 44100
)
```

#### パラメータ
- `rate`: 変調周波数（Hz）
- `depth`: 変調の深さ（秒）
- `feedback`: フィードバック量（0.0 - 1.0）

### RingModulate

リングモジュレーションエフェクト。

#### コンストラクタ
```python
def __init__(
    self,
    frequency: float = 440.0,
    mix: float = 0.5,
    sample_rate: int = 44100
)
```

#### パラメータ
- `frequency`: モジュレーション周波数（Hz）

## ステレオエフェクト

### UnisonEffect

ユニゾンエフェクト。複数のデチューンされた声部を生成。

#### コンストラクタ
```python
def __init__(
    self,
    voices: int = 7,
    detune: float = 0.02,
    stereo_spread: float = 1.0,
    mix: float = 1.0,
    sample_rate: int = 44100
)
```

#### パラメータ
- `voices`: 声部数
- `detune`: デチューン量（セント）
- `stereo_spread`: ステレオ幅（0.0 - 1.0）

## 使用例

### 基本的なディレイの使用
```python
# ディレイエフェクトの作成
delay = Delay(
    delay_time=0.5,   # 500ms のディレイ
    feedback=0.4,     # 40% のフィードバック
    mix=0.3          # 30% ウェット信号
)

# 信号の処理
output = delay.process(input_samples)
```

### リバーブとコーラスの組み合わせ
```python
# エフェクトチェーンの作成
reverb = Reverb(room_size=0.8, damping=0.5, mix=0.3)
chorus = Chorus(rate=1.5, depth=0.002, mix=0.4)

# 信号の順次処理
processed = reverb.process(input_samples)
processed = chorus.process(processed)
```

### ユニゾンエフェクトの使用
```python
# 豊かなステレオ効果の作成
unison = UnisonEffect(
    voices=7,
    detune=0.02,
    stereo_spread=0.8,
    mix=1.0
)

# 信号の処理
output = unison.process(input_samples)
```
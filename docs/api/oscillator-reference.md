# オシレーターモジュール (osc.py)

## 概要

オシレーターモジュールは、基本波形の生成から高度な音声合成まで、様々な音源生成機能を提供します。すべてのオシレーターは抽象基底クラス `Oscillator` を継承しており、ステレオ出力に対応しています。

## 基底クラス

### Oscillator (ABC)

抽象基底クラスとして、すべてのオシレーターの基本機能を定義します。

#### プロパティ
- `sample_rate: int` - サンプリングレート（デフォルト: 44100 Hz）
- `phase_left: float` - 左チャンネルの位相
- `phase_right: float` - 右チャンネルの位相
- `frequency: float` - 基本周波数（デフォルト: 440.0 Hz）
- `pan: float` - パンニング位置（0.0 = 左, 0.5 = 中央, 1.0 = 右）

#### メソッド
```python
def set_phase(self, phase_left: int|None=None, phase_right: int|None=None)
```
- 位相の設定
- **引数**:
  - `phase_left`: 左チャンネルの位相（オプショナル）
  - `phase_right`: 右チャンネルの位相（オプショナル）

```python
def set_frequency(self, freq: float)
```
- 周波数の設定
- **引数**:
  - `freq`: 周波数（Hz）

```python
def set_pan(self, pan: float)
```
- パンニング位置の設定
- **引数**:
  - `pan`: パンニング値（0.0 = 左, 0.5 = 中央, 1.0 = 右）

## 基本波形オシレーター

### SineOscillator

正弦波を生成するオシレーター。

```python
def process(self, num_samples: int) -> np.ndarray:
```
- 純粋な正弦波を生成
- 最も基本的な波形で、倍音を含まない
- **戻り値**: shape (num_samples, 2) のステレオ信号

### SquareOscillator

矩形波を生成するオシレーター。

```python
def process(self, num_samples: int) -> np.ndarray:
```
- デジタル的な特徴を持つ矩形波を生成
- 奇数次の倍音を含む
- **戻り値**: shape (num_samples, 2) のステレオ信号

### PulseOscillator

可変デューティ比のパルス波を生成するオシレーター。

#### プロパティ
- `pulse_width: float` - パルス幅（0.0 - 1.0）

#### メソッド
```python
def set_pulse_width(self, width: float)
```
- パルス幅の設定
- **引数**:
  - `width`: パルス幅（0.0 - 1.0）

### SawtoothOscillator

ノコギリ波を生成するオシレーター。

```python
def process(self, num_samples: int) -> np.ndarray:
```
- 全ての倍音を含む豊かな波形を生成
- シンセベースなどに適した特性
- **戻り値**: shape (num_samples, 2) のステレオ信号

## 高度なオシレーター

### AudioOscillator

オーディオファイルをサンプリングするオシレーター。

#### コンストラクタ
```python
def __init__(self, audio_data: np.ndarray|str, sample_rate:int = 44100)
```
- **引数**:
  - `audio_data`: オーディオデータまたはファイルパス
  - `sample_rate`: サンプリングレート

#### メソッド
```python
def create_sub_oscillator(self)
```
- 同じ音源を参照する新しいオシレーターを作成
- **戻り値**: 新しい `AudioOscillator` インスタンス

### FMOscillator

FM合成を行うオシレーター。

#### プロパティ
- `mod_ratio: float` - モジュレーター周波数比
- `mod_index: float` - モジュレーション・インデックス
- `mod_envelope: Optional[ADSR]` - モジュレーター用エンベロープ

#### メソッド
```python
def set_mod_envelope(self, envelope: ADSR)
```
- モジュレーター用エンベロープの設定
- **引数**:
  - `envelope`: ADSRエンベロープインスタンス

### MetallicOscillator

金属的な音色を生成する特殊なオシレーター。

#### プロパティ
- `partials: List[List[float]]` - 倍音パラメータ（周波数比, 振幅, 減衰速度）
- `normalize: float` - 出力の正規化係数

#### メソッド
```python
def reset_time(self)
```
- 内部タイマーをリセット

## 使用例

### 基本的な使用例
```python
# 基本的な正弦波オシレーター
osc = SineOscillator(sample_rate=44100)
osc.set_frequency(440.0)  # A4音
osc.set_pan(0.5)  # センター
audio_data = osc.process(44100)  # 1秒分の音声データを生成
```

### FM合成の例
```python
# FM合成オシレーター
fm_osc = FMOscillator(sample_rate=44100, mod_ratio=2.0, mod_index=5.0)
fm_osc.set_frequency(440.0)
audio_data = fm_osc.process(44100)
```

### オーディオサンプリングの例
```python
# オーディオファイルからのサンプリング
audio_osc = AudioOscillator("sample.wav", sample_rate=44100)
audio_osc.set_pan(0.7)  # 右寄りのパン
audio_data = audio_osc.process(44100)
```
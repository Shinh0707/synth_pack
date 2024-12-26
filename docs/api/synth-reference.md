# シンセサイザーモジュール (synth.py)

## 概要

シンセサイザーモジュールは、音声合成のコアとなる機能を提供します。基本的なモジュラーシンセサイザーから、複合シンセサイザー、グラニュラーシンセサイザーまで、様々な音声合成手法を実装しています。

## 基底クラス

### Synth (ABC)

すべてのシンセサイザーの基底となる抽象クラス。

#### プロパティ
```python
sample_rate: int  # サンプリングレート
```

#### 抽象メソッド
```python
@abstractmethod
def note_on(self, frequency: float, offset_phase_time: float = 0.0):
    """音の発音を開始"""
    pass

@abstractmethod
def note_off(self):
    """音の発音を終了"""
    pass

@abstractmethod
def process(self, num_samples: int) -> np.ndarray:
    """音声データを生成"""
    pass
```

## 主要クラス

### ModularSynth

基本的なモジュラーシンセサイザー。オシレーター、エフェクト、エンベロープを組み合わせて音声を生成。

#### コンストラクタ
```python
def __init__(self, sample_rate: int = 44100)
```

#### プロパティ
- `oscillators: list[Oscillator]` - オシレーターのリスト
- `effects: list[Effect]` - エフェクトのリスト
- `envelope: ADSR` - エンベロープ

#### メソッド

```python
def add_oscillator(self, osc: Oscillator)
```
- オシレーターの追加
- **引数**: `osc`: 追加するオシレーター

```python
def add_filter(self, filter: Filter)
```
- フィルターの追加
- **引数**: `filter`: 追加するフィルター

```python
def add_effect(self, effect: Effect)
```
- エフェクトの追加
- **引数**: `effect`: 追加するエフェクト

```python
def set_envelope(self, attack: float, decay: float, sustain: float, release: float)
```
- エンベロープの設定
- **引数**:
  - `attack`: アタック時間（秒）
  - `decay`: ディケイ時間（秒）
  - `sustain`: サステインレベル（0-1）
  - `release`: リリース時間（秒）

### CompositeModularSynth

複数のModularSynthを組み合わせた複合シンセサイザー。

#### コンストラクタ
```python
def __init__(self, sample_rate: int = 44100)
```

#### プロパティ
- `synths: list[ModularSynth]` - シンセサイザーのリスト

#### メソッド
```python
def add_synth(self, synth: ModularSynth) -> ModularSynth
```
- シンセサイザーの追加
- **戻り値**: 追加されたシンセサイザー

### GranularSynthesizer

グラニュラー合成を行うシンセサイザー。

#### コンストラクタ
```python
def __init__(
    self,
    audio_source: str | np.ndarray,
    sample_rate: int = 44100,
    grain_size: float = 0.1,
    grain_density: float = 10,
    position_jitter: float = 0.1,
    pitch_jitter: float = 0.1,
    max_grains: int = 20
)
```

#### 内部クラス

##### Grain (ModularSynth)
個々のグレインを表現するクラス。

```python
class Grain(ModularSynth):
    def __init__(
        self,
        oscillator: AudioOscillator,
        start_time: float,
        duration: float,
        position: int,
        pitch_shift: float = 1.0,
        pan: float = 0.5
    )
```

#### メソッド

```python
def set_grain_size(self, size: float)
```
- グレインサイズの設定（秒）
- **引数**: `size`: 0.01から1.0の範囲

```python
def set_grain_density(self, density: float)
```
- グレイン密度の設定（グレイン/秒）
- **引数**: `density`: 1以上の値

```python
def set_position_jitter(self, amount: float)
```
- 位置のランダム性を設定（0-1）
- **引数**: `amount`: ジッター量

```python
def set_pitch_jitter(self, amount: float)
```
- ピッチのランダム性を設定（0-1）
- **引数**: `amount`: ジッター量

## 使用例

### 基本的なモジュラーシンセの使用
```python
# 基本的なシンセサイザーの作成
synth = ModularSynth(44100)

# オシレーターとエフェクトの追加
synth.add_oscillator(SawtoothOscillator())
synth.add_oscillator(PulseOscillator(pulse_width=0.3))

# フィルターの追加
lpf = LowPassFilter()
lpf.set_cutoff(2000)
synth.add_filter(lpf)

# エンベロープの設定
synth.set_envelope(
    attack=0.1,
    decay=0.2,
    sustain=0.7,
    release=0.3
)

# 音声の生成
synth.note_on(440.0)  # A4音
audio_data = synth.process(44100)  # 1秒分の音声生成
synth.note_off()
```

### 複合シンセサイザーの使用
```python
# 複合シンセサイザーの作成
composite = CompositeModularSynth(44100)

# 2つのシンセを追加
synth1 = composite.add_synth(ModularSynth())
synth1.add_oscillator(SawtoothOscillator())
synth1.set_envelope(0.1, 0.2, 0.7, 0.3)

synth2 = composite.add_synth(ModularSynth())
synth2.add_oscillator(SquareOscillator())
synth2.set_envelope(0.2, 0.3, 0.6, 0.4)

# 音声の生成
composite.note_on(440.0)
audio_data = composite.process(44100)
composite.note_off()
```

### グラニュラーシンセサイザーの使用
```python
# グラニュラーシンセサイザーの作成
granular = GranularSynthesizer(
    "sample.wav",
    grain_size=0.1,
    grain_density=10,
    position_jitter=0.1,
    pitch_jitter=0.1
)

# パラメータの動的な調整
granular.set_grain_size(0.05)
granular.set_grain_density(15)

# 音声の生成
granular.note_on()
audio_data = granular.process(44100)
granular.note_off()
```
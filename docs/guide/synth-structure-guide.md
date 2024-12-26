# シンセサイザー構築ガイド

## 基本構造

シンセサイザーを構築する際の基本的な構造を示します。すべてのカスタムシンセサイザーは`ModularSynth`を継承して作成します。

### 1. 基本的なクラス構造

```python
from synth.synth import ModularSynth
from synth.osc import SawtoothOscillator, SineOscillator, PulseOscillator
from synth.filter import LowPassFilter
from synth.effect import Chorus, Reverb, Delay, UnisonEffect
from synth.env import ADSR

class CustomSynth(ModularSynth):
    def __init__(self, sample_rate: int = 44100):
        super().__init__(sample_rate)
        self._initialize_oscillators()
        self._initialize_filters()
        self._initialize_effects()
        self._initialize_envelope()
```

### 2. コンポーネントの初期化

基本的に4つの初期化メソッドを実装します：

```python
def _initialize_oscillators(self):
    # オシレーターの追加
    osc = SawtoothOscillator(self.sample_rate)
    self.add_oscillator(osc)

def _initialize_filters(self):
    # フィルターの追加
    lpf = LowPassFilter()
    lpf.set_cutoff(1000)
    lpf.set_resonance(0.8)
    self.add_filter(lpf)

def _initialize_effects(self):
    # エフェクトの追加
    reverb = Reverb(room_size=0.8, mix=0.3)
    self.add_effect(reverb)

def _initialize_envelope(self):
    # エンベロープの設定
    self.set_envelope(
        attack=0.1,
        decay=0.2,
        sustain=0.7,
        release=0.3
    )
```

## 利用可能なコンポーネント

### 1. オシレーター
```python
from synth.osc import (
    SineOscillator,      # 正弦波
    SquareOscillator,    # 矩形波
    PulseOscillator,     # パルス波（可変デューティ比）
    SawtoothOscillator,  # ノコギリ波
    AudioOscillator,     # サンプリング音源
    FMOscillator,        # FM合成
    MetallicOscillator   # 金属的な音源
)
```

### 2. エフェクト
```python
from synth.effect import (
    Delay,          # ディレイ
    Reverb,         # リバーブ
    Chorus,         # コーラス
    RingModulate,   # リングモジュレーション
    Flanger,        # フランジャー
    UnisonEffect    # ユニゾン
)
```

### 3. フィルター
```python
from synth.filter import LowPassFilter  # ローパスフィルター
```

## シンセサイザーの拡張方法

### 1. 複数オシレーターの使用

```python
def _initialize_oscillators(self):
    # 基本音用のノコギリ波
    base_saw = SawtoothOscillator(self.sample_rate)
    self.add_oscillator(base_saw)
    
    # サブオシレーター
    sub_osc = PulseOscillator(self.sample_rate, pulse_width=0.3)
    self.add_oscillator(sub_osc)
```

### 2. フィルターエンベロープの実装

```python
def __init__(self, sample_rate: int = 44100):
    super().__init__(sample_rate)
    self._initialize_components()
    self.filter_envelope = self._create_filter_envelope()

def _create_filter_envelope(self):
    return ADSR(
        attack=0.2,
        decay=0.4,
        sustain=0.3,
        release=0.6,
        sample_rate=self.sample_rate
    )

def process(self, num_samples: int) -> np.ndarray:
    # フィルターエンベロープの適用
    filter_mod = self.filter_envelope.process(num_samples)
    self.lpf.set_cutoff(500 + filter_mod * 3000)
    return super().process(num_samples)
```

### 3. FMシンセシスの実装

```python
def note_on(self, frequency: float, offset_phase_time: float = 0.0):
    # キャリア周波数の設定
    self.carrier.set_frequency(frequency)
    # モジュレータ周波数の設定
    self.modulator.set_frequency(frequency * 2.0)  # 2倍の周波数
    super().note_on(frequency, offset_phase_time)
```

## 効果的な音作りのパターン

### 1. 厚みのある音作り
```python
def _initialize_effects(self):
    # ユニゾンで厚みを作る
    unison = UnisonEffect(
        voices=7,
        detune=0.002,
        stereo_spread=1.0,
        mix=1.0
    )
    self.add_effect(unison)
    
    # コーラスで揺らぎを加える
    chorus = Chorus(
        rate=0.6,
        depth=0.002,
        mix=0.3
    )
    self.add_effect(chorus)
```

### 2. 空間的な広がりの作成
```python
def _initialize_effects(self):
    # リバーブで空間を作る
    reverb = Reverb(
        room_size=0.8,
        damping=0.5,
        mix=0.3
    )
    self.add_effect(reverb)
    
    # ディレイで立体感を出す
    delay = Delay(
        delay_time=0.25,
        feedback=0.3,
        mix=0.2
    )
    self.add_effect(delay)
```

## シンセサイザーの使用

### 1. 基本的な使用方法
```python
from synth.player import SimpleSynthPlayer, Note

# シンセサイザーの作成
synth = CustomSynth(44100)

# プレイヤーの作成
player = SimpleSynthPlayer(synth, bpm=120)

# シーケンスの定義と再生
sequence = [
    Note("C4", 1.0),
    Note("E4", 1.0),
    Note("G4", 1.0)
]
audio_data = player.play_sequence(sequence, loop=False)
```

### 2. 音声ファイルとしての保存
```python
import soundfile as sf

sf.write(
    "output.wav",
    audio_data,
    synth.sample_rate,
    subtype="PCM_24"
)
```
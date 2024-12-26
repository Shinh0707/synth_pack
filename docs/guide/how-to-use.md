# シンセサイザーライブラリ使用例

## 1. 基本的な使い方

### 1.1 シンプルなシンセサイザーの作成

```python
from synth.synth import ModularSynth
from synth.osc import SineOscillator

# 基本的なシンセサイザーの作成
synth = ModularSynth(sample_rate=44100)

# オシレーターの追加
osc = SineOscillator()
synth.add_oscillator(osc)

# エンベロープの設定
synth.set_envelope(
    attack=0.1,   # アタック時間（秒）
    decay=0.2,    # ディケイ時間（秒）
    sustain=0.7,  # サステインレベル（0-1）
    release=0.3   # リリース時間（秒）
)

# 音を生成（440Hz = A4音）
synth.note_on(frequency=440.0)
audio_data = synth.process(44100)  # 1秒分の音声データを生成
synth.note_off()
```

### 1.2 エフェクトの追加

```python
from synth.effect import Reverb, Delay
from synth.filter import LowPassFilter

# リバーブの追加
reverb = Reverb(room_size=0.8, damping=0.5, mix=0.3)
synth.add_effect(reverb)

# ディレイの追加
delay = Delay(delay_time=0.5, feedback=0.4, mix=0.3)
synth.add_effect(delay)

# フィルターの追加
lpf = LowPassFilter()
lpf.set_cutoff(2000)  # カットオフ周波数の設定（Hz）
synth.add_filter(lpf)
```

## 2. 音楽シーケンスの作成

### 2.1 シンプルなメロディーの再生

```python
from synth.player import SimpleSynthPlayer, Note

# プレイヤーの作成
player = SimpleSynthPlayer(synth, bpm=120)

# メロディーシーケンスの定義
sequence = [
    Note(60, 0.5),  # C4（ド）、0.5秒
    Note(64, 0.5),  # E4（ミ）、0.5秒
    Note(67, 0.5),  # G4（ソ）、0.5秒
    Note(72, 0.5),  # C5（ド）、0.5秒
]

# シーケンスの再生と音声データの取得
audio_data = player.play_sequence(sequence, loop=False)

# 音声データの保存（WAVファイル）
import soundfile as sf
sf.write("melody.wav", audio_data, synth.sample_rate)
```

## 3. 高度な使用例

### 3.1 複合シンセサイザーの作成

```python
from synth.synth import CompositeModularSynth
from synth.osc import SawtoothOscillator, SquareOscillator

# 複合シンセサイザーの作成
composite = CompositeModularSynth(sample_rate=44100)

# 第1シンセサイザー（ノコギリ波）
synth1 = composite.add_synth(ModularSynth())
synth1.add_oscillator(SawtoothOscillator())
synth1.set_envelope(attack=0.1, decay=0.2, sustain=0.7, release=0.3)

# 第2シンセサイザー（矩形波）
synth2 = composite.add_synth(ModularSynth())
synth2.add_oscillator(SquareOscillator())
synth2.set_envelope(attack=0.2, decay=0.3, sustain=0.6, release=0.4)

# 音声の生成
composite.note_on(frequency=440.0)
audio_data = composite.process(44100)
composite.note_off()
```

### 3.2 グラニュラーシンセサイザーの使用

```python
from synth.synth import GranularSynthesizer

# グラニュラーシンセサイザーの作成
granular = GranularSynthesizer(
    audio_source="sample.wav",  # オーディオファイルのパス
    grain_size=0.1,            # グレインサイズ（秒）
    grain_density=10,          # 1秒あたりのグレイン数
    position_jitter=0.1,       # 位置のランダム性（0-1）
    pitch_jitter=0.1,          # ピッチのランダム性（0-1）
    max_grains=20             # 最大グレイン数
)

# パラメータの動的な変更
granular.set_grain_size(0.05)
granular.set_grain_density(15)

# 音声の生成
granular.note_on()
audio_data = granular.process(44100)
granular.note_off()
```

## 4. 音声ファイルの入出力

```python
import soundfile as sf
import numpy as np

# 音声データの保存
def save_audio(audio_data: np.ndarray, filename: str, sample_rate: int = 44100):
    sf.write(
        filename,
        audio_data,
        sample_rate,
        subtype='PCM_24'  # 24ビットPCM形式
    )

# 音声データの読み込み（グラニュラー合成用など）
def load_audio(filename: str) -> tuple[np.ndarray, int]:
    audio_data, sample_rate = sf.read(filename, always_2d=True)
    return audio_data, sample_rate
```

## 5. パフォーマンスとメモリの考慮事項

- 大量の音声データを生成する場合は、適切なバッファサイズを使用する
- エフェクトチェーンが長い場合は、CPU使用率に注意
- グラニュラー合成では、max_grainsパラメータでメモリ使用量を制御
- 長時間の音声生成では、ストリーミング処理の使用を検討

## 6. エラー処理の例

```python
try:
    synth = ModularSynth(sample_rate=44100)
    synth.add_oscillator(SineOscillator())
    synth.note_on(frequency=440.0)
    audio_data = synth.process(44100)
except Exception as e:
    print(f"シンセサイザーエラー: {str(e)}")
finally:
    # リソースのクリーンアップ
    synth.note_off()
```

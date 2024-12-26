# Synth Pack

Synth Packは、Pythonで実装されたモジュラー型シンセサイザーライブラリです。基本的な波形生成から高度なFM合成まで、様々な音声合成機能を提供します。

## 特徴

- モジュラー設計による柔軟な音声合成
- 複数のオシレーター波形（サイン波、矩形波、ノコギリ波、パルス波）
- 実用的なエフェクト（リバーブ、ディレイ、コーラス、フランジャー）
- ADSRエンベロープによる音量制御
- ローパスフィルターによる音色加工
- グラニュラー合成対応
- 44.1kHzサンプリングレート、ステレオ出力対応

## インストール

```bash
# 開発版をインストール
pip install -e .

# または通常インストール
pip install .
```

## 必要要件

- Python 3.8以上
- NumPy
- SoundFile

## 基本的な使用法

### シンプルなシンセサイザーの作成

```python
from synth_pack import ModularSynth, SawtoothOscillator, LowPassFilter
from synth_pack import SimpleSynthPlayer, Note

# シンセサイザーの作成
synth = ModularSynth(44100)

# オシレーターの追加
synth.add_oscillator(SawtoothOscillator())

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

# シーケンスの再生
player = SimpleSynthPlayer(synth, bpm=120)
sequence = [
    Note("C4", 1.0),
    Note("E4", 1.0),
    Note("G4", 1.0)
]
audio_data = player.play_sequence(sequence)

# 音声ファイルとして保存
import soundfile as sf
sf.write("output.wav", audio_data, synth.sample_rate)
```

### FMシンセシスの例

```python
from synth_pack import ModularSynth, FMOscillator

# FMシンセの作成
fm_synth = ModularSynth(44100)
fm_osc = FMOscillator(mod_ratio=2.0, mod_index=5.0)
fm_synth.add_oscillator(fm_osc)

# 音声生成
fm_synth.note_on(440.0)  # A4音
audio_data = fm_synth.process(44100)  # 1秒分の音声
fm_synth.note_off()
```

## 高度な使用例

example/kawaii.pyには、以下の実装例が含まれています：

- SuperSawSynth: Future Bass向けの厚みのあるパッド音源
- KawaiiPadSynth: フィルターエンベロープを使用した特徴的なパッド
- KawaiiSynthBell: FMシンセシスを使用したベル系音源

```python
from synth_pack.examples.kawaii import SuperSawSynth

# Super Sawシンセの使用
super_saw = SuperSawSynth(44100)
player = SimpleSynthPlayer(super_saw, bpm=128)
sequence = [Note("C4", 1.0), Note("E4", 1.0), Note("G4", 1.0)]
audio_data = player.play_sequence(sequence)
```

## ドキュメント

詳細なドキュメントは`docs/`ディレクトリにあります：

- `docs/api/`: 各モジュールのAPIリファレンス
- `docs/examples/`: 実装例の説明
- `docs/guide/`: シンセサイザー構築ガイド

## 開発

### 開発環境のセットアップ

```bash
# リポジトリのクローン
git clone https://github.com/Shinh0707/synth_pack.git
cd synth_pack

# 開発用依存パッケージのインストール
pip install -e ".[dev]"
```

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は`LICENSE`ファイルを参照してください。
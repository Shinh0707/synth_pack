# プレイヤーモジュール (player.py)

## 概要

プレイヤーモジュールは、シンセサイザーを使用して実際の音楽演奏を制御するためのコンポーネントを提供します。音符のシーケンス管理、テンポ制御、MIDIノート変換などの機能を実装しています。

## クラス定義

### Note

音符を表現するクラス。

#### コンストラクタ
```python
def __init__(self, value: Union[int, str], duration: float = 0.5)
```

#### パラメータ
- `value`: MIDIノートナンバーまたはコード名（例: "C4"）
- `duration`: 音符の長さ（秒）

### SimpleSynthPlayer

シンセサイザーを制御するための基本的なプレイヤー。

#### コンストラクタ
```python
def __init__(self, synth: Synth, bpm: float = 120.0)
```

#### パラメータ
- `synth`: 使用するシンセサイザーインスタンス
- `bpm`: テンポ（Beats Per Minute）

#### プロパティ
- `synth: Synth` - 制御対象のシンセサイザー
- `bpm: float` - 現在のテンポ
- `is_playing: bool` - 再生状態

#### メソッド

##### note_to_freq (staticmethod)
```python
@staticmethod
def note_to_freq(note: Union[int, str]) -> float
```
- MIDIノートナンバーまたはコード名を周波数に変換
- **引数**:
  - `note`: MIDIノートナンバー（0-127）またはコード名（例: "C4"）
- **戻り値**: 周波数（Hz）
- **対応コード名**:
  ```python
  note_map = {
      "C": 60,  "C#": 61, "Db": 61,
      "D": 62,  "D#": 63, "Eb": 63,
      "E": 64,
      "F": 65,  "F#": 66, "Gb": 66,
      "G": 67,  "G#": 68, "Ab": 68,
      "A": 69,  "A#": 70, "Bb": 70,
      "B": 71
  }
  ```

##### beat_to_seconds
```python
def beat_to_seconds(self, beats: float) -> float
```
- 拍数を秒数に変換
- **引数**: `beats`: 拍数
- **戻り値**: 秒数

##### set_bpm
```python
def set_bpm(self, bpm: float)
```
- テンポ（BPM）の設定
- **引数**: `bpm`: 新しいテンポ値（1.0以上）

##### play_sequence
```python
def play_sequence(
    self,
    sequence: List[Note],
    loop: bool = False
) -> np.ndarray
```
- 音符のシーケンスを再生
- **引数**:
  - `sequence`: Note オブジェクトのリスト
  - `loop`: ループ再生の有無
- **戻り値**: 生成された音声データ（shape (total_samples, 2)）

##### stop
```python
def stop(self)
```
- 再生の停止

## シーケンス制御の仕様

### タイミング制御
1. BPMベースのタイミング
   - BPMから1拍あたりの秒数を計算
   - 実時間でのタイミング制御

2. ノート長の管理
   - 秒単位での音符の長さ指定
   - リリース時間の自動制御

### シーケンス再生の流れ
1. 各ノートの処理
   - 周波数の計算と設定
   - ノートオンイベントの発行
   - 指定時間の音声データ生成
   - ノートオフイベントの発行
   - リリース時間の処理

2. ループ制御
   - ループフラグによる繰り返し制御
   - 停止フラグによる中断処理

## 使用例

### 基本的な使用法
```python
# プレイヤーの作成
synth = ModularSynth(44100)
player = SimpleSynthPlayer(synth, bpm=120)

# MIDIノートナンバーによるシーケンス
sequence1 = [
    Note(60, 0.5),  # C4
    Note(64, 0.5),  # E4
    Note(67, 0.5),  # G4
    Note(72, 0.5),  # C5
]

# コード名によるシーケンス
sequence2 = [
    Note("C4", 1.0),
    Note("F4", 1.0),
    Note("G4", 1.0),
    Note("C4", 1.0),
]

# シーケンスの再生
audio_data = player.play_sequence(sequence1, loop=False)
```

### テンポの動的な変更
```python
# テンポの変更
player.set_bpm(140)  # テンポを140BPMに変更

# 新しいテンポでの再生
audio_data = player.play_sequence(sequence)
```

### ファイル出力
```python
import soundfile as sf

def save_sequence(player: SimpleSynthPlayer, sequence: List[Note], filename: str):
    audio_data = player.play_sequence(sequence)
    sf.write(
        filename,
        audio_data,
        player.synth.sample_rate,
        subtype="PCM_24"
    )
```

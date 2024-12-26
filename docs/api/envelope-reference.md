# エンベロープモジュール (env.py)

## 概要

エンベロープモジュールは、音の時間的な振幅変化を制御するADSR（Attack, Decay, Sustain, Release）エンベロープを実装します。このモジュールは音の立ち上がり、減衰、持続、消音の特性を細かく制御することができます。

## ADSRState 列挙型

エンベロープの状態を表す列挙型です。

```python
class ADSRState(Enum):
    IDLE = 0      # 非アクティブ状態
    ATTACK = 1    # アタック（立ち上がり）フェーズ
    DECAY = 2     # ディケイ（減衰）フェーズ
    SUSTAIN = 3   # サステイン（持続）フェーズ
    RELEASE = 4   # リリース（消音）フェーズ
```

## ADSR クラス

### コンストラクタ
```python
def __init__(
    self,
    attack: float = 0.1,
    decay: float = 0.1,
    sustain: float = 0.7,
    release: float = 0.2,
    sample_rate: int = 44100
)
```

#### パラメータ
- `attack`: アタック時間（秒）
- `decay`: ディケイ時間（秒）
- `sustain`: サステインレベル（0.0 - 1.0）
- `release`: リリース時間（秒）
- `sample_rate`: サンプリングレート（Hz）

### プロパティ

- `state: ADSRState` - 現在のエンベロープ状態
- `current_level: float` - 現在の出力レベル（0.0 - 1.0）
- `samples_processed: int` - 処理済みサンプル数

### メソッド

#### note_on
```python
def note_on(self, samples_processed: int = 0)
```
- エンベロープをトリガーし、アタックフェーズを開始
- **引数**:
  - `samples_processed`: 処理開始位置のサンプル数（オプション）

#### note_off
```python
def note_off(self, samples_processed: int = 0)
```
- リリースフェーズを開始
- **引数**:
  - `samples_processed`: 処理開始位置のサンプル数（オプション）

#### process
```python
def process(self, num_samples: int) -> np.ndarray
```
- エンベロープの振幅値を生成
- **引数**:
  - `num_samples`: 生成するサンプル数
- **戻り値**: shape (num_samples, 2) のステレオ振幅値

### 状態遷移図

```
     note_on()
IDLE ---------> ATTACK
                  |
                  | attack_time経過
                  v
                DECAY
                  |
                  | decay_time経過
                  v
                SUSTAIN
                  |
                  | note_off()
                  v
                RELEASE
                  |
                  | release_time経過
                  v
                IDLE
```

## 詳細な動作説明

### アタックフェーズ
- 振幅が0から最大値（1.0）まで線形に増加
- 時間はattackパラメータで指定
- サンプル数 = attack_time * sample_rate

### ディケイフェーズ
- 振幅が最大値からサステインレベルまで減衰
- 時間はdecayパラメータで指定
- サンプル数 = decay_time * sample_rate

### サステインフェーズ
- 振幅をsustainレベルで維持
- note_off()が呼ばれるまで継続
- レベルはsustainパラメータで指定（0.0 - 1.0）

### リリースフェーズ
- 振幅がサステインレベルから0まで減衰
- 時間はreleaseパラメータで指定
- サンプル数 = release_time * sample_rate

## 使用例

### 基本的な使用例
```python
# ADSRエンベロープの作成
env = ADSR(
    attack=0.1,    # 100ms のアタック
    decay=0.2,     # 200ms のディケイ
    sustain=0.7,   # 70% のサステインレベル
    release=0.3    # 300ms のリリース
)

# ノートオン（発音開始）
env.note_on()

# エンベロープ値の生成（1秒分）
envelope_data = env.process(44100)

# ノートオフ（発音終了）
env.note_off()

# リリース部分の生成
release_data = env.process(44100)
```

### シンセサイザーでの使用例
```python
# シンセサイザーと組み合わせる例
class SimpleSynth:
    def __init__(self):
        self.envelope = ADSR(
            attack=0.1,
            decay=0.2,
            sustain=0.7,
            release=0.3
        )
        self.oscillator = SineOscillator()

    def process(self, num_samples: int):
        # オシレーター出力にエンベロープを適用
        osc_data = self.oscillator.process(num_samples)
        env_data = self.envelope.process(num_samples)
        return osc_data * env_data
```

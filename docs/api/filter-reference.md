# フィルターモジュール (filter.py)

## 概要

フィルターモジュールは、オーディオ信号の周波数特性を制御するためのコンポーネントを提供します。現在の実装では、ローパスフィルターを中心に、ステレオ信号に対するフィルタリング処理を実現しています。

## 基底クラス

### Filter

エフェクトクラスを継承した、フィルターの基底クラス。

#### コンストラクタ
```python
def __init__(self, sample_rate: int = 44100)
```

#### プロパティ
- `sample_rate: int` - サンプリングレート
- `cutoff: float` - カットオフ周波数（Hz）
- `resonance: float` - レゾナンス（Q値）
- `_buffer: np.ndarray` - ステレオバッファ (2, 2)

#### メソッド

```python
def set_cutoff(self, freq: np.ndarray | float)
```
- カットオフ周波数の設定
- **引数**:
  - `freq`: カットオフ周波数（Hz）またはその配列
  - 自動的にナイキスト周波数（sample_rate/2）以下に制限

```python
def set_resonance(self, res: np.ndarray | float)
```
- レゾナンスの設定
- **引数**:
  - `res`: レゾナンス値（0.707 - 2.0）またはその配列
  - 0.707が最小値（バターワース特性）

## フィルター実装

### LowPassFilter

2次のローパスフィルター（バイクアッド）実装。

#### 継承
- `Filter` クラスを継承

#### 実装の特徴
1. ステレオ信号の個別処理
2. バイクアッドフィルターのアルゴリズム使用
3. 動的なパラメータ変更に対応
4. 位相の連続性を保持

#### フィルター係数
フィルターは以下の係数を使用：
- a0: ノーマライゼーション係数
- a1, a2: フィードバック係数
- b0, b1, b2: フィードフォワード係数

```python
def _process(self, samples: np.ndarray) -> np.ndarray
```
- フィルター処理の本体実装
- **引数**:
  - `samples`: shape (num_samples, 2) のステレオ信号
- **戻り値**: 
  - フィルター処理された shape (num_samples, 2) のステレオ信号

## フィルターの数学的基礎

### 伝達関数
バイクアッドフィルターの伝達関数：

$$
H(z) = \frac{b_0 + b_1z^{-1} + b_2z^{-2}}{1 + a_1z^{-1} + a_2z^{-2}}
$$

### 係数計算
カットオフ周波数（$$\omega_c$$）とレゾナンス（Q）から係数を計算：

```python
omega = 2 * np.pi * cutoff / sample_rate
alpha = np.sin(omega) / (2 * resonance)

a0 = 1 + alpha
a1 = -2 * np.cos(omega)
a2 = 1 - alpha
b0 = (1 - np.cos(omega)) / 2
b1 = 1 - np.cos(omega)
b2 = (1 - np.cos(omega)) / 2
```

## 使用例

### 基本的な使用例
```python
# ローパスフィルターの作成
lpf = LowPassFilter(sample_rate=44100)

# パラメータの設定
lpf.set_cutoff(1000)  # カットオフ周波数 1000Hz
lpf.set_resonance(0.707)  # バターワース特性

# 信号の処理
filtered_audio = lpf.process(input_audio)
```

### 動的なパラメータ変更
```python
# カットオフ周波数の時間変化
cutoff_envelope = np.linspace(100, 5000, num_samples)
lpf.set_cutoff(cutoff_envelope)

# レゾナンスの時間変化
resonance_envelope = np.linspace(0.707, 2.0, num_samples)
lpf.set_resonance(resonance_envelope)

# 信号の処理
filtered_audio = lpf.process(input_audio)
```

### シンセサイザーでの使用例
```python
# シンセサイザーのフィルターとして使用
class SynthWithFilter:
    def __init__(self):
        self.oscillator = SawtoothOscillator()
        self.filter = LowPassFilter()
        self.filter.set_cutoff(1000)
        self.filter.set_resonance(1.2)
    
    def process(self, num_samples: int):
        # オシレーター出力をフィルタリング
        osc_output = self.oscillator.process(num_samples)
        return self.filter.process(osc_output)
```

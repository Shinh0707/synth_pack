from enum import Enum
import random
from typing import Callable, TypeVar, overload
import numpy as np
from abc import ABC, abstractmethod
from synth.env import ADSR

"""シンセサイザー用オシレータークラス群

基本波形の生成から高度な音声合成まで、
様々な方式のオシレーターを提供する。
全てのオシレーターはステレオ出力に対応。

Classes:
    Oscillator: オシレーターの基底クラス
    SineOscillator: サイン波オシレーター
    SquareOscillator: 矩形波オシレーター
    PulseOscillator: パルス波オシレーター
    TriangleOscillator: 三角波オシレーター
    SawtoothOscillator: のこぎり波オシレーター
    AudioOscillator: 音声ファイルベースのオシレーター
    FMOscillator: FM合成オシレーター
    SuperSawOscillator: SuperSawオシレーター
    NoiseOscillator: ノイズジェネレーター
"""

class Oscillator(ABC):
    """オシレーターの基底クラス

    全てのオシレータークラスはこのクラスを継承して実装する。
    位相管理とステレオパンニング機能を提供する。

    Attributes:
        sample_rate: サンプリングレート (Hz), Default: 44100
        amplitude: 出力振幅 (0.0-1.0), Default: 1.0
        frequency: 基本周波数 (Hz), Default: 440.0
        pan: パンニング位置 (0.0=左, 0.5=中央, 1.0=右), Default: 0.5
    """

    def __init__(self, sample_rate: int = 44100, amplitude: float=1.0, frequency_shift: Callable[[float],float] = lambda freq:freq):
        """オシレーターを初期化する

        Args:
            sample_rate: サンプリングレート (Hz), Default: 44100
            amplitude: 初期振幅 (0.0-1.0), Default: 1.0
            frequency_shift: 周波数変調関数 (Hz -> Hz), Default: 恒等関数
        """
        self.sample_rate = sample_rate
        self.amplitude = amplitude
        self.phase_left = 0.0
        self.phase_right = 0.0
        self.phase_max = 1.0
        self.phase_left_sample = 0
        self.phase_right_sample = 0
        self.phase_max_sample = int(self.phase_max * self.sample_rate)
        self.frequency = 440.0
        self.frequency_shift = frequency_shift
        self.pan = 0.5  # 0.0 = 完全左, 0.5 = 中央, 1.0 = 完全右

    @overload
    def set_phase_max(self, phase_max: int) -> None:
        self.phase_max_sample = phase_max
        self.phase_max = phase_max / self.sample_rate

    @overload
    def set_phase_max(self, phase_max: float) -> None:
        self.phase_max = phase_max
        self.phase_max = int(self.phase_max * self.sample_rate)

    def set_frequency_shift(self,  frequency_shift: Callable[[float],float]):
        self.frequency_shift = frequency_shift

    def process(self, num_samples: int) -> np.ndarray:
        output = self._process(num_samples, self.get_times(num_samples))
        self.phase_left_sample += num_samples
        self.phase_right_sample += num_samples
        self.phase_left_sample %= self.phase_max_sample
        self.phase_right_sample %= self.phase_max_sample
        self.phase_left = self.phase_left_sample/self.sample_rate
        self.phase_right = self.phase_right_sample / self.sample_rate
        return self.apply_panning(output * self.amplitude)

    def get_times(self, num_samples: int):
        return (
            np.column_stack(
                [
                    np.linspace(
                        self.phase_left_sample,
                        self.phase_left_sample + num_samples,
                        num_samples,
                        False,
                    ),
                    np.linspace(
                        self.phase_right_sample,
                        self.phase_right_sample + num_samples,
                        num_samples,
                        False,
                    ),
                ]
            )
            / self.sample_rate
        )

    @abstractmethod
    def _process(self, num_samples: int, t: np.ndarray) -> np.ndarray:
        pass

    def set_phase(self, phase_left: int|float|None=None, phase_right: int|float|None=None):
        if phase_left is not None:
            if isinstance(phase_left, int):
                self.phase_left_sample = phase_left
                self.phase_left = phase_left/self.sample_rate
            elif isinstance(phase_left, float):
                self.phase_left = phase_left
                self.phase_left_sample = int(phase_left * self.sample_rate)
        if phase_right is not None:
            if isinstance(phase_right, int):
                self.phase_right_sample = phase_right
                self.phase_right = phase_right / self.sample_rate
            elif isinstance(phase_right, float):
                self.phase_right = phase_right
                self.phase_right_sample = int(phase_right * self.sample_rate)

    def set_frequency(self, freq: float):
        self.frequency = self.frequency_shift(freq)

    def set_pan(self, pan: float):
        """パンニング位置を設定 (0.0 = 左, 0.5 = 中央, 1.0 = 右)"""
        self.pan = np.clip(pan, 0, 1)

    def apply_panning(self, signal: np.ndarray) -> np.ndarray:
        """信号にパンニングを適用してステレオ信号を生成"""
        left_gain = np.sqrt(1 - self.pan)  # 左チャンネルのゲイン
        right_gain = np.sqrt(self.pan)  # 右チャンネルのゲイン

        # ステレオ信号を生成 (samples, channels)
        if signal.ndim == 1:
            stereo = np.zeros((signal.size, 2))
            stereo[:, 0] = signal * left_gain  # 左チャンネル
            stereo[:, 1] = signal * right_gain  # 右チャンネル
            signal = stereo
        else:
            signal[:, 0] *= left_gain
            signal[:, 1] *= right_gain

        return signal

class SineOscillator(Oscillator):
    """サイン波オシレーター

    純粋な単一周波数の正弦波を生成する。
    最も基本的な波形で、倍音を含まない。
    """
    def _process(self, num_samples: int, t: np.ndarray) -> np.ndarray:
        return np.sin(2 * np.pi * self.frequency * t)


class SquareOscillator(Oscillator):
    """矩形波オシレーター

    デューティ比50%の矩形波を生成する。
    奇数次高調波を含み、パルスオシレーターの特殊ケース。
    """
    def _process(self, num_samples: int, t: np.ndarray) -> np.ndarray:
        return np.sign(np.sin(2 * np.pi * self.frequency * t))


class PulseOscillator(Oscillator):
    """パルス波オシレーター

    可変デューティ比のパルス波を生成する。
    デューティ比を変更することで音色を大きく変化させることが可能。
    """
    def __init__(self, sample_rate: int = 44100, pulse_width: float = 0.5, **kwargs):
        """パルスオシレーターを初期化する

        Args:
            sample_rate: サンプリングレート (Hz), Default: 44100
            pulse_width: パルス幅 (0.0-1.0), Default: 0.5
                0.5で矩形波、それ以外でパルス波となる
            **kwargs: Oscillatorクラスのパラメータ
        """
        super().__init__(sample_rate, **kwargs)
        self.pulse_width = max(0.0, min(1.0, pulse_width))

    def set_pulse_width(self, width: float):
        self.pulse_width = max(0.0, min(1.0, width))

    def _process(self, num_samples: int, t: np.ndarray) -> np.ndarray:
        return np.where(np.sin(2 * np.pi * self.frequency * t) > 2 * self.pulse_width - 1, 1.0, -1.0)


class TriangleOscillator(Oscillator):
    """三角波オシレーター

    三角波を生成する。
    奇数次高調波を含むが、高次になるほど急速に減衰する
    特徴的な倍音構造を持つ。
    """
    def _process(self, num_samples: int, t: np.ndarray) -> np.ndarray:
        # 鋸波を絶対値を取って三角波に変換
        sawtooth = 2.0 * (self.frequency * t % 1.0) - 1.0
        return 2.0 * np.abs(sawtooth) - 1.0


class SawtoothOscillator(Oscillator):
    """のこぎり波オシレーター

    のこぎり波を生成する。
    全ての整数次高調波を含み、
    高次になるほど1/nで減衰する特徴を持つ。
    """
    def _process(self, num_samples: int, t: np.ndarray) -> np.ndarray:
        return 2.0 * (self.frequency * t % 1.0) - 1.0


class AudioOscillator(Oscillator):
    """音声ファイルベースのオシレーター

    音声ファイルやバッファをループ再生する。
    """
    def __init__(self, audio_data: np.ndarray|str,sample_rate:int = 44100, **kwargs):
        """音声オシレーターを初期化する

        Args:
            audio_data: 音声データ配列またはファイルパス
                numpy配列の場合は (samples, 2) の形状
                文字列の場合は音声ファイルのパス
            sample_rate: サンプリングレート (Hz), Default: 44100
            **kwargs: Oscillatorクラスのパラメータ
        """
        super().__init__(sample_rate, **kwargs)
        if isinstance(audio_data, str):
            import soundfile as sf
            self.audio_data, sr = sf.read(audio_data,always_2d=True)
            if sr != sample_rate:
                import librosa
                self.audio_data = librosa.resample(self.audio_data, orig_sr=sr, target_sr=sample_rate)
        elif audio_data is None:
            self.audio_data = None
            return
        else:
            self.audio_data = audio_data
        self.audio_samples: int = self.audio_data.shape[0]
        self.phase_max_sample = int(self.audio_samples)

    def _process_(self, num_samples: int, phase_left: int, phase_right: int):
        output = np.zeros((num_samples, 2))
        remain_left = self.audio_samples - phase_left
        output[: min(remain_left, num_samples), 0] = self.audio_data[
            phase_left : phase_left + min(remain_left, num_samples),
            0,
        ].copy()
        if remain_left < num_samples:
            output[remain_left:, 0] = self.audio_data[: (num_samples - remain_left), 0]
        remain_right = self.audio_samples - phase_right
        output[: min(remain_right, num_samples), 1] = self.audio_data[
            phase_right : phase_right
            + min(remain_right, num_samples),
            1,
        ].copy()
        if remain_right < num_samples:
            output[remain_right:, 1] = self.audio_data[
                : (num_samples - remain_right), 1
            ]
        return output

    def _process(self, num_samples: int, t: np.ndarray):
        return self._process_(
            num_samples,
            self.phase_left_sample,
            self.phase_right_sample,
        )

    def create_sub_oscillator(self):
        """このインスタンスと同じ音源をサンプリングするオシレーターを作成"""
        sub_osc = AudioOscillator(None,self.sample_rate)
        sub_osc._process_ = self._process_
        sub_osc.audio_samples = self.audio_samples
        return sub_osc

class FMOscillator(Oscillator):
    """FM合成オシレーター

    周波数変調(FM)による音声合成を行う。
    キャリア波とモジュレータ波の関係により
    複雑な倍音構造を持つ音色を生成可能。
    """
    def __init__(
        self, sample_rate: int = 44100, mod_ratio: float = 1.0, mod_index: float = 1.0, **kwargs
    ):
        """FM合成オシレーターを初期化する

        Args:
            sample_rate: サンプリングレート (Hz), Default: 44100
            mod_ratio: モジュレーター周波数比, Default: 1.0
                キャリア周波数との比率を指定
            mod_index: 変調指数, Default: 1.0
                変調の深さを指定。大きいほど倍音が豊富になる
            **kwargs: Oscillatorクラスのパラメータ
        """
        super().__init__(sample_rate, **kwargs)
        self.mod_ratio = mod_ratio  # モジュレーター周波数比
        self.mod_index = mod_index  # モジュレーション・インデックス
        self.mod_envelope = None  # モジュレーター用エンベロープ

    def set_mod_envelope(self, envelope: ADSR):
        self.mod_envelope = envelope

    def _process(self, num_samples: int, t: np.ndarray) -> np.ndarray:
        # モジュレーター信号の生成
        mod_freq = self.frequency * self.mod_ratio
        if self.mod_envelope is not None:
            mod_env = self.mod_envelope.process(num_samples)
        else:
            mod_env = np.ones((num_samples, 2))

        modulator = np.sin(2 * np.pi * mod_freq * t) * self.mod_index * mod_env

        # キャリア信号の生成（周波数変調を適用）
        return np.sin(2 * np.pi * self.frequency * t + modulator)


class SuperSawOscillator(Oscillator):
    """SuperSawオシレーター

    複数のデチューンされたのこぎり波を重ね合わせ、
    豊かなユニゾン効果を持つ音色を生成する。
    ステレオ拡がりを制御可能で、より立体的な音場を実現。

    Attributes:
        num_voices: ボイス数
        detune: デチューン量（半音単位）
        stereo_spread: ステレオの広がり (0.0-1.0)
            0.0: モノラル（中央定位）
            1.0: 最大のステレオ効果
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        num_voices: int = 7,
        detune: float = 0.1,
        stereo_spread: float = 1.0,
        **kwargs
    ):
        """SuperSawオシレーターを初期化する

        Args:
            sample_rate: サンプリングレート (Hz), Default: 44100
            num_voices: ボイス数, Default: 7
                重ね合わせるのこぎり波の数
            detune: デチューン量, Default: 0.1
                各ボイスのピッチずれの最大値（半音単位）
            stereo_spread: ステレオの広がり (0.0-1.0), Default: 1.0
                ステレオフィールドでの音の拡がりを制御
            **kwargs: Oscillatorクラスのパラメータ
        """
        super().__init__(sample_rate, **kwargs)
        self.num_voices = num_voices
        self.detune = detune
        self.stereo_spread = np.clip(stereo_spread, 0.0, 1.0)

        # 各ボイスのデチューン値を計算
        self.detune_factors = np.linspace(-self.detune, self.detune, self.num_voices)

        # 各ボイスのパンニング位置を計算
        # 中央を0.5として、stereo_spreadに応じて左右に配置
        self.pan_positions = np.linspace(
            0.5 - 0.5 * self.stereo_spread,
            0.5 + 0.5 * self.stereo_spread,
            self.num_voices,
        )

    def _process(self, num_samples: int, t: np.ndarray) -> np.ndarray:
        output = np.zeros((num_samples, 2))  # ステレオ出力用の配列

        # 各ボイスを生成して合成
        for detune_factor, pan_pos in zip(self.detune_factors, self.pan_positions):
            detuned_freq = self.frequency * (1.0 + detune_factor)
            # 鋸波を生成
            voice = 2.0 * (detuned_freq * t[:, 0] % 1.0) - 1.0  # モノラル信号として生成

            # ステレオパンニングを適用
            left_gain = np.sqrt(1 - pan_pos)  # 左チャンネルのゲイン
            right_gain = np.sqrt(pan_pos)  # 右チャンネルのゲイン

            # 各チャンネルに振幅を適用
            output[:, 0] += voice * left_gain  # 左チャンネル
            output[:, 1] += voice * right_gain  # 右チャンネル

        # 正規化（-1.0から1.0の範囲に収める）
        return output / np.sqrt(self.num_voices)  # より正確なゲイン補正


class NoiseOscillator(Oscillator):
    """ノイズジェネレーター

    各種ノイズを生成する。
    ホワイトノイズ、ピンクノイズ、ブラウンノイズに対応。

    Attributes:
        sample_rate: サンプリングレート (Hz)
        noise_type: ノイズの種類
        prev_value: ブラウンノイズ用の前回値
        b0-b6: ピンクノイズ用のフィルター係数
    """

    class NoiseType(Enum):
        WHITE = 0
        PINK = 1
        BROWN = 2

    def __init__(
        self,
        sample_rate: int = 44100,
        noise_type: "NoiseOscillator.NoiseType | str | int | None" = None,
        type: "NoiseOscillator.NoiseType | str | int | None" = None,
        **kwargs,
    ):
        """ノイズジェネレーターを初期化する

        Args:
            sample_rate: サンプリングレート (Hz), Default: 44100
            noise_type: ノイズの種類, Default: WHITE
            type: noise_typeの別名（どちらでも指定可能）
                Enumまたは文字列で指定可能:
                - WHITE/white/0: ホワイトノイズ（全周波数で同じパワー）
                - PINK/pink/1: ピンクノイズ（1/fスペクトル）
                - BROWN/brown/2: ブラウンノイズ（1/f^2スペクトル）
            **kwargs: Oscillatorクラスのパラメータ

        Raises:
            ValueError: 無効なnoise_typeが指定された場合
        """
        super().__init__(sample_rate, **kwargs)

        # noise_typeとtypeの両方が指定された場合はエラー
        if noise_type is not None and type is not None:
            raise ValueError(
                "Cannot specify both 'noise_type' and 'type'. Please use only one."
            )

        # typeかnoise_typeのどちらかを使用（typeを優先）
        noise_type_value = (
            type if type is not None else (noise_type if noise_type is not None else 0)
        )

        # noise_typeの型に応じて適切に処理
        if isinstance(noise_type_value, NoiseOscillator.NoiseType):
            self.noise_type = noise_type_value
        elif isinstance(noise_type_value, str):
            # 文字列の場合、大文字に変換して処理
            try:
                self.noise_type = NoiseOscillator.NoiseType[noise_type_value.upper()]
            except KeyError:
                raise ValueError(
                    f"Invalid noise type: {noise_type_value}. "
                    "Must be one of: WHITE, PINK, BROWN"
                )
        elif isinstance(noise_type_value, int):
            try:
                self.noise_type = NoiseOscillator.NoiseType(noise_type_value)
            except ValueError:
                raise ValueError(
                    f"Invalid noise type value: {noise_type_value}. "
                    "Must be 0 (WHITE), 1 (PINK), or 2 (BROWN)"
                )
        else:
            raise ValueError(
                f"noise_type must be NoiseType, str, or int, not {type(noise_type_value)}"
            )

        # フィルター係数の初期化
        self.prev_value = 0.0  # ブラウンノイズ用
        self.b0 = 0.0  # ピンクノイズ用
        self.b1 = 0.0
        self.b2 = 0.0
        self.b3 = 0.0
        self.b4 = 0.0
        self.b5 = 0.0
        self.b6 = 0.0

    def set_noise_type(self, noise_type: str | int):
        """ノイズタイプを設定する

        Args:
            noise_type: 新しいノイズタイプ
                Enumまたは文字列で指定可能:
                - WHITE/white/0: ホワイトノイズ
                - PINK/pink/1: ピンクノイズ
                - BROWN/brown/2: ブラウンノイズ

        Raises:
            ValueError: 無効なnoise_typeが指定された場合
        """
        # 既存のインスタンスを再利用して型チェック
        temp = NoiseOscillator(noise_type=noise_type)
        self.noise_type = temp.noise_type

    def _process(self, num_samples: int, t: np.ndarray) -> np.ndarray:
        if self.noise_type == NoiseOscillator.NoiseType.WHITE:
            return self._white_noise(num_samples)
        elif self.noise_type == NoiseOscillator.NoiseType.PINK:
            return self._pink_noise(num_samples)
        elif self.noise_type == NoiseOscillator.NoiseType.BROWN:
            return self._brown_noise(num_samples)
        else:
            return self._white_noise(num_samples)

    def _white_noise(self, num_samples: int) -> np.ndarray:
        # ホワイトノイズの生成
        noise = np.random.uniform(-1.0, 1.0, (num_samples, 2))
        return noise

    def _pink_noise(self, num_samples: int) -> np.ndarray:
        # ピンクノイズの生成（Voss-McCartney アルゴリズム）
        output = np.zeros((num_samples, 2))

        for i in range(num_samples):
            white = np.random.uniform(-1.0, 1.0, 2)

            # 各オクターブのフィルター更新
            self.b0 = white
            self.b1 = 0.99886 * self.b1 + white * 0.0555179
            self.b2 = 0.99332 * self.b2 + white * 0.0750759
            self.b3 = 0.96900 * self.b3 + white * 0.1538520
            self.b4 = 0.86650 * self.b4 + white * 0.3104856
            self.b5 = 0.55000 * self.b5 + white * 0.5329522
            self.b6 = -0.7616 * self.b6 - white * 0.0168980

            # フィルター出力の合成
            output[i] = (
                self.b0 + self.b1 + self.b2 + self.b3 + self.b4 + self.b5 + self.b6
            ) / 7.0

        return output

    def _brown_noise(self, num_samples: int) -> np.ndarray:
        # ブラウンノイズの生成
        output = np.zeros((num_samples, 2))
        prev_left = self.prev_value
        prev_right = self.prev_value

        for i in range(num_samples):
            # ランダムな変化量を生成
            delta_left = np.random.uniform(-0.1, 0.1)
            delta_right = np.random.uniform(-0.1, 0.1)

            # 前の値に変化量を加える
            prev_left += delta_left
            prev_right += delta_right

            # -1.0から1.0の範囲に制限
            prev_left = np.clip(prev_left, -1.0, 1.0)
            prev_right = np.clip(prev_right, -1.0, 1.0)

            output[i] = [prev_left, prev_right]

        self.prev_value = (prev_left + prev_right) / 2.0  # 次回用に値を保存
        return output


class PluckOscillator(Oscillator):
    """カルプスストロング・アルゴリズムによる弦楽器シミュレーター

    物理モデリングにより弦の振動をシミュレートし、
    ギターやハープのような弦楽器音を生成する。

    Attributes:
        damping: 弦の減衰係数 (0.0-1.0), Default: 0.5
        string_tension: 弦のテンション (0.0-1.0), Default: 0.9
        decay: 音の減衰時間 (sec), Default: 1.0
        noise_color: ノイズの色 ('white' or 'pink'), Default: 'white'
    """

    def __init__(
        self,
        damping: float = 0.5,
        string_tension: float = 0.9,
        decay: float = 1.0,
        noise_color: str = "white",
        sample_rate: int = 44100,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.damping = np.clip(damping, 0.0, 1.0)
        self.string_tension = np.clip(string_tension, 0.0, 1.0)
        self.decay = max(0.1, decay)
        self.noise_color = noise_color
        self.buffer = None
        self.buffer_size = 0
        self.current_pos = 0
        self.start_time = None

    def _init_string(self, frequency: float):
        """弦のバッファを初期化"""
        # バッファサイズを周波数から計算
        self.buffer_size = int(self.sample_rate / frequency)

        # 初期ノイズの生成
        if self.noise_color == "pink":
            # ピンクノイズの生成（1/fスペクトル）
            noise = np.zeros(self.buffer_size)
            for i in range(self.buffer_size):
                noise[i] = sum(random.uniform(-1, 1) / (j + 1) for j in range(5))
        else:
            # ホワイトノイズの生成
            noise = np.random.uniform(-1, 1, self.buffer_size)

        self.buffer = noise
        self.current_pos = 0

    def _process(self, num_samples: int, t: np.ndarray) -> np.ndarray:
        if self.buffer is None or self.frequency != getattr(self, "_last_freq", None):
            self._init_string(self.frequency)
            self._last_freq = self.frequency
            self.start_time = self.phase_left

        output = np.zeros((num_samples, 2))
        current_time = t[0]

        # 経過時間に基づく減衰の計算
        elapsed_time = (
            current_time - self.start_time if self.start_time is not None else 0
        )
        decay_factor = np.exp(-elapsed_time / self.decay)

        for i in range(num_samples):
            # 現在の出力を取得（減衰を適用）
            output[i] = self.buffer[self.current_pos] * decay_factor

            # カルプスストロングアルゴリズムの適用
            next_pos = (self.current_pos + 1) % self.buffer_size
            self.buffer[self.current_pos] = (
                self.string_tension
                * (self.buffer[self.current_pos] + self.buffer[next_pos])
                / 2
                * (1.0 - self.damping)
            )

            self.current_pos = next_pos
            # 時間ごとの減衰の更新
            decay_factor *= np.exp(-1.0 / (self.decay * self.sample_rate))

        return output

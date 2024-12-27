from abc import ABC, abstractmethod
from enum import Enum
import random
from typing import Callable
import numpy as np
from scipy import signal

"""オーディオエフェクト処理の基底クラスと実装クラス群

基底クラスEffectを継承して各種エフェクトを実装する。
全てのエフェクトはステレオ信号(num_samples, 2)の形式で処理を行う。

Classes:
    Effect: エフェクト処理の基底クラス
    Delay: ディレイエフェクト
    Reverb: リバーブエフェクト
    Echo: エコーエフェクト 
    Chorus: コーラスエフェクト
    RingModulate: リングモジュレーションエフェクト
    Flanger: フランジャーエフェクト
    Unison: ユニゾンエフェクト
"""

class Effect(ABC):
    """エフェクト処理の基底クラス

    全てのエフェクトクラスはこのクラスを継承して実装する。
    エフェクト処理は_process()メソッドをオーバーライドして実装する。

    Attributes:
        sample_rate: サンプリングレート (Hz)
        mix: ドライ/ウェット比 (0.0-1.0)
        phase: 現在の位相時間 (sec)
    """
    def __init__(
        self, mix: float = 0.5, offset_phase_time: float = 0.0, sample_rate: int = 44100
    ):
        self.sample_rate = sample_rate
        self.mix = mix
        self.phase = offset_phase_time

    def set_mix(self, mix: float):
        self.mix = max(0.0, min(1.0, mix))

    def reset_phase(self, offset_phase_time: float = 0.0):
        self.phase = offset_phase_time

    @abstractmethod
    def _process(self, samples: np.ndarray) -> np.ndarray:
        """これをオーバーライドしてエフェクトを実装"""
        return samples

    def process(self, samples: np.ndarray) -> np.ndarray:
        """
        samples: shape (num_samples, 2) のステレオ信号
        returns: shape (num_samples, 2) のステレオ信号
        """
        if self.mix == 0:
            self.phase += samples.shape[0] / self.sample_rate
            return samples
        process_samples = self._process(samples)
        self.phase += samples.shape[0] / self.sample_rate
        return samples * (1 - self.mix) + process_samples * self.mix


class Delay(Effect):
    """ディレイエフェクト

    入力信号を遅延させて出力する。
    フィードバックにより複数回の遅延を生成可能。

    Attributes:
        delay_time: 遅延時間 (sec), Default: 0.5
        feedback: フィードバック量 (0.0-0.99), Default: 0.5
        mix: エフェクトのミックス量 (0.0-1.0), Default: 0.5
    """
    def __init__(
        self,
        delay_time: float = 0.5,
        feedback: float = 0.5,
        mix: float = 0.5,
        sample_rate: int = 44100,
    ):
        super().__init__(mix=mix, sample_rate=sample_rate)
        self.delay_samples = int(delay_time * sample_rate)
        self.feedback = max(0.0, min(0.99, feedback))
        # ステレオバッファ (delay_samples, 2)
        self.buffer = np.zeros((self.delay_samples, 2))
        self.buffer_pos = 0

    def _process(self, samples: np.ndarray) -> np.ndarray:
        buffer_len = self.buffer.shape[0]
        samples_len = samples.shape[0]

        # バッファの古い部分を取得 (samples_len, 2)
        old_samples = self.buffer[
            np.arange(self.buffer_pos, self.buffer_pos + samples_len) % buffer_len
        ]

        # 出力を計算（ステレオ）
        output = samples + old_samples * self.feedback

        # バッファを更新
        new_buffer_indices = (
            np.arange(self.buffer_pos, self.buffer_pos + samples_len) % buffer_len
        )
        self.buffer[new_buffer_indices] = output

        self.buffer_pos = (self.buffer_pos + samples_len) % buffer_len
        return output


class Reverb(Effect):
    """リバーブエフェクト

    複数のディレイラインを組み合わせて残響効果を生成する。
    左右のチャンネルで異なる遅延時間を設定し、立体的な効果を実現。
    stereo_spreadパラメータでステレオの広がりを調整可能。

    Attributes:
        room_size: 空間の大きさ (0.0-1.0), Default: 0.8
        damping: 高域の減衰量 (0.0-1.0), Default: 0.5
        stereo_spread: ステレオの広がり (0.0-1.0), Default: 1.0
        mix: エフェクトのミックス量 (0.0-1.0), Default: 0.5
    """

    def __init__(
        self,
        room_size: float = 0.8,
        damping: float = 0.5,
        stereo_spread: float = 1.0,
        mix: float = 0.5,
        sample_rate: int = 44100,
    ):
        """リバーブエフェクトを初期化する

        Args:
            room_size: 空間の大きさ (0.0-1.0), Default: 0.8
            damping: 高域の減衰量 (0.0-1.0), Default: 0.5
            stereo_spread: ステレオの広がり (0.0-1.0), Default: 1.0
                0.0: モノラル（左右同じ遅延時間）
                1.0: 最大のステレオ効果
            mix: エフェクトのミックス量 (0.0-1.0), Default: 0.5
            sample_rate: サンプリングレート (Hz), Default: 44100
        """
        super().__init__(mix=mix, sample_rate=sample_rate)
        self.room_size = max(0.0, min(1.0, room_size))
        self.damping = max(0.0, min(1.0, damping))
        self.stereo_spread = max(0.0, min(1.0, stereo_spread))

        # 基本となる遅延時間
        self.base_delay_times = [0.03, 0.05, 0.07, 0.11]
        # ステレオ拡がりに使用する最大オフセット時間（秒）
        self.max_offsets = [0.001, 0.003, 0.003, 0.003]
        self.feedbacks = [0.3, 0.25, 0.2, 0.15]

        # stereo_spreadに基づいて左右の遅延時間を計算
        self.delay_times_l = [
            t - offset * self.stereo_spread
            for t, offset in zip(self.base_delay_times, self.max_offsets)
        ]
        self.delay_times_r = [
            t + offset * self.stereo_spread
            for t, offset in zip(self.base_delay_times, self.max_offsets)
        ]

        # 左右のディレイラインを作成
        self.delays_l = [
            Delay(delay_time=dt, feedback=fb, mix=1.0, sample_rate=sample_rate)
            for dt, fb in zip(self.delay_times_l, self.feedbacks)
        ]
        self.delays_r = [
            Delay(delay_time=dt, feedback=fb, mix=1.0, sample_rate=sample_rate)
            for dt, fb in zip(self.delay_times_r, self.feedbacks)
        ]

    def set_stereo_spread(self, spread: float):
        """ステレオの広がりを設定する

        Args:
            spread: ステレオの広がり (0.0-1.0)
                0.0: モノラル（左右同じ遅延時間）
                1.0: 最大のステレオ効果
        """
        self.stereo_spread = max(0.0, min(1.0, spread))

        # 遅延時間を更新
        for i, (delay_l, delay_r, base_time, offset) in enumerate(
            zip(self.delays_l, self.delays_r, self.base_delay_times, self.max_offsets)
        ):
            delay_l.delay_samples = int(
                (base_time - offset * self.stereo_spread) * self.sample_rate
            )
            delay_r.delay_samples = int(
                (base_time + offset * self.stereo_spread) * self.sample_rate
            )

    def _process(self, samples: np.ndarray) -> np.ndarray:
        output = np.zeros_like(samples)

        # 左右のチャンネルを分離して処理
        left_samples = samples[:, [0]]
        right_samples = samples[:, [1]]

        # 左チャンネルの処理
        for delay in self.delays_l:
            output[:, 0] += delay.process(left_samples)[:, 0] * self.room_size

        # 右チャンネルの処理
        for delay in self.delays_r:
            output[:, 1] += delay.process(right_samples)[:, 0] * self.room_size

        return output * (1.0 - self.damping)


class Echo(Effect):
    """エコーエフェクト

    単一のディレイラインによるシンプルなエコー効果を生成する。

    Attributes:
        delay_time: エコーの間隔 (sec), Default: 1.0
        feedback: エコーの減衰量 (0.0-1.0), Default: 0.3
        mix: エフェクトのミックス量 (0.0-1.0), Default: 0.5
    """
    def __init__(
        self,
        delay_time: float = 1.0,
        feedback: float = 0.3,
        mix: float = 0.5,
        sample_rate: int = 44100,
    ):
        super().__init__(mix=mix, sample_rate=sample_rate)
        # ステレオディレイを作成
        self.delay = Delay(
            delay_time=delay_time, feedback=feedback, mix=1.0, sample_rate=sample_rate
        )

    def _process(self, samples: np.ndarray) -> np.ndarray:
        return self.delay.process(samples)


class Chorus(Effect):
    """コーラスエフェクト

    LFOで変調された短い遅延時間により、音に厚みを加える。
    ステレオモード時は左右で異なる位相のLFOを使用して効果を強調。
    モノラルモード時は単一のLFOを使用。

    Attributes:
        rate: LFOの周波数 (Hz), Default: 1.0
        depth: 変調の深さ (sec), Default: 0.002
        mix: エフェクトのミックス量 (0.0-1.0), Default: 0.5
        stereo: ステレオモードの有効/無効, Default: True
    """

    def __init__(
        self,
        rate: float = 1.0,
        depth: float = 0.002,
        mix: float = 0.5,
        stereo: bool = True,
        sample_rate: int = 44100,
    ):
        """コーラスエフェクトを初期化する

        Args:
            rate: LFOの周波数 (Hz), Default: 1.0
            depth: 変調の深さ (sec), Default: 0.002
            mix: エフェクトのミックス量 (0.0-1.0), Default: 0.5
            stereo: ステレオモードの有効/無効, Default: True
            sample_rate: サンプリングレート (Hz), Default: 44100
        """
        super().__init__(mix=mix, sample_rate=sample_rate)
        self.rate = rate
        self.depth = depth
        self.stereo = stereo
        # ステレオバッファ (buffer_size, 2)
        self.buffer = np.zeros((int(0.05 * sample_rate), 2))
        self.buffer_pos = 0

    def _process(self, samples: np.ndarray) -> np.ndarray:
        buffer_len = self.buffer.shape[0]
        samples_len = samples.shape[0]

        # 基本のLFO波形を生成
        lfo_phases = (
            2
            * np.pi
            * self.rate
            * (np.arange(samples_len) / self.sample_rate + self.phase)
        )
        lfo_l = np.sin(lfo_phases)

        if self.stereo:
            # ステレオモード: 右チャンネルは90度位相シフト
            lfo_r = np.sin(lfo_phases + np.pi / 2)
        else:
            # モノラルモード: 左右同じLFO
            lfo_r = lfo_l

        # 遅延時間の計算
        delay_samples_l = (self.depth * 0.5 * (lfo_l + 1) * self.sample_rate).astype(
            int
        )
        delay_samples_r = (self.depth * 0.5 * (lfo_r + 1) * self.sample_rate).astype(
            int
        )

        # バッファへの書き込み
        write_indices = (
            np.arange(self.buffer_pos, self.buffer_pos + samples_len) % buffer_len
        )
        self.buffer[write_indices] = samples

        # 左右のチャンネルを個別に読み出し
        read_indices_l = (write_indices - delay_samples_l) % buffer_len
        read_indices_r = (write_indices - delay_samples_r) % buffer_len

        output = np.zeros_like(samples)
        output[:, 0] = self.buffer[read_indices_l, 0]
        output[:, 1] = self.buffer[read_indices_r, 1]

        self.buffer_pos = (self.buffer_pos + samples_len) % buffer_len
        return output


class RingModulate(Effect):
    """リングモジュレーションエフェクト

    入力信号に正弦波キャリアを乗算し、周波数変調効果を生成する。
    ステレオモード時は左右で位相の異なるキャリアを使用。

    Attributes:
        frequency: キャリア周波数 (Hz), Default: 440.0
        mix: エフェクトのミックス量 (0.0-1.0), Default: 0.5
        stereo: ステレオモードの有効/無効, Default: True
    """

    def __init__(
        self,
        frequency: float = 440.0,
        mix: float = 0.5,
        stereo: bool = True,
        sample_rate: int = 44100,
    ):
        """リングモジュレーションエフェクトを初期化する

        Args:
            frequency: キャリア周波数 (Hz), Default: 440.0
            mix: エフェクトのミックス量 (0.0-1.0), Default: 0.5
            stereo: ステレオモードの有効/無効, Default: True
                True: 左右で異なる位相のキャリアを使用
                False: 両チャンネルで同じキャリアを使用
            sample_rate: サンプリングレート (Hz), Default: 44100
        """
        super().__init__(mix=mix, sample_rate=sample_rate)
        self.frequency = frequency
        self.stereo = stereo

    def _process(self, samples: np.ndarray) -> np.ndarray:
        t = np.arange(samples.shape[0]) / self.sample_rate + self.phase

        # 左チャンネル用のキャリア生成
        carrier_l = np.sin(2 * np.pi * self.frequency * t)

        if self.stereo:
            # ステレオモード: 右チャンネルは45度位相シフト
            carrier_r = np.sin(2 * np.pi * self.frequency * t + np.pi / 4)
        else:
            # モノラルモード: 両チャンネルで同じキャリアを使用
            carrier_r = carrier_l

        output = np.zeros_like(samples)
        output[:, 0] = samples[:, 0] * carrier_l
        output[:, 1] = samples[:, 1] * carrier_r
        return output


class Flanger(Effect):
    """フランジャーエフェクト
    
    短い遅延時間をLFOで変調し、特徴的なスイープ効果を生成する。
    フィードバックループにより効果を強調。
    
    Attributes:
        rate: LFOの基本周波数 (Hz), Default: 0.5
        depth: 変調の深さ (sec), Default: 0.002
        feedback: フィードバック量 (0.0-1.0), Default: 0.5
        mix: エフェクトのミックス量 (0.0-1.0), Default: 0.5
    """
    def __init__(
        self,
        rate: float = 0.5,
        depth: float = 0.002,
        feedback: float = 0.5,
        mix: float = 0.5,
        sample_rate: int = 44100,
    ):
        super().__init__(mix=mix, sample_rate=sample_rate)
        self.rate = rate
        self.depth = depth
        self.feedback = feedback
        # ステレオバッファ
        self.buffer = np.zeros((int(0.05 * sample_rate), 2))
        self.buffer_pos = 0

        self.automation_phase = 0.0
        self.automation_rate = 0.1

    def _process(self, samples: np.ndarray) -> np.ndarray:
        buffer_len = self.buffer.shape[0]
        samples_len = samples.shape[0]

        automation_factor = np.clip(np.sin(2 * np.pi * self.automation_phase), 0, 1)
        current_depth = self.depth * (1 + automation_factor * 2)
        current_feedback = self.feedback * (1 - automation_factor * 0.5)

        lfo_freq = self.rate * (1 + automation_factor * 4)
        time = np.arange(samples_len) / self.sample_rate + self.phase

        # 左右で位相の異なるLFOを生成
        lfo_l = (np.sin(2 * np.pi * lfo_freq * time) + 1) / 2
        lfo_r = (
            np.sin(2 * np.pi * lfo_freq * time + np.pi / 3) + 1
        ) / 2  # 60度位相シフト

        delay_samples_l = (current_depth * self.sample_rate * lfo_l).astype(int)
        delay_samples_r = (current_depth * self.sample_rate * lfo_r).astype(int)

        write_indices = (
            np.arange(self.buffer_pos, self.buffer_pos + samples_len) % buffer_len
        )

        # フィードバック処理（ステレオ）
        read_indices_l = (write_indices - delay_samples_l) % buffer_len
        read_indices_r = (write_indices - delay_samples_r) % buffer_len

        feedback_samples = np.zeros_like(samples)
        feedback_samples[:, 0] = (
            samples[:, 0] + self.buffer[read_indices_l, 0] * current_feedback
        )
        feedback_samples[:, 1] = (
            samples[:, 1] + self.buffer[read_indices_r, 1] * current_feedback
        )

        self.buffer[write_indices] = feedback_samples

        output = np.zeros_like(samples)
        output[:, 0] = self.buffer[read_indices_l, 0]
        output[:, 1] = self.buffer[read_indices_r, 1]

        self.buffer_pos = (self.buffer_pos + samples_len) % buffer_len
        self.automation_phase += self.automation_rate * samples_len / self.sample_rate

        return output


class Unison(Effect):
    """ユニゾンエフェクト

    入力信号を複数のボイスにコピーし、それぞれにデチューンとパンニングを適用する。
    位相のランダム化により、より豊かな効果を実現。

    Attributes:
        voices: ボイス数, Default: 7
        detune: デチューン量 (セント), Default: 0.02
        stereo_spread: ステレオの広がり (0.0-1.0), Default: 1.0
        mix: エフェクトのミックス量 (0.0-1.0), Default: 1.0
    """
    def __init__(
        self,
        voices: int = 7,
        detune: float = 0.02,
        stereo_spread: float = 1.0,
        mix: float = 1.0,
        sample_rate: int = 44100,
    ):
        super().__init__(mix=mix, sample_rate=sample_rate)
        self.voices = max(1, voices)
        self.detune = detune
        self.stereo_spread = stereo_spread
        # ステレオパンニング設定
        self.panning = np.linspace(-stereo_spread, stereo_spread, self.voices)
        self.phase_offsets = np.random.rand(self.voices) * 2 * np.pi

    def _process(self, samples: np.ndarray) -> np.ndarray:
        samples_len = samples.shape[0]

        detune_ratios = (
            1 + (np.arange(self.voices) - self.voices // 2) * self.detune / 1200
        )
        detune_factors = 2**detune_ratios

        time = np.arange(samples_len) / self.sample_rate + self.phase
        phases = (
            time[np.newaxis, :] * (detune_factors)[:, np.newaxis]
            + self.phase_offsets[:, np.newaxis]
        )
        osc = np.sin(2 * np.pi * phases)

        # ステレオパンニング係数の計算
        pan_values = np.zeros((self.voices, 2))
        pan_values[:, 0] = np.cos(self.panning * np.pi / 2)  # Left
        pan_values[:, 1] = np.sin(self.panning * np.pi / 2)  # Right

        output = np.zeros_like(samples, dtype=np.float64)

        for i in range(self.voices):
            voice_output = np.zeros_like(samples)
            # 各チャンネルに対して位相変調を適用
            for ch in range(2):
                modulated = samples[:, ch] * osc[i, :]
                voice_output[:, ch] = modulated * pan_values[i, ch]
            output += voice_output

        # ボイス数で正規化
        output /= np.sqrt(self.voices)

        return output

class Tremolo(Effect):
    """トレモロエフェクト

    音量を周期的に変調させるエフェクト。
    LFOを使用して振幅を制御する。

    Attributes:
        rate: LFOの周波数 (Hz), Default: 5.0
        depth: 変調の深さ (0.0-1.0), Default: 0.5
        wave_shape: LFOの波形 ('sine' or 'square'), Default: 'sine'
    """

    def __init__(
        self,
        rate: float = 5.0,
        depth: float = 0.5,
        wave_shape: str = "sine",
        mix: float = 1.0,
        sample_rate: int = 44100,
    ):
        super().__init__(mix=mix, sample_rate=sample_rate)
        self.rate = rate
        self.depth = np.clip(depth, 0.0, 1.0)
        self.wave_shape = wave_shape

    def _process(self, samples: np.ndarray) -> np.ndarray:
        # 時間配列の生成
        t = np.arange(samples.shape[0]) / self.sample_rate + self.phase

        # LFO波形の生成
        if self.wave_shape == "square":
            lfo = np.sign(np.sin(2 * np.pi * self.rate * t))
        else:  # sine
            lfo = np.sin(2 * np.pi * self.rate * t)

        # 変調信号の生成 (0.0-1.0の範囲に正規化)
        modulation = (lfo + 1) / 2 * self.depth + (1 - self.depth)

        # ステレオ信号への適用
        return samples * modulation[:, np.newaxis]


class BitCrusher(Effect):
    """ビットクラッシャーエフェクト

    サンプルレートとビット深度を低下させることで
    ローファイなサウンドを生成する。

    Attributes:
        bit_depth: ビット深度 (1-16), Default: 8
        sample_freq: 新しいサンプリング周波数 (Hz), Default: 8000
    """

    def __init__(
        self,
        bit_depth: int = 8,
        sample_freq: float = 8000,
        mix: float = 1.0,
        sample_rate: int = 44100,
    ):
        super().__init__(mix=mix, sample_rate=sample_rate)
        self.bit_depth = max(1, min(16, bit_depth))
        self.sample_freq = min(sample_freq, sample_rate / 2)
        self.hold_samples = int(sample_rate / sample_freq)

    def _process(self, samples: np.ndarray) -> np.ndarray:
        # ビット深度の削減
        max_val = 2 ** (self.bit_depth - 1)
        crushed = np.round(samples * max_val) / max_val

        # サンプルレートの削減（ホールド処理）
        output = np.zeros_like(samples)
        for i in range(0, samples.shape[0], self.hold_samples):
            end_idx = min(i + self.hold_samples, samples.shape[0])
            output[i:end_idx] = crushed[i : i + 1]  # サンプル値のホールド

        return output

    def set_bit_depth(self, depth: int):
        """ビット深度を設定 (1-16)"""
        self.bit_depth = max(1, min(16, depth))

    def set_sample_freq(self, freq: float):
        """サンプリング周波数を設定"""
        self.sample_freq = min(freq, self.sample_rate / 2)
        self.hold_samples = int(self.sample_rate / self.sample_freq)


class SpringReverb(Effect):
    """物理モデリングによるバネリバーブ

    バネの物理特性をモデリングし、
    ビンテージ機器特有の金属的な残響を再現する。

    Attributes:
        tension: バネの張力 (0.0-1.0), Default: 0.5
        density: バネの密度 (影響するモード数), Default: 8
        decay: 残響時間 (sec), Default: 2.0
        brightness: 高域の特性 (0.0-1.0), Default: 0.5
    """

    def __init__(
        self,
        tension: float = 0.5,
        density: int = 8,
        decay: float = 2.0,
        brightness: float = 0.5,
        mix: float = 0.5,
        sample_rate: int = 44100,
    ):
        super().__init__(mix=mix, sample_rate=sample_rate)
        self.tension = tension
        self.density = density
        self.decay = decay
        self.brightness = np.clip(brightness, 0.0, 1.0)

        # 各モードのパラメータを初期化
        self.modes = []
        for i in range(density):
            # 高域特性を反映した周波数とアンプリチュード計算
            mode_freq = 100 * (i + 1) * tension
            mode_decay = decay / (i + 1)
            brightness_factor = 1.0 + (self.brightness - 0.5) * 2.0  # 0.0-2.0の範囲
            self.modes.append(
                {
                    "freq": mode_freq,
                    "decay": mode_decay,
                    "phase": 0.0,
                    "amp": (1.0 / (i + 1))
                    * np.power(brightness_factor, i / self.density),
                }
            )

        # ディレイラインの初期化
        max_delay = int(0.1 * sample_rate)  # 100ms
        self.delay_line = np.zeros((max_delay, 2))
        self.delay_pos = 0

    def _process(self, samples: np.ndarray) -> np.ndarray:
        output = np.zeros_like(samples)
        num_samples = samples.shape[0]

        # 各モードの処理
        for mode in self.modes:
            # モード周波数での共振
            t = np.arange(num_samples) / self.sample_rate + self.phase
            resonance = np.sin(2 * np.pi * mode["freq"] * t + mode["phase"])

            # 減衰の適用
            decay_factor = np.exp(-t / mode["decay"])
            output += (
                mode["amp"] * resonance[:, np.newaxis] * decay_factor[:, np.newaxis]
            )

            # 位相の更新
            mode["phase"] = (
                mode["phase"]
                + 2 * np.pi * mode["freq"] * num_samples / self.sample_rate
            ) % (2 * np.pi)

        # ディレイライン処理
        for i in range(num_samples):
            # 入力と共振の合成
            self.delay_line[self.delay_pos] = samples[i] + output[i] * self.tension

            # ディレイタップの読み出し
            tap1 = (self.delay_pos - int(0.05 * self.sample_rate)) % len(
                self.delay_line
            )
            tap2 = (self.delay_pos - int(0.03 * self.sample_rate)) % len(
                self.delay_line
            )

            output[i] += self.delay_line[tap1] * 0.3 + self.delay_line[tap2] * 0.2

            self.delay_pos = (self.delay_pos + 1) % len(self.delay_line)

        return output


class PitchShifter(Effect):
    """ピッチシフトエフェクト

    時間伸縮とリサンプリングを組み合わせてピッチシフトを実現する。
    scipy.signal.resampleを使用した単純な実装。

    Attributes:
        semitones: ピッチシフト量（半音単位）
        mix: エフェクトのミックス量 (0.0-1.0)
    """

    def __init__(
        self, semitones: float = 12.0, mix: float = 1.0, sample_rate: int = 44100
    ):
        super().__init__(mix=mix, sample_rate=sample_rate)
        self.semitones = semitones
        # ピッチシフト量から変換係数を計算
        self.pitch_factor = 2.0 ** (self.semitones / 12.0)

    def _process(self, samples: np.ndarray) -> np.ndarray:
        """ピッチシフト処理を適用

        1. 信号を伸長/圧縮（ピッチを下げる/上げる）
        2. リサンプリングで元の長さに戻す

        Args:
            samples: 入力信号 (shape: (num_samples, 2))

        Returns:
            ピッチシフトされた信号 (shape: (num_samples, 2))
        """
        num_samples = len(samples)

        # リサンプリングで時間軸を伸縮
        output = np.zeros_like(samples)
        for ch in range(2):
            # 時間軸を伸縮（ピッチシフトの逆数）
            stretched = signal.resample(
                samples[:, ch], int(num_samples / self.pitch_factor)
            )
            # 元の長さにリサンプリング
            output[:, ch] = signal.resample(stretched, num_samples)

        return output


class ShimmerReverb(Effect):
    """シマーリバーブエフェクト

    通常のリバーブにピッチシフト処理を組み合わせた特殊なリバーブ。
    上方にピッチシフトした信号をリバーブに混ぜることで、
    キラキラとした空間的な効果を生成する。

    Attributes:
        pitch_shift: ピッチシフト量（半音単位）
        mix: ドライ/ウェット比 (0.0-1.0)
        decay: リバーブの減衰時間 (秒)
    """

    def __init__(
        self,
        pitch_shift: float = 12.0,  # 1オクターブ上
        mix: float = 0.5,
        decay: float = 2.0,
        sample_rate: int = 44100,
    ):
        """シマーリバーブを初期化する

        Args:
            pitch_shift: ピッチシフト量（半音単位）, Default: 12.0 (1オクターブ上)
            mix: ドライ/ウェット比 (0.0-1.0), Default: 0.5
            decay: リバーブの減衰時間 (秒), Default: 2.0
            sample_rate: サンプリングレート (Hz), Default: 44100
        """
        super().__init__(mix=mix, sample_rate=sample_rate)

        # エフェクトチェーンの構築
        self.pitch_shifter = PitchShifter(
            semitones=pitch_shift, mix=1.0, sample_rate=sample_rate
        )

        self.reverb = Reverb(
            room_size=decay / 3.0,  # room_sizeは0-1なのでスケーリング
            damping=0.5,
            mix=1.0,
            sample_rate=sample_rate,
        )

    def _process(self, samples: np.ndarray) -> np.ndarray:
        """シマーリバーブ処理を適用

        1. 入力信号をピッチシフト
        2. ピッチシフトした信号にリバーブを適用

        Args:
            samples: 入力信号 (shape: (num_samples, 2))

        Returns:
            処理済み信号 (shape: (num_samples, 2))
        """
        # ピッチシフト処理
        pitched = self.pitch_shifter.process(samples)

        # リバーブ処理
        reverbed = self.reverb.process(pitched)

        return reverbed


class StereoEnhancer(Effect):
    """ステレオエンハンサー

    ステレオイメージを拡大し、空間的な広がりを強調する。
    Haasエフェクトと周波数依存の位相シフトを組み合わせて
    自然な広がりを実現。

    Attributes:
        width: ステレオ幅 (0.0-1.0)
        shimmer: キラキラ効果の強さ (0.0-1.0)
    """

    def __init__(
        self, width: float = 0.5, shimmer: float = 0.2, sample_rate: int = 44100
    ):
        super().__init__(mix=1.0, sample_rate=sample_rate)
        self.width = np.clip(width, 0.0, 1.0)
        self.shimmer = np.clip(shimmer, 0.0, 1.0)

        # Haasエフェクト用のディレイ設定
        self.max_delay = int(0.03 * sample_rate)  # 30ms
        self.delay_buffer = np.zeros((self.max_delay, 2))
        self.delay_pos = 0

        # 周波数依存の位相シフト用パラメータ
        self.phase_buffer_size = 2048
        self.phase_buffer = np.zeros((self.phase_buffer_size, 2))
        self.phase_pos = 0

        # キラキラエフェクト用のモジュレーション
        self.mod_phase = 0.0
        self.mod_freq = 0.5  # Hz

    def _process(self, samples: np.ndarray) -> np.ndarray:
        num_samples = len(samples)
        output = np.zeros_like(samples)

        # ステレオ幅の拡大
        mid = (samples[:, 0] + samples[:, 1]) / 2
        side = (samples[:, 0] - samples[:, 1]) / 2

        # サイド信号の増幅
        enhanced_side = side * (1.0 + self.width)

        # Haasエフェクトの適用
        delay_time = int(self.max_delay * self.width)
        for i in range(num_samples):
            # 遅延信号の取得
            delay_idx = (self.delay_pos - delay_time) % self.max_delay
            delayed = self.delay_buffer[delay_idx]

            # キラキラエフェクトのモジュレーション
            mod = np.sin(
                2 * np.pi * self.mod_freq * i / self.sample_rate + self.mod_phase
            )
            shimmer_amount = self.shimmer * (0.5 + 0.5 * mod)

            # 出力の計算
            output[i, 0] = mid[i] + enhanced_side[i] + delayed[1] * shimmer_amount
            output[i, 1] = mid[i] - enhanced_side[i] + delayed[0] * shimmer_amount

            # バッファの更新
            self.delay_buffer[self.delay_pos] = samples[i]
            self.delay_pos = (self.delay_pos + 1) % self.max_delay

        # モジュレーション位相の更新
        self.mod_phase += 2 * np.pi * self.mod_freq * num_samples / self.sample_rate
        self.mod_phase %= 2 * np.pi

        return output

class Portamento(Effect):
    """ポルタメント（グライド）エフェクト

    周波数の滑らかな遷移を実現する。
    入力信号の周波数を検出し、目標周波数まで
    なめらかに変化させる。

    Attributes:
        time: 遷移時間 (sec), Default: 0.1
        curve: 遷移カーブの特性 ('linear' or 'exponential'), Default: 'exponential'
    """

    def __init__(
        self,
        time: float = 0.1,
        curve: str = "exponential",
        mix: float = 1.0,
        sample_rate: int = 44100,
    ):
        super().__init__(mix=mix, sample_rate=sample_rate)
        self.portamento_time = time
        self.curve = curve
        self.current_freq = 0.0
        self.target_freq = 0.0
        self.transition_samples = 0
        self.samples_processed = 0

    def set_target_frequency(self, freq: float):
        """目標周波数を設定"""
        if self.current_freq == 0.0:
            self.current_freq = freq
        self.target_freq = freq
        self.transition_samples = int(self.portamento_time * self.sample_rate)
        self.samples_processed = 0

    def _process(self, samples: np.ndarray) -> np.ndarray:
        if self.transition_samples == 0:
            return samples

        output = np.zeros_like(samples)
        num_samples = samples.shape[0]

        for i in range(num_samples):
            if self.samples_processed < self.transition_samples:
                # 進行度の計算
                progress = self.samples_processed / self.transition_samples

                # 周波数の補間
                if self.curve == "exponential":
                    # 指数カーブ
                    alpha = 1.0 - np.exp(-5 * progress)
                    current = (
                        self.current_freq * (1.0 - alpha) + self.target_freq * alpha
                    )
                else:
                    # 線形補間
                    current = (
                        self.current_freq * (1.0 - progress)
                        + self.target_freq * progress
                    )

                # 位相の積算
                phase = 2 * np.pi * current * (i / self.sample_rate)
                output[i] = samples[i] * np.exp(1j * phase)

                self.samples_processed += 1
            else:
                output[i] = samples[i]

        return output.real


class Compressor(Effect):
    """ダイナミックレンジコンプレッサー

    信号の動的範囲を制御し、
    音圧と音質を最適化する。

    Attributes:
        threshold: 圧縮開始レベル (dB), Default: -20
        ratio: 圧縮比, Default: 4.0
        attack: アタック時間 (sec), Default: 0.005
        release: リリース時間 (sec), Default: 0.1
        knee: ニー幅 (dB), Default: 6.0
        makeup_gain: メイクアップゲイン (dB), Default: 0.0
    """

    def __init__(
        self,
        threshold: float = -20.0,
        ratio: float = 4.0,
        attack: float = 0.005,
        release: float = 0.1,
        knee: float = 6.0,
        makeup_gain: float = 0.0,
        mix: float = 1.0,
        sample_rate: int = 44100,
    ):
        super().__init__(mix=mix, sample_rate=sample_rate)
        self.threshold = threshold
        self.ratio = max(1.0, ratio)
        self.attack_time = attack
        self.release_time = release
        self.knee = abs(knee)
        self.makeup_gain = makeup_gain

        # 時定数の計算
        self.attack_coef = np.exp(-1.0 / (self.sample_rate * attack))
        self.release_coef = np.exp(-1.0 / (self.sample_rate * release))

        # 状態変数
        self.envelope = 0.0
        self.gain_reduction = 0.0

    def _compute_gain(self, level_db: float) -> float:
        """圧縮のゲイン計算"""
        # ニー領域の処理
        if self.knee > 0:
            knee_start = self.threshold - self.knee / 2
            knee_end = self.threshold + self.knee / 2

            if level_db < knee_start:
                # ニー以下
                return 0.0
            elif level_db > knee_end:
                # ニー以上
                return (
                    self.threshold - level_db + (level_db - self.threshold) / self.ratio
                )
            else:
                # ニー領域
                knee_factor = ((level_db - knee_start) / self.knee) ** 2
                return (level_db - self.threshold) * (1 / self.ratio - 1) * knee_factor
        else:
            # ハードニー
            if level_db < self.threshold:
                return 0.0
            else:
                return (
                    self.threshold - level_db + (level_db - self.threshold) / self.ratio
                )

    def _process(self, samples: np.ndarray) -> np.ndarray:
        output = np.zeros_like(samples)
        num_samples = samples.shape[0]

        for i in range(num_samples):
            # 信号レベルの検出（RMS）
            level = np.sqrt(np.mean(samples[i] ** 2))
            level_db = 20 * np.log10(level + 1e-10)

            # エンベロープの更新
            if level > self.envelope:
                self.envelope = (
                    1 - self.attack_coef
                ) * level + self.attack_coef * self.envelope
            else:
                self.envelope = (
                    1 - self.release_coef
                ) * level + self.release_coef * self.envelope

            # ゲイン計算
            env_db = 20 * np.log10(self.envelope + 1e-10)
            target_gain_db = self._compute_gain(env_db)

            # ゲイン適用
            gain_db = target_gain_db + self.makeup_gain
            gain_linear = 10 ** (gain_db / 20)
            output[i] = samples[i] * gain_linear

        return output

    def set_threshold(self, threshold: float):
        """圧縮開始レベルを設定 (dB)"""
        self.threshold = threshold

    def set_ratio(self, ratio: float):
        """圧縮比を設定"""
        self.ratio = max(1.0, ratio)

    def set_attack(self, attack: float):
        """アタック時間を設定 (sec)"""
        self.attack_time = max(0.0, attack)
        self.attack_coef = np.exp(-1.0 / (self.sample_rate * attack))

    def set_release(self, release: float):
        """リリース時間を設定 (sec)"""
        self.release_time = max(0.0, release)
        self.release_coef = np.exp(-1.0 / (self.sample_rate * release))

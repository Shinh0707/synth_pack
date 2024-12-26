import numpy as np
from abc import ABC, abstractmethod
from synth.env import ADSR


class Oscillator(ABC):
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.phase_left = 0.0
        self.phase_right = 0.0
        self.frequency = 440.0
        self.pan = 0.5  # 0.0 = 完全左, 0.5 = 中央, 1.0 = 完全右

    @abstractmethod
    def process(self, num_samples: int) -> np.ndarray:
        pass

    def set_phase(self, phase_left: int|None=None, phase_right: int|None=None):
        if phase_left is not None:
            self.phase_left = phase_left
        if phase_right is not None:
            self.phase_right = phase_right

    def set_frequency(self, freq: float):
        self.frequency = freq

    def set_pan(self, pan: float):
        """パンニング位置を設定 (0.0 = 左, 0.5 = 中央, 1.0 = 右)"""
        self.pan = max(0.0, min(1.0, pan))

    def apply_panning(self, signal: np.ndarray) -> np.ndarray:
        """信号にパンニングを適用してステレオ信号を生成"""
        left_gain = np.sqrt(1 - self.pan)  # 左チャンネルのゲイン
        right_gain = np.sqrt(self.pan)  # 右チャンネルのゲイン

        # ステレオ信号を生成 (samples, channels)
        if signal.ndim == 1:
            stereo = np.zeros((signal.size, 2))
            stereo[:, 0] = signal * left_gain  # 左チャンネル
            stereo[:, 1] = signal * right_gain  # 右チャンネル
        else:
            stereo = np.zeros_like(signal)
            stereo[:, 0] = signal[:, 0] * left_gain
            stereo[:, 1] = signal[:, 1] * right_gain

        return stereo

class SineOscillator(Oscillator):
    def process(self, num_samples: int) -> np.ndarray:
        t = np.linspace(
            self.phase_left,
            self.phase_left + (num_samples / self.sample_rate),
            num_samples,
            False,
        )
        self.phase_left += num_samples / self.sample_rate
        self.phase_left %= 1.0

        mono = np.sin(2 * np.pi * self.frequency * t)
        return self.apply_panning(mono)


class SquareOscillator(Oscillator):
    def process(self, num_samples: int) -> np.ndarray:
        t = np.linspace(
            self.phase_left,
            self.phase_left + (num_samples / self.sample_rate),
            num_samples,
            False,
        )
        self.phase_left += num_samples / self.sample_rate
        self.phase_left %= 1.0

        mono = np.sign(np.sin(2 * np.pi * self.frequency * t))
        return self.apply_panning(mono)


class PulseOscillator(Oscillator):
    def __init__(self, sample_rate: int = 44100, pulse_width: float = 0.5):
        super().__init__(sample_rate)
        self.pulse_width = max(0.0, min(1.0, pulse_width))

    def set_pulse_width(self, width: float):
        self.pulse_width = max(0.0, min(1.0, width))

    def process(self, num_samples: int) -> np.ndarray:
        t = np.linspace(
            self.phase_left,
            self.phase_left + (num_samples / self.sample_rate),
            num_samples,
            False,
        )
        self.phase_left += num_samples / self.sample_rate
        self.phase_left %= 1.0

        wave = np.sin(2 * np.pi * self.frequency * t)
        mono = np.where(wave > 2 * self.pulse_width - 1, 1.0, -1.0)
        return self.apply_panning(mono)


class SawtoothOscillator(Oscillator):
    def process(self, num_samples: int) -> np.ndarray:
        t = np.linspace(
            self.phase_left,
            self.phase_left + (num_samples / self.sample_rate),
            num_samples,
            False,
        )
        self.phase_left += num_samples / self.sample_rate
        self.phase_left %= 1.0

        mono = 2.0 * (self.frequency * t % 1.0) - 1.0
        return self.apply_panning(mono)

class AudioOscillator(Oscillator):
    """ループ音源をサンプリングするオシレーター"""
    def __init__(self, audio_data: np.ndarray|str,sample_rate:int = 44100):
        super().__init__(sample_rate)
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
    
    def _process(self, num_samples: int, phase_left: int, phase_right: int):
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

    def process(self, num_samples: int):
        output = self._process(num_samples,self.phase_left,self.phase_right)
        self.phase_left = (self.phase_left + num_samples) % self.audio_samples
        self.phase_right = (self.phase_right + num_samples) % self.audio_samples
        return self.apply_panning(output)
    
    def create_sub_oscillator(self):
        """このインスタンスと同じ音源をサンプリングするオシレーターを作成"""
        sub_osc = AudioOscillator(None,self.sample_rate)
        sub_osc._process = self._process
        sub_osc.audio_samples = self.audio_samples
        return sub_osc

class FMOscillator(Oscillator):
    def __init__(
        self, sample_rate: int = 44100, mod_ratio: float = 1.0, mod_index: float = 1.0
    ):
        super().__init__(sample_rate)
        self.mod_ratio = mod_ratio  # モジュレーター周波数比
        self.mod_index = mod_index  # モジュレーション・インデックス
        self.mod_envelope = None  # モジュレーター用エンベロープ

    def set_mod_envelope(self, envelope: ADSR):
        self.mod_envelope = envelope

    def process(self, num_samples: int) -> np.ndarray:
        t = np.linspace(
            self.phase_left,
            self.phase_left + (num_samples / self.sample_rate),
            num_samples,
            False,
        )
        self.phase_left += num_samples / self.sample_rate
        self.phase_left %= 1.0

        # モジュレーター信号の生成
        mod_freq = self.frequency * self.mod_ratio
        if self.mod_envelope is not None:
            mod_env = self.mod_envelope.process(num_samples)
        else:
            mod_env = np.ones(num_samples)

        modulator = np.sin(2 * np.pi * mod_freq * t) * self.mod_index * mod_env

        # キャリア信号の生成（周波数変調を適用）
        mono = np.sin(2 * np.pi * self.frequency * t + modulator)
        return self.apply_panning(mono)


class MetallicOscillator(Oscillator):
    """金属的な倍音を生成するオシレーター"""

    def __init__(self, sample_rate: int = 44100):
        super().__init__(sample_rate)
        self.partials = [
            # [周波数比率, 初期振幅, 減衰速度]
            [1.0, 0.2, 2.0],  # 基本周波数
            [1.7, 0.8, 1.8],  # メジャーサードの上の倍音
            [3.8, 0.5, 1.5],  # 5度上の倍音
            [5.6, 0.4, 2.0],  # 非調和的な高次倍音
            [7.1, 0.4, 2.5],  # より高次の非調和倍音
            [10.6, 0.3, 3.0],  # さらに高次の非調和倍音
        ]
        self.normalize = 1 / max(sum([param[1] for param in self.partials]), 1)
        self.phase = np.zeros(len(self.partials))
        self.time = 0

    def reset_time(self):
        self.time = 0

    def process(self, num_samples: int) -> np.ndarray:
        output = np.zeros(num_samples)
        t = np.linspace(
            self.time, self.time + num_samples / self.sample_rate, num_samples
        )

        for i, (ratio, amp, decay) in enumerate(self.partials):
            freq = 1550 * ratio  # 基本周波数の倍音
            partial = amp * np.sin(2 * np.pi * freq * t + self.phase[i])
            partial *= np.exp(-decay * t)  # 個別の減衰
            output += partial * self.normalize  # 振幅を調整

            # 位相を更新
            self.phase[i] += 2 * np.pi * freq * num_samples / self.sample_rate
            self.phase[i] %= 2 * np.pi

        self.time += num_samples / self.sample_rate
        return self.apply_panning(output)

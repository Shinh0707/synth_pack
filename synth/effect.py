from abc import ABC, abstractmethod
from enum import Enum
import random
from typing import Callable
import numpy as np


class Effect(ABC):
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
    def __init__(
        self,
        room_size: float = 0.8,
        damping: float = 0.5,
        mix: float = 0.5,
        sample_rate: int = 44100,
    ):
        super().__init__(mix=mix, sample_rate=sample_rate)
        self.room_size = max(0.0, min(1.0, room_size))
        self.damping = max(0.0, min(1.0, damping))
        # ステレオ効果を高めるため、左右で異なる遅延時間を設定
        self.delay_times_l = [0.03, 0.05, 0.07, 0.11]
        self.delay_times_r = [0.031, 0.053, 0.073, 0.113]  # 右チャンネルは少し長め
        self.feedbacks = [0.3, 0.25, 0.2, 0.15]

        # 左右のディレイラインを個別に作成
        self.delays_l = [
            Delay(delay_time=dt, feedback=fb, mix=1.0, sample_rate=sample_rate)
            for dt, fb in zip(self.delay_times_l, self.feedbacks)
        ]
        self.delays_r = [
            Delay(delay_time=dt, feedback=fb, mix=1.0, sample_rate=sample_rate)
            for dt, fb in zip(self.delay_times_r, self.feedbacks)
        ]

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
    def __init__(
        self,
        rate: float = 1.0,
        depth: float = 0.002,
        mix: float = 0.5,
        sample_rate: int = 44100,
    ):
        super().__init__(mix=mix, sample_rate=sample_rate)
        self.rate = rate
        self.depth = depth
        # ステレオバッファ (buffer_size, 2)
        self.buffer = np.zeros((int(0.05 * sample_rate), 2))
        self.buffer_pos = 0

    def _process(self, samples: np.ndarray) -> np.ndarray:
        buffer_len = self.buffer.shape[0]
        samples_len = samples.shape[0]

        # 左右で位相の異なるLFOを生成
        lfo_phases = (
            2
            * np.pi
            * self.rate
            * (np.arange(samples_len) / self.sample_rate + self.phase)
        )
        lfo_l = np.sin(lfo_phases)
        lfo_r = np.sin(lfo_phases + np.pi / 2)  # 右チャンネルは90度位相シフト

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
    def __init__(
        self, frequency: float = 440.0, mix: float = 0.5, sample_rate: int = 44100
    ):
        super().__init__(mix=mix, sample_rate=sample_rate)
        self.frequency = frequency

    def _process(self, samples: np.ndarray) -> np.ndarray:
        t = np.arange(samples.shape[0]) / self.sample_rate + self.phase
        # 左右で位相の異なるキャリアを生成
        carrier_l = np.sin(2 * np.pi * self.frequency * t)
        carrier_r = np.sin(2 * np.pi * self.frequency * t + np.pi / 4)  # 45度位相シフト

        output = np.zeros_like(samples)
        output[:, 0] = samples[:, 0] * carrier_l
        output[:, 1] = samples[:, 1] * carrier_r
        return output


class Flanger(Effect):
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


class UnisonEffect(Effect):
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
from abc import ABC, abstractmethod
import numpy as np
from synth.osc import Oscillator, AudioOscillator
from synth.filter import Filter
from synth.effect import Delay, Effect, Reverb, Chorus
from synth.env import ADSR, ADSRState
import random

class Synth(ABC):
    sample_rate: int

    @abstractmethod
    def note_on(self, frequency: float, offset_phase_time: float = 0.0):
        pass
    @abstractmethod
    def note_off(self):
        pass
    @abstractmethod
    def process(self, num_samples: int) -> np.ndarray:
        pass

class ModularSynth(Synth):
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.oscillators: list[Oscillator] = []
        self.effects: list[Effect] = []
        self.envelope = ADSR(sample_rate=sample_rate)

    def add_oscillator(self, osc: Oscillator):
        self.oscillators.append(osc)

    def add_filter(self, filter: Filter):
        filter.sample_rate = self.sample_rate
        self.effects.append(filter)

    def set_envelope(self, attack: float, decay: float, sustain: float, release: float):
        self.envelope = ADSR(attack, decay, sustain, release, self.sample_rate)

    def add_effect(self, effect:Effect):
        effect.sample_rate = self.sample_rate
        self.effects.append(effect)

    def set_delay(
        self, delay_time: float = 0.5, feedback: float = 0.5, mix: float = 0.5
    ):
        # add_effectを使ってほしい
        self.effects.append(Delay(sample_rate=self.sample_rate, delay_time=delay_time, feedback=feedback, mix=mix))

    def set_reverb(
        self, room_size: float = 0.8, damping: float = 0.5, mix: float = 0.5
    ):
        # add_effectを使ってほしい
        self.effects.append(Reverb(sample_rate=self.sample_rate, room_size=room_size, damping=damping, mix=mix))

    def set_chorus(self, rate: float = 1.0, depth: float = 0.003, mix: float = 0.5):
        # add_effectを使ってほしい
        self.effects.append(Chorus(sample_rate=self.sample_rate, rate=rate, depth=depth, mix=mix))

    def note_on(self, frequency: float, offset_phase_time: float = 0.0):
        for osc in self.oscillators:
            osc.set_frequency(frequency)
        self.envelope.note_on()
        for effect in self.effects:
            effect.reset_phase(offset_phase_time)

    def note_off(self):
        if self.envelope.state != ADSRState.RELEASE:
            self.envelope.note_off()

    def process(self, num_samples: int) -> np.ndarray:
        if not self.oscillators:
            return np.zeros((num_samples, 2))
        # Mix oscillators
        mixed = np.zeros((num_samples,2))
        for osc in self.oscillators:
            mixed += osc.process(num_samples)
        mixed /= len(self.oscillators)
        # Apply effects
        for effect in self.effects:
            mixed = effect.process(mixed)
        # Apply envelope
        return mixed * self.envelope.process(num_samples)

class CompositeModularSynth(Synth):
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.synths: list[ModularSynth] = []

    def add_synth(self, synth: ModularSynth):
        synth.sample_rate = self.sample_rate
        self.synths.append(synth)
        return synth

    def note_on(self, frequency: float, offset_phase_time: float = 0.0):
        for synth in self.synths:
            synth.note_on(frequency, offset_phase_time)

    def note_off(self):
        for synth in self.synths:
            synth.note_off()

    def process(self, num_samples: int) -> np.ndarray:
        mixed = np.zeros(num_samples)
        for synth in self.synths:
            mixed += synth.process(num_samples)
        mixed /= len(self.synths)
        return mixed


class GranularSynthesizer(Synth):
    class Grain(ModularSynth):
        def __init__(
            self,
            oscillator: AudioOscillator,
            start_time: float,
            duration: float,
            position: int,
            pitch_shift: float = 1.0,
            pan: float = 0.5
        ):
            oscillator = oscillator.create_sub_oscillator()  # サブオシレーターを作成
            super().__init__(sample_rate=oscillator.sample_rate)
            self.start_time = start_time
            self.duration = duration
            self.position = position
            self.pitch_shift = pitch_shift
            self.envelope = ADSR(
                attack=0.1,
                decay=0.2,
                sustain=0.5,
                release=0.7,
            )
            oscillator.set_pan(pan)
            oscillator.phase_left = position
            oscillator.phase_right = position
            self.add_oscillator(oscillator)
            self.is_active = True

        def process(self, num_samples: int) -> np.ndarray:
            if not self.is_active:
                return np.zeros((num_samples, 2))
            audio = self.oscillators[0].process(num_samples)
            if len(self.oscillators) > 1:
                for osc in self.oscillators[1:]:
                    audio += osc.process(num_samples)
                audio /= len(self.oscillators)
            # Apply effects
            for effect in self.effects:
                audio = effect.process(audio)
            envelope = self.envelope.process(num_samples)
            return audio * envelope

    def __init__(
        self,
        audio_source: str | np.ndarray,
        sample_rate: int = 44100,
        grain_size: float = 0.1,
        grain_density: float = 10,
        position_jitter: float = 0.1,
        pitch_jitter: float = 0.1,
        max_grains: int = 20
    ):
        """
        Parameters:
            audio_source: オーディオファイルパスまたは波形データ
            sample_rate: サンプリングレート
            grain_size: グレインの基本サイズ（秒）
            grain_density: 1秒あたりの新規グレイン生成数
            position_jitter: 位置のランダム性（0-1）
            pitch_jitter: ピッチのランダム性（0-1）
            max_grains: グレインの最大数
        """
        self.base_oscillator = AudioOscillator(audio_source, sample_rate)
        self.sample_rate = sample_rate
        self.grain_size = grain_size
        self.grain_density = grain_density
        self.position_jitter = position_jitter
        self.pitch_jitter = pitch_jitter
        self.max_grains = max_grains
        self.frequency = 0

        self.grains: list[GranularSynthesizer.Grain] = []
        self.current_time = 0.0
        self.playhead_position = 0
        self.is_playing = True

    def _create_grain(self) -> Grain:
        """新しいグレインを生成"""
        # ランダム性を加えたパラメータの計算
        duration = self.grain_size * (1 + random.uniform(-0.2, 0.2))
        position_offset = int(
            random.uniform(-self.position_jitter, self.position_jitter)
            * self.sample_rate
        )
        position = (
            self.playhead_position + position_offset
        ) % self.base_oscillator.audio_samples
        pitch_shift = 1.0 + random.uniform(-self.pitch_jitter, self.pitch_jitter)
        pan = random.uniform(0.3, 0.7)  # ステレオフィールドでの位置

        # グレインの生成
        return GranularSynthesizer.Grain(
            oscillator=self.base_oscillator,
            start_time=self.current_time,
            duration=duration,
            position=position,
            pitch_shift=pitch_shift,
            pan=pan,
        )

    def process(self, num_samples: int) -> np.ndarray:
        if (not self.is_playing) and len(self.grains) == 0:
            return np.zeros((num_samples, 2))

        output = np.zeros((num_samples, 2))
        time_step = num_samples / self.sample_rate

        # 新しいグレインの生成
        if self.is_playing:
            num_new_grains = max(int(self.grain_density * time_step),1)
            living_grains = len([
                1
                for grain in self.grains
                if not (
                    grain.envelope.state == ADSRState.IDLE
                    or grain.envelope.state == ADSRState.RELEASE
                )
            ])
            num_new_grains = max(0, min(self.max_grains-living_grains,num_new_grains))
            for _ in range(num_new_grains):
                new_grain = self._create_grain()
                new_grain.note_on(self.frequency)
                self.grains.append(new_grain)
        # 各グレインの処理と合成
        active_grains = []
        for grain in self.grains:
            if (self.current_time - grain.start_time) <= grain.duration:
                grain_output = grain.process(num_samples)
                output += grain_output
                active_grains.append(grain)
            elif grain.envelope.state != ADSRState.IDLE:
                if grain.envelope.state != ADSRState.RELEASE:
                    grain.note_off()
                grain_output = grain.process(num_samples)
                output += grain_output
                active_grains.append(grain)
        # アクティブなグレインのみを保持
        self.grains = active_grains

        # 正規化（オーバーフロー防止）
        if len(self.grains) > 0:
            output /= np.sqrt(len(self.grains))

        # 時間と再生位置の更新
        self.current_time += time_step
        self.playhead_position = (
            self.playhead_position + num_samples
        ) % self.base_oscillator.audio_samples

        return output

    def note_on(self, frequency: float = None, offset_phase_time: float = 0.0):
        """再生開始"""
        self.is_playing = True
        self.frequency = frequency
        self.grains = []

    def note_off(self):
        """再生停止"""
        self.is_playing = False
        for grain in self.grains:
            grain.note_off()

    def set_grain_size(self, size: float):
        """グレインサイズの設定（秒）"""
        self.grain_size = max(0.01, min(1.0, size))

    def set_grain_density(self, density: float):
        """グレイン密度の設定（グレイン/秒）"""
        self.grain_density = max(1, density)

    def set_position_jitter(self, amount: float):
        """位置のランダム性を設定（0-1）"""
        self.position_jitter = max(0.0, min(1.0, amount))

    def set_pitch_jitter(self, amount: float):
        """ピッチのランダム性を設定（0-1）"""
        self.pitch_jitter = max(0.0, min(1.0, amount))

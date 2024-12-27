from abc import ABC, abstractmethod
from typing import Generic, Optional
import numpy as np
from synth.osc import Oscillator, AudioOscillator
from synth.filter import Filter
from synth.effect import Delay, Effect, Reverb, Chorus
from synth.env import ADSR, ADSRState
import random

"""シンセサイザーの基本クラスと各種実装

基本的なシンセサイザーから高度な音声合成まで、
様々な方式のシンセサイザーを提供する。
各シンセサイザーは独自の音声合成方式を持ち、
オシレーター、フィルター、エフェクトを組み合わせて
複雑な音色を生成することが可能。

Classes:
    Synth: シンセサイザーの基底クラス
    ModularSynth: モジュラー方式のシンセサイザー
    CompositeModularSynth: 複数のModularSynthを組み合わせたシンセサイザー
    GranularSynth: グラニュラー合成方式のシンセサイザー
"""


class Synth(ABC):
    """シンセサイザーの基底クラス

    全てのシンセサイザークラスはこのクラスを継承して実装する。
    ADSRエンベロープによる音量制御機能を提供する。

    Attributes:
        sample_rate: サンプリングレート (Hz), Default: 44100
        envelope: ADSRエンベロープ

    Methods:
        note_on(frequency: float, offset_phase_time: float = 0.0):
            指定された周波数でノートをオンにする
        note_off():
            現在鳴っているノートをオフにする
        process(num_samples: int, samples: Optional[np.ndarray] = None):
            指定されたサンプル数の音声を生成する
    """
    sample_rate: int
    envelope: ADSR
    @abstractmethod
    def __init__(self, sample_rate:int=44100):
        """シンセサイザーを初期化する

        Args:
            sample_rate: サンプリングレート (Hz), Default: 44100
        """
        self.sample_rate = sample_rate
        self.envelope = ADSR(sample_rate=sample_rate)

    def note_on(self, frequency: float, offset_phase_time: float = 0.0):
        self.envelope.note_on()

    def note_off(self):
        if self.envelope.state != ADSRState.RELEASE:
            self.envelope.note_off()
    def process(self, num_samples: int, samples: Optional[np.ndarray]=None) -> np.ndarray:
        output = self._process(
            num_samples,
            np.zeros((num_samples, 2)) if samples is None else samples
        )
        return output * self.envelope.process(num_samples)

    @abstractmethod
    def _process(self, num_samples: int, samples: np.ndarray) -> np.ndarray:
        pass

    def getADSR(self):
        return self.envelope

    def changeADSR(self, envelope:ADSR):
        self.envelope = envelope

class ModularSynth(Synth):
    """モジュラー方式のシンセサイザー

    複数のオシレーター、フィルター、エフェクトを
    自由に組み合わせて音色を作り出すことができる。
    モジュラーシンセサイザーの基本的な機能を実装。

    Attributes:
        oscillators: オシレーターのリスト
        effects: フィルターとエフェクトのリスト

    Methods:
        add_oscillator(osc: Oscillator):
            オシレーターを追加する
        add_filter(filter: Filter):
            フィルターを追加する
        add_effect(effect: Effect):
            エフェクトを追加する
        set_envelope(attack: float, decay: float, sustain: float, release: float):
            エンベロープのパラメータを設定する
    """
    def __init__(self, sample_rate: int = 44100):
        """モジュラーシンセを初期化する

        Args:
            sample_rate: サンプリングレート (Hz), Default: 44100

        Example:
            >>> synth = ModularSynth(44100)
            >>> synth.add_oscillator(SawtoothOscillator())
            >>> synth.add_filter(LowPassFilter())
            >>> synth.set_envelope(0.1, 0.2, 0.7, 0.3)
        """
        super().__init__(sample_rate=sample_rate)
        self.oscillators: list[Oscillator] = []
        self.effects: list[Effect] = []

    def add_oscillator(self, osc: Oscillator):
        self.oscillators.append(osc)
        return self.oscillators[-1]

    def add_filter(self, filter: Filter) -> Filter:
        filter.sample_rate = self.sample_rate
        self.effects.append(filter)
        return self.effects[-1]

    def set_envelope(self, attack: float, decay: float, sustain: float, release: float):
        self.envelope.attack_time = attack
        self.envelope.decay_time = decay
        self.envelope.sustain_level = sustain
        self.envelope.release_time = release

    def add_effect(self, effect:Effect):
        effect.sample_rate = self.sample_rate
        self.effects.append(effect)
        return self.effects[-1]

    def note_on(self, frequency: float, offset_phase_time: float = 0.0):
        super().note_on(frequency=frequency, offset_phase_time=offset_phase_time)
        for osc in self.oscillators:
            osc.set_frequency(frequency)
        for effect in self.effects:
            effect.reset_phase(offset_phase_time)

    def _process(self, num_samples:int, samples: np.ndarray) -> np.ndarray:
        if not self.oscillators:
            return samples
        # Mix oscillators
        for osc in self.oscillators:
            samples += osc.process(num_samples)
        samples /= len(self.oscillators)
        # Apply effects
        for effect in self.effects:
            samples = effect.process(samples)
        return samples

class CompositeModularSynth(Synth):
    """複数のModularSynthを組み合わせたシンセサイザー

    複数のモジュラーシンセを重ね合わせることで、
    より複雑な音色を作り出すことができる。
    レイヤー化された音色の生成に適している。

    Attributes:
        synths: ModularSynthのリスト

    Methods:
        add_synth(synth: ModularSynth):
            新しいシンセサイザーレイヤーを追加する
    """
    def __init__(self, sample_rate: int = 44100):
        """複合モジュラーシンセを初期化する

        Args:
            sample_rate: サンプリングレート (Hz), Default: 44100

        Example:
            >>> composite = CompositeModularSynth(44100)
            >>> synth1 = ModularSynth()
            >>> synth2 = ModularSynth()
            >>> composite.add_synth(synth1)
            >>> composite.add_synth(synth2)
        """
        super().__init__(sample_rate=sample_rate)
        self.synths: list[ModularSynth] = []

    def add_synth(self, synth: ModularSynth):
        synth.sample_rate = self.sample_rate
        self.synths.append(synth)
        return synth

    def note_on(self, frequency: float, offset_phase_time: float = 0.0):
        super().note_on(frequency=frequency, offset_phase_time=offset_phase_time)
        for synth in self.synths:
            synth.note_on(frequency, offset_phase_time)

    def note_off(self):
        super().note_off()
        for synth in self.synths:
            synth.note_off()

    def _process(self, num_samples: int, samples: np.ndarray) -> np.ndarray:
        for synth in self.synths:
            samples += synth.process(num_samples)
        samples /= len(self.synths)
        return samples


class GranularSynth(Synth):
    """グラニュラー合成方式のシンセサイザー

    音声をマイクロ単位の粒子（グレイン）に分解し、
    それらを重ね合わせることで新しい音色を生成する。
    テクスチャー的な音色や時間軸の操作が可能。

    Attributes:
        grain_size: グレインの大きさ (sec)
        grain_density: 1秒あたりのグレイン生成数
        position_jitter: 位置のランダム性 (0.0-1.0)
        pitch_jitter: ピッチのランダム性 (0.0-1.0)
        max_grains: 同時に発音可能な最大グレイン数

    Methods:
        set_grain_size(size: float):
            グレインサイズを設定する (sec)
        set_grain_density(density: float):
            グレイン密度を設定する (grains/sec)
        set_position_jitter(amount: float):
            位置のランダム性を設定する (0.0-1.0)
        set_pitch_jitter(amount: float):
            ピッチのランダム性を設定する (0.0-1.0)
    """
    class Grain(ModularSynth):
        """グレイン（音の粒子）を表現するクラス

        GranularSynthで使用される個々のグレインを管理する。
        ModularSynthを継承し、グレイン特有の機能を追加。

        Attributes:
            start_time: グレインの開始時間 (sec)
            duration: グレインの継続時間 (sec)
            position: 音源内の開始位置 (samples)
            pitch_shift: ピッチシフト量
            pan: パンニング位置 (0.0-1.0)
            is_active: グレインがアクティブかどうか
        """
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

        def _process(self, samples: np.ndarray, num_samples: int) -> np.ndarray:
            if not self.is_active:
                return samples
            samples = self.oscillators[0].process(num_samples)
            if len(self.oscillators) > 1:
                for osc in self.oscillators[1:]:
                    samples += osc.process(num_samples)
                samples /= len(self.oscillators)
            # Apply effects
            for effect in self.effects:
                samples = effect.process(samples)
            return samples

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
        """グラニュラーシンセを初期化する

        Args:
            audio_source: 音源データまたは音声ファイルパス
            sample_rate: サンプリングレート (Hz), Default: 44100
            grain_size: グレインの基本サイズ (sec), Default: 0.1
            grain_density: 1秒あたりの新規グレイン生成数, Default: 10
            position_jitter: 位置のランダム性 (0.0-1.0), Default: 0.1
            pitch_jitter: ピッチのランダム性 (0.0-1.0), Default: 0.1
            max_grains: 最大同時発音グレイン数, Default: 20

        Example:
            >>> # 音声ファイルからグラニュラーシンセを作成
            >>> granular = GranularSynth("sample.wav",
            ...                         grain_size=0.05,
            ...                         grain_density=20)
            >>> # パラメータを調整
            >>> granular.set_position_jitter(0.2)
            >>> granular.set_pitch_jitter(0.15)
        """
        self.base_oscillator = AudioOscillator(audio_source, sample_rate)
        self.sample_rate = sample_rate
        self.grain_size = grain_size
        self.grain_density = grain_density
        self.position_jitter = position_jitter
        self.pitch_jitter = pitch_jitter
        self.max_grains = max_grains
        self.frequency = 0

        self.grains: list[GranularSynth.Grain] = []
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
        return GranularSynth.Grain(
            oscillator=self.base_oscillator,
            start_time=self.current_time,
            duration=duration,
            position=position,
            pitch_shift=pitch_shift,
            pan=pan,
        )

    def _process(self, num_samples: int, samples: np.ndarray) -> np.ndarray:
        if (not self.is_playing) and len(self.grains) == 0:
            return samples

        output = samples
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
        super().note_on(frequency=frequency,offset_phase_time=offset_phase_time)
        self.is_playing = True
        self.frequency = frequency
        self.grains = []

    def note_off(self):
        """再生停止"""
        super().note_off()
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

from synth.env import ADSR
from synth.synth import ModularSynth
from synth.osc import SawtoothOscillator, SineOscillator, PulseOscillator
from synth.filter import LowPassFilter
from synth.effect import Chorus, Reverb, Delay, UnisonEffect
import numpy as np


class SuperSawSynth(ModularSynth):
    """
    Future Bass用のSuper Sawシンセ
    厚みのある温かいパッド系サウンドを生成
    """

    def __init__(self, sample_rate: int = 44100):
        super().__init__(sample_rate)
        self._initialize_oscillators()
        self._initialize_effects()
        self._initialize_filters()
        self._initialize_envelope()

    def _initialize_oscillators(self):
        # 基本音用のノコギリ波
        base_saw = SawtoothOscillator(self.sample_rate)
        self.add_oscillator(base_saw)
        # オクターブ上のノコギリ波
        high_saw = SawtoothOscillator(self.sample_rate)
        self.add_oscillator(high_saw)

    def _initialize_effects(self):
        # 基本音用のユニゾン
        base_unison = UnisonEffect(voices=7, detune=0.002, mix=1.0)
        self.add_effect(base_unison)
        # オクターブ上用のユニゾン（より広がり大きめ）
        high_unison = UnisonEffect(voices=5, detune=0.004, mix=0.5)
        self.add_effect(high_unison)
        # コーラスで温かみを追加
        chorus = Chorus(rate=0.6, depth=0.002, mix=0.3)
        self.add_effect(chorus)

    def _initialize_filters(self):
        lpf = LowPassFilter()
        lpf.set_cutoff(2000.0)
        lpf.set_resonance(0.8)
        self.add_filter(lpf)

    def _initialize_envelope(self):
        self.set_envelope(
            attack=0.1, decay=0.3, sustain=0.6, release=0.4  # なめらかな立ち上がり
        )

    def note_on(self, frequency: float, offset_phase_time: float = 0.0):
        self.oscillators[0].set_frequency(frequency)
        self.oscillators[1].set_frequency(frequency * 2.0)  # オクターブ上
        super().note_on(frequency, offset_phase_time)


class KawaiiPadSynth(ModularSynth):
    """
    ホワンとしたKawaiiパッドサウンド
    フィルターエンベロープで特徴的な音色を作る
    """

    def __init__(self, sample_rate: int = 44100):
        super().__init__(sample_rate)
        self._initialize_oscillators()
        self._initialize_filters()
        self._initialize_effects()
        self._initialize_envelope()
        self.filter_envelope = self._create_filter_envelope()

    def _initialize_oscillators(self):
        # メインのノコギリ波
        main_saw = SawtoothOscillator(self.sample_rate)
        self.add_oscillator(main_saw)
        # サブのパルス波
        sub_pulse = PulseOscillator(self.sample_rate, pulse_width=0.3)
        self.add_oscillator(sub_pulse)

    def _initialize_filters(self):
        self.lpf = LowPassFilter()
        self.lpf.set_cutoff(500)  # 最初は閉じ気味
        self.lpf.set_resonance(1.2)  # レゾナンス強め
        self.add_filter(self.lpf)

    def _initialize_effects(self):
        # コーラスで広がりを
        chorus = Chorus(rate=0.7, depth=0.003, mix=0.4)
        self.add_effect(chorus)
        # リバーブで空間を作る
        reverb = Reverb(room_size=0.6, damping=0.4, mix=0.3)
        self.add_effect(reverb)

    def _create_filter_envelope(self):
        filter_env = ADSR(
            attack=0.2,  # ゆっくり開く
            decay=0.4,
            sustain=0.3,  # やや閉じた状態を維持
            release=0.6,
            sample_rate=self.sample_rate,
        )
        return filter_env

    def _initialize_envelope(self):
        self.set_envelope(attack=0.15, decay=0.3, sustain=0.5, release=0.8)

    def process(self, num_samples: int) -> np.ndarray:
        # フィルターエンベロープの処理
        filter_mod = self.filter_envelope.process(num_samples)
        self.lpf.set_cutoff(500 + filter_mod * 3000)  # フィルターを動的に制御
        return super().process(num_samples)


class KawaiiSynthBell(ModularSynth):
    """
    金属的なベル系シンセサウンド
    FMシンセシスで特徴的な倍音を生成
    """

    def __init__(self, sample_rate: int = 44100):
        super().__init__(sample_rate)
        self._initialize_oscillators()
        self._initialize_filters()
        self._initialize_effects()
        self._initialize_envelope()
        self.modulation_index = 2.0
        self.time_passed = 0

    def _initialize_oscillators(self):
        # キャリア（基本となる音）
        self.carrier = SineOscillator(self.sample_rate)
        self.add_oscillator(self.carrier)
        # モジュレータ（倍音を作る）
        self.modulator = SineOscillator(self.sample_rate)
        self.add_oscillator(self.modulator)
        # 輝きを加えるサイン波
        bright_sine = SineOscillator(self.sample_rate)
        self.add_oscillator(bright_sine)

    def _initialize_filters(self):
        # ハイパスフィルター
        self.hpf = LowPassFilter()  # 実際はローパスだが逆に使う
        self.hpf.set_cutoff(1000)
        self.hpf.set_resonance(0.9)
        self.add_filter(self.hpf)

    def _initialize_effects(self):
        # キラキラ感のためのコーラス
        chorus = Chorus(rate=1.2, depth=0.001, mix=0.3)
        self.add_effect(chorus)
        # 空間系のディレイ
        delay = Delay(delay_time=0.25, feedback=0.3, mix=0.2)
        self.add_effect(delay)
        # 広がり用のリバーブ
        reverb = Reverb(room_size=0.8, damping=0.2, mix=0.25)
        self.add_effect(reverb)

    def _initialize_envelope(self):
        self.set_envelope(
            attack=0.02,  # 素早い立ち上がり
            decay=0.3,  # 適度な減衰
            sustain=0.2,  # 控えめなサステイン
            release=1.5,  # 長めの余韻
        )

    def note_on(self, frequency: float, offset_phase_time: float = 0.0):
        # キャリア周波数の設定
        self.carrier.set_frequency(frequency)
        # モジュレータ周波数の設定（3オクターブ上）
        self.modulator.set_frequency(frequency * 8.0)
        # 輝き用の周波数設定
        self.oscillators[2].set_frequency(frequency * 4.0)
        self.time_passed = 0
        super().note_on(frequency, offset_phase_time)

    def process(self, num_samples: int) -> np.ndarray:
        # 時間経過による変調指数の減衰
        self.time_passed += num_samples / self.sample_rate
        current_mod_index = self.modulation_index * np.exp(-2 * self.time_passed)

        # モジュレータ信号の生成
        mod_signal = self.modulator.process(num_samples) * current_mod_index

        # FM処理
        t = np.repeat(np.arange(num_samples)[:,np.newaxis],2,-1) / self.sample_rate
        carrier_phase = 2 * np.pi * self.carrier.frequency * t + mod_signal
        fm_signal = np.sin(carrier_phase)

        # 他のオシレータと合成
        bright_signal = self.oscillators[2].process(num_samples) * 0.3
        mixed = (fm_signal + bright_signal) / 2

        # エフェクト処理
        for effect in self.effects:
            mixed = effect.process(mixed)

        return mixed * self.envelope.process(num_samples)


# 使用例
if __name__ == "__main__":
    from synth.player import SimpleSynthPlayer, Note
    import soundfile as sf

    def test_synth(synth, sequence, filename):
        player = SimpleSynthPlayer(synth, bpm=128)  # Future Base typical BPM
        audio_data = player.play_sequence(sequence, loop=False)
        sf.write(
            filename,
            audio_data,
            synth.sample_rate,
            subtype="PCM_24",
        )

    # Super Saw test
    super_saw = SuperSawSynth(44100)
    saw_sequence = [
        Note("C4", 1.0),
        Note("E4", 1.0),
        Note("G4", 1.0),
        Note("B4", 1.0),
    ]
    test_synth(super_saw, saw_sequence, "kawaii_super_saw.wav")

    # Kawaii Pad test
    kawaii_pad = KawaiiPadSynth(44100)
    pad_sequence = [
        Note("C5", 2.0),
        Note("G4", 2.0),
        Note("A4", 2.0),
    ]
    test_synth(kawaii_pad, pad_sequence, "kawaii_pad.wav")

    # Bell test
    bell_synth = KawaiiSynthBell(44100)
    bell_sequence = [
        Note("C6", 0.5),
        Note("G5", 0.5),
        Note("E6", 0.5),
        Note("C6", 0.5),
    ]
    test_synth(bell_synth, bell_sequence, "kawaii_bell.wav")

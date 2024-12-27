from synth import *
import numpy as np


class CuteSynth(ModularSynth):
    def __init__(self, sample_rate: int = 44100):
        super().__init__(sample_rate)

        # メインの音色: キラキラした高音成分
        main_saw: SuperSawOscillator = self.add_oscillator(SuperSawOscillator(self.sample_rate))
        main_saw.set_frequency(0)  # 基準周波数は後でnote_onで設定
        main_saw.detune = 0.15  # デチューンで広がりを持たせる
        main_saw.stereo_spread = 0.3

        # サブの音色: 可愛らしさを強調する正弦波
        sub_sine = self.add_oscillator(SineOscillator(self.sample_rate))
        sub_sine.set_frequency(0)
        sub_sine.amplitude = 0.4

        # キラキラ感を出すFM音源
        sparkle = self.add_oscillator(
            FMOscillator(self.sample_rate, mod_ratio=4.0, mod_index=2.0)  # 高倍音を強調
        )
        sparkle.amplitude = 0.3

        # 音の輪郭を整えるフィルター
        bright_filter: BandPassFilter = self.add_filter(BandPassFilter())  # 新規フィルター提案
        bright_filter.set_cutoff(3000)
        bright_filter.set_resonance(2.0)

        # エフェクトチェーン

        # 音を広げるコーラス
        chorus: Chorus = self.add_effect(Chorus())
        chorus.rate = 0.6
        chorus.depth = 0.02
        chorus.mix = 0.3

        # キラキラ感を強調するFlanger
        flanger: Flanger = self.add_effect(Flanger())
        flanger.rate = 0.8
        flanger.depth = 0.004
        flanger.feedback = 0.3
        flanger.mix = 0.2

        # ステレオを広げる
        enhancer = self.add_effect(
            StereoEnhancer(width=0.6, shimmer=0.3)  # 新規エフェクト提案
        )

        # ADSRエンベロープ設定
        # 可愛らしさを表現するため、アタックは短めでサステインを高めに
        self.set_envelope(
            attack=0.02,  # 素早いアタック
            decay=0.1,  # 短めのディケイ
            sustain=0.7,  # 高めのサステイン
            release=0.2,  # 適度なリリース
        )


# 使用例
if __name__ == "__main__":
    from synth.player import SimpleSynthPlayer, Note
    import soundfile as sf

    # かわいらしいメロディのテストシーケンス
    test_sequence = [
        Note("C5", 0.25),  # 高めの音域で
        Note("E5", 0.25),
        Note("G5", 0.25),
        Note("A5", 0.25),
        Note("G5", 0.5),
        Note("E5", 0.25),
        Note("C5", 0.25),
        Note("D5", 0.5),
        Note("C5", 0.5),
    ]

    synth = CuteSynth(44100)
    player = SimpleSynthPlayer(synth, bpm=140)  # テンポアップでポップに
    audio_data = player.play_sequence(test_sequence, loop=False)
    sf.write("cute_synth_test.wav", audio_data, synth.sample_rate, subtype="PCM_24")

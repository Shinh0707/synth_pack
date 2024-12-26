import time
from typing import List, Union, Optional
import numpy as np

from synth.filter import LowPassFilter
from synth.osc import PulseOscillator, SawtoothOscillator
from synth.synth import ModularSynth,Synth


class Note:
    def __init__(self, value: Union[int, str], duration: float = 0.5):
        self.value = value  # MIDIノートナンバーまたはコード名
        self.duration = duration  # 秒単位の長さ


class SimpleSynthPlayer:
    def __init__(self, synth: Synth, bpm: float = 120.0):
        self.synth = synth
        self.bpm = bpm
        self.is_playing = False

    @staticmethod
    def note_to_freq(note: Union[int, str]) -> float:
        """MIDIノートナンバーまたはコード名を周波数に変換"""
        if isinstance(note, int):
            # MIDI Note Number to frequency (A4 = 69 = 440Hz)
            return 440.0 * (2.0 ** ((note - 69) / 12.0))
        else:
            # コード名から周波数を計算（簡易版）
            note_map = {
                "C": 60,
                "C#": 61,
                "Db": 61,
                "D": 62,
                "D#": 63,
                "Eb": 63,
                "E": 64,
                "F": 65,
                "F#": 66,
                "Gb": 66,
                "G": 67,
                "G#": 68,
                "Ab": 68,
                "A": 69,
                "A#": 70,
                "Bb": 70,
                "B": 71,
            }
            base_note = "".join(c for c in note if not c.isdigit())
            octave = int("".join(c for c in note if c.isdigit()) or "4")
            midi_note = note_map.get(base_note, 69) + (octave - 4) * 12
            return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))

    def beat_to_seconds(self, beats: float) -> float:
        """拍数を秒数に変換"""
        return beats * (60.0 / self.bpm)

    def set_bpm(self, bpm: float):
        """BPMを設定"""
        self.bpm = max(1.0, bpm)

    def play_sequence(self, sequence: List[Note], loop: bool = False) -> np.ndarray:
        """シーケンスを再生して、すべての音声データを返す"""
        self.is_playing = True
        total_samples = 0

        # 合計サンプル数を計算
        for note in sequence:
            samples = int(note.duration * self.synth.sample_rate)
            total_samples += samples

        # 出力バッファを準備
        output = np.zeros((total_samples, 2))
        current_sample = 0

        while self.is_playing:
            for note in sequence:
                if not self.is_playing:
                    break
                # サンプル数を計算
                samples = int(note.duration * self.synth.sample_rate)
                # ノートの周波数を設定して発音
                if note.value is not None:
                    freq = self.note_to_freq(note.value)
                    self.synth.note_on(freq)
                # 音声データを生成
                samples = min(total_samples - current_sample, samples)
                audio_data = self.synth.process(samples)
                # 出力バッファに書き込み
                output[current_sample:current_sample+samples] += audio_data
                current_sample += samples
                # ノートオフ
                self.synth.note_off()
                # 次のノートまでの無音部分を生成
                release_samples = int(0.01 * self.synth.sample_rate)  # 10ms
                release_samples = min(total_samples - current_sample, release_samples)
                if release_samples > 0:
                    release_data = self.synth.process(release_samples)
                    output[current_sample : current_sample + release_samples] += release_data
                    current_sample += release_samples

            if not loop:
                break

        return output

    def stop(self):
        """再生を停止"""
        self.is_playing = False


# example_usage.py
if __name__ == "__main__":
    # シンセサイザーの作成
    synth = ModularSynth(44100)

    # オシレーターの追加
    synth.add_oscillator(SawtoothOscillator())
    synth.add_oscillator(PulseOscillator(pulse_width=0.3))

    # フィルターとエフェクトの設定
    lpf = LowPassFilter()
    lpf.set_cutoff(2000)
    synth.add_filter(lpf)
    synth.set_reverb(room_size=0.7, mix=0.2) #非推奨

    # エンベロープの設定
    synth.set_envelope(attack=0.1, decay=0.2, sustain=0.7, release=0.3)

    # プレイヤーの作成
    player = SimpleSynthPlayer(synth, bpm=120)

    # MIDIノートナンバーを使用したシーケンス
    sequence1 = [
        Note(60, 0.5),  # C4
        Note(64, 0.5),  # E4
        Note(67, 0.5),  # G4
        Note(72, 0.5),  # C5
    ]

    # コード名を使用したシーケンス
    sequence2 = [
        Note("C4", 1.0),
        Note("F4", 1.0),
        Note("G4", 1.0),
        Note("C4", 1.0),
    ]

    # シーケンスを再生して音声データを取得
    audio_data = player.play_sequence(sequence1, loop=False)
    print(audio_data)
    # ここで audio_data を WAV ファイルとして保存したり
    # オーディオデバイスで再生したりできます

    import soundfile as sf

    sf.write(
        "test.wav",
        audio_data,
        synth.sample_rate,
        subtype="PCM_24",
    )
    print("テスト音声を保存しました。")


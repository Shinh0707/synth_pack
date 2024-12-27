import numpy as np
from enum import Enum

"""ADSR (Attack, Decay, Sustain, Release) エンベロープの実装

シンセサイザーの音量エンベロープを制御するクラス群を提供する。
エンベロープはnote_on時にAttack->Decay->Sustain、note_off時にReleaseの状態遷移を行う。

Classes:
    ADSRState: エンベロープの状態を表す列挙型
    ADSR: ADSRエンベロープの実装クラス
"""

class ADSRState(Enum):
    """ADSRエンベロープの状態を表す列挙型

    Attributes:
        IDLE: 非アクティブ状態
        ATTACK: アタック状態
        DECAY: ディケイ状態
        SUSTAIN: サステイン状態
        RELEASE: リリース状態
    """
    IDLE = 0
    ATTACK = 1
    DECAY = 2
    SUSTAIN = 3
    RELEASE = 4

class ADSR:
    """ADSRエンベロープジェネレータ

    音量エンベロープを生成するクラス。
    各状態でのレベル変化を制御し、出力レベルを計算する。

    Attributes:
        attack_time: アタック時間 (sec), Default: 0.1
        decay_time: ディケイ時間 (sec), Default: 0.1
        sustain_level: サステインレベル (0.0-1.0), Default: 0.7
        release_time: リリース時間 (sec), Default: 0.2
        sample_rate: サンプリングレート (Hz), Default: 44100
        state: 現在のエンベロープ状態
        current_level: 現在の出力レベル (0.0-1.0)
    """
    def __init__(
        self,
        attack: float = 0.1,
        decay: float = 0.1,
        sustain: float = 0.7,
        release: float = 0.2,
        sample_rate: int = 44100,
    ):
        self.attack_time = attack
        self.decay_time = decay
        self.sustain_level = sustain
        self.release_time = release
        self.sample_rate = sample_rate

        self.state = ADSRState.IDLE
        self.current_level = 0.0
        self.samples_processed = 0

    def note_on(self, samples_processed: int = 0):
        self.state = ADSRState.ATTACK
        self.samples_processed = samples_processed

    def note_off(self, samples_processed:int = 0):
        self.state = ADSRState.RELEASE
        self.samples_processed = samples_processed

    def process(self, num_samples: int) -> np.ndarray:
        output = np.zeros((num_samples, 2))
        if self.state == ADSRState.IDLE:
            return output

        attack_samples = int(self.attack_time * self.sample_rate)
        decay_samples = int(self.decay_time * self.sample_rate)
        release_samples = int(self.release_time * self.sample_rate)
        processed_up_to = 0  # 処理済みのインデックスを追跡

        if self.state == ADSRState.ATTACK:
            attack_end_index = min(num_samples, attack_samples - self.samples_processed)
            attack_indices = np.repeat(
                np.arange(
                    self.samples_processed, self.samples_processed + attack_end_index
                )[:, np.newaxis],
                repeats=2,
                axis=1,
            )
            output[processed_up_to : processed_up_to + attack_end_index] = (
                attack_indices / attack_samples
            )
            self.samples_processed += attack_end_index
            processed_up_to += attack_end_index

            if self.samples_processed >= attack_samples:
                self.state = ADSRState.DECAY
                self.samples_processed = 0

        if self.state == ADSRState.DECAY and processed_up_to < num_samples:
            decay_end_index = min(
                num_samples - processed_up_to, decay_samples - self.samples_processed
            )
            decay_indices = np.repeat(
                np.arange(
                    self.samples_processed, self.samples_processed + decay_end_index
                )[:, np.newaxis],
                repeats=2,
                axis=1,
            )
            output[processed_up_to : processed_up_to + decay_end_index] = 1.0 - (
                1.0 - self.sustain_level
            ) * (decay_indices / decay_samples)
            self.samples_processed += decay_end_index
            processed_up_to += decay_end_index

            if self.samples_processed >= decay_samples:
                self.state = ADSRState.SUSTAIN
                self.samples_processed = 0

        if self.state == ADSRState.SUSTAIN and processed_up_to < num_samples:
            output[processed_up_to:] = self.sustain_level
            processed_up_to = num_samples  # SUSTAIN 以降の処理をスキップ

        if self.state == ADSRState.RELEASE and processed_up_to < num_samples:
            release_end_index = min(
                num_samples - processed_up_to, release_samples - self.samples_processed
            )
            release_indices = np.repeat(
                np.arange(
                    self.samples_processed, self.samples_processed + release_end_index
                )[:, np.newaxis],
                repeats=2,
                axis=1,
            )
            output[processed_up_to : processed_up_to + release_end_index] = (
                self.sustain_level * (1 - release_indices / release_samples)
            )
            self.samples_processed += release_end_index
            processed_up_to += release_end_index

            if self.samples_processed >= release_samples:
                self.state = ADSRState.IDLE
                self.current_level = 0.0
                self.samples_processed = 0

        return output

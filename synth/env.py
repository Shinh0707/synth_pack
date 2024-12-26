import numpy as np
from enum import Enum

class ADSRState(Enum):
    IDLE = 0
    ATTACK = 1
    DECAY = 2
    SUSTAIN = 3
    RELEASE = 4

class ADSR:
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
        attack_samples = int(self.attack_time * self.sample_rate)
        decay_samples = int(self.decay_time * self.sample_rate)
        release_samples = int(self.release_time * self.sample_rate)

        for i in range(num_samples):
            if self.samples_processed >= 0:
                if self.state == ADSRState.ATTACK:
                    if self.samples_processed < attack_samples:
                        self.current_level = self.samples_processed / attack_samples
                    else:
                        self.state = ADSRState.DECAY
                        self.samples_processed = 0

                elif self.state == ADSRState.DECAY:
                    if self.samples_processed < decay_samples:
                        self.current_level = 1.0 - (
                            (1.0 - self.sustain_level)
                            * (self.samples_processed / decay_samples)
                        )
                    else:
                        self.state = ADSRState.SUSTAIN

                elif self.state == ADSRState.RELEASE:
                    if self.samples_processed < release_samples:
                        self.current_level = self.sustain_level * (
                            1 - self.samples_processed / release_samples
                        )
                    else:
                        self.state = ADSRState.IDLE
                        self.current_level = 0.0

                output[i] = self.current_level
            self.samples_processed += 1

        return output

from .synth import ModularSynth, CompositeModularSynth, GranularSynth
from .osc import (
    SineOscillator,
    SquareOscillator,
    PulseOscillator,
    SawtoothOscillator,
    TriangleOscillator,
    AudioOscillator,
    FMOscillator,
    SuperSawOscillator,
    NoiseOscillator,
    PluckOscillator
)
from .effect import (
    Delay,
    Reverb,
    Chorus,
    RingModulate,
    Flanger,
    Unison,
    Tremolo,
    BitCrusher,
    SpringReverb,
    ShimmerReverb,
    StereoEnhancer,
    Portamento,
    Compressor
)
from .filter import LowPassFilter, HighPassFilter, BandPassFilter, MultiModeFilter
from .env import ADSR, ADSRState
from .player import SimpleSynthPlayer, Note

__version__ = "0.1.1"

__all__ = [
    # シンセサイザー
    "ModularSynth",
    "CompositeModularSynth",
    "GranularSynth",
    # オシレーター
    "SineOscillator",
    "SquareOscillator",
    "PulseOscillator",
    "TriangleOscillator",
    "SawtoothOscillator",
    "AudioOscillator",
    "FMOscillator",
    "SuperSawOscillator",
    "NoiseOscillator",
    "PluckOscillator",
    # エフェクト
    "Delay",
    "Reverb",
    "Chorus",
    "RingModulate",
    "Flanger",
    "Unison",
    "Tremolo",
    "BitCrusher",
    "SpringReverb",
    "ShimmerReverb",
    "StereoEnhancer",
    "Portamento",
    "Compressor",
    # フィルター
    "LowPassFilter",
    "HighPassFilter",
    "BandPassFilter",
    "MultiModeFilter",
    # エンベロープ
    "ADSR",
    "ADSRState",
    # プレイヤー
    "SimpleSynthPlayer",
    "Note",
]

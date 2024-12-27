オシレーターモジュール
======================

.. automodule:: synth.osc
   :members:
   :undoc-members:
   :show-inheritance:

オシレーターの基底クラス
------------------------

.. autoclass:: synth.osc.Oscillator
   :members:
   :undoc-members:
   :show-inheritance:

基本波形オシレーター
--------------------

.. autoclass:: synth.osc.SineOscillator
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: synth.osc.SquareOscillator
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: synth.osc.PulseOscillator
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: synth.osc.TriangleOscillator
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: synth.osc.SawtoothOscillator
   :members:
   :undoc-members:
   :show-inheritance:

高度な合成オシレーター
----------------------

.. autoclass:: synth.osc.AudioOscillator
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: synth.osc.FMOscillator
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: synth.osc.SuperSawOscillator
   :members:
   :undoc-members:
   :show-inheritance:

ノイズジェネレーター
--------------------

.. autoclass:: synth.osc.NoiseOscillator
   :members:
   :undoc-members:
   :show-inheritance:

使用例
---------

各種オシレーターの基本的な使用例を示します::

    # 基本波形の生成
    sine_osc = SineOscillator()
    sine_osc.set_frequency(440)  # A4音を設定
    
    # パルス波の生成（矩形波の応用）
    pulse_osc = PulseOscillator(pulse_width=0.3)
    pulse_osc.set_frequency(440)
    
    # FM合成の例
    fm_osc = FMOscillator(mod_ratio=2.0, mod_index=1.0)
    fm_osc.set_frequency(440)
    
    # SuperSawの例（豊かなユニゾン音色）
    supersaw = SuperSawOscillator(num_voices=7, detune=0.1)
    supersaw.set_frequency(440)
    
    # ノイズ生成の例
    noise = NoiseOscillator(noise_type=NoiseOscillator.NoiseType.PINK)
    
    # 音声ファイルを使用したオシレーターの例
    audio_osc = AudioOscillator("sample.wav")
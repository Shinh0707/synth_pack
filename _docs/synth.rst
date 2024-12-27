シンセサイザーモジュール
========================

.. automodule:: synth.synth
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

シンセサイザーの基底クラス
---------------------------

.. autoclass:: synth.synth.Synth
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

モジュラーシンセサイザー
--------------------------

.. autoclass:: synth.synth.ModularSynth
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

複合モジュラーシンセサイザー
-----------------------------

.. autoclass:: synth.synth.CompositeModularSynth
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

グラニュラーシンセサイザー
---------------------------

.. autoclass:: synth.synth.GranularSynth
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

使用例
---------

基本的なシンセサイザーの使用例を示します::

    # モジュラーシンセの基本的な使用例
    synth = ModularSynth(44100)
    
    # オシレーターの追加
    synth.add_oscillator(SawtoothOscillator())
    synth.add_oscillator(PulseOscillator(pulse_width=0.3))
    
    # フィルターとエフェクトの設定
    lpf = LowPassFilter()
    lpf.set_cutoff(2000)
    synth.add_filter(lpf)
    synth.add_effect(Reverb())
    
    # エンベロープの設定
    synth.set_envelope(attack=0.1, decay=0.2, sustain=0.7, release=0.3)
    
    # グラニュラーシンセの使用例
    granular = GranularSynth("sample.wav",
                            grain_size=0.1,
                            grain_density=10,
                            position_jitter=0.1)
    
    granular.set_grain_size(0.05)      # グレインサイズの調整
    granular.set_grain_density(20)     # 密度の調整
    granular.set_position_jitter(0.2)  # ランダム性の調整
    
    # 複合シンセの使用例
    composite = CompositeModularSynth()
    synth1 = ModularSynth()
    synth2 = ModularSynth()
    
    # 異なる設定の2つのシンセを組み合わせる
    composite.add_synth(synth1)
    composite.add_synth(synth2)

高度な使用例
-------------

より複雑な音色設計の例を示します::

    # FM音源とフィルターを組み合わせた例
    synth = ModularSynth(44100)
    
    # FMオシレーターを追加
    fm_osc = FMOscillator(mod_ratio=2.0, mod_index=3.0)
    synth.add_oscillator(fm_osc)
    
    # エフェクトチェーンの構築
    lpf = LowPassFilter()
    lpf.set_cutoff(2000)
    synth.add_filter(lpf)
    
    chorus = Chorus(rate=0.8, depth=0.002)
    synth.add_effect(chorus)
    
    reverb = Reverb(room_size=0.8, damping=0.5)
    synth.add_effect(reverb)
    
    # 複雑なエンベロープ設定
    synth.set_envelope(
        attack=0.05,   # 素早いアタック
        decay=0.3,     # やや長めのディケイ
        sustain=0.4,   # 控えめなサステイン
        release=0.8    # 長めのリリース
    )
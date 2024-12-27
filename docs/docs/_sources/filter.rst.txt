フィルターモジュール
======================

.. automodule:: synth.filter
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

フィルターの基底クラス
----------------------

.. autoclass:: synth.filter.Filter
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

ローパスフィルター
------------------

.. autoclass:: synth.filter.LowPassFilter
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

ハイパスフィルター
------------------

.. autoclass:: synth.filter.HighPassFilter
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

バンドパスフィルター
--------------------

.. autoclass:: synth.filter.BandPassFilter
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

使用例
---------

各フィルターの基本的な使用例を示します::

    # ローパスフィルターの例
    lpf = LowPassFilter()
    lpf.set_cutoff(1000)    # カットオフ周波数を1000Hzに設定
    lpf.set_resonance(1.2)  # レゾナンスを設定
    
    # 信号処理
    filtered_audio = lpf.process(input_audio)
    
    # ハイパスフィルターの例
    hpf = HighPassFilter()
    hpf.set_cutoff(500)     # カットオフ周波数を500Hzに設定
    
    # バンドパスフィルターの例
    bpf = BandPassFilter()
    bpf.set_cutoff(1000)    # 中心周波数を1000Hzに設定
    bpf.set_bandwidth(200)  # バンド幅を200Hzに設定
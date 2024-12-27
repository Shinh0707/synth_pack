エンベロープモジュール
======================

.. automodule:: synth.env
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

エンベロープの状態
--------------------

.. autoclass:: synth.env.ADSRState
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

ADSRエンベロープ
------------------

.. autoclass:: synth.env.ADSR
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

使用例
---------

以下は、ADSRエンベロープの基本的な使用例です::

    # エンベロープの作成
    env = ADSR(
        attack=0.1,    # アタック時間 0.1秒
        decay=0.2,     # ディケイ時間 0.2秒
        sustain=0.7,   # サステインレベル 0.7
        release=0.3    # リリース時間 0.3秒
    )
    
    # ノートオン（発音開始）
    env.note_on()
    
    # エンベロープ値の取得（サンプル数分）
    samples = env.process(1000)
    
    # ノートオフ（発音終了）
    env.note_off()
    
    # リリース部分の処理
    release_samples = env.process(1000)
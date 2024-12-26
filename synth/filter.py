import numpy as np
from synth.effect import Effect


class Filter(Effect):
    def __init__(self, sample_rate: int = 44100):
        super().__init__(sample_rate=sample_rate, mix=1.0)
        self.sample_rate = sample_rate
        self.cutoff = 1000.0
        self.resonance = 0.707
        # ステレオバッファ: (2, 2) - [チャンネル, バッファ]
        self._buffer = np.zeros((2, 2))

    def set_cutoff(self, freq: np.ndarray | float):
        if isinstance(freq, float):
            self.cutoff = min(freq, self.sample_rate / 2)
        else:
            self.cutoff = np.minimum(freq, self.sample_rate / 2)

    def set_resonance(self, res: np.ndarray | float):
        if isinstance(res, float):
            self.resonance = max(0.707, min(2.0, res))
        else:
            self.resonance = np.clip(res, 0.707, 2.0)


class LowPassFilter(Filter):
    def _process(self, samples: np.ndarray) -> np.ndarray:
        """
        ステレオ信号に対するローパスフィルター処理
        samples: shape (num_samples, 2) のステレオ信号
        returns: shape (num_samples, 2) のステレオ信号
        """
        # カットオフ周波数とレゾナンスのブロードキャスト用の形状調整
        if isinstance(self.cutoff, (float, int)):
            cutoff = np.full(samples.shape, self.cutoff)
        else:
            # 入力がndarray の場合、ステレオチャンネル分に拡張
            cutoff = self.cutoff

        if isinstance(self.resonance, (float, int)):
            resonance = np.full(samples.shape, self.resonance)
        else:
            resonance = self.resonance

        # フィルタ係数の計算（ステレオチャンネル別）
        omega = 2 * np.pi * cutoff / self.sample_rate
        alpha = np.sin(omega) / (2 * resonance)

        a0 = 1 + alpha
        a1 = -2 * np.cos(omega)
        a2 = 1 - alpha
        b0 = (1 - np.cos(omega)) / 2
        b1 = 1 - np.cos(omega)
        b2 = (1 - np.cos(omega)) / 2

        # チャンネルごとのバッファ初期化
        x_b = np.zeros((samples.shape[0], 2, 2))  # (samples, channels, buffer)
        y_b = np.zeros((samples.shape[0], 2, 2))  # (samples, channels, buffer)

        # 初期バッファ状態の設定（チャンネルごと）
        for ch in range(2):
            x_b[0, ch] = self._buffer[ch]

        # 出力バッファの準備
        output = np.zeros_like(samples)

        # チャンネルごとにフィルタ処理
        for ch in range(2):
            # ベクトル化された初期計算
            output[:, ch] = (
                b0[:, ch] / a0[:, ch] * samples[:, ch]
                + b1[:, ch]
                / a0[:, ch]
                * np.pad(x_b[:, ch, 0], ((1, 0)), "constant")[:-1]
                + b2[:, ch]
                / a0[:, ch]
                * np.pad(x_b[:, ch, 1], ((2, 0)), "constant")[:-2]
                - a1[:, ch]
                / a0[:, ch]
                * np.pad(y_b[:, ch, 0], ((1, 0)), "constant")[:-1]
                - a2[:, ch]
                / a0[:, ch]
                * np.pad(y_b[:, ch, 1], ((2, 0)), "constant")[:-2]
            )

            # 再帰的なフィルタ処理
            for i in range(samples.shape[0]):
                output[i, ch] = (
                    b0[i, ch] / a0[i, ch] * samples[i, ch]
                    + b1[i, ch] / a0[i, ch] * x_b[i, ch, 0]
                    + b2[i, ch] / a0[i, ch] * x_b[i, ch, 1]
                    - a1[i, ch] / a0[i, ch] * y_b[i, ch, 0]
                    - a2[i, ch] / a0[i, ch] * y_b[i, ch, 1]
                )

                # バッファの更新
                if i < samples.shape[0] - 2:
                    x_b[i + 1, ch, 1] = x_b[i, ch, 0]
                    x_b[i + 1, ch, 0] = samples[i, ch]
                    y_b[i + 1, ch, 1] = y_b[i, ch, 0]
                    y_b[i + 1, ch, 0] = output[i, ch]
                elif i < samples.shape[0] - 1:
                    x_b[i + 1, ch, 0] = samples[i, ch]
                    y_b[i + 1, ch, 0] = output[i, ch]

            # チャンネルごとの最終バッファ状態を保存
            self._buffer[ch] = x_b[-1, ch]

        return output

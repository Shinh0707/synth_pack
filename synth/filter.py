from typing import Literal
import numpy as np
from synth.effect import Effect

"""デジタルフィルタの実装

2次IIRフィルタを使用した各種フィルタークラスを提供する。
全てのフィルターはステレオ信号に対応し、チャンネルごとに独立して処理を行う。

Classes:
    Filter: フィルター処理の基底クラス
    LowPassFilter: ローパスフィルター
    HighPassFilter: ハイパスフィルター
    BandPassFilter: バンドパスフィルター
"""

class Filter(Effect):
    """フィルター処理の基底クラス

    全てのフィルタークラスはこのクラスを継承して実装する。
    カットオフ周波数とレゾナンスのパラメータを持つ。

    Attributes:
        cutoff: カットオフ周波数 (Hz), Default: 1000.0
        resonance: レゾナンス (0.707-2.0), Default: 0.707
    """

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
    """ローパスフィルター

    カットオフ周波数以下の成分を通過させ、
    それ以上の周波数成分を減衰させる。

    2次IIRフィルターによる実装で、
    -12dB/octaveのスロープ特性を持つ。
    """

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

class HighPassFilter(Filter):
    """ハイパスフィルター

    カットオフ周波数以上の成分を通過させ、
    それ以下の周波数成分を減衰させる。

    2次IIRフィルターによる実装で、
    -12dB/octaveのスロープ特性を持つ。
    """
    def _process(self, samples: np.ndarray) -> np.ndarray:
        """
        ステレオ信号に対するハイパスフィルター処理
        samples: shape (num_samples, 2) のステレオ信号
        returns: shape (num_samples, 2) のステレオ信号
        """
        # カットオフ周波数とレゾナンスのブロードキャスト用の形状調整
        if isinstance(self.cutoff, (float, int)):
            cutoff = np.full(samples.shape, self.cutoff)
        else:
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
        b0 = (1 + np.cos(omega)) / 2
        b1 = -(1 + np.cos(omega))
        b2 = (1 + np.cos(omega)) / 2

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


class BandPassFilter(Filter):
    """バンドパスフィルター

    指定された周波数帯域の成分のみを通過させる。
    中心周波数とバンド幅のパラメータで特性を制御。

    Attributes:
        bandwidth: フィルターのバンド幅 (Hz), Default: 100.0

    2次IIRフィルターによる実装で、
    中心周波数を境に±6dB/octaveのスロープ特性を持つ。
    """

    def __init__(self, sample_rate: int = 44100, bandwidth: float = 100.0):
        super().__init__(sample_rate=sample_rate)
        self.bandwidth = bandwidth

    def set_bandwidth(self, bw: np.ndarray | float):
        if isinstance(bw, float):
            self.bandwidth = max(1.0, min(bw, self.sample_rate / 2))
        else:
            self.bandwidth = np.clip(bw, 1.0, self.sample_rate / 2)

    def _process(self, samples: np.ndarray) -> np.ndarray:
        """
        ステレオ信号に対するバンドパスフィルター処理
        samples: shape (num_samples, 2) のステレオ信号
        returns: shape (num_samples, 2) のステレオ信号
        """

        # カットオフ周波数、バンド幅とレゾナンスのブロードキャスト用の形状調整
        if isinstance(self.cutoff, (float, int)):
            cutoff = np.full(samples.shape, self.cutoff)
        else:
            cutoff = self.cutoff

        if isinstance(self.bandwidth, (float, int)):
            bandwidth = np.full(samples.shape, self.bandwidth)
        else:
            bandwidth = self.bandwidth

        if isinstance(self.resonance, (float, int)):
            resonance = np.full(samples.shape, self.resonance)
        else:
            resonance = self.resonance

        # フィルタ係数の計算（ステレオチャンネル別）
        omega = 2 * np.pi * cutoff / self.sample_rate
        bw = 2 * np.pi * bandwidth / self.sample_rate
        alpha = np.tan(bw / 2)

        # バンドパスフィルタの係数
        b0 = alpha
        b1 = np.zeros_like(alpha)
        b2 = -alpha
        a0 = 1 + alpha
        a1 = -2 * np.cos(omega)
        a2 = 1 - alpha

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

class MultiModeFilter(Filter):
    """マルチモードフィルター

    1つのフィルターで複数のフィルターモードを切り替え可能。
    ローパス、ハイパス、バンドパス、ノッチの4モードを提供。

    Attributes:
        mode: フィルターモード ("lowpass", "highpass", "bandpass", "notch")
        cutoff: カットオフ周波数 (Hz)
        resonance: レゾナンス (Q値、0.707-2.0)
    """

    def __init__(
        self,
        mode: Literal["lowpass", "highpass", "bandpass", "notch"] = "lowpass",
        cutoff: float = 1000.0,
        resonance: float = 1.0,
        sample_rate: int = 44100,
    ):
        super().__init__(sample_rate=sample_rate)
        self.mode = mode
        self.cutoff = cutoff
        self.resonance = max(0.707, min(2.0, resonance))
        self._buffer = np.zeros((2, 2))  # ステレオバッファ (channels, history)

    def set_mode(self, mode:  Literal["lowpass", "highpass", "bandpass", "notch"]):
        self.mode = mode

    def _process(self, samples: np.ndarray) -> np.ndarray:
        """
        マルチモードフィルター処理
        samples: shape (num_samples, 2) のステレオ信号
        returns: shape (num_samples, 2) のステレオ信号
        """
        # 正規化されたカットオフ周波数
        omega = 2.0 * np.pi * self.cutoff / self.sample_rate
        cos_omega = np.cos(omega)
        sin_omega = np.sin(omega)
        alpha = sin_omega / (2.0 * self.resonance)

        # フィルター係数の計算
        if self.mode == "lowpass":
            b0 = (1.0 - cos_omega) / 2.0
            b1 = 1.0 - cos_omega
            b2 = (1.0 - cos_omega) / 2.0
        elif self.mode == "highpass":
            b0 = (1.0 + cos_omega) / 2.0
            b1 = -(1.0 + cos_omega)
            b2 = (1.0 + cos_omega) / 2.0
        elif self.mode == "bandpass":
            b0 = alpha
            b1 = 0.0
            b2 = -alpha
        else:  # notch
            b0 = 1.0
            b1 = -2.0 * cos_omega
            b2 = 1.0

        a0 = 1.0 + alpha
        a1 = -2.0 * cos_omega
        a2 = 1.0 - alpha

        # 係数の正規化
        b0 /= a0
        b1 /= a0
        b2 /= a0
        a1 /= a0
        a2 /= a0

        output = np.zeros_like(samples)

        # ステレオチャンネル別処理
        for ch in range(2):
            x1, x2 = self._buffer[ch]
            y1, y2 = 0.0, 0.0

            for i in range(samples.shape[0]):
                x0 = samples[i, ch]

                # フィルター方程式の適用
                y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2

                # 出力と状態更新
                output[i, ch] = y0
                x2, x1 = x1, x0
                y2, y1 = y1, y0

            # バッファの更新
            self._buffer[ch] = [x1, x2]

        return output
from dataclasses import dataclass, field, asdict
import hashlib
import json
import numpy as np


PIPELINE_VERSION = "xsm-pipeline-0.2.0"


@dataclass
class PipelineConfig:
    baseline_window_s: float = 301.0
    sigma_seed: float = 3.5
    grow_sigma: float = 1.3
    wavelet_sigma: float = 4.0
    wavelet_scales_s: tuple = (4.0, 16.0, 64.0)
    bb_p0: float = 0.01
    bb_sigma: float = 2.5
    bb_max_points: int = 1000
    min_segment_bins: int = 20
    fit_context_s: float = 100.0
    duplicate_policy: str = "error"  # 'identical' drops exact duplicate records only.
    bootstrap_samples: int = 0
    random_seed: int = 42

    def __post_init__(self):
        for k in ("baseline_window_s", "sigma_seed", "grow_sigma", "wavelet_sigma", "bb_sigma", "fit_context_s"):
            v = getattr(self, k)
            if not np.isfinite(v) or v <= 0:
                raise ValueError(f"{k} must be finite and positive")
        if self.grow_sigma >= self.sigma_seed:
            raise ValueError("grow_sigma must be below sigma_seed")
        if not 0 < self.bb_p0 < 1 or self.bb_max_points < 20:
            raise ValueError("Invalid Bayesian Blocks settings")
        if self.min_segment_bins < 12 or self.bootstrap_samples < 0:
            raise ValueError("Invalid minimum segment or bootstrap setting")
        if self.duplicate_policy not in {"error", "identical"}:
            raise ValueError("Invalid duplicate policy")
        if not self.wavelet_scales_s or any(not np.isfinite(x) or x <= 0 for x in self.wavelet_scales_s):
            raise ValueError("Wavelet scales must be positive seconds")

    def fingerprint(self):
        return hashlib.sha256(json.dumps({"implementation": PIPELINE_VERSION, "config": asdict(self)}, sort_keys=True).encode()).hexdigest()


@dataclass
class LightCurve:
    """Nonoverlapping bins; time is bin center in seconds relative to metadata epoch.

    quantity: 'counts' (raw integer counts), 'count_rate', or 'flux'.
    For corrected rates/flux, positive measurement errors are mandatory.
    quality==0 means usable. Instrument states should mark transition bins invalid.
    Exposure is assumed uniformly distributed within each bin; partial GTI bins
    must be excluded or handled upstream with the actual live-time geometry.
    """
    time: np.ndarray
    value: np.ndarray
    error: np.ndarray | None
    bin_width: np.ndarray
    exposure: np.ndarray
    quality: np.ndarray
    metadata: dict
    warnings: list = field(default_factory=list)

    def prepared(self, duplicate_policy="error"):
        fields = [np.asarray(getattr(self, x), dtype=float) for x in
                  ("time", "value", "bin_width", "exposure", "quality")]
        if any(a.ndim != 1 for a in fields) or len({len(a) for a in fields}) != 1:
            raise ValueError("All light-curve arrays must be equal-length vectors")
        t, y, width, exposure, quality = fields
        if len(t) < 3 or not np.all(np.isfinite(t)):
            raise ValueError("At least three finite timestamps are required")
        if not np.all(np.isfinite(quality)) or np.any(quality < 0) or np.any(quality != np.floor(quality)):
            raise ValueError("Quality must contain nonnegative integer flags")
        if np.any(~np.isfinite(width)) or np.any(width <= 0):
            raise ValueError("Explicit finite positive bin widths are required")
        if np.any(~np.isfinite(exposure)) or np.any(exposure < 0) or np.any(exposure > width * (1 + 1e-8)):
            raise ValueError("Exposure must lie between zero and bin width")
        q = self.metadata.get("quantity")
        if q not in {"counts", "count_rate", "flux"} or not self.metadata.get("unit"):
            raise ValueError("Explicit quantity and unit are required")
        if not self.metadata.get("epoch") or not self.metadata.get("time_scale"):
            raise ValueError("Explicit epoch (or 'relative') and time_scale are required")
        if self.metadata["epoch"] == "relative":
            if self.metadata["time_scale"] != "relative":
                raise ValueError("Relative epoch requires relative time_scale")
        else:
            from astropy.time import Time
            if self.metadata["time_scale"] not in Time.SCALES:
                raise ValueError("Unsupported/unresolved time scale")
            try:
                Time(self.metadata["epoch"], scale=self.metadata["time_scale"])
            except (ValueError, TypeError) as exc:
                raise ValueError("Epoch must be a verified date string or 'relative'") from exc
        err = np.full(len(t), np.nan) if self.error is None else np.asarray(self.error, dtype=float)
        if err.shape != t.shape:
            raise ValueError("Error vector shape mismatch")
        if q != "counts" and self.error is None:
            raise ValueError("Corrected rates/flux require supplied measurement errors")
        if q == "counts":
            raw = y[np.isfinite(y) & (quality == 0)]
            if np.any(raw < 0) or not np.allclose(raw, np.round(raw), atol=1e-7, rtol=0):
                raise ValueError("Raw counts must be nonnegative integers; corrected rates are not counts")
        order = np.argsort(t, kind="stable")
        t, y, err, width, exposure, quality = [a[order] for a in (t, y, err, width, exposure, quality)]
        warnings = list(self.warnings)
        dup = np.r_[False, np.diff(t) == 0]
        if dup.any():
            if duplicate_policy != "identical":
                raise ValueError("Duplicate timestamps: resolve independent exposures or set identical policy")
            for i in np.flatnonzero(dup):
                if any(not np.array_equal(a[i:i+1], a[i-1:i], equal_nan=True)
                       for a in (y, err, width, exposure, quality)):
                    raise ValueError("Conflicting duplicate timestamps cannot be combined automatically")
            warnings.append(f"Dropped {int(dup.sum())} exact duplicate records")
            keep = ~dup
            t, y, err, width, exposure, quality = [a[keep] for a in (t, y, err, width, exposure, quality)]
        overlap = (t[:-1] + width[:-1]/2) - (t[1:] - width[1:]/2)
        if np.any(overlap > np.maximum(1e-7, np.minimum(width[:-1], width[1:])*1e-6)):
            raise ValueError("Overlapping bin intervals: resolve exposure geometry before analysis")
        return LightCurve(t, y, err, width, exposure, quality.astype(np.int64), dict(self.metadata), warnings)

    @property
    def valid(self):
        v = (self.quality == 0) & np.isfinite(self.value) & (self.exposure > 0)
        if self.metadata["quantity"] != "counts":
            v &= np.isfinite(self.error) & (self.error > 0)
        return v

    def rates(self):
        if self.metadata["quantity"] == "counts":
            ex = np.where(self.exposure > 0, self.exposure, np.nan)
            # Detection approximation only. Final count fits use Poisson likelihood.
            return self.value/ex, np.sqrt(np.maximum(self.value, 1))/ex
        return self.value.copy(), self.error.copy()

    def segments(self):
        idx = np.flatnonzero(self.valid)
        if not len(idx):
            return []
        right = self.time[idx[:-1]] + self.bin_width[idx[:-1]]/2
        left = self.time[idx[1:]] - self.bin_width[idx[1:]]/2
        tol = 1e-6 * np.minimum(self.bin_width[idx[:-1]], self.bin_width[idx[1:]]) + 1e-7
        cuts = np.flatnonzero((np.diff(idx) > 1) | (left-right > tol)) + 1
        return np.split(idx, cuts)

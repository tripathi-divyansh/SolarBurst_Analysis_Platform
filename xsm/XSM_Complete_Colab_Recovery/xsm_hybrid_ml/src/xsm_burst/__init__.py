"""Scientific engine. Scores require domain validation before scientific use."""
__version__ = "0.2.0"
from .schema import LightCurve, PipelineConfig
from .pipeline import analyze

__all__ = ["LightCurve", "PipelineConfig", "analyze"]

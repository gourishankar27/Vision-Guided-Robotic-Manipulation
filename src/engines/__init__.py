"""Optional accelerated engines."""

try:
    from .warp_planar_arm import WarpPlanarArmEngine, is_warp_available
except Exception:  # pragma: no cover - optional dependency guard
    WarpPlanarArmEngine = None  # type: ignore

    def is_warp_available() -> bool:
        return False

__all__ = ["WarpPlanarArmEngine", "is_warp_available"]

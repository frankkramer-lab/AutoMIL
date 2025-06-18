from pathlib import Path

import openslide
from slideflow import log


def get_slide_magnification(slide_path: str) -> str | None:
    """Retrieves magnification from slide properties or estimates it using microns per pixel (mpp)

    Args:
        slide_path (str): path to slide as string

    Returns:
        str | None: magnification as string (e.g. '20') or None if unable
    """
    try:
        slide = openslide.OpenSlide(slide_path)
        # Typical property for objective magnification
        if (mag := slide.properties.get("openslide.objective-power")):
            return mag
        # Estimate magnification from mpp_x
        if (mpp_x := slide.properties.get("openslide.mpp-x")):
            estimated_mag = 10 / float(mpp_x)
            return str(estimated_mag)
    # Estimation (or opening slide) failed
    except Exception:
        return None

if __name__ == "__main__":

    for slide in Path("test_dataset").iterdir():
        if not slide.suffix == ".svs":
            continue
        mag = get_slide_magnification(str(slide))
        log.info(f"{slide} | [cyan]{mag}{'x' if mag is not None else ''}[/]")







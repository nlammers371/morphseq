import xml.etree.ElementTree as ET
from typing import Dict, Any, Union
from pathlib import Path

def scrape_keyence_metadata(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Parse the <Data> … </Data> chunk inside a Keyence .tif and
    return the exact same keys your downstream code expects.
    """
    path = Path(path)

    # read only until </Data> tag – avoids loading the full image bytes twice
    with path.open("rb") as fh:
        chunk = fh.read().split(b"</Data>", 1)[0] + b"</Data>"

    root = ET.fromstring(chunk.decode(errors="ignore"))

    def text(tag: str) -> str:
        node = root.find(f".//{tag}")
        return node.text if node is not None else ""

    # original column names preserved ↓
    sec = int(text("ShootingDateTime")) / 1e7           # 100 ns → s
    width_px  = int(text("Width"))
    height_px = int(text("Height"))
    width_um  = int(text("Width_um"))  / 1000
    height_um = int(text("Height_um")) / 1000

    return {
        "Time (s)"   : sec,
        "Objective"  : text("LensName"),
        "Channel"    : text("Observation_Type"),
        "Width (px)" : width_px,
        "Height (px)": height_px,
        "Width (um)" : width_um,
        "Height (um)": height_um,
    }

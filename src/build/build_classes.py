# metadata_utils.py
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional, Dict, Any
import pathlib

@dataclass
class KeyenceMeta:
    time_s:        float
    objective:     str
    channel:       str
    width_px:      int
    height_px:     int
    width_um:      float
    height_um:     float

    @classmethod
    def from_file(cls, path: str | pathlib.Path) -> "KeyenceMeta":
        with open(path, "rb") as fh:
            # read only up to </Data> for speed
            xml_chunk = fh.read().split(b"</Data>", 1)[0] + b"</Data>"
        root = ET.fromstring(xml_chunk.decode("utf-8", errors="ignore"))

        def tag(txt: str) -> Optional[str]:
            node = root.find(f".//{txt}")
            return node.text if node is not None else None

        width_px  = int(tag("Width"))
        height_px = int(tag("Height"))
        meta = cls(
            time_s      = int(tag("ShootingDateTime")) / 1e7,           # 100 ns → s
            objective   = tag("LensName"),
            channel     = tag("Observation_Type"),
            width_px    = width_px,
            height_px   = height_px,
            width_um    = int(tag("Width_um"))  / 1000.0,
            height_um   = int(tag("Height_um")) / 1000.0,
        )
        return meta

    def as_dict(self) -> Dict[str, Any]:
        return {f.name.replace("_", " ").title(): getattr(self, f.name)
                for f in self.__dataclass_fields__.values()}
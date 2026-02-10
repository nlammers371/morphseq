# ---------------------------------------------------------------------------
# Frame‑aware parser (understands snip_id suffix frame numbers)
# ---------------------------------------------------------------------------
class FrameAwareRangeParser(RangeParser):
    """Parse slice strings by **frame number** contained in snip_id suffix."""

    FRAME_RE = re.compile(r"[_s](\d{4})$")

    # -------- frame helpers -------------------------------------------------
    @staticmethod
    def extract_frame_number(snip_id: str) -> int:
        match = FrameAwareRangeParser.FRAME_RE.search(snip_id)
        if match:
            return int(match.group(1))
        raise ValueError(f"Cannot extract frame number from {snip_id}")

    # -------- public API ----------------------------------------------------
    @staticmethod
    def parse_snip_range(
        range_spec: Union[str, List[str]],
        available_snips: List[str],
        *,
        mode: str = "auto",
    ) -> List[str]:
        """Return snip_ids matching *range_spec*.

        * If `range_spec` is already a list → validate membership.
        * Otherwise treat it as slice string – either index‑based or frame‑based.
        """
        if isinstance(range_spec, list):
            return [s for s in range_spec if s in available_snips]

        range_str = str(range_spec).strip().strip("[]")

        # auto‑detect frame vs index
        if mode == "auto":
            nums = [int(p) for p in range_str.split(":") if p.isdigit()]
            mode = "frame" if any(n > 100 for n in nums) else "index"

        if mode == "frame":
            return FrameAwareRangeParser._parse_by_frame_number(range_str, available_snips)
        else:
            return RangeParser._parse_string_range(f"[{range_str}]", available_snips)

    # -------- internals -----------------------------------------------------
    @staticmethod
    def _parse_by_frame_number(range_str: str, available_snips: List[str]) -> List[str]:
        mapping: Dict[int, str] = {}
        for sid in available_snips:
            try:
                mapping[FrameAwareRangeParser.extract_frame_number(sid)] = sid
            except ValueError:
                continue
        if not mapping:
            return []

        parts = range_str.split(":")
        frames_sorted = sorted(mapping)

        # start, end, step
        start = int(parts[0]) if parts[0] else frames_sorted[0]
        end = int(parts[1]) if len(parts) > 1 and parts[1] else frames_sorted[-1] + 1
        step = int(parts[2]) if len(parts) > 2 and parts[2] else 1

        return [mapping[f] for f in frames_sorted if start <= f < end and (f - start) % step == 0]

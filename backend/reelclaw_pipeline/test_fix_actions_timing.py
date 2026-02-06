from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Ensure `backend/` is on sys.path so `import reelclaw_pipeline.*` works when running from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reelclaw_pipeline.fix_actions import apply_fix_actions  # noqa: E402


class TestFixActionsTiming(unittest.TestCase):
    def test_shift_inpoint_clamps_to_shot_window(self) -> None:
        timeline = {
            "timeline_segments": [
                {
                    "id": 1,
                    "shot_id": "a#00001000_00003000",  # [1.0s, 3.0s]
                    "asset_in_s": 1.9,
                    "duration_s": 1.0,
                    "speed": 1.0,
                }
            ]
        }
        patched, rep = apply_fix_actions(timeline, [{"type": "shift_inpoint", "segment_id": 1, "seconds": 0.6}])
        seg = patched["timeline_segments"][0]
        # Max start = 3.0 - 1.0 = 2.0
        self.assertAlmostEqual(float(seg["asset_in_s"]), 2.0, places=6)
        self.assertTrue(rep.get("applied"))

        applied0 = rep["applied"][0]
        # Effective shift is only +0.1 (1.9 -> 2.0)
        self.assertAlmostEqual(float(applied0.get("seconds")), 0.1, places=6)

    def test_set_speed_clamps_and_adjusts_inpoint(self) -> None:
        timeline = {
            "timeline_segments": [
                {
                    "id": 1,
                    "shot_id": "a#00001000_00003000",  # [1.0s, 3.0s]
                    "asset_in_s": 1.8,
                    "duration_s": 1.0,
                    "speed": 1.0,
                }
            ]
        }
        patched, rep = apply_fix_actions(timeline, [{"type": "set_speed", "segment_id": 1, "value": 1.25}])
        seg = patched["timeline_segments"][0]
        self.assertAlmostEqual(float(seg["speed"]), 1.25, places=6)
        # Max start = 3.0 - 1.25 = 1.75
        self.assertAlmostEqual(float(seg["asset_in_s"]), 1.75, places=6)
        self.assertTrue(rep.get("applied"))

    def test_invalid_segment_id_rejected(self) -> None:
        timeline = {"timeline_segments": [{"id": 1, "asset_in_s": 0.0, "duration_s": 1.0, "speed": 1.0}]}
        patched, rep = apply_fix_actions(timeline, [{"type": "shift_inpoint", "segment_id": 9, "seconds": 0.2}])
        self.assertEqual(float(patched["timeline_segments"][0]["asset_in_s"]), 0.0)
        self.assertTrue(rep.get("rejected"))

    def test_shift_inpoint_without_shot_id_clamps_nonnegative_and_delta(self) -> None:
        timeline = {"timeline_segments": [{"id": 1, "asset_in_s": 0.1, "duration_s": 1.0, "speed": 1.0}]}
        patched, rep = apply_fix_actions(timeline, [{"type": "shift_inpoint", "segment_id": 1, "seconds": -9.0}])
        seg = patched["timeline_segments"][0]
        self.assertAlmostEqual(float(seg["asset_in_s"]), 0.0, places=6)
        # Requested delta is clamped to -0.60, but effective delta is only -0.10 due to floor at 0.0.
        applied0 = rep["applied"][0]
        self.assertAlmostEqual(float(applied0.get("seconds")), -0.1, places=6)


if __name__ == "__main__":
    unittest.main()


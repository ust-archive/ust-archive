"""Ensure a failed preflight gates every expensive or publishing step."""

from pathlib import Path
import unittest


class WorkflowGateTests(unittest.TestCase):
    def test_preflight_precedes_download_generation_and_publication(self):
        root = Path(__file__).resolve().parents[1]
        for workflow in ("schedule", "catalog", "credit-transfer", "ust-space"):
            with self.subTest(workflow=workflow):
                text = (root / ".github" / "workflows" / f"{workflow}.yml").read_text()
                steps = text.split("\n      - ")[1:]
                gates = [i for i, step in enumerate(steps) if "scripts/hf_preflight.py" in step]
                self.assertEqual(len(gates), 1, "Missing fail-fast HF credential/access gate")
                gate = gates[0]
                self.assertIn(f"ust-archive/{workflow}", steps[gate])
                self.assertIn("HF_TOKEN: ${{ secrets.HF_TOKEN }}", steps[gate])
                self.assertNotIn("continue-on-error", steps[gate])
                for i, step in enumerate(steps):
                    if "uvx hf " in step or "bun go " in step:
                        self.assertGreater(i, gate)
                        self.assertNotIn("always()", step)
                        self.assertNotIn("continue-on-error", step)
                        self.assertIn("working-directory: scraper/packages/updater", step)


if __name__ == "__main__":
    unittest.main()

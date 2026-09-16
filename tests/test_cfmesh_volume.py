"""Volume failures must identify invalid cells before suggesting a tolerance issue."""

import unittest

from tools.cfmesh_pipeline import validate_mesh_volume


class MeshVolumeTests(unittest.TestCase):
    def test_negative_cells_without_total_volume(self):
        log = (
            "***Zero or negative cell volume detected. Minimum negative volume: -1.8534e-13, "
            "Number of negative volume cells: 2\nFailed 10 mesh checks."
        )
        with self.assertRaisesRegex(ValueError, r"negative cell volumes \(2 cells\)"):
            validate_mesh_volume(log, 0.452857, 0.005)

    def test_negative_cells_are_rejected_even_with_total_volume(self):
        with self.assertRaisesRegex(ValueError, "negative cell volumes"):
            validate_mesh_volume("Zero or negative cell volume detected\nTotal volume = 1", 1, 1)

    def test_missing_volume(self):
        with self.assertRaisesRegex(ValueError, "did not report"):
            validate_mesh_volume("Failed 1 mesh checks.", 1, 0.005)

    def test_volume_mismatch_reports_actual_error(self):
        with self.assertRaisesRegex(ValueError, "2.000%"):
            validate_mesh_volume("Total volume = 1.02", 1, 0.005)

    def test_valid_volume(self):
        validate_mesh_volume("Total volume = 4.52857e-1", 0.452857137, 0.005)


if __name__ == "__main__":
    unittest.main()

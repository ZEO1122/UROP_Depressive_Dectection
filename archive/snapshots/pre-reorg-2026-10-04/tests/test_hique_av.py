import unittest

import numpy as np

from experiments.hique_av import audio_samples, validate_spans, visual_features


class AVTests(unittest.TestCase):
    def test_visual_wallclock_sampling_and_order(self):
        times = np.arange(90) / 30
        xy = np.repeat(times[:, None], 136, axis=1)
        xy[:, 68:] += 10
        values, usable, coverage, count = visual_features(times, xy, np.ones(90, bool), [[0, 3]])
        self.assertTrue(usable)
        self.assertEqual(count, 3)
        self.assertAlmostEqual(coverage, 1)
        self.assertEqual(values.shape, (272,))
        np.testing.assert_allclose(values[:68], 1)
        np.testing.assert_allclose(values[68:136], 11)
        np.testing.assert_allclose(values[136:], 2 / 3)

    def test_truncated_video_does_not_extrapolate(self):
        times = np.arange(90) / 30
        xy = np.ones((90, 136))
        values, usable, coverage, count = visual_features(times, xy, np.ones(90, bool), [[10, 13]])
        self.assertFalse(usable)
        self.assertEqual((coverage, count), (0, 0))
        self.assertFalse(values.any())

    def test_invalid_frames_reduce_coverage(self):
        times = np.arange(90) / 30
        valid = np.arange(90) % 3 == 0
        _, usable, coverage, _ = visual_features(times, np.ones((90, 136)), valid, [[0, 3]])
        self.assertFalse(usable)
        self.assertAlmostEqual(coverage, 1 / 3)

    def test_temporal_gaps_and_scrubbed_spans(self):
        times = np.r_[np.arange(30) / 30, 2 + np.arange(30) / 30]
        xy = np.ones((60, 136))
        _, usable, coverage, count = visual_features(times, xy, np.ones(60, bool), [[0, 1], [2, 3]])
        self.assertTrue(usable)
        self.assertEqual(count, 2)
        self.assertAlmostEqual(coverage, 1)
        _, _, all_coverage, _ = visual_features(times, xy, np.ones(60, bool), [[0, 4]])
        self.assertAlmostEqual(all_coverage, .5)

    def test_audio_clamps_bounds_and_rejects_insufficient_coverage(self):
        samples, valid, coverage = audio_samples(np.ones(100), 10, [[9, 12]])
        self.assertEqual(len(samples), 10)
        self.assertFalse(valid)
        self.assertAlmostEqual(coverage, 1 / 3)

    def test_invalid_spans(self):
        for spans in ([[2, 1]], [[-1, 2]], [[1, 3], [2, 4]], [[2, 3], [0, 1]], [[0, float('nan')]]):
            with self.assertRaises(ValueError):
                validate_spans(spans)

    def test_empty_and_nonfinite(self):
        _, usable, coverage = audio_samples(np.ones(20), 10, [])
        self.assertFalse(usable)
        self.assertEqual(coverage, 0)
        xy = np.ones((90, 136))
        xy[:] = np.nan
        _, usable, coverage, count = visual_features(np.arange(90) / 30, xy, np.ones(90, bool), [[0, 3]])
        self.assertFalse(usable)
        self.assertEqual((coverage, count), (0, 0))


if __name__ == '__main__':
    unittest.main()

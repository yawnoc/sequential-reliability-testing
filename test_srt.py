#!/usr/bin/env python3

"""
# test_srt.py

Unit testing for `srt.py`.
"""

import unittest

from srt import maximum_failure_count, maximum_test_time


class TestSrt(unittest.TestCase):
    def test_maximum_failure_count(self):
        # Epstein (1954) > Section 5 (Examples) > Problem 1
        self.assertEqual(
            maximum_failure_count(
                alpha=0.05,
                beta=0.05,
                theta_0=10000,
                theta_1=2000,
            ),
            5,
        )

        # MIL-HDBK-781A > Section 5.9.5 (Sequential test example)
        self.assertEqual(
            maximum_failure_count(
                alpha=0.1,
                beta=0.1,
                theta_0=200,
                theta_1=100,
            ),
            15,
        )

    def test_maximum_test_time(self):
        # MIL-HDBK-781A > Section 5.9.5 (Sequential test example)
        self.assertAlmostEqual(
            maximum_test_time(
                alpha=0.1,
                beta=0.1,
                theta_0=200,
                theta_1=100,
            ),
            2060,
            delta=0.1,
        )


if __name__ == '__main__':
    unittest.main()

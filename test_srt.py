#!/usr/bin/env python3

"""
# test_srt.py

Unit testing for `srt.py`.
"""

import unittest

from srt import (
    acceptance_intercept,
    rejection_intercept,
    decision_slope,
    maximum_failure_count,
    maximum_test_time,
)


class TestSrt(unittest.TestCase):
    def test_acceptance_intercept(self):
        # MIL-HDBK-781A > Section 5.9.5 (Sequential test example)
        self.assertAlmostEqual(
            acceptance_intercept(
                theta_0=200,
                theta_1=100,
                alpha=0.1,
                beta=0.1,
            ),
            -3.17,
            delta=0.005,
        )

    def test_rejection_intercept(self):
        # MIL-HDBK-781A > Section 5.9.5 (Sequential test example)
        self.assertAlmostEqual(
            rejection_intercept(
                theta_0=200,
                theta_1=100,
                alpha=0.1,
                beta=0.1,
            ),
            2.75,
            delta=0.005,
        )

    def test_decision_slope(self):
        # MIL-HDBK-781A > Section 5.9.5 (Sequential test example)
        self.assertAlmostEqual(
            decision_slope(theta_0=200, theta_1=100),
            0.00721,
            delta=0.000005,
        )

    def test_maximum_failure_count(self):
        # Epstein (1954) > Section 5 (Examples) > Problem 1
        self.assertEqual(
            maximum_failure_count(
                theta_0=10000,
                theta_1=2000,
                alpha=0.05,
                beta=0.05,
            ),
            5,
        )

        # MIL-HDBK-781A > Section 5.9.5 (Sequential test example)
        self.assertEqual(
            maximum_failure_count(
                theta_0=200,
                theta_1=100,
                alpha=0.1,
                beta=0.1,
            ),
            15,
        )

    def test_maximum_test_time(self):
        # MIL-HDBK-781A > Section 5.9.5 (Sequential test example)
        self.assertAlmostEqual(
            maximum_test_time(
                theta_0=200,
                theta_1=100,
                alpha=0.1,
                beta=0.1,
            ),
            2060,
            delta=0.5,
        )


if __name__ == '__main__':
    unittest.main()

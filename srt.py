#!/usr/bin/env python3

"""
# srt.py

Sequential Reliability Testing.


## Epstein & Sobel (1955)

Specifically:

Epstein & Sobel (1955). Sequential Life Tests in the Exponential Case.
Annals of Mathematical Statistics, 26(1) 82--93.
<https://doi.org/10.1214/aoms/1177728595>

Summary:

- We are testing items with length of life having exponential density
          f(x, theta) = exp(-x/theta) / theta,    x > 0.
- H_0: theta = theta_0    (with Type I error at probability alpha)
- H_1: theta = theta_1    (with Type II error at probability beta)
- theta_1 < theta_0
- k = d = theta_0/theta_1 > 1    (discrimination factor)
- Continue the test if
          B < (theta_0/theta_1)^r exp[-(1/theta_1 - 1/theta_0) V(t)] < A,
  where, per Wald,
          B = beta / (1 - alpha)
          A = (1 - beta) / alpha.
  Here
          r = failure count
          t = time
          V(t) = cumulative time    (n*t for instantaneous replacement)
          alpha = producer's risk
          beta = consumer's risk.
- Accept H_0 if the first inequality is violated.
- Accept H_1 if the second inequality is violated.
- The decision inequality can be written
          -h_1 + r s < V(t) < h_0 + r s
  where
          h_0 = -log(B) / (1/theta_1 - 1/theta_0)
          h_1 = log(A) / (1/theta_1 - 1/theta_0)
          s = log(theta_0/theta_1) / (1/theta_1 - 1/theta_0).
- The operating characteristic curve is given by
          L(theta) = (A^h - 1) / (A^h - B^h)
          theta = [(theta_0/theta_1)^h - 1] / [h (1/theta_1 - 1/theta_0)]
  where h runs through the reals.
- L(theta) is the probability of accepting H_0 when theta is true parameter.


## Epstein (1954)

Specifically:

Epstein (1954). Truncated Life Tests in the Exponential Case.
Annals of Mathematical Statistics, 25(3) 555--564.
Available <https://doi.org/10.1214/aoms/1177728723>.

Among the results is how to find n (the item count) and r_0 (the maximum
failure count) for truncating a test (with replacement) to time T_0.
See Section 4 (Some computational remarks):

- Take r_0 to be the smallest integer r such that
          chi^2(1-alpha; 2r) / chi^2(beta; 2r) >= theta_1/theta_0.
- Require T_0 = theta_0 * chi^2(1-alpha; 2r) / (2n), and hence
          n = ceil[theta_0 * chi^2(1-alpha; 2r_0) / (2T_0)].


## MIL-HDBK-781A

Specifically:

Department of Defense (1996). MIL-HDBK-781A.
<https://www.tc.faa.gov/its/worldpac/Standards/Mil/mil-std-781a.pdf>

See Section 5.9 (Sequential test plans).
Of note:

- There is a typo in the expression for probability ratio.
  The factor (theta_1/theta_0)^r should be (theta_0/theta_1)^r
  per equation (2) of Epstein & Sobel (1955).
- The decision inequality is given in terms of bounds on failure count
  rather than time, as
          a + b t < r < c + b t.
  Here t is accumulated operating time, which Epstein & Sobel (1955)
  called V(t). Comparing this against
          -h_1 + r s < V(t) < h_0 + r s,
  we see that
          a = -h_0/s = log(B) / log(theta_0/theta_1)
          b = 1/s = (1/theta_1 - 1/theta_0) / log(theta_0/theta_1)
          c = +h_1/s = log(A) / log(theta_0/theta_1).
- It is incorrect claimed that Epstein & Sobel (1955) develops the method of
  truncation of a sequential test. It is Epstein (1954) that does this.
- For truncation, it presents the maximum (accumulated) time T_0,
  rather than the item count n in Epstein (1954), as that to be determined.
  We cap the failure count at r_0 being the smallest integer r such that
          chi^2(1-alpha; 2r) / chi^2(beta; 2r) >= theta_1/theta_0,
  and hence the accumulated time at
          T_0 = theta_0 * chi^2(1-alpha; 2r_0) / 2.
"""
import argparse
import random

import matplotlib.pyplot as plt
from numpy import log
from scipy.stats import chi2


class Trial:
    TERMINATE_ACCEPTED = 1
    TERMINATE_TIME_REACHED = 2
    TERMINATE_REJECTED = -1
    TERMINATE_FAILURES_REACHED = -2

    COLOUR_ACCEPTED = 'green'
    COLOUR_TIME_REACHED = 'turquoise'
    COLOUR_REJECTED = 'crimson'
    COLOUR_FAILURES_REACHED = 'orangered'

    AXIS_LIMIT_MARGIN_FACTOR = 1.05

    def __init__(self, theta_0, theta_1, theta, alpha, beta, item_count):
        """
        Run a single trial.
        """
        # Intercepts and slopes of the decision lines
        a = acceptance_intercept(theta_0, theta_1, alpha, beta)
        c = rejection_intercept(theta_0, theta_1, alpha, beta)
        b = decision_slope(theta_0, theta_1)

        # Hard cutoffs
        r_0 = maximum_failure_count(theta_0, theta_1, alpha, beta)
        t_0 = maximum_test_time(theta_0, theta_1, alpha, beta)

        # Corner cases (intersections between hard cutoffs and decision lines)
        r_corner = a + b * t_0  # between acceptance line and max time
        t_corner = (r_0 - c) / b  # between rejection line and max failures

        # Initialisation
        t = 0
        r = 0
        t_values = []
        r_values = []
        items = [Item(theta) for _ in range(0, item_count)]

        # Loop
        while True:
            # Time increment (horizontal step)
            next_failing_item = min(items, key=lambda item: item.time_left)
            individual_time_step = next_failing_item.time_left
            next_t = t + item_count * individual_time_step
            if r < r_corner:
                if r <= a + b * next_t:
                    t = (r - a) / b
                    t_values.append(t)
                    termination = Trial.TERMINATE_ACCEPTED
                    break
            else:
                if next_t >= t_0:
                    t = t_0
                    t_values.append(t)
                    termination = Trial.TERMINATE_TIME_REACHED
                    break
            for item in items:
                item.age(individual_time_step)
            t = next_t
            t_values.append(t)

            # Failure count increment (vertical step)
            next_r = r + 1
            if t < t_corner:
                if next_r >= c + b * t:
                    r = next_r
                    r_values.append(r)
                    termination = Trial.TERMINATE_REJECTED
                    break
            else:
                if next_r >= r_0:
                    r = r_0
                    r_values.append(r)
                    termination = Trial.TERMINATE_FAILURES_REACHED
                    break
            r = next_r
            r_values.append(r)

        self.a = a
        self.b = b
        self.c = c
        self.r_0 = r_0
        self.t_0 = t_0
        self.r_corner = r_corner
        self.t_corner = t_corner
        self.termination = termination
        self.t_values = t_values
        self.r_values = r_values

    def save_plot(self, file_name):
        a = self.a
        c = self.c
        r_0 = self.r_0
        t_0 = self.t_0
        r_corner = self.r_corner
        t_corner = self.t_corner
        t_values = self.t_values
        r_values = self.r_values

        figure, axes = plt.subplots()

        # Testing region
        axes.plot([0, t_0], [a, r_corner], Trial.COLOUR_ACCEPTED)
        axes.plot([t_0, t_0], [r_corner, r_0], Trial.COLOUR_TIME_REACHED)
        axes.plot([0, t_corner], [c, r_0], Trial.COLOUR_REJECTED)
        axes.plot([t_corner, t_0], [r_0, r_0], Trial.COLOUR_FAILURES_REACHED)

        axes.set_xlim([0, t_0 * Trial.AXIS_LIMIT_MARGIN_FACTOR])
        axes.set_ylim([0, r_0 * Trial.AXIS_LIMIT_MARGIN_FACTOR])

        plt.savefig(file_name)


class Item:
    def __init__(self, theta):
        self.time_left = random.expovariate(1/theta)

    def age(self, time_step):
        self.time_left -= time_step


def acceptance_probability_ratio(alpha, beta):
    """
    The lower bound B of the probability ratio, for acceptance.

    Given by
            B = beta / (1 - alpha),
    per Wald.
    """
    return beta / (1 - alpha)


def rejection_probability_ratio(alpha, beta):
    """
    The upper bound A of the probability ratio, for rejection.

    Given by
            A = (1 - beta) / alpha,
    per Wald. Currently ignores the correction factor of (d+1)/(2d).
    """
    return (1 - beta) / alpha


def acceptance_intercept(theta_0, theta_1, alpha, beta):
    """
    The failure count for the acceptance line at nil time.

    Denoted by a in MIL-HDBK-781A, and -h_0/s in Epstein & Sobel (1955).
    Given by
            a = -h_0/s = log(B) / log(theta_0/theta_1),
    where B is determined by `acceptance_probability_ratio`.
    """
    capital_b = acceptance_probability_ratio(alpha, beta)
    return log(capital_b) / log(theta_0/theta_1)


def rejection_intercept(theta_0, theta_1, alpha, beta):
    """
    The failure count for the rejection line at nil time.

    Denoted by c in MIL-HDBK-781A, and h_1/s in Epstein & Sobel (1955).
    Given by
            c = +h_1/s = log(A) / log(theta_0/theta_1),
    where A is determined by `rejection_probability_ratio`.
    """
    capital_a = rejection_probability_ratio(alpha, beta)
    return log(capital_a) / log(theta_0/theta_1)


def decision_slope(theta_0, theta_1):
    """
    The slope (failure count per time) of the decision lines.

    Denoted by b in MIL-HDBK-781A, and 1/s in Epstein & Sobel (1955).
    Given by
            b = 1/s = (1/theta_1 - 1/theta_0) / log(theta_0/theta_1).
    """
    return (1/theta_1 - 1/theta_0) / log(theta_0/theta_1)


def maximum_failure_count(theta_0, theta_1, alpha, beta):
    """
    The maximum failure count r_0 for truncation.

    Given by the smallest integer r such that
          chi^2(1-alpha; 2r) / chi^2(beta; 2r) >= theta_1/theta_0.
    """
    r = 1
    while chi2.ppf(alpha, df=2*r) / chi2.ppf(1-beta, df=2*r) < theta_1/theta_0:
        r += 1

    return r


def maximum_test_time(theta_0, theta_1, alpha, beta):
    """
    The maximum cumulative test time T_0.

    Given by
            theta_0 * chi^2(1-alpha; 2r_0) / 2,
    where r_0 is determined by `maximum_failure_count`.
    """
    r_0 = maximum_failure_count(theta_0, theta_1, alpha, beta)
    return theta_0 * chi2.ppf(alpha, df=2*r_0) / 2


def test(theta_0, theta_1, theta, alpha, beta, item_count, trial_count, seed):
    random.seed(a=seed)
    trials = [
        Trial(theta_0, theta_1, theta, alpha, beta, item_count)
        for _ in range(0, trial_count)
    ]

    name_prefix = (
        f'{theta_0} {theta_1} {theta} {alpha} {beta}'
        f' -i {item_count} -t {trial_count} -s {seed}'
    )

    for trial_index, trial in enumerate(trials):
        file_name = f'{name_prefix} - {trial_index}.svg'
        trial.save_plot(file_name)


DESCRIPTION = 'Perform sequential reliability testing.'
DEFAULT_ITEM_COUNT = 1
DEFAULT_TRIAL_COUNT = 10


def parse_command_line_arguments():
    argument_parser = argparse.ArgumentParser(description=DESCRIPTION)
    argument_parser.add_argument(
        'theta_0',
        type=float,
        help='greater MTBF (null hypothesis)',
    )
    argument_parser.add_argument(
        'theta_1',
        type=float,
        help='lesser MTBF (alternative hypothesis)',
    )
    argument_parser.add_argument(
        'theta',
        type=float,
        help='true MTBF for trials',
    )
    argument_parser.add_argument(
        'alpha',
        type=float,
        help="producer's risk",
    )
    argument_parser.add_argument(
        'beta',
        type=float,
        help="consumer's risk",
    )
    argument_parser.add_argument(
        '-i',
        default=DEFAULT_ITEM_COUNT,
        dest='item_count',
        metavar='ITEMS',
        type=int,
        help=f'item count (default {DEFAULT_ITEM_COUNT})',
    )
    argument_parser.add_argument(
        '-t',
        default=DEFAULT_TRIAL_COUNT,
        dest='trial_count',
        metavar='TRIALS',
        type=int,
        help=f'trial count (default {DEFAULT_TRIAL_COUNT})',
    )
    argument_parser.add_argument(
        '-s',
        dest='seed',
        type=int,
        help='seed (integer) for deterministic runs',
    )

    return argument_parser.parse_args()


def main():
    parsed_arguments = parse_command_line_arguments()

    theta_0 = parsed_arguments.theta_0
    theta_1 = parsed_arguments.theta_1
    theta = parsed_arguments.theta
    alpha = parsed_arguments.alpha
    beta = parsed_arguments.beta
    item_count = parsed_arguments.item_count
    trial_count = parsed_arguments.trial_count
    seed = parsed_arguments.seed

    test(theta_0, theta_1, theta, alpha, beta, item_count, trial_count, seed)


if __name__ == '__main__':
    main()

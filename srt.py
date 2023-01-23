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

from numpy import log
from scipy.stats import chi2

DESCRIPTION = 'Perform sequential reliability testing.'


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
    The upper bound A of the probability raio, for rejection.

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
        'item_count',
        metavar='n',
        type=float,
        help='item count',
    )

    return argument_parser.parse_args()


def main():
    parsed_arguments = parse_command_line_arguments()
    theta_0 = parsed_arguments.theta_0
    theta_1 = parsed_arguments.theta_1
    alpha = parsed_arguments.alpha
    beta = parsed_arguments.beta
    item_count = parsed_arguments.item_count


if __name__ == '__main__':
    main()

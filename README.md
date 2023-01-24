# sequential-reliability-testing

A Python implementation of Sequential Reliability Testing,
according unto MIL-HDBK-781A (Section 5.9).


## Usage

```bash
$ srt.py [-h] [-i ITEMS] [-t TRIALS] [-s SEED] theta_0 theta_1 theta alpha beta

Perform sequential reliability testing. Generates a LOG file and some SVGs.

positional arguments:
  theta_0     greater MTBF (null hypothesis)
  theta_1     lesser MTBF (alternative hypothesis)
  theta       true MTBF for trials
  alpha       producer's risk
  beta        consumer's risk

optional arguments:
  -h, --help  show this help message and exit
  -i ITEMS    item count (default 1)
  -t TRIALS   trial count (default 10)
  -s SEED     integer seed for deterministic runs
```


## License

**Copyright 2023 Conway** <br>
Licensed under the GNU General Public License v3.0 (GPL-3.0-only). <br>
This is free software with NO WARRANTY etc. etc., see [LICENSE]. <br>


[LICENSE]: LICENSE

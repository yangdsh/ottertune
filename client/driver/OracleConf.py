#
# OtterTune - OracleConf.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#

import sys
import json
from collections import OrderedDict


def main():
    if (len(sys.argv) != 3):
        raise Exception("Usage: python oracle_conf.py [Next Config] [Current Config]")

    with open(sys.argv[1], "r") as f:
        conf = json.load(f,
                         encoding="UTF-8",
                         object_pairs_hook=OrderedDict)
    conf = conf['recommendation']
    with open(sys.argv[2], "r+") as oracle_conf:
        lines = oracle_conf.readlines()
        signal = "# configurations recommended by ottertune:\n"
        if signal not in lines:
            oracle_conf.write('\n' + signal)
            oracle_conf.flush()
            oracle_conf = open(sys.argv[2], "r+")
            lines = oracle_conf.readlines()
        settings_idx = lines.index(signal)

        oracle_conf.seek(0)
        oracle_conf.truncate(0)

        lines = lines[0:(settings_idx + 1)]
        for line in lines:
            oracle_conf.write(line)

        for (knob_name, knob_value) in list(conf.items()):
            oracle_conf.write(str(knob_name) + " = " + str(knob_value).strip('B') + "\n")


if __name__ == "__main__":
    main()

#
# OtterTune - upload.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
import logging
import os
import sys
import requests

# Logging
LOG = logging.getLogger(__name__)
LOG.addHandler(logging.StreamHandler())
LOG.setLevel(logging.INFO)


def upload(upload_code, datadir, upload_url=None):
    params = {
        'summary': open(os.path.join(datadir, 'sample-0__summary.json'), 'rb'),
        'knobs': open(os.path.join(datadir, 'sample-0__knobs.json'), 'rb'),
        'metrics_start': open(os.path.join(datadir, 'sample-0__metrics_start.json'), 'rb'),
        'metrics_end': open(os.path.join(datadir, 'sample-0__metrics_end.json'), 'rb'),
    }

    response = requests.post(upload_url + "/new_result/",
                             files=params,
                             data={'upload_code': upload_code})
    LOG.info(response.content)


if __name__ == "__main__":
    if not (3 <= len(sys.argv) <= 4):
        LOG.error("Usage: python upload.py [upload_code] [path_to_sample_data] [upload_url]")
    UPLOAD_URL = sys.arv[3] if len(sys.argv) == 4 else "http://0.0.0.0:8000"
    upload(sys.argv[1], sys.argv[2], UPLOAD_URL)

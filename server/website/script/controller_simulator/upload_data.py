'''
Created on Nov 30, 2017

@author: dvanaken
'''

import glob
import os
import sys
import requests

def upload(basedir, upload_code, upload_url):
    for wkld_dir in sorted(glob.glob(os.path.join(basedir, '*'))):
        print(wkld_dir)
        sample_idx = 0
        while True:
            samples = glob.glob(os.path.join(wkld_dir, 'sample-{}__*').format(sample_idx))
            if len(samples) == 0:
                break
            assert len(samples) == 4
            basename = samples[0].split('__')[0]
            params = {
                'summary': open(basename + '__summary.json', 'rb'),
                'knobs': open(basename + '__knobs.json', 'rb'),
                'metrics_before':open(basename + '__metrics_start.json', 'rb'),
                'metrics_after':open(basename + '__metrics_end.json', 'rb'),
            }

            response = requests.post(upload_url+"/new_result/",
                                    files=params,
                                    data={'upload_code':  upload_code})
            print(response)

            sample_idx += 1

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python upload_data.py [datadir] [upload_code] <url>")
        sys.exit(1)
    url = sys.arv[3] if len(sys.argv) == 4 else "http://0.0.0.0:8000"
    upload(sys.argv[1], sys.argv[2], url)

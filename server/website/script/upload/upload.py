import os
import sys
import requests

def upload(upload_code, datadir):
    params = {
        'summary': open(os.path.join(datadir, 'sample-0__summary.json'), 'rb'),
        'knobs': open(os.path.join(datadir, 'sample-0__knobs.json'), 'rb'),
        'metrics_start':open(os.path.join(datadir, 'sample-0__metrics_start.json'), 'rb'),
        'metrics_end':open(os.path.join(datadir, 'sample-0__metrics_end.json'), 'rb'),
    }

    response = requests.post("http://0.0.0.0:8000/new_result/",
                            files=params,
                            data={'upload_code':  upload_code})
    print(response)
    
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python upload.py [upload_code] [path_to_sample_data]")
        sys.exit(1)
    upload(sys.argv[1], sys.argv[2])

import requests
import argparse
import glob
import pdb
import random

test_url = "http://172.17.0.2:5000/detect"

parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, 
        default=4, help='inference batch size')
args = parser.parse_args()        

images = glob.glob('/GIT/datasets/coco/images/val2017/*.jpg')

request = random.sample(images, args.batch)

# test_files = {}

# for i in range(len(request)):
#     test_files[f'img'] = open(request[i], 'rb')

test_files = []

for i in range(len(request)):
    test_files.append(('img', open(request[i], 'rb')))

test_response = requests.post(test_url, files=test_files)
if test_response.ok:
    pdb.set_trace()
    print("Upload completed successfully!")
    print(test_response.text)
else:
    print("Something went wrong!")

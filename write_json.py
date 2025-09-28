import json

nnUNet_dir = '/home/wensheng/gjq_workspace/nnUNet/DATASET/' #此路径根据自己实际修改

def sts_json():
    info = {
        "channel_names": {
            "0": "R",
            "1": "G",
            "2": "B"

        },
        "labels": {
            "background": 0,
            "forbidden area": 1,
            "treated area": 2,
            "available area": 3

        },
        "numTraining": 141,
        "file_ending": ".nii.gz"
    }
    with open(nnUNet_dir + 'nnUNet_raw/Dataset001_prp/dataset.json',
              'w') as f:
        json.dump(info, f, indent=4)

sts_json()

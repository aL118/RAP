import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import json
from configs.openvo import Parameters as OpenVOParameters


class Parameters(OpenVOParameters):
    '''
    Inference-only config for the custom YouTube sequence at data/YTB/10Hz.
    KITTI/Argoverse2/nuScenes-est aren't downloaded locally; openvo.py's
    guarded intrinsics loading (_load_intrinsics) makes those None instead
    of crashing, so we only need to wire up the YouTube dataset here.
    '''
    def __init__(self):
        super().__init__()

        self.data_path['YouTube'] = self.path2_prefix + '/YTB/10Hz'
        self.depth_path['YouTube'] = self.path2_prefix + '/YTB/10Hz/depth_est_intrs'

        with open(self.data_path['YouTube'] + '/ytb_est_intrs.json') as f:
            self.data_intrinsics['YouTube'] = json.load(f)

        self.inference_data = {
            'YouTube': {'YouTube': ['01']},
        }


args = Parameters()

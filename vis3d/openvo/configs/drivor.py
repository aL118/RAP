import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from configs.openvo import Parameters as OpenVOParameters


class Parameters(OpenVOParameters):
    """
    Inference config for DrivoR's estimate_ego_motion.py.

    Unlike my_test.py (OpenVO's own YouTube config) this reads nothing at
    import time. my_test.py opens data/YTB/10Hz/ytb_est_intrs.json while being
    imported, which only exists inside the OpenVO checkout; here the clip's
    frames, depth and intrinsics are all passed to InferenceVO at construction
    time, and it loads <depth_dir_parent>/<scene>_intrs.json itself.

    The other datasets' paths are inherited and left dangling on purpose:
    openvo.py's _load_intrinsics is guarded, so datasets that aren't present
    come out as None rather than raising, and nothing here inferences them.
    """

    def __init__(self):
        super().__init__()

        # Filled in per run by InferenceVO(frames_dir=..., depth_dir=...).
        self.data_path['YouTube'] = None
        self.depth_path['YouTube'] = None
        self.data_intrinsics['YouTube'] = {}
        self.inference_data = {'YouTube': {'YouTube': []}}

        # openvo.py sets these relative to OpenVO's repo root, and the worker
        # is a subprocess whose cwd is wherever the job was launched -- so they
        # have to be absolute here. RAP/weights/openvo/ holds symlinks to
        # the checkpoints rather than copies (the VO model alone is 2.6 GB).
        # FlowNetS reads kitti.yaml from pretrained_flownet_path at
        # construction time, so a wrong value fails only once the worker is
        # already running on a GPU.
        weights = os.path.join(os.path.dirname(ROOT), os.pardir, 'weights', 'openvo')
        self.pretrained_flownet_path = os.path.abspath(weights + '/init_weights')
        self.model_path = os.path.abspath(weights)


args = Parameters()

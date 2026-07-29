"""
Compat shim so `transformers>=4.56` (needed for DINOv3 support) can be imported
against torch==2.1.0.

`transformers/utils/generic.py` calls `torch.utils._pytree.register_pytree_node`
at import time (once for every `ModelOutput` subclass). That name was only made
public in torch 2.2 -- torch 2.1.0 still has the same function, just under the
old private name `_register_pytree_node`, which doesn't accept the newer
`serialized_type_name` kwarg (only used by torch.export/serialization, not by a
plain forward/backward pass).

Upgrading torch itself is blocked here: mmdet==3.3.0 (the latest release)
hard-requires mmcv<2.2.0, and OpenMMLab only ships prebuilt mmcv wheels at
2.2.0+ for torch>=2.2 -- so bumping torch would break the mmcv/mmdet stack
image_encoder.py's FPN neck depends on.

Import this module first, before anything that might import `transformers`
(directly or transitively, e.g. torchvision -> torch.onnx -> transformers).
"""
import torch.utils._pytree as _pytree

if not hasattr(_pytree, "register_pytree_node"):

    def _register_pytree_node_compat(cls, flatten_fn, unflatten_fn, *, serialized_type_name=None, **_ignored):
        return _pytree._register_pytree_node(cls, flatten_fn, unflatten_fn)

    _pytree.register_pytree_node = _register_pytree_node_compat

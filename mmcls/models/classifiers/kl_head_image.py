import copy
import warnings

from ..builder import CLASSIFIERS, build_backbone, build_head, build_neck
from ..utils.augment import Augments
from .image import ImageClassifier
import mmcv
from mmcv.runner import load_checkpoint
from pdb import set_trace
from .. import build_classifier
import torch
import os
from mmcv.parallel import MMDataParallel

@CLASSIFIERS.register_module()
class KLHeadImageClassifier(ImageClassifier):

    def __init__(self,
                 teacher_config,
                 teacher_ckpt,
                 eval_teacher=True,
                 init_cfg=None,
                 *args,
                 **kwargs):
        super(KLHeadImageClassifier, self).__init__(init_cfg=init_cfg, *args, **kwargs)
        self.eval_teacher = eval_teacher
        
        # Build teacher model
        if isinstance(teacher_config, str):

            teacher_config = mmcv.Config.fromfile(teacher_config)

        teacher_config['model']['head']['num_classes'] = self.head.fc.out_features

        self.teacher_model = build_classifier(teacher_config['model'])
        if teacher_ckpt is not None:

            files_path = os.listdir(teacher_ckpt)

            checkpoint_file = 'best_accuracy'
            for file in files_path:
                if file.startswith(checkpoint_file):
                    checkpoint_file = file

                    checkpoint = load_checkpoint(self.teacher_model, os.path.join(teacher_ckpt, checkpoint_file))

                    break

            


    def forward_train(self, img, gt_label, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            gt_label (Tensor): It should be of shape (N, 1) encoding the
                ground-truth label of input images for single label task. It
                shoulf be of shape (N, C) encoding the ground-truth label
                of input images for multi-labels task.
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        if self.augments is not None:
            img, gt_label = self.augments(img, gt_label)
        
        x = self.extract_feat(img)
        
        with torch.no_grad():
            teacher_x = self.teacher_model.extract_feat(img)
            out_teacher = self.teacher_model.head.fc(teacher_x)

        losses = self.head.forward_train(x, out_teacher, gt_label)

        return losses

    def simple_test(self, img, img_metas):
        """Test without augmentation."""
        x = self.extract_feat(img)
        x_dims = len(x.shape)
        if x_dims == 1:
            x.unsqueeze_(0)
        return self.head.simple_test(x)
    
    def cuda(self, device=None):
        """Since teacher_model is registered as a plain object, it is necessary
        to put the teacher model to cuda when calling cuda function."""
        self.teacher_model.cuda(device=device)
        return super().cuda(device=device)

    def train(self, mode=True):
        """Set the same train mode for teacher and student model."""
        if self.eval_teacher:
            self.teacher_model.train(False)
        else:
            self.teacher_model.train(mode)
        super().train(mode)

    def __setattr__(self, name, value):
        """Set attribute, i.e. self.name = value
        This reloading prevent the teacher model from being registered as a
        nn.Module. The teacher module is registered as a plain object, so that
        the teacher parameters will not show up when calling
        ``self.parameters``, ``self.modules``, ``self.children`` methods.
        """
        if name == 'teacher_model':
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)

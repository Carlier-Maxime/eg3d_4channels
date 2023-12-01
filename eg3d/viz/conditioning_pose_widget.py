# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from gui_utils import imgui_utils
from viz.pose_widget import BasePoseWidget


# ----------------------------------------------------------------------------

class ConditioningPoseWidget(BasePoseWidget):
    def __init__(self, viz):
        super().__init__(viz)

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        super().__call__('Cond Pose', '##frac', show=show)
        self.viz.args.conditioning_yaw = self.pose.yaw
        self.viz.args.conditioning_pitch = self.pose.pitch

# ----------------------------------------------------------------------------

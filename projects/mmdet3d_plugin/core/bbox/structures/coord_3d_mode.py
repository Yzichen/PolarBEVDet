# Copyright (c) OpenMMLab. All rights reserved.
from enum import IntEnum, unique
from typing import Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor

from mmdet3d.core.points import (BasePoints, CameraPoints, DepthPoints,
                                       LiDARPoints)
from .base_box3d import BaseInstance3DBoxes
from .box_3d_mode import Box3DMode


@unique
class Coord3DMode(IntEnum):
    """Enum of different ways to represent a box and point cloud.

    Coordinates in LiDAR:

    .. code-block:: none

                    up z
                       ^   x front
                       |  /
                       | /
        left y <------ 0

    The relative coordinate of bottom center in a LiDAR box is (0.5, 0.5, 0),
    and the yaw is around the z axis, thus the rotation axis=2.

    Coordinates in Camera:

    .. code-block:: none

                z front
               /
              /
             0 ------> x right
             |
             |
             v
        down y

    The relative coordinate of bottom center in a CAM box is (0.5, 1.0, 0.5),
    and the yaw is around the y axis, thus the rotation axis=1.

    Coordinates in Depth:

    .. code-block:: none

        up z
           ^   y front
           |  /
           | /
           0 ------> x right

    The relative coordinate of bottom center in a DEPTH box is (0.5, 0.5, 0),
    and the yaw is around the z axis, thus the rotation axis=2.
    """

    LIDAR = 0
    CAM = 1
    DEPTH = 2

    @staticmethod
    def convert(input: Union[Sequence[float], np.ndarray, Tensor,
                             BaseInstance3DBoxes, BasePoints],
                src: Union[Box3DMode, 'Coord3DMode'],
                dst: Union[Box3DMode, 'Coord3DMode'],
                rt_mat: Optional[Union[np.ndarray, Tensor]] = None,
                with_yaw: bool = True,
                correct_yaw: bool = False,
                is_point: bool = True):
        """Convert boxes or points from ``src`` mode to ``dst`` mode.

        Args:
            input (Sequence[float] or np.ndarray or Tensor or
                :obj:`BaseInstance3DBoxes` or :obj:`BasePoints`): Can be a
                k-tuple, k-list or an Nxk array/tensor.
            src (:obj:`Box3DMode` or :obj:`Coord3DMode`): The source mode.
            dst (:obj:`Box3DMode` or :obj:`Coord3DMode`): The target mode.
            rt_mat (np.ndarray or Tensor, optional): The rotation and
                translation matrix between different coordinates.
                Defaults to None. The conversion from ``src`` coordinates to
                ``dst`` coordinates usually comes along the change of sensors,
                e.g., from camera to LiDAR. This requires a transformation
                matrix.
            with_yaw (bool): If ``box`` is an instance of
                :obj:`BaseInstance3DBoxes`, whether or not it has a yaw angle.
                Defaults to True.
            correct_yaw (bool): If the yaw is rotated by rt_mat.
                Defaults to False.
            is_point (bool): If ``input`` is neither an instance of
                :obj:`BaseInstance3DBoxes` nor an instance of
                :obj:`BasePoints`, whether or not it is point data.
                Defaults to True.

        Returns:
            Sequence[float] or np.ndarray or Tensor or
            :obj:`BaseInstance3DBoxes` or :obj:`BasePoints`: The converted box
            or points of the same type.
        """
        if isinstance(input, BaseInstance3DBoxes):
            return Coord3DMode.convert_box(
                input,
                src,
                dst,
                rt_mat=rt_mat,
                with_yaw=with_yaw,
                correct_yaw=correct_yaw)
        elif isinstance(input, BasePoints):
            return Coord3DMode.convert_point(input, src, dst, rt_mat=rt_mat)
        elif isinstance(input, (tuple, list, np.ndarray, Tensor)):
            if is_point:
                return Coord3DMode.convert_point(
                    input, src, dst, rt_mat=rt_mat)
            else:
                return Coord3DMode.convert_box(
                    input,
                    src,
                    dst,
                    rt_mat=rt_mat,
                    with_yaw=with_yaw,
                    correct_yaw=correct_yaw)
        else:
            raise NotImplementedError

    @staticmethod
    def convert_box(
        box: Union[Sequence[float], np.ndarray, Tensor, BaseInstance3DBoxes],
        src: Box3DMode,
        dst: Box3DMode,
        rt_mat: Optional[Union[np.ndarray, Tensor]] = None,
        with_yaw: bool = True,
        correct_yaw: bool = False
    ) -> Union[Sequence[float], np.ndarray, Tensor, BaseInstance3DBoxes]:
        """Convert boxes from ``src`` mode to ``dst`` mode.

        Args:
            box (Sequence[float] or np.ndarray or Tensor or
                :obj:`BaseInstance3DBoxes`): Can be a k-tuple, k-list or an Nxk
                array/tensor.
            src (:obj:`Box3DMode`): The source box mode.
            dst (:obj:`Box3DMode`): The target box mode.
            rt_mat (np.ndarray or Tensor, optional): The rotation and
                translation matrix between different coordinates.
                Defaults to None. The conversion from ``src`` coordinates to
                ``dst`` coordinates usually comes along the change of sensors,
                e.g., from camera to LiDAR. This requires a transformation
                matrix.
            with_yaw (bool): If ``box`` is an instance of
                :obj:`BaseInstance3DBoxes`, whether or not it has a yaw angle.
                Defaults to True.
            correct_yaw (bool): If the yaw is rotated by rt_mat.
                Defaults to False.

        Returns:
            Sequence[float] or np.ndarray or Tensor or
            :obj:`BaseInstance3DBoxes`: The converted box of the same type.
        """
        return Box3DMode.convert(
            box,
            src,
            dst,
            rt_mat=rt_mat,
            with_yaw=with_yaw,
            correct_yaw=correct_yaw)

    @staticmethod
    def convert_point(
        point: Union[Sequence[float], np.ndarray, Tensor, BasePoints],
        src: 'Coord3DMode',
        dst: 'Coord3DMode',
        rt_mat: Optional[Union[np.ndarray, Tensor]] = None,
    ) -> Union[Sequence[float], np.ndarray, Tensor, BasePoints]:
        """Convert points from ``src`` mode to ``dst`` mode.

        Args:
            box (Sequence[float] or np.ndarray or Tensor or :obj:`BasePoints`):
                Can be a k-tuple, k-list or an Nxk array/tensor.
            src (:obj:`Coord3DMode`): The source point mode.
            dst (:obj:`Coord3DMode`): The target point mode.
            rt_mat (np.ndarray or Tensor, optional): The rotation and
                translation matrix between different coordinates.
                Defaults to None. The conversion from ``src`` coordinates to
                ``dst`` coordinates usually comes along the change of sensors,
                e.g., from camera to LiDAR. This requires a transformation
                matrix.

        Returns:
            Sequence[float] or np.ndarray or Tensor or :obj:`BasePoints`: The
            converted point of the same type.
        """
        if src == dst:
            return point

        is_numpy = isinstance(point, np.ndarray)
        is_InstancePoints = isinstance(point, BasePoints)
        single_point = isinstance(point, (list, tuple))
        if single_point:
            assert len(point) >= 3, (
                'Coord3DMode.convert takes either a k-tuple/list or '
                'an Nxk array/tensor, where k >= 3')
            arr = torch.tensor(point)[None, :]
        else:
            # avoid modifying the input point
            if is_numpy:
                arr = torch.from_numpy(np.asarray(point)).clone()
            elif is_InstancePoints:
                arr = point.tensor.clone()
            else:
                arr = point.clone()

        # convert point from `src` mode to `dst` mode.
        if src == Coord3DMode.LIDAR and dst == Coord3DMode.CAM:
            if rt_mat is None:
                rt_mat = arr.new_tensor([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
        elif src == Coord3DMode.CAM and dst == Coord3DMode.LIDAR:
            if rt_mat is None:
                rt_mat = arr.new_tensor([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        elif src == Coord3DMode.DEPTH and dst == Coord3DMode.CAM:
            if rt_mat is None:
                rt_mat = arr.new_tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        elif src == Coord3DMode.CAM and dst == Coord3DMode.DEPTH:
            if rt_mat is None:
                rt_mat = arr.new_tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        elif src == Coord3DMode.LIDAR and dst == Coord3DMode.DEPTH:
            if rt_mat is None:
                rt_mat = arr.new_tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        elif src == Coord3DMode.DEPTH and dst == Coord3DMode.LIDAR:
            if rt_mat is None:
                rt_mat = arr.new_tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        else:
            raise NotImplementedError(
                f'Conversion from Coord3DMode {src} to {dst} '
                'is not supported yet')

        if not isinstance(rt_mat, Tensor):
            rt_mat = arr.new_tensor(rt_mat)
        if rt_mat.size(1) == 4:
            extended_xyz = torch.cat(
                [arr[..., :3], arr.new_ones(arr.size(0), 1)], dim=-1)
            xyz = extended_xyz @ rt_mat.t()
        else:
            xyz = arr[..., :3] @ rt_mat.t()

        remains = arr[..., 3:]
        arr = torch.cat([xyz[..., :3], remains], dim=-1)

        # convert arr to the original type
        original_type = type(point)
        if single_point:
            return original_type(arr.flatten().tolist())
        if is_numpy:
            return arr.numpy()
        elif is_InstancePoints:
            if dst == Coord3DMode.CAM:
                target_type = CameraPoints
            elif dst == Coord3DMode.LIDAR:
                target_type = LiDARPoints
            elif dst == Coord3DMode.DEPTH:
                target_type = DepthPoints
            else:
                raise NotImplementedError(
                    f'Conversion to {dst} through {original_type} '
                    'is not supported yet')
            return target_type(
                arr,
                points_dim=arr.size(-1),
                attribute_dims=point.attribute_dims)
        else:
            return arr

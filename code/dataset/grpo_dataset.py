import warnings
import hashlib
from typing import List, Tuple, Dict
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import json

# from act.utils import calc_pose_rel_with_angle, get_traj_rel_to_start, traj_rel_downsample
# from act.config import WDreamerConfig, ACTConfig

class GRPODataset(Dataset):
    def __init__(self, 
                 data_path: str, 
                 ) -> None:
        self.data_path = data_path
        with open(self.data_path, "r") as f:
            data = json.load(f)
        self.all_prompts = [item["caption"] for item in data]

    def __len__(self):
        return len(self.all_prompts)

    def __getitem__(self, index):
        return self.all_prompts[index]
        # sample_index = index // self.n_autoreg * self.n_skip_samples 
        # offset = index % self.n_autoreg
        # data = {}
        # file_name = self.h5_file_path.split('/')[-1]
        # load_img_feat = False
        # for (extrin_dir, extrin_attr_maps) in self.extrin_attrs.items():
        #     extrin_path = os.path.join(extrin_dir, file_name)
        #     with h5py.File(extrin_path, 'r') as h5_extrin:
        #         for (extrin_key, batch_key) in extrin_attr_maps:
        #             if batch_key.startswith("feat_"):
        #                 load_img_feat = True
        #             data[batch_key] = h5_extrin[extrin_key][sample_index]
        #             # print(f" extrin_key: {extrin_key}, batch_key: {batch_key}")
        #         if self.offline_RL:
        #             data['next_occ_map'] = h5_extrin['occ_map'][sample_index + self.rl_transit_len]
        #             data['next_feat_occv2'] = h5_extrin['feat'][sample_index + self.rl_transit_len]

        # with h5py.File(self.h5_file_path, 'r') as h5_file:
        #     data['offset'] = offset
        #     data['last_a'] = h5_file['last_a'][sample_index]
        #     data['trajs_rel'] = h5_file['trajs_rel'][sample_index]
        #     dim_traj = data['trajs_rel'].shape[1]
        #     data['trajs_abs'] = h5_file['trajs_abs'][sample_index]
        #     if 'timestamp' in h5_file.keys():
        #         data['timestamp'] = h5_file['timestamp'][sample_index]
        #     if 'tar_rel' in h5_file.keys():
        #         data['tar_rel'] = h5_file['tar_rel'][sample_index]
        #     if 'occ_map' in h5_file.keys():
        #         # print(f"smaple_index: {sample_index}")
        #         data['occ_map'] = h5_file['occ_map'][sample_index]
        #         if self.offline_RL:
        #             data['next_occ_map'] = h5_file['occ_map'][sample_index + self.rl_transit_len]

        #     # data['offline_rl'] = self.offline_RL
        #     # transition setting: (st,s_t+1, action_chunk_1)
        #     # 强化学习定义：两个state之间的transition,间隔时间由self.rl_transit_len决定
        #     if self.offline_RL:
        #         data['next_last_a'] = h5_file['last_a'][sample_index + self.rl_transit_len]
        #         data['next_tar_rel'] = h5_file['tar_rel'][sample_index + self.rl_transit_len]
        #         reward = self._get_reward(data)
        #         data['reward'] = reward
        #         # print(f"obs: {0.5*(obstacle_curr - obstacle_next)}, goal: {2*(dis_curr - dis_next)}, reward: {data['reward']}")

        #     if self.vlm_output_key in h5_file.keys():
        #         vlm_sample_index = sample_index
        #         # 没有数据的话就取该秒数下的第一个
        #         if len(h5_file[self.vlm_output_key][sample_index]) == 0:
        #             vlm_sample_index = int(sample_index // 10 * 10)
        #         vlm_output = h5_file[self.vlm_output_key][vlm_sample_index]
        #         vlm_output = json.loads(vlm_output)
        #         data['vlm_output'] = vlm_output

        #     # for auto-regression
        #     data['autoreg_x'] = np.zeros((5 * self.autoreg_step, dim_traj))
        #     if offset != 0:
        #         data['autoreg_x'][-offset * self.autoreg_step:] = data['trajs_rel'][:offset * self.autoreg_step]
        #     data['autoreg_y'] = data['trajs_rel'][offset * self.autoreg_step: (offset + 1) * self.autoreg_step]
            
        #     if not self.same_cameras:
        #         data['camera_param'] = h5_file['camera_params'][sample_index]
        #     # 是否load图像
        #     # training
        #     if self.config is not None:
        #         load_imgs = not self.config.use_disk_feat
        #         # print(f"load_imgs: {load_imgs}")
        #         load_temporal_imgs = not self.config.use_disk_feat
        #     # eval
        #     else:
        #         load_imgs = True # for visualization
        #         load_temporal_imgs = not load_img_feat
            
        #     # 有feat的前提下不再load图像了
        #     # load_imgs = True
        #     if load_imgs:
        #         for key in self.camera_keys:
        #             img_data = h5_file[key][sample_index]
        #             if len(img_data.shape) == 3:
        #                 data[key] = img_data
        #             else:
        #                 img_arr = self._img_to_arr(img_data)
        #                 data[key] = img_arr

        #     # load temporal data
        #     if self.use_temporal:
        #         if 'timestamp' in h5_file.keys():
        #             self._get_temporal_data(h5_file, data, sample_index, load_imgs=load_temporal_imgs)
        #         elif 'meta' in h5_file.keys():
        #             meta = json.loads(h5_file['meta'][sample_index])
        #             h5_extrin = h5py.File(meta['path'], 'r')
        #             self._get_temporal_data(h5_extrin, data, meta['idx'], load_imgs=load_temporal_imgs)
        #             h5_extrin.close()
        #         else:
        #             raise ValueError(f'No \'timestamp\' or \'meta\' in h5 file: {self.h5_file_path}')
        
        # # 把mask esdf预处理放在这里做来加速, 只在训练算loss会用到
        # if 'esdf' in data.keys():
        #     data['raw_esdf'] = data['esdf'].copy()
        #     if self.config is not None and self.config.use_esdf and self.config.esdf_weight > 0.0 and self.config.esdf_mask_size > 0:
        #         data['esdf'] = self._mask_esdf(data['esdf'], data['trajs_abs'], mask_size=self.config.esdf_mask_size, mask_value=self.config.esdf_mask_value)
        # # print(f"keys of data: {list(data.keys())}")
        # # self.vis.visualize(data,index)
        # # print(f"reward: {data['reward']}")
        # return data
    
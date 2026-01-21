import os
import numpy as np
import torch
import json




class DataLogger:
    def __init__(self, tpath, apath, cpath):
        self.T = self.load_pose(tpath, "T")
        self.A = self.load_pose(apath, "A")
        self.C = self.load_pose(cpath, "C")

    def load_pose(self, path, name):
        camera_matrix = torch.tensor([[[4.9270 * 3, 0.0000, -0.0519, 0.0000],
                                       [0.0000, 4.9415 * 3, 0.0000, 0.0000],
                                       [0.0000, 0.0000, 1.0001, -0.0101],
                                       [0.0000, 0.0000, 1.0000, 0.0000]]], device='cuda')
        camera_transform = torch.tensor([[[1., 0., 0., 0.],
                                          [0., 1., 0., 0.9],
                                          [0., 0., 1., 15.2122],
                                          [0., 0., 0., 1.]]], device="cuda")

        with open(path, 'r') as f:
            data = json.load(f)

        smplx_params = {}
        smplx_params["name"] = name
        smplx_params["transl"] = torch.FloatTensor(data["transl"])
        smplx_params["global_orient"] = torch.FloatTensor(data["global_orient"])
        smplx_params["body_pose"] = torch.FloatTensor(data["body_pose"])
        smplx_params["right_hand_pose"] = torch.FloatTensor(data["right_hand_pose"])
        smplx_params["left_hand_pose"] = torch.FloatTensor(data["left_hand_pose"])
        smplx_params["leye_pose"] = torch.FloatTensor(data["leye_pose"])
        smplx_params["reye_pose"] = torch.FloatTensor(data["reye_pose"])
        smplx_params["jaw_pose"] = torch.FloatTensor(data["jaw_pose"])
        smplx_params["expression"] = torch.FloatTensor(data["expression"])
        smplx_params["betas"] = torch.FloatTensor(data["betas"])
        smplx_params["camera_matrix"] = camera_matrix
        smplx_params["camera_transform"] = camera_transform

        return smplx_params

        

    def _log_data_val(self, save_folder, batch, global_step):

        # keys_to_save = [
        #     "mesh_rasterization",
        #     "rgb_image",
        #     "gaussian_rasterization",
        #     "merged_mask",
        #     "gaussian_depth",
        #     "mesh_depth",
        #     "rasterization",
        #     "skin_mask",
        #     "gaussian_alpha",
        #     "texture"
        # ]
        
        # os.makedirs(save_folder, exist_ok=True)

        # filename = "s{:06}_b{:03}.json".format(global_step, batch["batch_idx"])
        
        # # output = dict2device(batch,'cpu')


        # os.makedirs(save_folder, exist_ok=True)
        # for key_name in keys_to_save:
        #     if key_name not in batch:
        #         continue
        #     data = batch[key_name]
        #     if key_name in ["texture"]:
        #         data = data[None]
        #     if isinstance(data, torch.Tensor):
        #         data = data.cpu().tolist()

        #     sub_folder = os.path.join(save_folder, key_name)
        #     os.makedirs(sub_folder, exist_ok=True)

        #     with open(os.path.join(sub_folder, filename),'w') as f:
        #         json.dump(data, f)
        pass


    def on_train_batch_end(
            self,
            runner,
            outputs
    ):
        pass

    def on_validation_batch_end(
            self,
            runner,
            outputs
    ):
        # save_folder = os.path.join(runner.logger.log_dir, 'val')
        # self._log_data_val(save_folder, outputs, runner.global_step)
        pass

    def on_test_batch_end(
            self,
            runner,
            outputs
    ):
        pass

    def on_test_end(self, runner):
        save_folder = os.path.join(runner.logger.log_dir, 'data')
        data_dict = {}
        data_dict["pid"] = ["000"]

        for smplx_params in [self.T, self.A, self.C]:

            smplx_params["transl"] = smplx_params["transl"].to(runner.device)
            smplx_params["global_orient"] = smplx_params["global_orient"].to(runner.device)
            smplx_params["body_pose"] = smplx_params["body_pose"].to(runner.device)
            smplx_params["right_hand_pose"] = smplx_params["right_hand_pose"].to(runner.device)
            smplx_params["left_hand_pose"] = smplx_params["left_hand_pose"].to(runner.device)

            data_dict["smplx_params"] = smplx_params
            
            with torch.no_grad():
                output = runner.predict_smplx_vertices(data_dict, calc_gaussians=True)
            
            runner.save_play(os.path.join(save_folder, f'output_{smplx_params["name"]}.ply'),
                            output['gaussians_xyz'],
                            output['gaussians_colors'],
                            output['gaussians_opacity'],
                            output['gaussians_scales'],
                            output['gaussians_rotations'])
            
            for key,value in output.items():
                if isinstance(value, torch.Tensor):
                    output[key] = value.cpu().tolist()
            with open(os.path.join(save_folder, f'output_{smplx_params["name"]}.json','w')) as f:
                json.dump(output,f)


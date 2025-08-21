import copy
import json
import os
import pickle
import shutil
import tempfile
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.distributed as dist
import mmengine

from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from no_time_to_train.pl_wrapper.sam2ref_pl import RefSam2LightningModel
from no_time_to_train.pl_wrapper.sam2matcher_pl import Sam2MatcherLightningModel


def collect_results_cpu(result_part, size=None, tmpdir=None):
    
    # Check if distributed training is initialized
    if not dist.is_initialized():
        return result_part
    
    # Reference: MMDetection
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmengine.mkdir_or_exist('/tmp/.mydist_test')
            tmpdir = tempfile.mkdtemp(dir='/tmp/.mydist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmengine.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmengine.dump(result_part, os.path.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = os.path.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmengine.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        if size is not None:
            ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


class SAM2RefLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--out_path", default=None, type=str)
        parser.add_argument("--out_support_res", default=None, required=False, type=str)
        parser.add_argument("--out_neg_pkl", default=None, required=False, type=str)
        parser.add_argument("--out_neg_json", default=None, required=False, type=str)
        parser.add_argument("--export_result", default=None, type=str)
        parser.add_argument("--seed", default=None, type=int)
        parser.add_argument("--n_shot", default=None, type=int)
        parser.add_argument("--coco_semantic_split", default=None, type=str)
        parser.add_argument("--export_gaga_masks", default=None, type=str, 
                           help="Export masks to Gaga format at specified scene path")
        parser.add_argument("--gaga_seg_method", type=str, default="no_time_to_train")
        parser.add_argument("--gaga_conf_thresh", type=float, default=0.45)
        parser.add_argument("--gaga_min_area", type=int, default=64)

    def before_test(self):
        memory_bank_cfg = self.model.model_cfg["memory_bank_cfg"]

        if self.model.test_mode == "fill_memory":
            self.model.dataset_cfgs["fill_memory"]["memory_length"] = memory_bank_cfg["length"]
        elif self.model.test_mode == "fill_memory_neg":
            self.model.dataset_cfgs["fill_memory"]["memory_length"] = memory_bank_cfg["length_negative"]
            self.model.dataset_cfgs["fill_memory"]["root"] = self.model.dataset_cfgs["support"]["root"]
            self.model.dataset_cfgs["fill_memory"]["json_file"] = self.config.test.out_neg_json
            self.model.dataset_cfgs["fill_memory"]["memory_pkl"] = self.config.test.out_neg_pkl
        else:
            pass


    def after_test(self):
        if (
                self.model.test_mode == "fill_memory"
                or self.model.test_mode == "postprocess_memory"
                or self.model.test_mode == "fill_memory_neg"
                or self.model.test_mode == "postprocess_memory_neg"
        ):
            
            if self.config.test.out_path is not None:
                save_path = self.config.test.out_path
            else:
                raise RuntimeError(
                    "A saving path for the temporary checkpoint is required to store the model with memory bank."
                )
            print("======= Trainer save checkpoint ==========")
            self.trainer.save_checkpoint(save_path)

            if self.model.test_mode == "fill_memory":
                print("Checkpoint with memory is saved to %s" % save_path)
            elif self.model.test_mode == "postprocess_memory":
                print("Checkpoint with post-processed memory is saved to %s" % save_path)
            elif self.model.test_mode == "fill_memory_neg":
                print("Checkpoint with negative memory is saved to %s" % save_path)
            elif self.model.test_mode == "postprocess_memory_neg":
                print("Checkpoint with post-processed negative memory is saved to %s" % save_path)
            else:
                raise NotImplementedError
        elif self.model.test_mode == "test" or self.model.test_mode == "test_support":
            print("Start testing")
            results = copy.deepcopy(self.trainer.model.output_queue)
            for single in results:
                print(f"Single: {single}")
            results_all = collect_results_cpu(
                results, size=len(self.trainer.model.eval_dataset)
            )

            if len(self.trainer.model.scalars_queue) > 0:
                scalars = copy.deepcopy(self.trainer.model.scalars_queue)
                scalars_all = collect_results_cpu(
                    scalars, size=len(self.trainer.model.eval_dataset)
                )
            else:
                scalars_all = None

            if not dist.is_initialized() or dist.get_rank() == 0:
                if scalars_all is not None:
                    with open("./scalars_all.pkl", "wb") as f:
                        pickle.dump(scalars_all, f)

                results_unpacked = []
                for results_per_img in results_all:
                    results_unpacked.extend(results_per_img)
                # if self.config.test.export_result is not None:
                #     with open(self.config.test.export_result, 'w') as f:
                #         json.dump(results_unpacked, f)
                
                # # Export masks for Gaga if requested
                # if (hasattr(self.config.test, 'export_gaga_masks') and 
                #     self.config.test.export_gaga_masks is not None):
                #     print(f"🎯 Exporting masks for Gaga to: {self.config.test.export_gaga_masks}")
                #     for result in results_all:
                #         if hasattr(self.trainer.model.seg_model, 'export_masks_for_gaga'):
                #             for single_result in result:  # results_all contains lists of results
                #                 self.trainer.model.seg_model.export_masks_for_gaga(
                #                     single_result, self.config.test.export_gaga_masks
                #                 )
                # exp_dir = getattr(self.config.test, "export_gaga_masks", None)
                # print(f"EXP DIR: {exp_dir}")
                # if exp_dir:
                #     print(f"🎯 Exporting masks for Gaga to: {exp_dir}")

                #     # Resolve the image root used at test time (prefer model_cfg.test.imgs_path)
                #     imgs_root = None
                #     try:
                #         imgs_root = self.model.model_cfg["test"]["imgs_path"]
                #     except Exception:
                #         pass
                #     if not imgs_root:
                #         imgs_root = self.model.dataset_cfgs["test"]["root"]

                #     print(f"Image root: {imgs_root}")
                    
                #     gaga_seg_method = getattr(self.config.test, "gaga_seg_method", "no_time_to_train")
                #     conf_thr = float(getattr(self.config.test, "gaga_conf_thresh", 0.45))
                #     min_area = int(getattr(self.config.test, "gaga_min_area", 64))

                #     # results_all is a list (batches) of lists (per-image dicts)
                #     for batch_results in results_all:
                #         for single_result in batch_results:
                #             # Derive image_path from the same filename the dataloader used
                #             # print(f"Single result: {single_result}")
                #             image_info = single_result.get("image_info", {})
                #             # print(f"Image info: {image_info}")
                #             file_name = image_info.get("file_name") or single_result.get("filename")
                #             if not file_name:
                #                 print("Cannot export without a name.")
                #                 # Cannot export without a name; skip gracefully
                #                 continue

                #             image_path = os.path.join(imgs_root, file_name)

                #             # Call the model's exporter (your function in seg_model)
                #             print(f"F{self.trainer.model.seg_model}")
                #             if hasattr(self.trainer.model.seg_model, "export_masks_for_gaga"):
                #                 print("========= exporting gaga mask =========")
                #             self.trainer.model.seg_model.export_masks_for_gaga(
                #                 output_dict=single_result,
                #                 gaga_scene_path=exp_dir,
                #                 image_path=image_path,
                #                 confidence_threshold=conf_thr,
                #                 min_area=min_area,
                #                 gaga_seg_method=gaga_seg_method,
                #             )
                
                
                # if self.model.test_mode == "test":
                #     # Naming the output file
                #     output_name = ""
                #     if self.config.test.coco_semantic_split is not None:
                #         output_name += f"semantic_split_{self.config.test.coco_semantic_split}_"
                #     if self.config.test.n_shot is not None and self.config.test.seed is not None:
                #         output_name += f"{self.config.test.n_shot}shot_{self.config.test.seed}seed"
                #     # Evaluating the results
                #     self.trainer.model.eval_dataset.evaluate(results_unpacked, output_name=output_name)
                # elif self.model.test_mode == "test_support":
                #     self.trainer.model.eval_dataset.evaluate(results_unpacked)
                #     with open(self.config.test.out_support_res, "wb") as f:
                #         pickle.dump(results_unpacked, f)

                    # out_pkl = self.config.test.out_neg_pkl
                    # out_json = self.config.test.out_neg_json
                    # n_sample = self.trainer.model.seg_model.mem_length_negative
                    # self.trainer.model.eval_dataset.sample_negative(
                    #     results_unpacked, out_pkl, out_json, n_sample
                    # )
                # else:
                #     raise NotImplementedError
        elif self.model.test_mode == "vis_memory":
            pass
        else:
            raise NotImplementedError(f"Unrecognized test mode {self.model.test_mode}")



if __name__ == "__main__":
    SAM2RefLightningCLI()


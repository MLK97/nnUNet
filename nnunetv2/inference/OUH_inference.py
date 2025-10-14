import inspect
import itertools
import multiprocessing
import os
from copy import deepcopy
from queue import Queue
from threading import Thread
from time import sleep
from typing import Tuple, Union, List, Optional

import numpy as np
import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p, isdir, subdirs, \
    save_json
from torch import nn
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

import nnunetv2
from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy, preprocessing_iterator_fromfiles, \
    preprocessing_iterator_fromnpy
from nnunetv2.inference.export_prediction import export_prediction_from_logits, \
    convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.inference.sliding_window_prediction import compute_gaussian, \
    compute_steps_for_sliding_window
from nnunetv2.utilities.file_path_utilities import get_output_folder, check_workers_alive_and_busy
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from typing import Union, List

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
from batchgenerators.utilities.file_and_folder_operations import load_json, save_pickle

from nnunetv2.configuration import default_num_processes
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager

def convert_predicted_logits_fold_to_segmentation_with_correct_shape(predicted_logits_fold: Union[torch.Tensor, np.ndarray],
                                                                plans_manager: PlansManager,
                                                                configuration_manager: ConfigurationManager,
                                                                label_manager: LabelManager,
                                                                properties_dict: dict,
                                                                return_probabilities: bool = False,
                                                                num_threads_torch: int = default_num_processes):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)

    # resample to original shape
    spacing_transposed = [properties_dict['spacing'][i] for i in plans_manager.transpose_forward]
    current_spacing = configuration_manager.spacing if \
        len(configuration_manager.spacing) == \
        len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [spacing_transposed[0], *configuration_manager.spacing]
    # print('PREDICTED LOGITS FOLD')
    # print(predicted_logits_fold.shape)
    # print(type(predicted_logits_fold.shape))
    predicted_logits_fold = [configuration_manager.resampling_fn_probabilities(predicted_logits,
                                            properties_dict['shape_after_cropping_and_before_resampling'],
                                            current_spacing,
                                            [properties_dict['spacing'][i] for i in plans_manager.transpose_forward]) for predicted_logits in predicted_logits_fold]
                                            
    # print('PREDICTED LOGITS FOLD POST')
    # print(len(predicted_logits_fold))
    # print(type(predicted_logits_fold[0]))
    # return value of resampling_fn_probabilities can be ndarray or Tensor but that does not matter because
    # apply_inference_nonlin will convert to torch
    if not return_probabilities:
        # this has a faster computation path becasue we can skip the softmax in regular (not region based) trainig
        predicted_logits = np.mean(predicted_logits_fold, axis=0)
        segmentation = label_manager.convert_logits_to_segmentation(predicted_logits)
        # now we need to normalize (softmax) the logits for the computation of the entropies
        predicted_probabilities_folds = [label_manager.apply_inference_nonlin(predicted_logits) for predicted_logits in predicted_logits_fold]
        predicted_probabilities_folds = torch.stack(predicted_probabilities_folds, dim=0)
        predicted_probabilities = torch.mean(predicted_probabilities_folds, dim=0)
        jsd, entropy_mean, mean_entropy = jensen_shannon_divergence_torch(predicted_probabilities_folds)
    else:
        predicted_logits = np.mean(predicted_logits_fold, axis=0)
        # now we need to normalize (softmax) the logits for the computation of the entropies
        predicted_probabilities_folds = [label_manager.apply_inference_nonlin(predicted_logits) for predicted_logits in predicted_logits_fold]
        predicted_probabilities_folds = torch.stack(predicted_probabilities_folds, dim=0)
        
        jsd, entropy_mean, mean_entropy = jensen_shannon_divergence_torch(predicted_probabilities_folds)
        
        predicted_probabilities = label_manager.apply_inference_nonlin(predicted_logits)
        segmentation = label_manager.convert_probabilities_to_segmentation(predicted_probabilities)
 
    del predicted_logits, predicted_logits_fold

    # put segmentation in bbox (revert cropping)
    segmentation_reverted_cropping = np.zeros(properties_dict['shape_before_cropping'],
                                              dtype=np.uint8 if len(label_manager.foreground_labels) < 255 else np.uint16)
    segmentation_reverted_cropping = insert_crop_into_image(segmentation_reverted_cropping, segmentation, properties_dict['bbox_used_for_cropping'])
    del segmentation

    # segmentation may be torch.Tensor but we continue with numpy
    if isinstance(segmentation_reverted_cropping, torch.Tensor):
        segmentation_reverted_cropping = segmentation_reverted_cropping.cpu().numpy()

    # revert transpose
    segmentation_reverted_cropping = segmentation_reverted_cropping.transpose(plans_manager.transpose_backward)
    # revert cropping and transposition of uncertainty masks
    jsd = label_manager.revert_cropping_on_uncertainties(jsd,
                                                         properties_dict['bbox_used_for_cropping'],
                                                         properties_dict['shape_before_cropping'])
    jsd = jsd.cpu().numpy()
    
    jsd = jsd.transpose([0] + [i + 1 for i in plans_manager.transpose_backward])
    mean_entropy = label_manager.revert_cropping_on_uncertainties(mean_entropy,
                                                         properties_dict['bbox_used_for_cropping'],
                                                         properties_dict['shape_before_cropping'])
    mean_entropy = mean_entropy.cpu().numpy()
    mean_entropy = mean_entropy.transpose([0] + [i + 1 for i in plans_manager.transpose_backward])
    if return_probabilities:
        # revert cropping
        predicted_probabilities = label_manager.revert_cropping_on_probabilities(predicted_probabilities,
                                                                                 properties_dict[
                                                                                     'bbox_used_for_cropping'],
                                                                                 properties_dict[
                                                                                     'shape_before_cropping'])
        predicted_probabilities = predicted_probabilities.cpu().numpy()
        # revert transpose
        predicted_probabilities = predicted_probabilities.transpose([0] + [i + 1 for i in
                                                                           plans_manager.transpose_backward])
        torch.set_num_threads(old_threads)
        return segmentation_reverted_cropping, predicted_probabilities, jsd, mean_entropy
    else:
        torch.set_num_threads(old_threads)
        return segmentation_reverted_cropping, jsd, mean_entropy

def export_prediction_from_logits(predicted_array_or_file: Union[np.ndarray, torch.Tensor], properties_dict: dict,
                                  configuration_manager: ConfigurationManager,
                                  plans_manager: PlansManager,
                                  dataset_json_dict_or_file: Union[dict, str], output_file_truncated: str,
                                  save_probabilities: bool = False,
                                  compute_entropy: bool = False,
                                  num_threads_torch: int = default_num_processes):
    # if isinstance(predicted_array_or_file, str):
    #     tmp = deepcopy(predicted_array_or_file)
    #     if predicted_array_or_file.endswith('.npy'):
    #         predicted_array_or_file = np.load(predicted_array_or_file)
    #     elif predicted_array_or_file.endswith('.npz'):
    #         predicted_array_or_file = np.load(predicted_array_or_file)['softmax']
    #     os.remove(tmp)

    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
    ret = convert_predicted_logits_fold_to_segmentation_with_correct_shape(
        predicted_array_or_file, plans_manager, configuration_manager, label_manager, properties_dict,
        return_probabilities=save_probabilities, num_threads_torch=num_threads_torch
    )
    del predicted_array_or_file
    
    # save
    if save_probabilities:
        segmentation_final, probabilities_final, jsd_final, mean_entropy_final = ret  
        np.savez_compressed(output_file_truncated + '.npz', probabilities=probabilities_final)
        save_pickle(properties_dict, output_file_truncated + '.pkl')
        del probabilities_final, ret
    else:
        segmentation_final, jsd_final, mean_entropy_final = ret
        del ret

    rw = plans_manager.image_reader_writer_class()
    
    jsd_final = np.squeeze(jsd_final, axis=0)
    mean_entropy_final = np.squeeze(mean_entropy_final, axis=0)
    rw.write_seg(jsd_final, output_file_truncated + '_epistemic_uncertainty' + dataset_json_dict_or_file['file_ending'],
                 properties_dict)
    rw.write_seg(mean_entropy_final, output_file_truncated + '_aleatoric_uncertainty' + dataset_json_dict_or_file['file_ending'],
                 properties_dict)
    rw.write_seg(segmentation_final, output_file_truncated + dataset_json_dict_or_file['file_ending'],
                 properties_dict)


def entropy_torch(p: torch.Tensor, dim: int = 1, eps: float = 1e-12) -> torch.Tensor:
    """
    Compute entropy along `dim` using base-2 logarithm.
    
    Args:
        p (torch.Tensor): Tensor of probabilities (non-negative, summing to 1 along `dim`)
        dim (int): Dimension along which to compute entropy (usually class dimension)
        eps (float): Small value to prevent log(0)
    
    Returns:
        torch.Tensor: Entropy values with `dim` reduced
    """
    p = torch.clamp(p, eps, 1.0)
    return -(p * torch.log2(p)).sum(dim=dim)


def jensen_shannon_divergence_torch(softmax_folds: torch.Tensor) -> torch.Tensor:
    """
    Compute voxel-wise Jensen–Shannon divergence for an ensemble of softmax outputs.

    Args:
        softmax_folds (torch.Tensor): Tensor of shape (F, C, Z, X, Y)
            F = number of folds
            C = number of classes
            Z, X, Y = spatial dimensions

    Returns:
        torch.Tensor: Jensen–Shannon divergence map of shape (Z, X, Y)
    """
    print('starting uncertainty calculation')
    # Mean probability across folds: (C, Z, X, Y)
    mean_probs = softmax_folds.mean(dim=0)

    # Entropy over mean probabilities
    entropy_mean = entropy_torch(mean_probs, dim=0)  # (Z, X, Y) total uncertainty

    # Entropy per fold: (F, Z, X, Y)
    entropy_per_fold = entropy_torch(softmax_folds, dim=1)

    # Mean entropy across folds: (Z, X, Y)
    mean_entropy = entropy_per_fold.mean(dim=0) # aleatoric uncertainty

    # Need to add a dimension for cropping later on
    entropy_mean = entropy_mean.unsqueeze(0)
    mean_entropy = mean_entropy.unsqueeze(0)
    
    # Jensen–Shannon divergence (in bits)
    jsd = entropy_mean - mean_entropy # epistemic uncertainty

    # print('JSD check')
    # print(f'jsd {torch.mean(jsd)}, {torch.unique(jsd)}')
    # print(f'mean_entropy {torch.mean(mean_entropy)}, {torch.unique(mean_entropy)}')

    return jsd, entropy_mean, mean_entropy


class UncertaintyPredictor(nnUNetPredictor):
    @torch.inference_mode()
    def predict_logits_from_preprocessed_data(self, data: torch.Tensor) -> torch.Tensor:
        """
        IMPORTANT! IF YOU ARE RUNNING THE CASCADE, THE SEGMENTATION FROM THE PREVIOUS STAGE MUST ALREADY BE STACKED ON
        TOP OF THE IMAGE AS ONE-HOT REPRESENTATION! SEE PreprocessAdapter ON HOW THIS SHOULD BE DONE!

        RETURNED LOGITS HAVE THE SHAPE OF THE INPUT. THEY MUST BE CONVERTED BACK TO THE ORIGINAL IMAGE SIZE.
        SEE convert_predicted_logits_to_segmentation_with_correct_shape
        """
        n_threads = torch.get_num_threads()
        torch.set_num_threads(default_num_processes if default_num_processes < n_threads else n_threads)
        # prediction = None
        prediction = []

        for fold_idx, params in enumerate(self.list_of_parameters):

            # messing with state dict names...
            if not isinstance(self.network, OptimizedModule):
                self.network.load_state_dict(params)
            else:
                self.network._orig_mod.load_state_dict(params)

            # why not leave prediction on device if perform_everything_on_device? Because this may cause the
            # second iteration to crash due to OOM. Grabbing that with try except cause way more bloated code than
            # this actually saves computation time
            pred = self.predict_sliding_window_return_logits(data).to('cpu')
            prediction.append(pred)
            
        # print(type(pred))
	
        if len(self.list_of_parameters) > 1:
            # we average over the different folds in logit space, just
            # like the standard nnU-Net pipeline
            fold_pred_tensor = torch.stack(prediction, dim=0)
            prediction = torch.mean(fold_pred_tensor, dim=0)
            # prediction /= len(self.list_of_parameters) # this is the standard nnU-Net implementation
            
        # print('IS HE SOFTMAXXING?')
        # print(torch.max(prediction))
        # print(torch.min(prediction))

        if self.verbose: print('Prediction done')
        torch.set_num_threads(n_threads)
        return fold_pred_tensor
        
    def predict_from_data_iterator(self,
                                   data_iterator,
                                   save_probabilities: bool = False,
                                   num_processes_segmentation_export: int = default_num_processes):
        """
        each element returned by data_iterator must be a dict with 'data', 'ofile' and 'data_properties' keys!
        If 'ofile' is None, the result will be returned instead of written to a file
        """
        with multiprocessing.get_context("spawn").Pool(num_processes_segmentation_export) as export_pool:
            worker_list = [i for i in export_pool._pool]
            r = []
            for preprocessed in data_iterator:
                data = preprocessed['data']
                if isinstance(data, str):
                    delfile = data
                    data = torch.from_numpy(np.load(data))
                    os.remove(delfile)

                ofile = preprocessed['ofile']
                if ofile is not None:
                    print(f'\nPredicting {os.path.basename(ofile)}:')
                else:
                    print(f'\nPredicting image of shape {data.shape}:')

                print(f'perform_everything_on_device: {self.perform_everything_on_device}')

                properties = preprocessed['data_properties']

                # let's not get into a runaway situation where the GPU predicts so fast that the disk has to be swamped with
                # npy files
                proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)

                # convert to numpy to prevent uncatchable memory alignment errors from multiprocessing serialization of torch tensors
                prediction = self.predict_logits_from_preprocessed_data(data).cpu().detach().numpy()
                # print('DATA ITERATOR')
                # print(type(prediction))
                # print(prediction.shape)

                if ofile is not None:
                    print('sending off prediction to background worker for resampling and export')
                    r.append(
                        export_pool.starmap_async(
                            export_prediction_from_logits,
                            ((prediction, properties, self.configuration_manager, self.plans_manager,
                              self.dataset_json, ofile, save_probabilities),)
                        )
                    )
                else:
                    print('sending off prediction to background worker for resampling')
                    r.append(
                        export_pool.starmap_async(
                            convert_predicted_logits_fold_to_segmentation_with_correct_shape, (
                                (prediction, self.plans_manager,
                                 self.configuration_manager, self.label_manager,
                                 properties,
                                 save_probabilities),)
                        )
                    )
                if ofile is not None:
                    print(f'done with {os.path.basename(ofile)}')
                else:
                    print(f'\nDone with image of shape {data.shape}:')
            ret = [i.get()[0] for i in r]

        if isinstance(data_iterator, MultiThreadedAugmenter):
            data_iterator._finish()

        # clear lru cache
        compute_gaussian.cache_clear()
        # clear device cache
        empty_cache(self.device)
        return ret

        
if __name__ == '__main__':
    ########################## predict a bunch of files
    from nnunetv2.paths import nnUNet_results, nnUNet_raw

    predictor = UncertaintyPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    predictor.initialize_from_trained_model_folder(
        join(nnUNet_results, 'Dataset514_MalePelvis_MRL/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres'),
        use_folds=(0,1),
        checkpoint_name='checkpoint_final.pth',
    )


    ## we need to create list of list of input files
    #input_caseids = subdirs(args.input_dir, join=False)
    #input_files = [[join(args.input_dir, i, 'ct.nii.gz')] for i in input_caseids]
    #output_folders = [join(args.output_dir, i) for i in input_caseids]

    predictor.predict_from_files(
        '/home/ai/fastshare/PersonalStorage/Maximilian/Nifti_Female_Pelvis/',
        '/home/ai/fastshare/PersonalStorage/Maximilian/Nifti_Female_Pelvis/output_OUH_predictor/',
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=3,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0
    )

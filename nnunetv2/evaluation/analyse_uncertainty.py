import argparse
import os.path
from pathlib import Path
import torch
import numpy as np
from copy import deepcopy
from typing import Union, List, Tuple

from batchgenerators.utilities.file_and_folder_operations import subfiles, load_json, join, isfile, maybe_mkdir_p, isdir, subdirs, \
    save_json
from nnunetv2.configuration import default_num_processes
from nnunetv2.evaluation.accumulate_cv_results import accumulate_cv_results
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder, load_summary_json
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw, nnUNet_results
from nnunetv2.postprocessing.remove_connected_components import determine_postprocessing
from nnunetv2.utilities.file_path_utilities import maybe_convert_to_dataset_name, get_output_folder, \
    convert_identifier_to_trainer_plans_config, get_ensemble_name, folds_tuple_to_string
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.inference.OUH_inference import UncertaintyPredictor
        
from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy, preprocessing_iterator_fromfiles, \
    preprocessing_iterator_fromnpy

default_trained_model = tuple([
    {'plans': 'nnUNetPlans', 'configuration': '3d_fullres', 'trainer': 'nnUNetTrainer'},
])


def filter_available_models(model_dict: Union[List[dict], Tuple[dict, ...]], dataset_name_or_id: Union[str, int]):
    valid = []
    for trained_model in model_dict:
        plans_manager = PlansManager(join(nnUNet_preprocessed, maybe_convert_to_dataset_name(dataset_name_or_id),
                               trained_model['plans'] + '.json'))
        # check if configuration exists
        # 3d_cascade_fullres and 3d_lowres do not exist for each dataset so we allow them to be absent IF they are not
        # specified in the plans file
        if trained_model['configuration'] not in plans_manager.available_configurations:
            print(f"Configuration {trained_model['configuration']} not found in plans {trained_model['plans']}.\n"
                  f"Inferred plans file: {join(nnUNet_preprocessed, maybe_convert_to_dataset_name(dataset_name_or_id), trained_model['plans'] + '.json')}.")
            continue

        # check if trained model output folder exists. This is a requirement. No mercy here.
        expected_output_folder = get_output_folder(dataset_name_or_id, trained_model['trainer'], trained_model['plans'],
                                                   trained_model['configuration'], fold=None)
        if not isdir(expected_output_folder):
            raise RuntimeError(f"Trained model {trained_model} does not have an output folder. "
                  f"Expected: {expected_output_folder}. Please run the training for this model! (don't forget "
                  f"the --npz flag if you want to ensemble multiple configurations)")

        valid.append(trained_model)
    return valid
    
def generate_inference_command(dataset_name_or_id: Union[int, str], configuration_name: str,
                               model_folder: str, plans_identifier: str = 'nnUNetPlans', trainer_name: str = 'nnUNetTrainer',
                               folds: Union[List[int], Tuple[int, ...]] = (0, 1, 2, 3, 4),
                               checkpoint_name: str = 'checkpoint_final.pth',
                               input_folder: str = 'INPUT_FOLDER',
                               output_folder: str = 'OUTPUT_FOLDER',
                               save_npz: bool = False):
    plan_and_preprocess = f'nnUNetv2_plan_and_preprocess -d FINETUNING_DATASET'
    extract_fingerprint = f'nnUNetv2_extract_fingerprint -d {dataset_name_or_id}'
    move_plans_between_datasets = f'nnUNetv2_move_plans_between_datasets -s FINETUNING_DATASET -t {dataset_name_or_id} -sp {plans_identifier} -tp {plans_identifier}_finetuning'
    finetune_training = f'nnUNetv2_train FINETUNING_DATASET {configuration_name} {folds[0]}...{folds[-1]} -pretrained_weights {model_folder}/{checkpoint_name}'
    return f'\n***FOR FINETUNING FOLLOW THIS PIPELINE***:\n{plan_and_preprocess}\n{extract_fingerprint}\n{move_plans_between_datasets}\n{finetune_training}'


def analyse_uncertainty(dataset_name_or_id,
                            list_of_lists_or_source_folder: Union[str, List[List[str]]],
                            output_folder: Union[str, None],
                            trainer: str= 'nnUNetTrainer',
                            config: str= '3d_fullres',
                            plans: str = 'nnUNetPlans',
                            checkpoint_name: str = 'checkpoint_final.pth',
                            uncertainty_type: str = 't',
                            num_processes: int = default_num_processes,
                            device: torch.device = torch.device('cuda'),
                            folds: Union[List[int], Tuple[int, ...]] = (0, 1, 2, 3, 4),
                            tile_step_size: float = 0.5,
                            strict: bool = False,
                            verbose: bool = False,
                            tqdm: bool = False):
    dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
    all_results = {}

    model_folder = get_output_folder(dataset_name_or_id, trainer, plans, config, fold=None)
    if not isdir(output_folder) and strict:
        raise RuntimeError(f'{dataset_name}: The model folder of plans {m["plans"]} configuration '
                               f'{m["configuration"]} is missing. Please train the model (all requested folds!) first!')
    uncertainty_results = {}
    predictor = UncertaintyPredictor(uncertainty_type=uncertainty_type,
                                tile_step_size=tile_step_size,
                                use_gaussian=True,
	                        use_mirroring=False,
	                        perform_everything_on_device=True,
	                        device=device,
	                        verbose=verbose,
	                        verbose_preprocessing=verbose,
	                        allow_tqdm=not tqdm)
                                  
    predictor.initialize_from_trained_model_folder(
       model_folder,
       folds,
       checkpoint_name=checkpoint_name
    )
    predictor.predict_from_files(list_of_lists_or_source_folder, output_folder)

    # evaluate ensembled predictions
    plans_manager = PlansManager(join(model_folder, 'plans.json'))
    dataset_json = load_json(join(model_folder, 'dataset.json'))
    label_manager = plans_manager.get_label_manager(dataset_json)
    rw = plans_manager.image_reader_writer_class()
        
    files_tu = subfiles(output_folder, suffix='total_uncertainty.nii.gz', join=False)
    files_tu = [join(output_folder, i) for i in files_tu]
        
    files_au = subfiles(output_folder, suffix='aleatoric_uncertainty.nii.gz', join=False)
    files_au = [join(output_folder, i) for i in files_au]
        
    files_eu = subfiles(output_folder, suffix='epistemic_uncertainty.nii.gz', join=False)
    files_eu = [join(output_folder, i) for i in files_eu]
        
    for (file_tu, file_eu, file_au) in zip(files_tu, files_eu, files_au):
        tu_ref, tu_ref_dict = rw.read_seg(file_tu)
        eu_ref, eu_ref_dict = rw.read_seg(file_eu)
        au_ref, au_ref_dict = rw.read_seg(file_au)

        filename = Path(file_tu).stem
        identifier = filename.split("_0_total_uncertainty")[0]
        uncertainty_results[identifier] = \
            {
            'total_uncertainty': float(np.sum(tu_ref)),
            'epistemic_uncertainty': float(np.sum(eu_ref)),
            'aleatoric_uncertainty': float(np.sum(au_ref))
            }
    save_json(uncertainty_results, join(output_folder, 'uncertainty_summary.json'))

    ranks_tu = rank_uncertainties("total_uncertainty", uncertainty_results)
    ranks_eu = rank_uncertainties("epistemic_uncertainty", uncertainty_results)
    ranks_au = rank_uncertainties("aleatoric_uncertainty", uncertainty_results)
        
    sorted_cases = sorted(
        uncertainty_results.items(),
        key=lambda x: x[1]["total_uncertainty"],
        reverse=True
    )
    print_uncertainty_ranking(sorted_cases, ranks_tu, ranks_eu, ranks_au)
    print(generate_inference_command(dataset_name_or_id, config, model_folder,
                           plans, trainer, folds, checkpoint_name))


def rank_uncertainties(metric, uncertainty_results):
    sorted_items = sorted(uncertainty_results.items(), key=lambda x: x[1][metric], reverse=True)
    return {case: i + 1 for i, (case, _) in enumerate(sorted_items)}

def print_uncertainty_ranking(sorted_cases, ranks_tu, ranks_eu, ranks_au):
    header = (
        "Case".ljust(35)
        + "| Total Uncertainty".ljust(27)
        + "| Epistemic Uncertainty".ljust(30)
        + "| Aleatoric Uncertainty"
    )
    line = "-" * len(header)

    print(header)
    print(line)
    for case, vals in sorted_cases:
        short_name = case[:25] + "..." + case[-5:]
        print(
            f"{short_name.ljust(35)}"
            + f"| #{ranks_tu[case]:<2} ({vals['total_uncertainty']:.1f})".ljust(27)
            + f"| #{ranks_eu[case]:<2} ({vals['epistemic_uncertainty']:.1f})".ljust(30)
            + f"| #{ranks_au[case]:<2} ({vals['aleatoric_uncertainty']:.1f})"
        )

def _internal_get_data_iterator_from_lists_of_filenames(self,
                                                        input_list_of_lists: List[List[str]],
                                                        seg_from_prev_stage_files: Union[List[str], None],
                                                        output_filenames_truncated: Union[List[str], None],
                                                        num_processes: int):
    return preprocessing_iterator_fromfiles(input_list_of_lists, seg_from_prev_stage_files,
                                            output_filenames_truncated, self.plans_manager, self.dataset_json,
                                            self.configuration_manager, num_processes, self.device.type == 'cuda',
                                            self.verbose_preprocessing)
    
def _getDefaultValue(env: str, dtype: type, default: any,) -> any:
    try:
        val = dtype(os.environ.get(env) or default)
    except:
        val = default
    return val


def analyse_uncertainty_entry_point():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, required=True,
                        help='input folder. Remember to use the correct channel numberings for your files (_0000 etc). '
                             'File endings must be the same as the training dataset!')
    parser.add_argument('-o', type=str, required=True,
                        help='Output folder. If it does not exist it will be created. Predicted segmentations will '
                             'have the same name as their source images.')
    parser.add_argument('-d', type=str, required=True,
                        help='Dataset with which you would like to predict. You can specify either dataset name or id')
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans',
                        help='Plans identifier. Specify the plans in which the desired configuration is located. '
                             'Default: nnUNetPlans')
    parser.add_argument('-tr', type=str, required=False, default='nnUNetTrainer',
                        help='What nnU-Net trainer class was used for training? Default: nnUNetTrainer')
    parser.add_argument('-c', type=str, required=True,
                        help='nnU-Net configuration that should be used for prediction. Config must be located '
                             'in the plans specified with -p')
    parser.add_argument('-f', nargs='+', type=str, required=False, default=(0, 1, 2, 3, 4),
                        help='Specify the folds of the trained model that should be used for prediction. '
                             'Default: (0, 1, 2, 3, 4)')
    parser.add_argument('-u', type=str, required=False, default=None, help='Define if you want to generate uncertainty maps.'
                             'Uncertainty maps are based on entropy and jensen-shannon divergence. Default: None. '
                             'Options: t (total) or c (channelwise)')
    parser.add_argument('-step_size', type=float, required=False, default=0.5,
                        help='Step size for sliding window prediction. The larger it is the faster but less accurate '
                             'the prediction. Default: 0.5. Cannot be larger than 1. We recommend the default.')
    parser.add_argument('--disable_tta', action='store_true', required=False, default=False,
                        help='Set this flag to disable test time data augmentation in the form of mirroring. Faster, '
                             'but less accurate inference. Not recommended.')
    parser.add_argument('--verbose', action='store_true', help="Set this if you like being talked to. You will have "
                                                               "to be a good listener/reader.")
    parser.add_argument('--save_probabilities', action='store_true',
                        help='Set this to export predicted class "probabilities". Required if you want to ensemble '
                             'multiple configurations.')
    parser.add_argument('--continue_prediction', action='store_true',
                        help='Continue an aborted previous prediction (will not overwrite existing files)')
    parser.add_argument('-chk', type=str, required=False, default='checkpoint_final.pth',
                        help='Name of the checkpoint you want to use. Default: checkpoint_final.pth')
    parser.add_argument('-npp', type=int, required=False, default=_getDefaultValue('nnUNet_npp', int, 3),
                        help='Number of processes used for preprocessing. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-nps', type=int, required=False, default=_getDefaultValue('nnUNet_nps', int, 3),
                        help='Number of processes used for segmentation export. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-prev_stage_predictions', type=str, required=False, default=None,
                        help='Folder containing the predictions of the previous stage. Required for cascaded models.')
    parser.add_argument('-num_parts', type=int, required=False, default=1,
                        help='Number of separate nnUNetv2_predict call that you will be making. Default: 1 (= this one '
                             'call predicts everything)')
    parser.add_argument('-part_id', type=int, required=False, default=0,
                        help='If multiple nnUNetv2_predict exist, which one is this? IDs start with 0 can end with '
                             'num_parts - 1. So when you submit 5 nnUNetv2_predict calls you need to set -num_parts '
                             '5 and use -part_id 0, 1, 2, 3 and 4. Simple, right? Note: You are yourself responsible '
                             'to make these run on separate GPUs! Use CUDA_VISIBLE_DEVICES (google, yo!)')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                        help="Use this to set the device the inference should run with. Available options are 'cuda' "
                             "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
                             "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_predict [...] instead!")
    parser.add_argument('--disable_progress_bar', action='store_true', required=False, default=False,
                        help='Set this flag to disable progress bar. Recommended for HPC environments (non interactive '
                             'jobs)')
    args = parser.parse_args()
    
    if not isdir(args.o):
        maybe_mkdir_p(args.o)

    dataset_name = maybe_convert_to_dataset_name(args.d)
    
    assert args.device in ['cpu', 'cuda','mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}.'
    if args.device == 'cpu':
        # let's allow torch to use hella threads
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    analyse_uncertainty(dataset_name, args.i, args.o, args.tr, args.c, args.p, args.chk, args.u, device=device,
                            num_processes=args.npp, folds=args.f, tile_step_size=args.step_size,
                            strict=False, verbose=args.verbose, tqdm=args.disable_progress_bar)


def accumulate_crossval_results_entry_point():
    parser = argparse.ArgumentParser('Copies all predicted segmentations from the individual folds into one joint '
                                     'folder and evaluates them')
    parser.add_argument('dataset_name_or_id', type=str, help='Dataset Name or id')
    parser.add_argument('-c', type=str, required=True,
                        default='3d_fullres',
                        help="Configuration")
    parser.add_argument('-o', type=str, required=False, default=None,
                        help="Output folder. If not specified, the output folder will be located in the trained " \
                             "model directory (named crossval_results_folds_XXX).")
    parser.add_argument('-f', nargs='+', type=int, default=(0, 1, 2, 3, 4),
                        help='Folds to use. Default: 0 1 2 3 4')
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans',
                        help='Plan identifier in which to search for the specified configuration. Default: nnUNetPlans')
    parser.add_argument('-tr', type=str, required=False, default='nnUNetTrainer',
                        help='Trainer class. Default: nnUNetTrainer')
    args = parser.parse_args()
    trained_model_folder = get_output_folder(args.dataset_name_or_id, args.tr, args.p, args.c)

    if args.o is None:
        merged_output_folder = join(trained_model_folder, f'crossval_results_folds_{folds_tuple_to_string(args.f)}')
    else:
        merged_output_folder = args.o
        if isdir(merged_output_folder) and len(listdir(merged_output_folder)) > 0:
            raise FileExistsError(
                f"Output folder {merged_output_folder} exists and is not empty. "
                f"To avoid data loss, nnUNet requires an empty output folder."
            )

    accumulate_cv_results(trained_model_folder, merged_output_folder, args.f)


if __name__ == '__main__':
    find_best_configuration(4,
                            default_trained_models,
                            True,
                            8,
                            False,
                            (0, 1, 2, 3, 4))

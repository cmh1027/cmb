
import nnunet
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.experiment_planning.DatasetAnalyzer import DatasetAnalyzer
from nnunet.experiment_planning.utils import crop
from nnunet.paths import *
import shutil
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
from nnunet.preprocessing.sanity_checks import verify_dataset_integrity
from nnunet.training.model_restore import recursive_find_python_class
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.run.default_configuration import get_default_configuration
from nnunet.paths import default_plans_identifier
from nnunet.run.load_pretrained_weights import load_pretrained_weights
from nnunet.training.cascade_stuff.predict_next_stage import predict_next_stage
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.training.network_training.nnUNetTrainerCascadeFullRes import nnUNetTrainerCascadeFullRes
from nnunet.training.network_training.nnUNetTrainerV2_CascadeFullRes import nnUNetTrainerV2CascadeFullRes
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--task")
parser.add_argument("-sc", "--skip_crop", action='store_true')
parser.add_argument("-sp", "--skip_preprocess", action='store_true')
parser.add_argument("-s", "--source_task", required=True)
parser.add_argument("--max_epoch", default=1000, type=int)
parser.add_argument("-cb", "--checkpoint_base", required=True)
parser.add_argument("-scb", "--source_checkpoint_base", required=True)
args = parser.parse_args()

TASK = args.task

if not args.skip_crop:
    assert not args.skip_preprocess
    thread_count = 50
    #### try one time ####
    verify_dataset_integrity(join(nnUNet_raw_data, TASK))
    crop(TASK, False, thread_count)
    ######################
if not args.skip_preprocess:
    search_in = join(nnunet.__path__[0], "experiment_planning")
    planner_name3d = 'ExperimentPlanner3D_v21'
    planner_3d = recursive_find_python_class([search_in], planner_name3d, current_module="nnunet.experiment_planning")
    cropped_out_dir = os.path.join(nnUNet_cropped_data, TASK)
    preprocessing_output_dir_this_task = os.path.join(preprocessing_output_dir, TASK)
    dataset_json = load_json(join(cropped_out_dir, 'dataset.json'))
    modalities = list(dataset_json["modality"].values())
    collect_intensityproperties = True if (("CT" in modalities) or ("ct" in modalities)) else False
    dataset_analyzer = DatasetAnalyzer(cropped_out_dir, overwrite=False, num_processes=1)
    _ = dataset_analyzer.analyze_dataset(collect_intensityproperties)

    maybe_mkdir_p(preprocessing_output_dir_this_task)
    shutil.copy(join(cropped_out_dir, "dataset_properties.pkl"), preprocessing_output_dir_this_task)
    shutil.copy(join(nnUNet_raw_data, TASK, "dataset.json"), preprocessing_output_dir_this_task)
    threads = (50, 50)
    exp_planner = planner_3d(cropped_out_dir, preprocessing_output_dir_this_task)
    exp_planner.plan_experiment()
    exp_planner.run_preprocessing(threads)


network = '3d_fullres'
network_trainer = "nnUNetTrainer"
plans_identifier = default_plans_identifier
fold = 0
use_compressed_data = False
decompress_data = not use_compressed_data
deterministic = False
fp32 = False
run_mixed_precision = not fp32
validation_only = False
checkpoint_base = args.checkpoint_base

plans_file, output_folder_name, dataset_directory, batch_dice, stage, \
    trainer_class = get_default_configuration(network, TASK, network_trainer, plans_identifier, plan_task=args.source_task)
assert issubclass(trainer_class, nnUNetTrainer), "network_trainer was found but is not derived from nnUNetTrainer"

trainer = trainer_class(plans_file, fold, output_folder=output_folder_name, dataset_directory=dataset_directory, checkpoint_base=checkpoint_base,
                            batch_dice=batch_dice, stage=stage, unpack_data=decompress_data,
                            deterministic=deterministic,
                            fp16=run_mixed_precision)
trainer.initialize(not validation_only)


ckpt = os.path.join(f"/data000/mhchoi/CMB/RESULTS_FOLDER/nnUNet/3d_fullres/{args.source_task}/nnUNetTrainer__nnUNetPlansv2.1/fold_0", args.source_checkpoint_base, "model_best.model")
trainer.load_checkpoint(ckpt)

print("Max epoch is", args.max_epoch)
trainer.max_num_epochs = args.max_epoch

trainer.run_training()

#!/bin/bash

echo "Extract keypoints and descriptors"
python SuperGlueMatching/extract_descriptors_api.py --matterport_dataset_folder $1 \
--matterport_descriptors_outputdir './data/scan/descriptors/'

echo "Extract image descriptor and query"
python SuperGlueMatching/visual_database_api.py --matterport_descriptors_folder $1/descriptors/ \
--matterport_output_folder $1/descriptors/ \
--azure_dataset_folder $2

echo "Match query and database images"
python SuperGlueMatching/match_pairs_api.py \
--input_pairs $2/vlad_best_match/queries.pkl \
--starting_index 0 --ending_index -1 \
--output_dir $2/superglue_match_results/

echo "Obtain poses with PnP"
python pnp_api.py \
--azure_dataset_folder $2 \
--matterport_descriptors_folder $1/descriptors/ \
--output_dir $2/poses_reloc/
#!/bin/bash

echo "Extract image descriptor"
python SuperGlueMatching/extract_descriptors_api.py \
--matterport_dataset_folder $1 \
--matterport_descriptors_outputdir $1/descriptors

echo "Undistort image"
python undistort_image_api.py \
--ego_dataset_folder $2 --crop_x 300 --crop_y 300

echo "Extract image descriptor and query"
python visual_database_api.py \
--matterport_descriptors_folder $1/descriptors/ \
--matterport_output_folder $1/descriptors/ \
--ego_dataset_folder $2

echo "Match query and database images"
python SuperGlueMatching/match_pairs_api.py \
--input_pairs $2/vlad_best_match/queries.pkl \
--starting_index 0 --ending_index 0 \
--output_dir $2/superglue_match_results/

echo "Obtain poses with PnP"
python pnp_api.py \
--ego_dataset_folder $2 \
--matterport_descriptors_folder $1/descriptors/ \
--output_dir $2/poses_reloc/

echo "Extract temporal constraints by matching pairwise images"
python build_feature_track_api.py \
--ego_dataset_folder $2 \
--extract_descriptor

echo "Incremental unordered sfm"
python sfm_api.py \
--ego_dataset_folder $2

echo "Image overlaid visualization"
python Visualization/visualize_render_images.py \
--matterport_dataset_folder $1 \
--ego_dataset_folder $2
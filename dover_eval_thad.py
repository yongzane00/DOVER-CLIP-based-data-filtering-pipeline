# Modifying evaluate_a_set_of_videos into a callable function instead of having to run it from CLI

import torch
import argparse
import os
import pickle as pkl
import decord
import numpy as np
import yaml
import pandas
from pathlib import Path
from tqdm import tqdm

from dover.datasets import (
    UnifiedFrameSampler,
    ViewDecompositionDataset,
    spatial_temporal_view_decomposition,
)
from dover.models import DOVER

def fuse_results(results: list):
    ## results[0]: aesthetic, results[1]: technical
    ## thank @dknyxh for raising the issue
    t, a = (results[1] - 0.1107) / 0.07355, (results[0] + 0.08285) / 0.03774
    x = t * 0.6104 + a * 0.3896
    return {
        "aesthetic": 1 / (1 + np.exp(-a)),
        "technical": 1 / (1 + np.exp(-t)),
        "overall": 1 / (1 + np.exp(-x)),
    }

# To be quite honest I don't really know what I'm doing
# The main idea is that I change the original script to be a callable function from Jupyter notebook
# Let's see how that goes

# Changes:
#   Instead of passing in an ArgumentParser and parsing it within the function, just take a tuple of arguments in instead (should be exactly equivalent)
#   Removed some print statements
#   Return a DF with url, aesthetic score, technical score, overall score (weighted using their method)

def evaluate_videos(args: tuple):

    # Create lists to store video paths, aesthetic, technical, and overall scores
    vid_paths = []
    aesthetic_scores = []
    technical_scores = []
    overall_scores = []

    with open(args.opt, "r") as f:
        opt = yaml.safe_load(f)

    ### Load DOVER
    evaluator = DOVER(**opt["model"]["args"]).to(args.device)
    evaluator.load_state_dict(
        torch.load(opt["test_load_path"], map_location=args.device)
    )

    all_results = {}

    with open(args.output_result_csv, "w") as w:
        w.write(f"path, aesthetic score, technical score, overall/final score\n")

    dopt = opt["data"]["val-l1080p"]["args"]

    dopt["anno_file"] = None
    dopt["data_prefix"] = args.input_video_dir

    dataset = ViewDecompositionDataset(dopt)
    print("Dataset Length: ", dataset.__len__(), " files found.")

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=0,  pin_memory=True, # Override number of workers to 0, see if that prevents the hanging issue
    )

    sample_types = ["aesthetic", "technical"]

    for i, data in enumerate(tqdm(dataloader, desc="Testing")):

        error_id = 0

        if len(data.keys()) == 1: # Exception should return 1 item only
            error_id += 1
            print('Error #' + str(error_id) + ' of ' + str(i) + ' files in file: ' + str(data["name"][0]))
            continue

        current_fname = Path(data["name"][0]).stem
        # print(current_fname + "\n")

        video = {}
        for key in sample_types:
            if key in data:
                video[key] = data[key].to(args.device)
                b, c, t, h, w = video[key].shape
                video[key] = (
                    video[key]
                    .reshape(
                        b, c, data["num_clips"][key], t // data["num_clips"][key], h, w
                    )
                    .permute(0, 2, 1, 3, 4, 5)
                    .reshape(
                        b * data["num_clips"][key], c, t // data["num_clips"][key], h, w
                    )
                )

        with torch.no_grad():
            results = evaluator(video, reduce_scores=False)
            results = [np.mean(l.cpu().numpy()) for l in results]

        rescaled_results = fuse_results(results)
        
        with open("./DOVER/zero_shot_res_sensehdr.txt","a") as wf:
            wf.write(f'{current_fname},{rescaled_results["aesthetic"]*100:4f}, {rescaled_results["technical"]*100:4f},{rescaled_results["overall"]*100:4f}\n')

        with open(args.output_result_csv, "a") as w:
            w.write(
                f'{current_fname}, {rescaled_results["aesthetic"]*100:4f}, {rescaled_results["technical"]*100:4f},{rescaled_results["overall"]*100:4f}\n'
            )
        
        # Append results to list
        vid_paths.append(current_fname)
        aesthetic_scores.append(rescaled_results["aesthetic"]*100)
        technical_scores.append(rescaled_results["technical"]*100)
        overall_scores.append(rescaled_results["overall"]*100)
                              
    # At the very end, create a dataframe with results and return it
    results = pandas.DataFrame({'URL' : vid_paths, 'Aesthetic' : aesthetic_scores, 'Technical' : technical_scores, 'Overall' : overall_scores})
    return results

        

    


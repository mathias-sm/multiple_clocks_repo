#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 16:56:05 2025

@author: Svenja Küchenhoff

take the relevant entries of the behavioural table and only store those that
are actually required in the remaining analysis.

"""

import pandas as pd
import numpy as np
import os
import ast
import fire


def transform_coord_x(x):
    map_transform_coord_x = {-0.21: 0.0, 0: 1.0, .21: 2.0}
    return map_transform_coord_x[x]

def transform_coord_y(y):
    map_transform_coord_y = {-0.29: 0.0, 0: 1.0, .29: 2.0}
    return map_transform_coord_y[y]

def delete_unnecessary_fields(og_df):
    # the first row is empty so delete to get indices right
    og_df = og_df.iloc[1:].reset_index(drop=True)

    # drop all unused columns for better readability.
    return og_df.drop(columns=['rep_runs.thisRepN', 'rep_runs.thisTrialN', 'rep_runs.thisN', 'rep_runs.thisIndex',
                               't_step_end_global', 'sand_box.started', 'sand_box.stopped', 'foot.started', 'foot.stopped',
                               'reward.started', 'reward.stopped', 'TR_key.keys', 'TR_key.rt', 'TR_key.started',
                               'nav_key_task.stopped', 'break_key.keys', 'break_key.started', 'break_key.stopped',
                               'progressbar_background.started', 'progressbar_background.stopped', 'progress_bar.started',
                               'progress_bar.stopped', 'reward_progress.started', 'reward_progress.stopped',
                               'plus_coin_txt.started', 'plus_coin_txt.stopped', 'reward_A_feedback.started',
                               'reward_A_feedback.stopped', 'TR_key.stopped', 'participant', 'date'])
    # 'Unnamed: 55']) # @Svenja: Why not remove this one as well?


def add_fields_I_want(df):
    # fill gaps in a few fields
    for field in ["round_no", "task_config", "repeat"]:
        df[field] = df[field].ffill()

    # so that I cann differenatiate task config and direction
    df['config_type'] = df['task_config'] + '_' + df['type']

    # per grid, the navigation keys are stored in the first column
    # thus, these can be used as indices of where a new grid starts.
    # there will be 10 different tasks per task half, with 5 repeats.
    indices_with_nav_keys = df[df['nav_key_task.started'].notna()].index.to_list()

    # loop to add a colum with nav_key presses that counted based on nav_key_task.rt and t_step_press_curr_run
    # and one with the actual keys they pressed based on nav_key_task.keys and t_step_press_curr_run
    # and one with keys presssed, but never executed.
    for grid_no, row_index in enumerate(indices_with_nav_keys):
        curr_list_of_keys = ast.literal_eval(df.at[row_index, 'nav_key_task.keys'])
        curr_key_times = ast.literal_eval(df.at[row_index, 'nav_key_task.rt'])
        count_error_keys = 0
        overall_error_counter = 0

        start = 0 if grid_no == 0 else indices_with_nav_keys[grid_no-1]+1
        for i_list,i in enumerate(range(start, indices_with_nav_keys[grid_no])):
            if grid_no == 0:
                # if the data stored a value smaller than t = 0, correct that
                if df.at[i, 't_step_press_curr_run'] < 0:
                    curr_key_times = np.insert(curr_key_times, 0, 0)
                    curr_list_of_keys = np.insert(curr_list_of_keys, 0, 0)
                    df.at[i, 't_step_press_curr_run'] = 0
            else:
                # for some sad reason, there are some (rare) glitches in the behavioural tables.
                # one glitch is that the first time of t_step_press_curr_run is shorter than 0
                if df.at[indices_with_nav_keys[grid_no-1]+1, 't_step_press_curr_run'] <= 0:
                    curr_key_times = np.insert(curr_key_times, 0, 0)
                    curr_list_of_keys = np.insert(curr_list_of_keys, 0, 0)
                    df.at[indices_with_nav_keys[grid_no-1]+1, 't_step_press_curr_run'] = 0
                # another glitch is that the first time of t_step_press_curr_run is even later than the last recorded press of this task
                if df.at[indices_with_nav_keys[grid_no-1]+1, 't_step_press_curr_run'] > df.at[indices_with_nav_keys[grid_no]-1, 't_step_press_curr_run']:
                    df.at[indices_with_nav_keys[grid_no-1]+1, 't_step_press_curr_run'] = curr_key_times[i_list]
                # another glitch is that there is a negative time somewhere in the middle of the task
                if df.at[i, 't_step_press_curr_run'] < 0:
                    df.at[i, 't_step_press_curr_run'] = curr_key_times[i_list]

            # then, test for what I am actually interested in:
            # which of the key presses was the recorded one?
            if np.isclose(df.at[i, 't_step_press_curr_run'], curr_key_times[i_list + overall_error_counter]):
                df.at[i, 'curr_key'] = curr_list_of_keys[i_list + overall_error_counter]
                df.at[i, 'curr_key_time'] = curr_key_times[i_list + overall_error_counter]
            else:
                wrong_keys = [str(curr_list_of_keys[i_list + overall_error_counter])]
                # Note: is there a reason for rounding this?
                wrong_times = [str(round(curr_key_times[i_list + overall_error_counter],4))]
                count_error_keys += 1
                overall_error_counter += 1

                while not np.isclose(df.at[i, 't_step_press_curr_run'], curr_key_times[i_list + overall_error_counter]):
                    wrong_keys.append(str(curr_list_of_keys[i_list + overall_error_counter]))
                    wrong_times.append(str(round(curr_key_times[i_list + overall_error_counter], 4)))
                    count_error_keys += 1
                    overall_error_counter +=1

                # if these columns don't exist yet, there will be an error if I try to fill with
                # several items. instead, first create with 0, then fill.
                df.at[i, 'non-exe_key_time'] = 0
                df.at[i, 'non-exe_key'] = 0

                # once back to a correct key, fill in the one that you missed previously
                df.at[i, 'non-exe_key'] = wrong_keys
                df.at[i, 'non-exe_key_time'] = wrong_times

                df.at[i, 'curr_key'] = curr_list_of_keys[i_list + overall_error_counter]
                df.at[i, 'curr_key_time'] = curr_key_times[i_list + overall_error_counter]
                df.at[i, 'non-exe_key_counter'] = count_error_keys
                count_error_keys = 0


    # Fix coordinates:
    df["curr_loc_y_coord"] = df["curr_loc_y"].apply(transform_coord_y)
    df["curr_loc_x_coord"] = df["curr_loc_x"].apply(transform_coord_x)
    df["curr_rew_y_coord"] = df["curr_rew_y"].apply(transform_coord_y)
    df["curr_rew_x_coord"] = df["curr_rew_x"].apply(transform_coord_x)

    # add columns whith field numbers
    for index, row in df.iterrows():
        # and prepare the regressors: config type, state and reward/walking specific.
        if not pd.isna(row['state']):
            if np.isnan(row['rew_loc_x']):
                df.at[index, 'time_bin_type'] = df.at[index, 'config_type'] + '_' + df.at[index, 'state'] + '_path'
            else:
                df.at[index, 'time_bin_type'] = df.at[index, 'config_type'] + '_' + df.at[index, 'state'] + '_reward'

    return df


def clean_behaviour_for_sub(sub="02", behavior_path="/Users/xpsy1114/Documents/projects/multiple_clocks/data/pilot/"):
    # Turn sub number into bids-like "sub-04" format
    sub = f"sub-{int(sub):02}"
    for task_half in [1,2]:
        # First load the raw behavior file
        raw_df = pd.read_csv(f"{behavior_path}/{sub}/beh/{sub}_fmri_pt{task_half}.csv")
        # second, delete all that isn't necessary
        df_cleaned = delete_unnecessary_fields(raw_df)
        # third, add fields I am going to make use of later
        df_completed = add_fields_I_want(df_cleaned)
        # fourth, store the new csv file for later use.
        df_completed.to_csv(f"{behavior_path}/{sub}/beh/{sub}_beh_clean_fmri_pt{task_half}_msm.csv")

if __name__ == "__main__":
    fire.Fire(clean_behaviour_for_sub)

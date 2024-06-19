# Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from scipy.spatial.distance import cdist, pdist, squareform
import math
import itertools
import warnings
from scipy.stats import ttest_ind, sem


# Function to measure Euclidean Distance
def calc_distance(x1, z1, x2, z2):
    return np.sqrt((x2 - x1) ** 2 + (z2 - z1) ** 2)

# experiment = "32O32B"
num_agents = 4
models = ['CUR', 'RND', 'CRF', 'CTR']

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    for model in models:
        # Load data
        DRIVE_PATH = f"E:\EXP3_SS_LOWIM_CSV\LOWIM_SS_{model}.csv"
        print(f"[INFO] Loading: {model}")
        df = pd.read_csv(DRIVE_PATH)
        # print(df.head)
        # print(df.columns)
        # print(len(df.columns))

        # List to hold images for the GIF
        images = []
        distance_matrix_list = []
        max_episodes = 200

        # Loop through unique episodes
        # for episode in df['episode'].unique():
        # Filter data for the episode
        # if episode < max_episodes:
        # df_episode = df[df['episode'] == episode]
        # Placeholder for positions
        positions = []
        for i in range(num_agents):
            positions.append([df[f'blueagent_{i + 1:02}.xposition'].mean(),
                            df[f'blueagent_{i + 1:02}.zposition'].mean()])
        for i in range(num_agents):
            positions.append([df[f'orangeagent_{i + 1:02}.xposition'].mean(),
                            df[f'orangeagent_{i + 1:02}.zposition'].mean()])
        # Calculate distances
        dist_matrix = squareform(pdist(positions))
        distance_matrix_list.append(dist_matrix)

        # Find the maximum distance.
        vmax = np.max([np.max(dm) for dm in distance_matrix_list])

        # for episode, dist_matrix in zip(df['episode'].unique(), distance_matrix_list):
        # Visualize matrix with Matplotlib
        plt.figure(figsize=(8, 8))
        plt.imshow(dist_matrix, cmap='viridis', vmin=0, vmax=vmax)
        plt.colorbar(label='Distance')
        plt.title(f'Self-Grouping Task from Separately Trained Agents\n({model} = 1.0)  \n', fontweight='bold', fontsize=15)
        # plt.savefig(f'E:\EXP3_SS_LOWIM_CSV\LOWIM_{model}_distance.png')
        plt.close()
        # print("[INFO] Grid Generated")

        # Extract the x and z positions for all agents
        orange_nni_data = []
        blue_nni_data = []
        mixed_nni_data = []

        each_step_distances = []
        each_episode_distances = []

        max_episodes = df['episode'].max()

        for episode in range(1, max_episodes):
            episode_data = df[df['episode'] == episode]

            fish_type = ["orangeagent", "blueagent"]
            pos_type = ["xposition", "zposition"]
            included_data = [f"{fish}_{str(num).zfill(2)}.{pos}" for fish in fish_type for num in range(1, num_agents + 1)
                            for pos in pos_type]
            positions = episode_data[included_data].values

            each_step_values = []
            for v in positions:  # We skip the first since it's init
                coords = np.array([(v[k], v[k + 1]) for k in range(0, len(v) - 1, 2)])
                distance_matrix = cdist(coords, coords)
                distance_matrix[distance_matrix == 0] = np.nan
                each_step_values.append(distance_matrix)

            episode_mean_distances = np.nanmean(each_step_values, axis=0)
            each_episode_distances.append(episode_mean_distances)

        each_episode_distances = np.array(each_episode_distances)

        # This is only to reshape so that it can be given into an Excel format
        reshaped_episode_means = each_episode_distances.reshape(-1, 64)

        output_df = pd.DataFrame(reshaped_episode_means)
        output_df.to_csv(f"E:\EXP3_SS_LOWIM_CSV\LOWIM_FISH_{model}_Self_Segregation_by_Episode.csv")
        ###############################################################################

        # Split the Orange Groups
        orange_to_orange_distance = each_episode_distances[:, :4, :4]
        orange_to_orange_mean = np.nanmean(orange_to_orange_distance, axis=1)

        # Split the Orange Groups
        orange_to_blue_distance = each_episode_distances[:, :4, 4:]
        orange_to_blue_mean = np.nanmean(orange_to_blue_distance, axis=1)

        blue_to_blue_distance = each_episode_distances[:, 4:, 4:]
        blue_to_blue_mean = np.nanmean(blue_to_blue_distance, axis=1)

        # Calculate Standard Error
        orange_to_orange_error = np.std(orange_to_orange_mean, axis=0) / np.sqrt(orange_to_orange_distance.shape[0])
        orange_to_blue_error = np.std(orange_to_blue_mean, axis=0) / np.sqrt(orange_to_orange_distance.shape[0])
        blue_to_blue_error = np.std(blue_to_blue_mean, axis=0) / np.sqrt(orange_to_orange_distance.shape[0])

        # Concatenate orange_to_orange and blue_to_blue then see if the mean is significnatly greater than orange_to_blue
        orange_to_orange_and_blue_to_blue = np.hstack((orange_to_orange_mean, blue_to_blue_mean))
        orange_to_orange_and_blue_to_blue_serror = np.std(orange_to_orange_and_blue_to_blue, axis=0) / np.sqrt(orange_to_orange_and_blue_to_blue.shape[0])
        print(orange_to_orange_and_blue_to_blue.shape)
        # Find the mean distance between all orange and blue fish

        # Perform t-test between orange_to_blue and combined_mean
        ttest_results = ttest_ind(np.nanmean(orange_to_orange_and_blue_to_blue, axis=0), np.nanmean(orange_to_blue_mean, axis=0), equal_var=False)
        
        # Describe the results of the T-Test and significance level
        print(f"Orange to Blue vs. Orange to Orange and Blue to Blue: {ttest_results}")

        # Indicate whether the result is significant in plain text. The truth value of ttest_results cannot be used directly because it is a tuple.
        # if ttest_results[1] < 0.01: # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
        if ttest_results[1] < 0.01:
            print(f"{model}: *** Significant Difference (p < 0.01) (p = {ttest_results[1]})")
        else:
            print(f"{model}: No Significant Difference (p > 0.01) (p = {ttest_results[1]})")

        width = 0.5
        # bars = [orange_to_orange_mean.mean(),
        #         blue_to_blue_mean.mean(),
        #         orange_to_blue_mean.mean()]

        bars = [orange_to_orange_and_blue_to_blue.mean(),
                orange_to_blue_mean.mean()]

        error_bars = [orange_to_orange_and_blue_to_blue_serror.mean(),
                    orange_to_blue_error.mean()]

        print(bars)
        print(error_bars)

        # Plotting
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.bar([0, 1], bars, width, color=["silver", "darkslategray"])

        plt.errorbar([0, 1], bars, yerr=error_bars, fmt='*', color='black',
                    markersize=0, capsize=5)

        plt.ylabel('Average Distance between Members (Units)', fontweight='bold', fontsize=14)
        # plt.title(f'\n  Average Distance Between Separately-Trained Fish During Self-Segregation \n({model} = 1)  \n',
        #         fontweight='bold', fontsize=12)

        # xticks()
        plt.xticks([0, 1], ['Within Groups', 'Across Groups'], fontweight='bold', fontsize=12)
        plt.yticks(fontsize=12)

        plt.savefig(f'E:\EXP3_SS_LOWIM_CSV\LOWIM_{model}_SS.png')
        plt.close()
        # print("[INFO] Bars Generated")

    print("[INFO] Finished")

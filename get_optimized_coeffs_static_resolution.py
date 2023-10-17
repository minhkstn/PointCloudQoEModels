from functools import lru_cache
from itertools import permutations
from multiprocessing import Pool
import json
import sys
import pandas as pd
import numpy as np
import time
import os

from itu_p1203 import P1203Standalone
from itu_p1203 import P1203Pv

a1_start = 13
a1_end = 14


f = open("./mode0.json")
input_data = json.load(f)
# input_data

ground_truth_qoe_df = pd.read_csv('./results/ratings.csv')
ground_truth_qoe_df[['video', 'start_quality', 'end_quality', 'viewing_distance']] = ground_truth_qoe_df['objects'].str.split('_', expand=True)
ground_truth_qoe_df['video'] = ground_truth_qoe_df['video'].str.lower()
ground_truth_qoe_df['qoe_scaled'] = ground_truth_qoe_df['qoe'].apply(lambda x: x*0.5)
ground_truth_qoe_df = ground_truth_qoe_df[['video', 'start_quality', 'end_quality', 'qoe_scaled']]

# ground_truth_qoe_df


# remove outliers using boxplot method
vpcc_group_columns = ['video', 'start_quality', 'end_quality']
configurations = ground_truth_qoe_df.groupby(vpcc_group_columns)

def boxplot_outlier_filter(frame):
    """
    Outlier filter using interquantile range (filter below Q1 - 1.5 IQR and above Q3 + 1.5 IQR)

    :param frame: data frame
    :return: filtered frame
    """
    q1 = frame["qoe_scaled"].quantile(0.25)
    q3 = frame["qoe_scaled"].quantile(0.75)
    # interquantile range
    iqr = q3 - q1
    fence_low = q1 - (1.5*iqr)
    fence_high = q3 + (1.5*iqr)
    filterd = (frame["qoe_scaled"] >= fence_low) & (frame["qoe_scaled"] <= fence_high)
    return frame.loc[filterd]

# for each configuration, filter outliers
df_vpcc_filtered = None
for _, frame in configurations:
    #print(boxplot_outlier_filter(frame))
    df_vpcc_filtered = pd.concat([df_vpcc_filtered, boxplot_outlier_filter(frame)], axis=0)

df_vpcc_filtered = df_vpcc_filtered.reset_index(drop=True)


ground_truth_qoe_df = df_vpcc_filtered.loc[df_vpcc_filtered['video'].isin(['longdress', 'loot'])]
ground_truth_qoe_grouped_df = ground_truth_qoe_df.groupby(['video', 'start_quality', 'end_quality']).aggregate(lambda x: tuple(x))
# ground_truth_qoe_grouped_df



bitratesMbps = {
    'longdress': {'r1': 4.64, 'r3': 14.05, 'r5':46.78},
    'loot': {'r1': 2.28, 'r3': 5.63, 'r5': 16.68},
    'redandblack': {'r1': 3.39, 'r3': 7.55, 'r5': 22.9},
    'soldier': {'r1': 4.38, 'r3': 11.58, 'r5': 35.29}
}
resolution_map = {
    'r1': '1920x1080',
    'r3': '1920x1080',
    'r5': '1920x1080'
}

min_rmse = sys.float_info.max

def calculate_p1203(coeffs):

    p1203_results = {
        'video': [],
        'start_quality': [],
        'end_quality': [],
        'start_bitrate': [],
        'end_bitrate': [],
        'p1203_qoe': []
    }

    for video in list(bitratesMbps.keys()):
        bitrate_permutations = permutations(list(bitratesMbps[video].values()), 2)
        quality_permutations = permutations(list(bitratesMbps[video].keys()), 2)

        bitrates = list(bitrate_permutations)
        qualities = list(quality_permutations)
        for bitrate in list(bitratesMbps[video].values()):
            bitrates.append((bitrate, bitrate))

        for quality in list(bitratesMbps[video].keys()):
            qualities.append((quality, quality))
        
        for bitrate, quality in zip(bitrates, qualities):
            p1203_results['video'].append(video)
            p1203_results['start_quality'].append(quality[0])
            p1203_results['end_quality'].append(quality[1])
            p1203_results['start_bitrate'].append(bitrate[0])
            p1203_results['end_bitrate'].append(bitrate[1])

            input_data['I13']['segments'][0]['bitrate'] = bitrate[0]
            input_data['I13']['segments'][0]['resolution'] = resolution_map[quality[0]]
            input_data['I13']['segments'][1]['bitrate'] = bitrate[1]
            input_data['I13']['segments'][1]['resolution'] = resolution_map[quality[1]]
            
            qoe_p1203 = P1203Standalone(input_data, coeffs=coeffs).calculate_complete()['O46']
            
            p1203_results['p1203_qoe'].append(qoe_p1203)

    return p1203_results


def calculate_rmse(p1203_results, mos):
    joined_qoe = p1203_results.join(mos)
    rmse_arr = []

    for idx in np.arange(joined_qoe.shape[0]):
        targets = joined_qoe.iloc[idx,:]['qoe_scaled']
        predictions = np.full(len(targets), joined_qoe.loc[joined_qoe.index[idx], 'p1203_qoe'])
        rmse_arr.append(np.sqrt(np.mean((predictions-targets)**2)))
    
    return np.average(rmse_arr)



# _COEFFS = {
#         "u1": 72.61,
#         "u2": 0.32,
#         "t1": 30.98,
#         "t2": 1.29,
#         "t3": 64.65,
#         "q1": 4.66,
#         "q2": -0.07,
#         "q3": 4.06,
#         "mode0": {
#             "a1": 11.9983519,
#             "a2": -2.99991847,
#             "a3": 41.2475074001,
#             "a4": 0.13183165961,
#         },
#         "mode1": {
#             "a1": 5.00011566,
#             "a2": -1.19630824,
#             "a3": 41.3585049,
#             "a4": 0,
#             "c0": -0.91562479,
#             "c1": 0,
#             "c2": -3.28579526,
#             "c3": 20.4098663,
#         },
#         "htv_1": -0.60293,
#         "htv_2": 2.12382,
#         "htv_3": -0.36936,
#         "htv_4": 0.03409,
#     }

coeffs_array = []

#    for a1 in np.arange(1, 20, 2):
#         for a2 in np.arange(-2, 0, 0.1):
#             for a3 in np.arange(120, 150, 5):
#                 for a4 in np.arange(3, 6, 0.2):
#                     for q1 in np.arange(1, 5, 0.2):
#                         for q2 in np.arange(-2, 2, 0.2):
#                             for q3 in np.arange(0.5, 5, 0.2): 

for a1 in np.arange(a1_start, a1_end, 0.5): # 8 -> 20
    for a2 in np.arange(-3, 0, 0.2):
        for a3 in np.arange(60, 150, 10):
            for a4 in np.arange(0, 5, 0.2):
                for q1 in np.arange(3, 5, 0.2):
                    for q2 in np.arange(-1.1, 1, 0.2):
                        for q3 in np.arange(0.5, 5, 0.5):
                            coeff = {
                                "u1": 72.61,
                                "u2": 0.32,
                                "t1": 30.98,
                                "t2": 1.29,
                                "t3": 64.65,
                                "q1": q1,
                                "q2": q2,
                                "q3": q3,
                                "mode0": {
                                    "a1": a1,
                                    "a2": a2,
                                    "a3": a3,
                                    "a4": a4,
                                },
                                "mode1": {
                                    "a1": 5.00011566,
                                    "a2": -1.19630824,
                                    "a3": 41.3585049,
                                    "a4": 0,
                                    "c0": -0.91562479,
                                    "c1": 0,
                                    "c2": -3.28579526,
                                    "c3": 20.4098663,
                                },
                                "htv_1": -0.60293,
                                "htv_2": 2.12382,
                                "htv_3": -0.36936,
                                "htv_4": 0.03409,
                            }
                            
                            coeff['mode0']['a1'] = a1
                            coeff['mode0']['a2'] = a2
                            coeff['mode0']['a3'] = a3
                            coeff['mode0']['a4'] = a4

                            coeffs_array.append(coeff)


def get_rmse_by_coeffs(coeff):
    global min_rmse
    qoe_p1203_dict = calculate_p1203(coeff)
    p1203_df = pd.DataFrame.from_dict(qoe_p1203_dict)

    p1203_df = p1203_df.loc[p1203_df['video'].isin(['longdress', 'loot'])]
    grouped_p1203_qoe = p1203_df.groupby(['video', 'start_quality', 'end_quality']).mean()

    rmse = calculate_rmse(grouped_p1203_qoe, ground_truth_qoe_grouped_df)
    if rmse < min_rmse:
        min_rmse = rmse
        print('Min RMSE for pid {}: {}.\t Coeffs: {} {} {} {}'.format(os.getpid(), rmse, coeff['q1'], coeff['q2'], coeff['q3'], coeff['mode0']))

        return coeff, rmse
    else:
        return 0, 5


if __name__ == '__main__':
    all_rmse = None
    current_time = int(time.time())
    with Pool(40) as pool:
        all_rmse = pool.map(get_rmse_by_coeffs, coeffs_array)

    all_rmse.sort(key= lambda x: x[1])
    # write all_rmse to files
    with open('all_rmse_static_resolution_{}.txt'.format(current_time), 'w') as fp:
        for item in all_rmse:
            fp.write('%s\n' % str(item))

    print("====== BEST RESULT =======")
    print(all_rmse[0])

    # save the best result
    with open('./optimized_p1203_coeff_mode0_static_resolution_{}_{}_{}.csv'.format(a1_start, a1_end, current_time), 'w') as f:
        f.write('====== Coeffs ====\n' + str(all_rmse[0][0]))
        f.write('\n====== RMSE ====\n' + str(all_rmse[0][1]))
    print("====== DONE =======")

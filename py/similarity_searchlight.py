# coding: utf-8
"""
similarity_searchlight.py
Written by Meir Meshulam, Princeton Neuroscience Institute, 2020
For the paper "Think Like an Expert: Neural Alignment Predicts
Understanding in Students Taking an Introduction to Computer Science Course"
Raw data is available on OpenNeuro (https://openneuro.org/datasets/ds003233)

This code runs a cortical searchlight for whole-brain analyses
Use to analyze:
- lectures: calculate spatial alignment-to-class in each voxel (figure 2)
- recaps: correlate alignment-to-class and alignment-to-experts (figure 3)
- exam:
    - 'same-question' correlation between alignment and exam score (figure 4)
    - 'knowledge-structure' correlation between alignment and exam score (figure 5)

:param
    argv[1] = 'wk1'..'wk6' (scanning week)
    argv[2] = 'placement' or 'vid1'..'vid5' or 'wk1recap'..'wk5recap' (task)
    argv[3] = 'same' or 'knowledge' (analysis type)
    argv[4] = 'skip' or 'within' or 'direct' (correlation with exam score)
    argv[5] = 'student-vs-experts' or 'student-vs-students' (alignment type)
    argv[6] = int (searchlight radius)
    argv[7] = int (number of perms for permutation tests)


:example calls

-to calculate alignment-to-class in a single video lecture (creates the non-thresholded map in figure 2)
./similarity_searchlight.py wk1 wk1_vid1 same skip student-vs-students 2 1000

-to correlate alignment-to-class and alignment-to-experts in a single recap (figure 3A)
./similarity_searchlight.py wk6 wk6_wk1recap same direct student-vs-experts 2 1000

-to correlate alignment-to-class and alignment-to-experts in exam (figure 3B)
./similarity_searchlight.py wk6 placement same direct student-vs-experts 2 1000

-to correlate same-question alignment-to-class during exam with exam score (figure 4)
./similarity_searchlight.py wk6 placement same within student-vs-students 2 1000

-to correlate same-question alignment-to-class during exam with exam score (figure 5)
./similarity_searchlight.py wk6 placement knowledge within student-vs-students 2 1000

"""

from os import listdir, getlogin
from os.path import join
import numpy as np
import re
from brainiak import image, io, isc
import brainiak.searchlight.searchlight as searchlight
from brainiak.fcma.util import compute_correlation
import nibabel as nib
import time
import sys
from scipy import stats
import socket
import pandas as pd
from mpi4py import MPI

# set paths
my_name = getlogin()

# set system
this_system = socket.gethostname()
print('Server: ' + this_system)

# DATA
bids_path = '/mnt/sink/scratch/{}/to_bids'.format(my_name)  # raw data
const_data_path = '/mnt/bucket/labs/hasson/' + my_name
const_study_path = join(const_data_path, 'onlineL', 'pred20')  # pre-processed data
input_fslfeat_students_path = join(const_study_path, 'scan_data_nii', 'students_mni',
                                   '6motion')  # str student pre-processed data
input_fslfeat_experts_path = join(const_study_path, 'scan_data_nii', 'experts_mni',
                                  '6motion')  # str expert pre-processed data
print('DATA: ' + const_study_path)

# CODE
code_path = '/mnt/bucket/people/{}/{}/notebooks/share'.format(my_name, my_name[:-2])
print('CODE: ' + code_path)

# SCORES
scores_path = join(bids_path, 'sourcedata', 'exam_scores.tsv')
timing_path = join(bids_path, 'sourcedata', 'exam_timing.tsv')

# MASKS
masks_path = join(code_path, 'masks')
mni_brain_file_name = join(masks_path, 'MNI152_T1_3mm_brain.nii.gz')  # MNI brain
mni_cortex_file_name = join(masks_path, 'MNI152_T1_3mm_cortex_mask.nii.gz')  # MNI cortex

# OUTPUT
out_path = join(const_data_path, 'onlineL', 'shared', 'outputs')
print('OUTPUT:' + out_path)

# import multiple comparisons tools
sys.path.insert(0, join(code_path, 'py'))
from multi_comp import fdr_correction  # FDR from the MNE-python package

# set re template for parsing file names
task_name_template = "(s\d{3})_(wk\d+)_([0-9a-zA-Z]*)_6motion_mni"  # template for data preprocessed with FSL + regressout 6 motion


def get_filenames(student_and_expert_files, task_name_template=task_name_template,
                  input_fslfeat_students_path=input_fslfeat_students_path):
    """
    build dict for data file names, hierarchical
    dict structure:
    filenames_dict['students']['s113']['wk2']['vid1']
    :param
    student_and_expert_files: list of data files
    task_name_template: re expression to parse file names
    input_fslfeat_students_path: path to student data files
    :returns
    filenames_dict
    """
    filenames_dict = dict()
    filenames_dict['students'] = {}
    filenames_dict['experts'] = {}
    for i_this_file, this_file in enumerate(student_and_expert_files):
        dk = re.search(task_name_template, this_file)
        this_subject = str(dk[1])
        this_session = str(dk[2])
        this_task = str(dk[3])
        if i_this_file >= len(listdir(input_fslfeat_students_path)):
            student_or_expert = 'experts'
        else:
            student_or_expert = 'students'
        try:
            temp = type(filenames_dict[student_or_expert][this_subject]) is dict
        except KeyError:
            filenames_dict[student_or_expert][this_subject] = {}
        try:
            temp = type(filenames_dict[student_or_expert][this_subject][this_session]) is dict
        except KeyError:
            filenames_dict[student_or_expert][this_subject][this_session] = {}
        # write filename to dict
        filenames_dict[student_or_expert][this_subject][this_session][this_task] = join(
            eval('input_fslfeat_{}_path'.format(student_or_expert)), this_file)
    return filenames_dict


def load_epi(input_filenames, brain_mask):
    """
    load, mask and z-score epi data, return data as list of subjects, no nans
    :param
    filenames (list)
    brain_mask (assumes it was already read with io.load_boolean_mask and thresholded)
    :returns
    output_list
    run time
    """

    # divide the work (and memory) of loading subject data across ranks.
    subject_idx_list = np.array_split([iSubject for iSubject in range(len(input_filenames))], comm.size)[comm.rank]
    print("Rank {}: Loading subjects -> {}".format(comm.rank, subject_idx_list))
    # on each rank, we need a list of the same size with None in place of subject data that is loaded on other ranks
    output_list = [None for i in input_filenames]
    t1 = time.time()  # timeit

    for iSubject in subject_idx_list:
        # load
        this_image = nib.load(input_filenames[iSubject])
        # mask
        this_image_masked = image.mask_image(this_image, brain_mask)
        # nan to 0
        this_image_masked[np.isnan(this_image_masked)] = 0
        # zscore
        this_image_masked_zscored = stats.zscore(this_image_masked, axis=1, ddof=1)
        # into x*y*z*tr format
        uncon = np.zeros(
            (brain_mask.shape[0], brain_mask.shape[1], brain_mask.shape[2], this_image_masked_zscored.shape[1]))
        coords = np.where(brain_mask)
        uncon[coords[0], coords[1], coords[2], :] = np.nan_to_num(this_image_masked_zscored)
        # update output
        output_list[iSubject] = uncon
    t2 = time.time()
    return output_list, t2 - t1  # data and runtime


def load_data(subject_groups):
    """
    :param
    subject_groups: 'students' or 'experts'
    :return:
    dict_epi_data: dictionary with all epi data for the subject groups specified
    """
    # init
    dict_epi_data = {'students': {}, 'experts': {}}
    dict_epi_data['students']['test_data'] = []
    dict_epi_data['experts']['test_data'] = []

    for students_or_experts in sorted(subject_groups):
        if rank == 0:
            print(students_or_experts)

        # get filenames for test, (+train if required)
        test_on_files_prep = filter_filenames(filenames_dict, students_or_experts, test_vid_week,
                                              test_vid_name)
        if train_vid_week:
            train_on_files_prep = filter_filenames(filenames_dict, students_or_experts, train_vid_week,
                                                   train_vid_name)
        # get same subjects for train,test
        # then get files, read into dict of lists
        if 'student' in students_or_experts:  # get only good students
            test_on_files_prep = intersect_filenames(test_on_files_prep, good_students)
        elif 'expert' in students_or_experts:
            test_on_files_prep = intersect_filenames(test_on_files_prep, good_experts)
        # load train + intersect train-test + intersect with good students
        if train_vid_week:
            if 'student' in students_or_experts:
                train_on_files_prep = intersect_filenames(train_on_files_prep,
                                                          good_students)  # get only good students/experts
            elif 'expert' in students_or_experts:
                train_on_files_prep = intersect_filenames(train_on_files_prep,
                                                          good_experts)  # get only good students/experts
            train_on_files, test_on_files = intersect_filenames(train_on_files_prep,
                                                                test_on_files_prep)  # intersect test-train
            # load train
            train_on_input, train_load_time = load_epi(train_on_files, cortex_mask_3mm)
            dict_epi_data[students_or_experts]['train_data'] = train_on_input.copy()  # list of subjects, vox x trs
        else:  # no train data
            test_on_files = test_on_files_prep  # skip intersect with train
        # load test
        test_on_input, test_load_time = load_epi(test_on_files, cortex_mask_3mm)
        dict_epi_data[students_or_experts]['test_data'] = test_on_input.copy()
    if rank == 0:
        print()
        if train_vid_week:
            print("Training set, {} subjects, load time: {:.2f}s".format(len(train_on_input), train_load_time))
        print("Test set, {} subjects, load time: {:.2f}s".format(len(test_on_input), test_load_time))
    return dict_epi_data


def isc_local(D, collapse_subj=True, external_signal=None, float_type=np.float16):
    """
    Internal ISC function for this notebook, no stats
    external_signal: signal to correlate with, instead of mean subjects (n-1) like in standard isc
    # but here added external mean for comparison (mean of experts to compare to the individual students)
    """
    n_vox = D.shape[0]
    n_subj = D.shape[2]
    ISC = np.zeros((n_vox, n_subj), dtype=float_type)
    for this_subject in range(n_subj):
        if external_signal is not None:
            group = external_signal
            assert np.logical_and(group.shape[0] == D.shape[0],
                                  group.shape[1] == D.shape[1]), 'dims mismatch for external signal input'
        else:
            group = np.mean(D[:, :, np.arange(n_subj) != this_subject], axis=2)
        subj = D[:, :, this_subject]
        for v in range(n_vox):
            ISC[v, this_subject] = stats.pearsonr(group[v, :], subj[v, :])[0]

    if collapse_subj:
        ISC = isc.compute_summary_statistic(ISC, axis=1)[np.newaxis, :]

    # Throw away first dimension if singleton
    if ISC.shape[0] == 1:
        ISC = ISC[0]

    return ISC


def isfc_local(D, collapse_subj=True, external_signal=None, float_type=np.float16):
    """
    Internal ISFC function for this notebook, no stats
    external_signal: signal to correlate with, instead of mean subjects (n-1) like in standard isfc
    # but here added external mean for comparison (mean of experts to compare to the individual students)
    """
    n_vox = D.shape[0]
    n_subj = D.shape[2]
    ISFC = np.zeros((n_vox, n_vox, n_subj), dtype=float_type)
    for this_subject in range(n_subj):
        if external_signal is not None:
            group = external_signal
            assert np.logical_and(group.shape[0] == D.shape[0],
                                  group.shape[1] == D.shape[1]), 'dims mismatch for external signal input'
        else:
            group = np.mean(D[:, :, np.arange(n_subj) != this_subject], axis=2)
        subj_data = D[:, :, this_subject]
        ISFC[:, :, this_subject] = compute_correlation(np.ascontiguousarray(subj_data), np.ascontiguousarray(
            group))  # order important because compute_correlation correlates rows of matrix 1 with rows of matrix 2
        # Symmetrize matrix - skip
        # if external_signal is None:
        #    ISFC[:, :, this_subject] = (ISFC[:, :, this_subject] +
        #                            ISFC[:, :, this_subject].T) / 2
    # collapse over subjects
    if collapse_subj:
        ISFC = isc.compute_summary_statistic(ISFC, axis=2)
    # Throw away first dimension if singleton
    if ISFC.shape[0] == 1:
        ISFC = ISFC[0]

    return ISFC


def corr_and_null_dist(x, y, num_perms=0):
    """
    run pearson correlation and return r-val and null distribution
    :param
    x,y: vectors to compute Pearson correlation on
    num_perms: number of permutations
    :returns
    rval
    null_dist: null distribution for correlation of x,y
    """
    rval = stats.pearsonr(x, y)[0]
    if num_perms > 0:
        null_dist = np.array([stats.pearsonr(np.random.permutation(x), y)[0] for n in np.arange(num_perms)])
    return rval, null_dist


def filter_filenames(filenames_dict, students_or_experts, week, vid_name):
    """
    gets filenames dict and returns file list as specified by parameters
    """
    subject_keys = sorted([k for k, v in filenames_dict[students_or_experts].items()])
    return_files = [None for i in range(len(subject_keys))]
    for i_this_student in range(len(subject_keys)):
        try:
            return_files[i_this_student] = (
                filenames_dict[students_or_experts][subject_keys[i_this_student]][week][vid_name])
        except KeyError:
            pass
    return [i for i in return_files if i]  # eliminate nans


def intersect_filenames(list1, list2):
    """
    omit filenames from list1,list2, for subjects that do not appear in both lists
    :param
    list1: file names
    list2: file names or list of subject names
    :return
    files_out_1 - list1 items, minus subjects that are not in list 2
    files_out_2 - list2 items, minus subjects that are not in list 1 [only returned in case list2 contained files, otherwise not retured]
    """
    list2_is_file_names = False
    subjects_in_1 = [str(re.search(task_name_template, f)[1]) for f in list1]
    try:
        subjects_in_2 = [str(re.search(task_name_template, f)[1]) for f in list2]
        list2_is_file_names = True
    except TypeError:  # not a list of files matching template, use subject names in list2 directly
        subjects_in_2 = list2
    shared_subjects = list(set(subjects_in_1) & set(subjects_in_2))  # find subjects in both sets
    files_out_1 = [f for f in list1 if str(re.search(task_name_template, f)[1]) in shared_subjects]
    if not list2_is_file_names:
        return files_out_1
    else:
        files_out_2 = [f for f in list2 if str(re.search(task_name_template, f)[1]) in shared_subjects]
    return files_out_1, files_out_2


def get_exam_behavior(timing_path=timing_path, scores_path=scores_path):
    """
    read exam grades and question timing data
    :param timing_path:
    :param scores_path:
    :return:
    timing_df: timing
    placement_by_q: grades
    """
    # get grades
    scores_df = pd.read_csv(scores_path, sep='\t', index_col=[0])
    scores_df.columns = range(16)
    scores_df.index.name = 'Subject'
    students_ind = [True if int(i[1:]) < 200 else False for i, s in scores_df.iterrows()]
    experts_ind = np.logical_not(students_ind)
    placement_by_q = {}
    placement_by_q['students'] = scores_df[students_ind]
    placement_by_q['experts'] = scores_df[experts_ind]
    placement_by_q['students'] = placement_by_q['students'].drop(['s112'], axis=0)  # no placement data
    placement_by_q['experts'] = placement_by_q['experts'].drop(['s201'], axis=0)  # no placement data
    # get timing
    timing_df = pd.read_csv(timing_path, sep='\t', index_col=[0])
    timing_df['subject'] = timing_df.index
    timing_df['student_or_expert'] = ['student' if int(i[1:]) < 200 else 'expert' for i, s in timing_df.iterrows()]
    timing_df = timing_df.rename(
        columns={'question': 'q_number', 'response_onset_TR': 'q_start_TR', 'response_offset_TR': 'q_end_TR'})
    timing_df['q_number'] -= 1
    timing_df.subject = timing_df.index

    return timing_df, placement_by_q


def get_bins(nTR, bin_size=30, tr_length=2):
    """
    get time bins for lecture videos
    :param nTR: number of TRs
    :param bin_size: in seconds
    :param tr_length: in seconds
    :return:
    time_stamps - time stamp per time bin (instead of for each question)
    """
    number_of_bins = np.int(np.floor(nTR / (bin_size / tr_length)))  # number of bins given the params
    end_of_trs = np.int(
        number_of_bins * bin_size / tr_length)  # round such that end of vid is cut, makes sure binsize always equal
    slices = np.linspace(0, end_of_trs, number_of_bins, endpoint=False).astype(np.int)  # bin edges
    slices = np.append(slices, end_of_trs)
    slices_cut = slices

    time_stamps = pd.DataFrame()
    for students_or_experts in filenames_dict.keys():  # st/ex
        if 'student' in students_or_experts:  # student
            subj_list = sorted(good_students())  # subj id
            if test_vid_week == 'wk3':  # no s103 data in wk3
                subj_list.remove('s103')
        else:  # expert
            subj_list = sorted(good_experts())  # subj id
        for this_subject in subj_list:
            update_df_NNvid_timestamps = pd.DataFrame(
                {'student_or_expert': students_or_experts[:-1], 'subject': this_subject,
                 'q_start_TR': slices_cut[:-1], 'q_end_TR': slices_cut[1:], 'q_number': np.arange(len(slices_cut) - 1)})
            time_stamps = time_stamps.append(update_df_NNvid_timestamps)

    return time_stamps


def print_sl_info(epi_list, sl_rad, loc_split):
    """
    print out searchlight details
    :param epi_list:
    :param sl_rad:
    :param loc_split:
    :return: Null
    """
    print('Searchlight cube edge size is {} -> {} voxels'.format(1 + 2 * sl_rad, (1 + 2 * sl_rad) ** 3))
    print('Number of subjects: {}'.format(int(len(epi_list))))
    print('Number of student datasets: {}, expert datasets: {}.'.format(loc_split, len(epi_list) - loc_split))
    print('Number of TRs in first list item: {}'.format(epi_list[0].shape[3]))
    return


def get_file_name(suffix=None):
    """
    :param suffix - last part of file name
    :return: name of file to save, no extension
    """
    file_name = 'data-'
    file_name += test_vid_week + '-' + test_vid_name + '_search-'
    file_name += str((1 + 2 * sl_rad) ** 3) + 'vox_'
    file_name += similarity_type + '_' + vs_mean_of + '_corrwscore-' + correlation_with_score + '_'
    file_name += 'perms={}'.format(num_perms)
    if suffix:
        file_name += '_' + suffix
    return file_name


def unpack_r_and_p(c, item='rval'):
    """
    function returns nparray with r or p value in each location from c[rval,pval]
    use to process arrays returned by searchlight with [rval,tval] in each voxel
    use item=rval to get r-values, item=pval to get p-values
    """
    if 'r' in item:
        loc = 0
    elif 'p' in item:
        loc = 1
    if c is None:
        return None
    else:
        return c[loc]


def unpack_multiple_vals(c=None, loc=0):
    """
    function returns nparray with single value in each location
    use to process arrays returned by searchlight with per-subj-rvals in each voxel from func diag
    use to process arrays returned by searchlight with save_direct_placement_perms in each voxel from func diag
    """
    if c is None:
        return None
    else:
        return c[loc]


def save_result(t1, t2, sim, out_path=out_path):
    """
    Save results
    :param t1: time.time() timestamp (start)
    :param t2: time.time() timestamp (finish)
    :param sim: searchlight result data
    :param out_path: output path
    :return: Null

    """
    # print searchlight duration
    print('Searchlight total time {0:.2f} seconds'.format(t2 - t1))
    # npz
    # get filaname
    output_filename = join(out_path, get_file_name())
    # save as npz, with separate arrays for r-val and p-val (uncorrected)

    if vids_per_subj_data_skip:  # per subj r-val data for videos (not for direct)
        # prep unpacking func
        unpack_multiple_vals_vectorized = np.vectorize(
            unpack_multiple_vals)  # vectorize unpack func to use directly on nparrays (vid version - lectures/recaps)
        brain_dims = epi_list[0].shape[0:3]
        multi_rvals = np.zeros([*epi_list[0].shape[0:3], loc_split])
        # unpack vals per subject
        for i in np.arange(loc_split):
            multi_rvals[:, :, :, i] = unpack_multiple_vals_vectorized(sim, i)  # dim: x,y,z,student
        np.savez_compressed("{}.npz".format(output_filename), rval=multi_rvals)
        print('{} npz saved'.format(output_filename))
        print('Finished searchlight')
        return  # no need to save maps for individual subjs

    elif save_direct_placement_perms:  # direct comparison, save perms for this recap
        # prep unpacking func
        unpack_multiple_vals_vectorized = np.vectorize(
            unpack_multiple_vals)  # vectorize unpack func to use directly on nparrays (direct on recaps)
        unpack_r_and_p_vectorized = np.vectorize(unpack_r_and_p)
        multi_pvals = np.zeros([*epi_list[0].shape[0:3], num_perms])
        # unpack p-vals
        for i in np.arange(num_perms):
            multi_pvals[:, :, :, i] = unpack_multiple_vals_vectorized(unpack_r_and_p_vectorized(sim, 'pval'),
                                                                      loc=i)  # dim: x,y,z,perm
        np.savez_compressed("{}.npz".format(output_filename), rval=unpack_r_and_p_vectorized(sim, 'rval'),
                            pval=multi_pvals)
        print('{} npz saved'.format(output_filename))
        print('Finished searchlight')
        return  # no need to save maps for 'direct' comparison on recap

    else:
        # prep unpacking func
        unpack_r_and_p_vectorized = np.vectorize(
            unpack_r_and_p)  # vectorize unpack func to use directly on nparrays (no vid version - placement/questions)
        np.savez_compressed("{}.npz".format(output_filename), rval=unpack_r_and_p_vectorized(sim, 'rval'),
                            pval=unpack_r_and_p_vectorized(sim, 'pval'))
        print('{} npz saved'.format(output_filename))

    try:
        # prep rval and pval maps to save
        map_rval = unpack_r_and_p_vectorized(sim, 'rval').astype('double')
        map_rval[np.isnan(map_rval)] = 0
        map_pval = unpack_r_and_p_vectorized(sim, 'pval').astype('double')
        map_pval[np.isnan(map_pval)] = 1

        # save as nii - unthresholded
        output_filename = join(out_path, get_file_name(suffix='no-threshold'))
        brain_nii = nib.load(mni_brain_file_name)
        ISC_nifti = nib.Nifti1Image(map_rval, brain_nii.affine, brain_nii.header)
        nib.save(ISC_nifti, "{}.nii.gz".format(output_filename))
        print('rval nii saved, no threshold')
    except:
        print('Could not save no-threshold map')

    # save thresholded, uncorrected map
    try:
        for this_threshold in threshold:
            output_filename = join(out_path,
                                   get_file_name(suffix='thr_noFDR_p={}'.format(str(this_threshold).split('.')[-1])))
            map_rval_nofdr = map_rval.copy()
            map_rval_nofdr[map_pval >= float(this_threshold)] = 0
            brain_nii = nib.load(mni_brain_file_name)
            ISC_nifti = nib.Nifti1Image(map_rval_nofdr, brain_nii.affine, brain_nii.header)
            nib.save(ISC_nifti, "{}.nii.gz".format(output_filename))
            print('rval nii saved, threshold={}, uncorrected'.format(this_threshold))
    except:
        print('Could not save no-FDR thresholded map')

    # save FDR-thresholded map
    try:
        for this_threshold in threshold:
            brain_nii = nib.load(mni_brain_file_name)
            # vol=np.zeros(brain_nii.shape)
            output_filename = join(out_path,
                                   get_file_name(suffix='thr_FDR_p={}'.format(str(this_threshold).split('.')[-1])))
            map_rval_fdr = unpack_r_and_p_vectorized(sim, 'rval').astype('double')  # reset to original rvals, with nans
            map_pval_fdr = unpack_r_and_p_vectorized(sim, 'pval').astype('double')  # reset to original pvals, with nans
            nonan_loc = np.where(~np.isnan(map_rval_fdr))  # run fdr only on this
            pvals_nonan_fdr = fdr_correction(map_pval_fdr[nonan_loc], alpha=float(this_threshold))[1]
            map_pval_fdr[nonan_loc] = pvals_nonan_fdr
            map_rval_fdr[np.where(np.isnan(map_rval_fdr))] = 0  # no nans
            map_rval_fdr[map_pval_fdr >= float(this_threshold)] = 0  # threshold
            vol_nifti = nib.Nifti1Image(map_rval_fdr, brain_nii.affine, brain_nii.header)
            # vol[coords] = map_rval_fdr
            # vol_nifti = nib.Nifti1Image(vol, brain_nii.affine, brain_nii.header)
            nib.save(vol_nifti, "{}.nii.gz".format(output_filename))
            print('rval nii saved, threshold={}, FDR corrected'.format(this_threshold))
    except:
        print('Could not save FDR thresholded map')
    print('Finished searchlight')
    return


def same_question_similarity(df_q_timestamps, placement_by_q, data_in_sl):
    """
    same-question ('diagonal') analysis
    :param df_q_timestamps: exam questions time stamps
    :param placement_by_q: exam questions grades
    :param data_in_sl: data in searchlight (input)
    :return:
    out_val
    """
    number_of_questions = len(df_q_timestamps['q_number'].unique())  # 16 q in placement exam
    corr_result = {}
    for student_or_expert in sorted(
            subject_groups):  # order is important- do expert before to allow comparison with student
        epi_data = data_in_sl[student_or_expert + 's'][
            'test_data']  # epi data is just the non-transformed test data
        number_of_subjects = len(epi_data)
        corr_result[student_or_expert] = {}
        corr_result[student_or_expert]['roi_epi_per_q'] = None  # timecourse data in roi, do isc on this
        corr_result[student_or_expert][
            'roi_isc_per_q_vs_group_rval'] = None  # r-value result of isc, questions are in the 'voxels' dim: student vs student, expert vs expert
        corr_result[student_or_expert][
            'roi_isc_per_q_vs_group_dist'] = None  # null dist, questions are in the 'voxels' dim: student vs student, expert vs expert
        if 'student' in student_or_expert:
            corr_result['student'][
                'roi_isc_per_q_student_vs_expert_rval'] = None  # rval result of pseudo-isc, student vs expert
            corr_result['student'][
                'roi_isc_per_q_student_vs_expert_dist'] = None  # null dist of pseudo-isc, student vs expert
        epi_temporal_mean_per_q = np.zeros(
            [number_of_questions, epi_data[0].shape[0], len(epi_data)])  # questions (instead of TR) X voxels X subjects
        for this_question in range(number_of_questions):
            # print(this_question)
            # slice this question: entries for this question, all subjects (expert or student separately)
            df_this_q = df_q_timestamps.loc[np.logical_and(df_q_timestamps['student_or_expert'] == student_or_expert,
                                                           df_q_timestamps['q_number'] == this_question)]
            df_this_q = df_this_q.sort_values(by=['subject'], ascending=True)
            # start and end points for this question (TR), for each subject
            trs_start = df_this_q['q_start_TR'].values + trs_to_add_to_start
            trs_end = df_this_q['q_end_TR'].values + trs_to_add_to_end
            assert np.sum((trs_end - trs_start) > 2), 'slice problem in {}-{}-{}-{}'.format(student_or_expert,
                                                                                            this_question, trs_end,
                                                                                            trs_start)
            # extract mean across TRs of epi data per question: vox X subjects
            epi_temporal_mean_per_q[this_question, :, :] = np.array(
                [np.nanmean(epi_data[i_subject][:, np.int(trs_start[i_subject]):np.int(trs_end[i_subject])], axis=1) for
                 i_subject in np.arange(len(trs_start))]).transpose()
        # update dict with epi data, mean over TRs
        corr_result[student_or_expert]['roi_epi_per_q'] = epi_temporal_mean_per_q
        # run spatial isc with questions as first dim (get per-question value) (output: question X subject)
        # student vs mean of student, expert-expert
        # run isc
        corr_result[student_or_expert]['roi_isc_per_q_vs_group_rval'] = isc_local(epi_temporal_mean_per_q,
                                                                                  collapse_subj=False)
        # get bootstrap dist
        if num_perms > 0:
            d = corr_result[student_or_expert]['roi_isc_per_q_vs_group_rval']  # questions X subjects
            corr_result[student_or_expert]['roi_isc_per_q_vs_group_dist'] = np.zeros([number_of_subjects, num_perms])
            for this_subject in np.arange(number_of_subjects):
                observed, ci, pval, dist = isc.bootstrap_isc(d[:, this_subject], pairwise=False,
                                                             summary_statistic='mean', n_bootstraps=num_perms,
                                                             ci_percentile=0, random_state=None)
                corr_result[student_or_expert]['roi_isc_per_q_vs_group_dist'][this_subject, :] = dist.squeeze()
        else:
            corr_result[student_or_expert]['roi_isc_per_q_vs_group_dist'] = None

        # student vs CLEAN mean of expert (omit patterns of expert that answered wrong)
        if 'student' in student_or_expert:
            if epi_data is not None:
                try:
                    expert_all = corr_result['expert']['roi_epi_per_q']  # get data
                    if not (('questions' in test_vid_name) or ('placement' in test_vid_name)):
                        # 1) non-clean version - for use when no questions are involved - running just on video - qualifies all 'answers'
                        expert_clean_collapsed = np.nanmean(expert_all, axis=2)
                    else:
                        # 2) CLEAN version
                        clean_mask = (placement_by_q[
                                          'experts'] >= expert_accept_question_threshold).values.transpose()  # mask out wrong answers
                        # mask out bad expert responses, replace with nans
                        for v in np.arange(expert_all.shape[1]):
                            expert_all[:, v, :][np.invert(clean_mask)] = np.nan
                        # collapse over experts
                        expert_clean_collapsed = np.nanmean(
                            np.array([expert_all[:, v, :] for v in np.arange(expert_all.shape[1])]),
                            axis=2).transpose()
                    # get rval
                    corr_result[student_or_expert]['roi_isc_per_q_student_vs_expert_rval'] = isc_local(
                        epi_temporal_mean_per_q, collapse_subj=False, external_signal=expert_clean_collapsed)
                    # get bootstrap dist
                    if num_perms > 0:
                        d = corr_result[student_or_expert][
                            'roi_isc_per_q_student_vs_expert_rval']  # questions X students
                        corr_result[student_or_expert]['roi_isc_per_q_student_vs_expert_dist'] = np.zeros(
                            [number_of_subjects, num_perms])
                        for this_subject in np.arange(number_of_subjects):  # iter students
                            observed, ci, pval, dist = isc.bootstrap_isc(d[:, this_subject], pairwise=False,
                                                                         summary_statistic='mean',
                                                                         n_bootstraps=num_perms, ci_percentile=0,
                                                                         random_state=None)
                            corr_result[student_or_expert]['roi_isc_per_q_student_vs_expert_dist'][this_subject,
                            :] = dist.squeeze()
                    else:
                        corr_result[student_or_expert]['roi_isc_per_q_student_vs_expert_dist'] = None
                except KeyError:  # no expert data
                    # zero out student-expert
                    corr_result[student_or_expert]['roi_isc_per_q_student_vs_expert_rval'] = 0
                    corr_result[student_or_expert]['roi_isc_per_q_student_vs_expert_dist'] = 0
                    corr_result[student_or_expert]['roi_isc_NOTS_per_q_student_vs_expert_rval'] = 0

    if 'experts' in vs_mean_of:
        # rval
        cx = corr_result['student']['roi_isc_per_q_student_vs_expert_rval'].copy()  # questions X students
        # pval for the mean value, variance source: subjects
        null_dist = np.tanh(np.nanmean(np.arctanh(corr_result['student']['roi_isc_per_q_student_vs_expert_dist']),
                                       axis=0))  # mean over subj
        if np.isnan(np.nanmean(cx)) or np.sum(np.isnan(null_dist)):
            px = np.nan
        else:
            px = isc.p_from_null(np.tanh(np.nanmean(np.arctanh(cx))), null_dist, side='right', exact=False)
    elif 'students' in vs_mean_of:
        cx = corr_result['student']['roi_isc_per_q_vs_group_rval']  # questions X students
        # pval for the mean value, variance source: subjects
        null_dist = np.tanh(
            np.nanmean(np.arctanh(corr_result['student']['roi_isc_per_q_vs_group_dist']), axis=0))  # mean over subj
        if np.isnan(np.nanmean(cx)) or np.sum(np.isnan(null_dist)):
            px = np.nan
        else:
            px = isc.p_from_null(np.tanh(np.nanmean(np.arctanh(cx))), null_dist, side='right', exact=False)

    if 'within' in correlation_with_score.lower():  # correlate similarity score with placement score within subjects
        corr_score_vec = np.zeros(cx.shape[1])  # rvals
        corr_perm_dist = np.zeros((num_perms, cx.shape[1]))  # rand dist for each rval
        for i_this_student in range(cx.shape[1]):
            # rvals and distribution for each, taken from that subject
            x = cx[:, i_this_student]
            y = placement_by_q['students'].iloc[i_this_student].values
            corr_score_vec[i_this_student], corr_perm_dist[:, i_this_student] = corr_and_null_dist(x, y,
                                                                                                   num_perms=num_perms)
        # rval
        rval = isc.compute_summary_statistic(corr_score_vec)
        # pval
        within_dist = np.tanh(np.nanmean(np.arctanh(corr_perm_dist), axis=1))  # mean over subjects for each rand perm
        if np.isnan(rval) or np.sum(np.isnan(within_dist)):
            pval = np.nan
        else:
            pval = isc.p_from_null(rval, within_dist, side='right', exact=False)
        out_val = np.array([rval, pval])

    elif 'direct' in correlation_with_score.lower():  # correlate similarity to group and similarity to experts
        x = corr_result['student']['roi_isc_per_q_student_vs_expert_rval']  # questions X students
        y = corr_result['student']['roi_isc_per_q_vs_group_rval']  # questions X students
        # corr separately in each bin
        n_bins = x.shape[0]
        direct_dists = np.zeros((n_bins, num_perms))
        direct_rvals = np.zeros((n_bins))
        for this_bin in range(n_bins):
            direct_rvals[this_bin], direct_dists[this_bin, :] = \
                corr_and_null_dist(x[this_bin, :], y[this_bin, :], num_perms=num_perms)
        rval = np.tanh(np.nanmean(np.arctanh(direct_rvals)))  # take mean across bins
        direct_mean_dist = np.tanh(np.nanmean(np.arctanh(direct_dists), axis=0))
        if np.isnan(rval) or np.sum(np.isnan(direct_mean_dist)):
            pval = np.nan
        else:
            pval = isc.p_from_null(rval, direct_mean_dist, side='right', exact=False)
        if save_direct_placement_perms:
            out_val = [rval, direct_mean_dist]  # for recaps direct comprison, save perms of placement, sl output is npz
        else:
            out_val = np.array([rval, pval])  # for placement direct comparison  - sl output is nii map

    elif 'skip' in correlation_with_score.lower():  # do not correlate with placement score, output sim score as is
        if vids_per_subj_data_skip:  # working on vids to get spatial isc, diag/skip, no mean over subjects
            out_val = np.tanh(
                np.nanmean(np.arctanh(cx), axis=0))  # similarity: [r-values per subject; mean across bins only)
        else:
            out_val = np.array(
                [np.tanh(np.nanmean(np.arctanh(cx))), px])  # standard option - similarity: [r-value, p-value]

    return out_val


def knowledge_similarity(df_q_timestamps, placement_by_q, data_in_sl):
    """
    knowledge structure ('isfc') analysis
    :param df_q_timestamps: exam questions time stamps
    :param placement_by_q: exam questions grades
    :param data_in_sl: data in searchlight (input)
    :return:
    out_val
    """

    number_of_questions = len(df_q_timestamps['q_number'].unique())  # 16 q in placement exam
    corr_result = {}

    # first, get ISFC matrices for experts (collapsed) and students (not collapsed)
    for student_or_expert in sorted(
            subject_groups):  # order is important - do expert before to allow comparison with student
        epi_data = data_in_sl[student_or_expert + 's'][
            'test_data']  # epi data is just the non-transformed test data
        corr_result[student_or_expert] = {}
        corr_result[student_or_expert]['roi_epi_per_q'] = None  # timecourse data in roi, do isc on this
        corr_result[student_or_expert]['q-q-similarity'] = None  # per participant, question-to-question similarity
        epi_temporal_mean_per_q = np.zeros(
            [number_of_questions, epi_data[0].shape[0], len(epi_data)])  # questions (instead of TR) X voxels X subjects
        for this_question in range(number_of_questions):
            # print(this_question)
            # slice this question: entries for this question, all subjects (expert or student separately)
            df_this_q = df_q_timestamps.loc[np.logical_and(df_q_timestamps['student_or_expert'] == student_or_expert,
                                                           df_q_timestamps['q_number'] == this_question)]
            df_this_q = df_this_q.sort_values(by=['subject'], ascending=True)
            # start and end points for this question (TR), for each subject
            trs_start = df_this_q['q_start_TR'].values + trs_to_add_to_start
            trs_end = df_this_q['q_end_TR'].values + trs_to_add_to_end
            assert np.sum((trs_end - trs_start) > 2), 'slice problem in {}-{}-{}-{}'.format(student_or_expert,
                                                                                            this_question, trs_end,
                                                                                            trs_start)
            # extract mean across TRs of epi data per question: vox X subjects
            epi_temporal_mean_per_q[this_question, :, :] = np.array(
                [np.nanmean(epi_data[i_subject][:, np.int(trs_start[i_subject]):np.int(trs_end[i_subject])], axis=1) for
                 i_subject in np.arange(len(trs_start))]).transpose()
        # update dict with epi data, mean over TRs
        corr_result[student_or_expert]['roi_epi_per_q'] = epi_temporal_mean_per_q

        if 'expert' in subject_groups:
            # calc expert template and use that for ISFC of students,experts
            if 'expert' in student_or_expert:
                if not (('questions' in test_vid_name) or ('placement' in test_vid_name)):
                    # 1) non-clean version - for use when no questions are involved - running just on video - qualifies all 'answers'
                    experts_clean_collapsed = np.nanmean(epi_temporal_mean_per_q, axis=2)
                else:
                    # 2) CLEAN version
                    clean_mask = (placement_by_q[
                                      'experts'] >= expert_accept_question_threshold).values.transpose()  # mask out wrong answers
                    # mask out bad expert responses - replace with nans
                    for v in np.arange(epi_temporal_mean_per_q.shape[1]):
                        epi_temporal_mean_per_q[:, v, :][np.invert(clean_mask)] = np.nan
                    # mean over experts
                    experts_clean_collapsed = np.nanmean(np.array(
                        [epi_temporal_mean_per_q[:, v, :] for v in np.arange(epi_temporal_mean_per_q.shape[1])]),
                        axis=2).transpose()
                # update dict (unused, here for clarity)
                corr_result[student_or_expert]['template'] = experts_clean_collapsed

            # for both students and experts
            # calc ISFC with experts as template
            # collapse over experts to create template, don't collapse over students to keep individual
            do_isfc_mean_over_subj = True if 'expert' in student_or_expert else False
            corr_result[student_or_expert]['isfc_mat'] = \
                isfc_local(corr_result[student_or_expert]['roi_epi_per_q'],
                           collapse_subj=do_isfc_mean_over_subj, external_signal=experts_clean_collapsed)

        # for students only: calc standard ISFC, leave one out
        if 'student' in student_or_expert:
            corr_result[student_or_expert]['isfc_mat'] = \
                isfc_local(corr_result[student_or_expert]['roi_epi_per_q'],
                           collapse_subj=False)

    # second, do row-by-row (q-by-q) correlation between st and ex matrices
    # get data
    students_mat = corr_result['student']['isfc_mat'].copy()
    if 'expert' in subject_groups:
        experts_mat = corr_result['expert']['isfc_mat'].copy()
    number_of_questions = students_mat.shape[0]
    number_of_subjects = students_mat.shape[2]
    # corr each question in each subject with experts' pattern for question and question-nots
    sim_rval = np.zeros([number_of_questions, number_of_subjects])  # similarity for each question, subject
    sim_dist = np.zeros([number_of_questions, number_of_subjects, num_perms])
    for this_question in np.arange(number_of_questions):
        # get (mean of) experts' pattern for this question
        if 'expert' in subject_groups:
            experts_this_q = experts_mat[this_question, :]
        for this_subject in np.arange(number_of_subjects):
            this_student_this_q = students_mat[this_question, :, this_subject]  # sim pattern for this st, this q
            group_this_q = np.mean(students_mat[this_question, :, np.arange(students_mat.shape[2]) != this_subject],
                                   axis=0)  # sim pattern for all other st, this q (ISC)
            # (1) sim_rval: correlate this question sim pattern in expert/group of students, this question sim pattern in student
            # but omit same-q corr, will be 1 for both experts and students, drive corr up artificially
            x = this_student_this_q[np.arange(len(this_student_this_q)) != this_question].copy()
            if 'experts' in vs_mean_of:
                # correlate this-question sim pattern in student, this-question patterns in experts
                y_experts = experts_this_q[np.arange(len(experts_this_q)) != this_question].copy()
                x_corr_y_rval, x_corr_y_dist = corr_and_null_dist(x, y_experts, num_perms=num_perms)
            elif 'students' in vs_mean_of:
                # correlate this-question sim pattern in student, this-question patterns in group of students
                y_students = group_this_q[np.arange(len(this_student_this_q)) != this_question].copy()
                x_corr_y_rval, x_corr_y_dist = corr_and_null_dist(x, y_students, num_perms=num_perms)
            sim_rval[this_question, this_subject] = x_corr_y_rval
            sim_dist[this_question, this_subject, :] = x_corr_y_dist

    # prep for out / corr and out
    # rval
    cx = sim_rval.copy()  # questions X students
    # pval
    # collapse across questions and students to get null dist for mean
    # sim_dist: questions X subjects X perms
    temp = np.tanh(np.nanmean(np.arctanh(sim_dist), axis=0))
    null_dist = np.tanh(np.nanmean(np.arctanh(temp), axis=0))
    if np.isnan(np.nanmean(cx)) or np.sum(np.isnan(null_dist)):
        px = np.nan
    else:
        px = isc.p_from_null(np.tanh(np.nanmean(np.arctanh(cx))), null_dist, side='right', exact=False)

    if 'within' in correlation_with_score.lower():  # correlate similarity score with placement score
        # for every subject, use per-q data, correlate similarity to experts (16 vals) with question score (16 vals): return mean over subjects
        corr_score_vec = np.zeros(cx.shape[1])  # rvals, vector size number of subjects
        corr_perm_dist = np.zeros((num_perms, cx.shape[1]))  # rand dist for each rval
        for this_subject in np.arange(number_of_subjects):
            # rvals and distribution for each, taken from that subject
            # x is sim
            x = cx[:, this_subject]
            y = placement_by_q['students'].iloc[this_subject].values  # y is vec q scores this subject
            # for within-length control: correlate sim with response length instead of question score
            if 'length' in correlation_with_score.lower():
                y = \
                    (df_q_timestamps[df_q_timestamps.subject == good_students[this_subject]]).sort_values(
                        by='q_number')[
                        'q_RT'].values
            corr_score_vec[this_subject], corr_perm_dist[:, this_subject] = corr_and_null_dist(x, y,
                                                                                               num_perms=num_perms)
        # rval summary - as in searchlight
        rval = isc.compute_summary_statistic(corr_score_vec)
        # pval
        within_dist = np.tanh(np.nanmean(np.arctanh(corr_perm_dist), axis=1))  # mean over subjects for each rand perm
        if np.isnan(rval) or np.sum(np.isnan(within_dist)):
            pval = np.nan
        else:
            pval = isc.p_from_null(rval, within_dist, side='right', exact=False)
        out_val = np.array([rval, pval])

    elif 'skip' in correlation_with_score.lower():  # do not correlate with placement score, output sim score as is
        out_val = np.array([np.tanh(np.nanmean(np.arctanh(cx))), px])

    return out_val


def sl_function(epi_list, msk, myrad, bcast_var):
    """
    searchlight function (do in every searchlight)
    :param epi_list: a list of 4D arrays, containing data from a single searchlight. len is double the number of subjects, because it is [students] followed by [experts plus nones]
    :param msk: a 3D binary array, mask of this searchlight
    :param myrad: -- an integer, sl_rad
    :param bcast_var: whatever is broadcast
    :return:
    sl_result
    """

    # time the searchlight call
    t0 = time.time()
    d1, d2, d3, ntr_train = epi_list[0].shape  # var ntr_train unused
    nvox = d1 * d2 * d3  # number of voxels

    # get data and reshape
    data_in_sl = dict()
    data_in_sl['students'] = {}
    data_in_sl['students']['test_data'] = []
    data_in_sl['experts'] = {}
    data_in_sl['experts']['test_data'] = []
    tt_data = 'test_data'

    # first half of dataset list ('test') is students, second half experts
    for s in epi_list[:bcast_var[2]]:  # students
        if s is not None:
            tt_ntr = s.shape[3]  # number of TRs, can vary between subjects
            data_in_sl['students'][tt_data].append(np.reshape(s, (nvox, tt_ntr)))
    for s in epi_list[bcast_var[2]:]:  # experts
        if s is not None:
            tt_ntr = s.shape[3]
            data_in_sl['experts'][tt_data].append(np.reshape(s, (nvox, tt_ntr)))

    # for testing with video instead of questions
    # adjust timestamps according to number of TRs read in epi list
    if not ('placement' in test_vid_name):
        local_df_q_timestamps = get_bins(
            epi_list[0].shape[-1])  # overwrite placement question time markers with equidistant bins
    else:  # placement exam data
        local_df_q_timestamps = df_q_timestamps

    # run corr
    if 'same' in similarity_type:  # 'diagonal/isc-like'
        sl_result = same_question_similarity(local_df_q_timestamps, placement_by_q, data_in_sl)  # 'diagonal'
    elif 'knowledge' in similarity_type:  # 'isfc-like'
        sl_result = knowledge_similarity(local_df_q_timestamps, placement_by_q, data_in_sl)  # 'isfc'

    t1 = time.time()

    return sl_result


def main(argv=None):
    # load argv
    if argv is None:
        argv = sys.argv

    # declare global params (for MPI)
    global test_vid_week
    global test_vid_name
    global similarity_type
    global correlation_with_score
    global vs_mean_of
    global sl_rad
    global num_perms
    global threshold
    global train_vid_week
    global train_vid_name

    # input params
    test_vid_week = sys.argv[1]  # 'wk1'..'wk6'
    test_vid_name = sys.argv[2]  # 'placement' or 'vid1'..'vid5' or 'wk1recap'..'wk5recap'
    similarity_type = sys.argv[3]  # 'same-question' or 'knowledge-structure'
    correlation_with_score = sys.argv[4]  # 'skip' or 'within' or 'direct'
    # vs mean of: calc similarity for student-vs-experts / student-vs-students
    vs_mean_of = sys.argv[5]  # 'student-vs-experts' or 'student-vs-students'
    # sl_rad #  searchlight radius
    sl_rad = int(sys.argv[6])  # usually 2
    # perms for stats: 0/1000
    num_perms = int(sys.argv[7])  # usually 1000
    # maps p-value threshold (FDR corrected) - one sided
    threshold = [0.05]


    # print
    print('Input video (=SRM test): {}-{}'.format(test_vid_week, test_vid_name))
    print('Similarity type: {}'.format(similarity_type))
    print('Correlate with score: {}'.format(correlation_with_score))
    print('student-experts or student-students: {}'.format(vs_mean_of))
    print('Searchlight edge: {}, size = {} voxels'.format(sl_rad, (1 + 2 * sl_rad) ** 3))
    print('Number of perms: {}'.format(num_perms))
    print('p-value threshold for maps, FDR corrected: {}'.format(threshold))

    # MPI - parallelization
    global comm, rank, size
    comm = MPI.COMM_WORLD
    rank = comm.rank  # rank = comm.Get_rank()
    size = comm.size  # size = comm.Get_size()
    print()
    print('mpi info')
    print(comm)
    print(rank)
    print(size)

    # load cortex mask
    global cortex_mask_3mm
    cortex_mask_3mm = io.load_boolean_mask(mni_cortex_file_name, lambda x: x > 0.05)  # CORTEX ONLY!

    # build dict for all filenames
    global student_and_expert_files, filenames_dict
    student_and_expert_files = listdir(input_fslfeat_students_path) + listdir(input_fslfeat_experts_path)
    filenames_dict = get_filenames(student_and_expert_files, task_name_template)

    # read logs and get behav data (placement and isq)
    global df_q_timestamps, placement_by_q, good_students, good_experts
    df_q_timestamps, placement_by_q = get_exam_behavior()
    good_students = placement_by_q['students'].index.tolist()
    good_experts = placement_by_q['experts'].index.tolist()

    # set parameters for questions
    global expert_accept_question_threshold, trs_to_add_to_end, trs_to_add_to_start
    expert_accept_question_threshold = 2  # for expert pattern used to compare student to, only consider equal-or-above-threshold (correct) answers: use 0 to qualify all answers
    if 'placement' in test_vid_name:  # exam: trim first 8s of question (before response)
        trs_to_add_to_end = 0
        trs_to_add_to_start = 4
    else:  # do not trim for video / recap
        trs_to_add_to_end = 0
        trs_to_add_to_start = 0

    # set param to get per-subject data when processing videos to get raw spatial isc
    global vids_per_subj_data_skip
    if ('vid' in test_vid_name or 'recap' in test_vid_name) and not ('direct' in correlation_with_score):
        vids_per_subj_data_skip = True
    else:
        # keep false for 'direct', no need for per-subj
        vids_per_subj_data_skip = False
    if vids_per_subj_data_skip:
        # only true for
        # lecture/recap vids when not direct; keep false for placement, in order to output map; keep false for recaps
        # and placement in direct
        print('Returns per-subject data - setting good for vids only')
    else:
        print(
            'Returns data collapsed over students - setting good for placement, not vids')

    # set param for 'direct' analysis
    global save_direct_placement_perms
    if 'recap' in test_vid_name and 'direct' in correlation_with_score:
        save_direct_placement_perms = True
    else:
        save_direct_placement_perms = False
    if save_direct_placement_perms:
        print('Returns perms for direct comparison within recaps, not map')

    # set parameters for searchlight
    # The size of the searchlight's radius, excluding the center voxel.
    # This means the total volume size of the searchlight, if using a cube, is defined as: ((2 * sl_rad) + 1) ^ 3.
    global max_blk_edge, pool_size, sl_mask, nfeature, niter
    max_blk_edge = 10  # size of block searchlight distributes
    pool_size = 1  # cores per task
    sl_mask = cortex_mask_3mm

    # for future SRM
    niter = 0
    nfeature = 0

    # sanity check for searchlight parameters
    if sl_rad <= 0:
        raise ValueError('sl_rad must be positive')

    # epi_list: data for searchlight
    # loc_split items in train dataset is students, the rest-experts
    global subject_groups, epi_list, loc_split
    if not 'wk6' in test_vid_week:  # expert data only in wk6; otherwise just st-st
        subject_groups = ['student']
        all_epi_data = load_data(['students'])
    else:  # wk6 placement/recaps
        subject_groups = ['expert', 'student']
        all_epi_data = load_data(['students', 'experts'])
    epi_list = []
    loc_split = len(all_epi_data['students']['test_data'])
    epi_list += all_epi_data['students']['test_data']
    epi_list += all_epi_data['experts']['test_data']

    # print searchlight info
    if rank == 0:
        print_sl_info(epi_list, sl_rad, loc_split)
    # call searchlight & time, setup a barrier so all tasks need to get here, save result
    comm.Barrier()
    t1 = time.time()
    comm.Barrier()
    # Create searchlight object
    sl = searchlight.Searchlight(sl_rad=sl_rad, max_blk_edge=max_blk_edge, min_active_voxels_proportion=0.2)
    # Distribute data to processes
    sl.distribute(epi_list, sl_mask)
    # broadcast something that should be shared by all ranks (for SRM params in future)
    sl.broadcast([niter, nfeature, loc_split])  # [srm:niter, srm:nfeature, student-expert split loc:loc_split]
    # Run searchlight
    sim = sl.run_searchlight(sl_function, pool_size=pool_size)  # output is a 3D array in shape (dim1,dim2,dim3)
    # Wait till every task is done then stop the timer.
    comm.Barrier()
    t2 = time.time()
    comm.Barrier()
    # save result
    if rank == 0:
        save_result(t1, t2, sim)


if __name__ == "__main__":
    main()

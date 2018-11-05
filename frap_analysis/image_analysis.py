import numpy as np
import pandas as pd
from datetime import timedelta
from skimage.morphology import binary_dilation, ball
from skimage.measure import regionprops

from tracker import track as tk
from img_manager import fv1000 as fv
from img_manager import corrector as corr
from img_manager import tifffile as tif
from foci_finder import foci_analysis as fa


def load_and_segment(path):
    """
    Loads the last version of an image and segments it returning the labeled stack.

    It first loads the last version of the image according to numbering procedure in fv1000. Runs find foci for each
    frame and returns the labeled stack.

    Parameters
    ----------
    path: pathlib.Path
        Path to the image file to load

    Returns
    -------
    foci_labeled: np.array
        Labeled array of the segmented image
    """
    img = fv.FV1000(str(path))
    path = img.get_last_path()
    img = fv.FV1000(str(path))

    stack = img.transpose_axes('CTZYX')[0]

    foci_labeled = np.zeros_like(stack)
    for frame, this_frame in enumerate(stack):
        foci_labeled[frame] = fa.find_foci(this_frame, LoG_size=[2, 2, 2])

    return foci_labeled


def load_and_track(path_labeled):
    """
    With path to labeled image, it looks for path to the original image and loads its last version. Calculates bleaching
    correction and corrects for bleaching and background. Generates a list of bleached particles inside the bleached
    region. Tracks every particle in the stack and returns a DataFrame with tracked particles information as well as
    particle labeled stack. Time is corrected to set 0 as the moment when bleaching ends.

    Parameters
    ----------
    path_labeled: pathlib.Path
        Path to the labeled image file to load

    Returns
    -------
    tracked: Pandas DataFrame
        DataFrame containing the information of each tracked particle in the set of images.
    particle_labeled: np.array
        stack labeled according to particle numbers
    """
    path = get_original_path(path_labeled)
    img = fv.FV1000(str(path))
    path = img.get_last_path()
    img = fv.FV1000(str(path))
    img_ble = fv.FV1000(str(img.get_other_path(kind='ble')))

    scales = img.get_scales()
    time_step = scales['T']
    stack = img.transpose_axes('CTZYX')[0]

    foci_img = tif.TiffFile(str(path_labeled))
    foci_labeled = foci_img.asarray().astype(int)

    corrector = corr.Corrector()

    bleached_bbox = img_ble.get_clip_bbox()
    bleached_bbox = expand_bbox(bleached_bbox, pixs=10)

    bleaching_stack = stack.copy()
    foci_mask = foci_labeled > 0
    foci_mask = np.asarray([binary_dilation(this_foci_mask, selem=ball(2)) for this_foci_mask in foci_mask])
    bleaching_stack[foci_mask] = np.nan
    bleaching_stack[:, :, bleached_bbox[0]:bleached_bbox[1], bleached_bbox[2]:bleached_bbox[3]] = np.nan
    corrector.find_bleaching(bleaching_stack, time_step)

    stack = corrector.subtract_and_normalize(stack, time_step)

    bbox = img_ble.get_clip_bbox()
    bleached_particles = find_bleached_particle(foci_labeled, bbox)
    tracked, particle_labeled = track(stack, foci_labeled, bleached_particles, scales, bbox)

    dead_time = find_bleching_end_time(img, img_ble)
    tracked['time'] = tracked.time.apply(lambda x: x + dead_time)

    return tracked, particle_labeled


def track(stack, foci_labeled, particles, scales, bbox):
    """Tracks the particles found in the foci_labeled stack, assigning them the mean intensity from stack. It generates
    a boolean column where True means that the particle was bleached. Returns time column as frames multiplied by scale.

    Parameters
    ----------
    stack: np.array
        image stack from which mean intensities are estimated
    foci_labeled: np.array
        foci labeled stack from which particles are tracked
    particles: list
        list of particle labels that have been bleached
    scales: dictionary
        dictionary containing the scale for each dimension
    bbox: list
        borders of the bleached region

    Returns
    -------
    tracked: pandas DataFrame
        DatFrame containing the information of the tracked particles
    particle_labeled: np.array
        labeled stack by each tracked particle
    """
    tracked = tk.track(foci_labeled.astype(int), max_dist=1, gap=0, scale=scales,
                       extra_attrs=['area', 'mean_intensity'], intensity_image=stack)
    tracked = tracked.reset_index(drop=True)
    tracked['bleached'] = tracked.label.isin(particles)
    tracked['time'] = tracked.frame.values * scales['T']

    particle_labeled = tk.relabel_by_track(foci_labeled, tracked)
    particles = find_bleached_particle(particle_labeled, bbox)
    tracked['bleached_particle'] = tracked.particle.isin(particles)

    return tracked, particle_labeled


def tracks_to_curves(df_pos_path, columns=['date', 'condition', 'experiment', 'cell'],
                     squash_columns=['time', 'X', 'Y', 'Z', 'mean_intensity', 'area', 'frame'],
                     df_pre_path=None):
    """Loads a post bleaching DataFrame, with its corresponding pre bleached DataFrame, groups them by columns and then
    squashes all the column information in squash columns make them a list in each cell where each row is one of the
    grouped by DataFrames. This allows having whole curve of timepoints in each cell.

    Parameters
    ----------
    df_pos_path: pathlib.Path
        path to the corresponding post bleached DataFrame. If df_pre_path is given, this variable can be a string.
    columns: (optional) list
        List of columns to be used to group by the DataFrames.
    squash_columns: (optional) list
        List of columns containing the timepoints that are to be saved in each cell as a list.
    df_pre_path: (optional) pathlib.Path, string
        path to the corresponding pre bleached DataFrame. Default is None and path is used to guess the pre bleached
        path

    Returns
    -------
    new_df: Pandas DataFrame
        DataFrame containing the information of intensity timepoints normalized to mean pre bleached intensities and
        ready to be fitted and analyzed by frap functions.

    """
    if df_pre_path is None:
        parts = df_pos_path.stem.split('_')
        parts[-1] = 'pre'
        df_pre_path = df_pos_path.with_name('_'.join(parts) + '.pandas')

    df_pos = pd.read_pickle(str(df_pos_path))
    df_pre = pd.read_pickle(str(df_pre_path))

    df = df_pos.append(df_pre, ignore_index=True)
    df['cell'] = df.cell.apply(lambda x: '_'.join(x.split('_')[:2]))
    columns.append('kind')

    # Squash DataFrame of bleached particles only
    new_df = pd.DataFrame()
    # TODO: take into account when more than one particle is bleached
    for column_values, this_df in df.groupby(columns):
        sel_df = this_df.query('bleached_particle')
        for particle, particle_df in sel_df.groupby('particle'):

            dict_to_df = {col: [particle_df[col].values] for col in squash_columns}
            this_df = pd.DataFrame(dict_to_df)
            this_df['particle'] = particle
            for val, col in zip(column_values, columns):
                this_df[col] = val

            new_df = new_df.append(this_df, ignore_index=True)

    # Find initial intensity and normalize it
    new_df['mean_pre_intensity'] = np.nan
    new_df['std_pre_intensity'] = np.nan
    new_df['intensity'] = np.nan
    new_df['intensity'] = new_df['intensity'].astype(object)

    for column_values, this_df in new_df.groupby(columns[:-1]):
        this_pre_df = this_df.query('kind == "pre"')
        this_pos_df = this_df.query('kind == "pos"')

        mean_pre_intensity = np.mean(this_pre_df.mean_intensity.values[0])
        std_pre_intensity = np.std(this_pre_df.mean_intensity.values[0])

        try:
            new_df.at[this_pre_df.index[0], 'mean_pre_intensity'] = mean_pre_intensity
            new_df.at[this_pos_df.index[0], 'mean_pre_intensity'] = mean_pre_intensity
            new_df.at[this_pre_df.index[0], 'std_pre_intensity'] = std_pre_intensity
            new_df.at[this_pos_df.index[0], 'std_pre_intensity'] = std_pre_intensity
            new_df.at[this_pre_df.index[0], 'intensity'] = np.asarray(
                new_df.at[this_pre_df.index[0], 'mean_intensity'] / mean_pre_intensity)
            new_df.at[this_pos_df.index[0], 'intensity'] = np.asarray(
                new_df.at[this_pos_df.index[0], 'mean_intensity'] / mean_pre_intensity)
        except IndexError:
            print('date: %s, condition: %s, exp: %s, cell: %s is missing a period.' % column_values)

    return new_df


def expand_bbox(bbox, pixs=10, max_bbox=256):
    """Expands bounding box by pixs limiting it to a maximum size of max_bbox."""
    bbox = np.asarray([_bbox - pixs if n % 2 == 0 else _bbox + pixs for n, _bbox in enumerate(bbox)])
    bbox = np.clip(bbox, 0, max_bbox)
    return bbox


def find_bleached_particle(stack, bbox):
    """Lists particles inside the bbox region."""
    bleached_box = stack[0, :, bbox[0]:bbox[1], bbox[2]:bbox[3]]
    particles = []
    for region in regionprops(bleached_box):
        particles.append(region.label)

    # If no particle is found in first image, intensity might be too small so keep searching

    if len(particles) < 1:
        t = 1
        while len(particles) < 1 and t < stack.shape[0]:
            bleached_box = stack[t, :, bbox[0]:bbox[1], bbox[2]:bbox[3]]
            t += 1
            for region in regionprops(bleached_box):
                particles.append(region.label)

    return particles


def get_original_path(path):
    """Uses given path to labeled image to look for the originating image."""
    label_dir = path.parents[4]
    label_dir = label_dir.joinpath('data')
    parts = path.parts[-4:]
    for i in range(3):
        label_dir = label_dir.joinpath(parts[i])
    return label_dir.joinpath(path.stem + '.oif')


def find_bleching_end_time(img, img_ble):
    """Calculates when was the end of bleaching period."""
    abs_start_pos = img.get_acquisition_time()
    abs_start_ble = img_ble.get_acquisition_time()
    t_step_ble = img_ble.get_t_step()
    len_ble = img_ble.get_axes_info('T')['MaxSize']
    len_ble_seconds = timedelta(seconds=len_ble * t_step_ble)
    dead_time = abs_start_pos - (abs_start_ble + len_ble_seconds)

    return dead_time.total_seconds()


# Citoplasm analysis

def get_intensity_df(stack, bbox):
    """Returns a DataFrame containing information about the intensity in the bbox region.

    Parameters
    ----------
    stack: np.array
        stack of images from which intensity must be estimated
    bbox: list
        borders of the bleached region

    Returns
    -------
    pandas DataFrame
        DataFrame containing information of the bleached area.

    """
    sel_stack = stack[:, bbox[0]:bbox[1], bbox[2]:bbox[3]]
    area = (bbox[1] - bbox[0]) * (bbox[3] - bbox[2])
    variable_dict = {'sum_intensity': np.nansum(sel_stack, axis=(1, 2)),
                     'mean_intensity': np.nanmean(sel_stack, axis=(1, 2)),
                     'std_intensity': np.nanstd(sel_stack, axis=(1, 2)),
                     'median_intensity': np.nanmedian(sel_stack, axis=(1, 2)),
                     'frame': [i for i in range(sel_stack.shape[0])],
                     'area': [area, ] * sel_stack.shape[0]}

    return pd.DataFrame(variable_dict)


def load_and_correct_citoplasmic(path):
    """Loads last version of the image at path and corrects for bleaching and background. Returns a DataFrame containing
    information about the bleached area.

    Parameters
    ----------
    path: pathlib.Path
        path to the image to be analyzed

    Returns
    -------
    df: pandas DataFrame
        DataFrame containing information of the set of bleached areas in the analyzed stack

    """
    img = fv.FV1000(str(path))
    path = img.get_last_path()
    img = fv.FV1000(str(path))
    img_ble = fv.FV1000(str(img.get_other_path(kind='ble')))

    scales = img.get_scales()
    time_step = scales['T']
    stack = img.transpose_axes('CTYX')[0]

    corrector = corr.Corrector()

    bleached_bbox = img_ble.get_clip_bbox()
    bleached_bbox = expand_bbox(bleached_bbox, pixs=10)

    bleaching_stack = stack.copy()
    bleaching_stack[:, bleached_bbox[0]:bleached_bbox[1], bleached_bbox[2]:bleached_bbox[3]] = np.nan
    corrector.find_bleaching(bleaching_stack, time_step)

    stack = corrector.subtract_and_normalize(stack, time_step)

    bbox = img_ble.get_clip_bbox()
    df = get_intensity_df(stack, bbox)
    df['time'] = df.frame.apply(lambda x: x * time_step)

    dead_time = find_bleching_end_time(img, img_ble)
    df['time'] = df.time.apply(lambda x: x + dead_time)

    return df


def citoplasm_to_curves(df_pos_path, columns=['date', 'condition', 'experiment', 'cell'],
                     squash_columns=['time', 'sum_intensity', 'mean_intensity',
                                     'std_intensity', 'median_intensity', 'frame', 'area'],
                     df_pre_path=None):
    """Loads a DataFrame containing the information of a various bleached areas and prepares a DataFrame that can be
    analyzed by frap analysis function. DataFrames are grouped by columns and then squashes all the column information
    in squash columns make them a list in each cell where each row is one of the grouped by DataFrames. This allows
    having whole curve of timepoints in each cell.

    Parameters
    ----------
    df_pos_path: pathlib.Path
        path to the corresponding post bleached DataFrame. If df_pre_path is given, this variable can be a string.
    columns: (optional) list
        List of columns to be used to group by the DataFrames.
    squash_columns: (optional) list
        List of columns containing the timepoints that are to be saved in each cell as a list.
    df_pre_path: (optional) pathlib.Path, string
        path to the corresponding pre bleached DataFrame. Default is None and path is used to guess the pre bleached
        path

    Returns
    -------
    new_df: Pandas DataFrame
        DataFrame containing the information of intensity timepoints normalized to mean pre bleached intensities and
        ready to be fitted and analyzed by frap functions.

    """

    if df_pre_path is None:
        parts = df_pos_path.stem.split('_')
        parts[-1] = 'pre'
        df_pre_path = df_pos_path.with_name('_'.join(parts) + '.pandas')

    df_pos = pd.read_pickle(str(df_pos_path))
    df_pre = pd.read_pickle(str(df_pre_path))

    df = df_pos.append(df_pre, ignore_index=True)
    df['cell'] = df.cell.apply(lambda x: '_'.join(x.split('_')[:2]))
    columns.append('kind')

    # Squash DataFrame of bleached particles only
    new_df = pd.DataFrame()
    for column_values, this_df in df.groupby(columns):

        dict_to_df = {col: [this_df[col].values] for col in squash_columns}
        this_df = pd.DataFrame(dict_to_df)
        this_df['particle'] = 1
        for val, col in zip(column_values, columns):
            this_df[col] = val

        new_df = new_df.append(this_df, ignore_index=True)

    # Find initial intensity and normalize it
    new_df['mean_pre_intensity'] = np.nan
    new_df['std_pre_intensity'] = np.nan
    new_df['intensity'] = np.nan
    new_df['intensity'] = new_df['intensity'].astype(object)

    for column_values, this_df in new_df.groupby(columns[:-1]):
        this_pre_df = this_df.query('kind == "pre"')
        this_pos_df = this_df.query('kind == "pos"')

        mean_pre_intensity = np.mean(this_pre_df.median_intensity.values[0])
        std_pre_intensity = np.std(this_pre_df.median_intensity.values[0])

        try:
            new_df.at[this_pre_df.index[0], 'mean_pre_intensity'] = mean_pre_intensity
            new_df.at[this_pos_df.index[0], 'mean_pre_intensity'] = mean_pre_intensity
            new_df.at[this_pre_df.index[0], 'std_pre_intensity'] = std_pre_intensity
            new_df.at[this_pos_df.index[0], 'std_pre_intensity'] = std_pre_intensity
            new_df.at[this_pre_df.index[0], 'intensity'] = np.asarray(
                new_df.at[this_pre_df.index[0], 'median_intensity'] / mean_pre_intensity)
            new_df.at[this_pos_df.index[0], 'intensity'] = np.asarray(
                new_df.at[this_pos_df.index[0], 'median_intensity'] / mean_pre_intensity)
        except IndexError:
            print('date: %s, condition: %s, exp: %s, cell: %s is missing a period.' % column_values)

    return new_df

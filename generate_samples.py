'''
This script will create samples composed of concatenations of utterances
from a same speaker


'''
import random, os, json, pickle
from itertools import combinations, permutations
import numpy as np
from scipy.io import wavfile
import pyroomacoustics as pra

def sampling(num_subsets, num_speakers, metadata_file, gender_balanced=False, seed=None):
    '''
    This function will pick automatically and at random subsets
    speech samples from a list generated using this file.
    '''

    # save current numpy RNG state and set a known seed
    if seed is not None:
        rng_state = random.getstate()
        random.seed(seed)

    samples_dir = os.path.split(metadata_file)[0]

    # load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    if gender_balanced:
        samples = [metadata['sorted']['male'], metadata['sorted']['female']]
        numbers = [num_speakers // 2, num_speakers - num_speakers // 2]
    else:
        samples = [metadata['sorted']['male'].copy()]
        samples[0].update(metadata['sorted']['female'])
        numbers = [num_speakers]

    #  now create a list of list of files
    all_lists = [[] for i in range(num_subsets)]

    for S, num in zip(samples, numbers):

        speakers = list(S.keys())
        random.shuffle(speakers)
        all_combs = list(combinations(speakers, num))

        for sub_list in all_lists:
            spkrs = list(random.choice(all_combs))
            random.shuffle(spkrs)
            for spkr in spkrs:
                sub_list.append(
                        os.path.join(samples_dir, random.choice(S[spkr])))

    # restore numpy RNG former state
    if seed is not None:
        random.setstate(rng_state)

    return all_lists


def wav_read_center(wav_list, center=True, seed=None):
    '''
    Read a bunch of wav files, equalize their length
    and puts them in a numpy array

    Parameters
    ----------
    wav_list: list of str
        A list of file names, the file names should be of format wav and monaural
    center: bool, optional
        When True (default), the signals will be centered, otherwise, only their end will be zero padded
    seed: int
        Provides a seed for the random number generator. When this is provided,
        center option is ignored and the beginning of segments is placed at
        random within the maximum length available

    Returns
    -------
    ndarray (n_files, n_samples)
        A 2D array that contains one signal per row
    '''

    if seed is not None:
        rng_state = np.random.get_state()
        np.random.seed(seed)

    rows = []
    fs = None

    for fn in wav_list:

        fs_loc, data = wavfile.read(fn)
        data = data.astype(np.float)

        if fs is None:
            fs = fs_loc

        if fs != fs_loc:
            raise ValueError('All the files should have the same sampling frequency')

        if data.ndim > 1:
            import warnings
            warnings.warn('Discarding extra channels of non-monaural file')
            data = data[:,0]

        rows.append(data)

    max_len = np.max([d.shape[0] for d in rows])

    output = np.zeros((len(rows), max_len), dtype=rows[0].dtype)

    for r,row in enumerate(rows):

        if seed is not None:
            slack =  max_len - row.shape[0]
            if slack > 0:
                b = np.random.randint(0, max_len - row.shape[0])
            else:
                b = 0
        elif center:
            b = (max_len - row.shape[0]) // 2
        else:
            b = 0

        output[r,b:b+row.shape[0]] = row

    if seed is not None:
        np.random.set_state(rng_state)

    output /= 2 ** 15

    return output

        
if __name__ == '__main__':

    n_speakers_per_sex = 7
    n_samples = 10
    duration = 15  # seconds
    blank = 1  # seconds

    # output filename pattern
    output_dir = 'samples'
    filename = 'cmu_arctic_{sex}_{spkr}_{ind}.wav'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # load database and cache locally, or load cached db if existing
    cmu_cache_file = '.cmu_arctic.dat'
    if not os.path.exists(cmu_cache_file):
        print('Cache DB doesn''t exist, loading from scratch')
        cmu_arctic = pra.datasets.CMUArcticCorpus(basedir='/Volumes/datanet/datasets/CMU_ARCTIC')
        with open(cmu_cache_file, 'wb') as f:
            pickle.dump(cmu_arctic, f)
    else:
        print('Cache DB exists, loading')
        with open(cmu_cache_file, 'rb') as f:
            cmu_arctic = pickle.load(f)

    # get data type and sampling frequency of dataset
    dtype = cmu_arctic.sentences[0].data.dtype
    fs = cmu_arctic.sentences[0].fs


    # a blank segment to insert between sentences
    lmin, lmax = int(0.5) * fs, int(2.5) * fs
    def new_blank():
        return np.zeros(lmin + np.random.randint(lmin, lmax), dtype=dtype)

    # keep track of metadata in a file
    metadata = {
            'fs' : fs,
            'files' : [],
            'sorted' : {},
            }

    # iterate over different speakers to create the concatenated sentences
    for sex in ['male', 'female']:

        ds_sex = cmu_arctic.filter(sex=sex)
        speakers = list(ds_sex.info['speaker'].keys())
        metadata['sorted'][sex] = {}

        for speaker in speakers[:n_speakers_per_sex]:

            ds_spkr = ds_sex.filter(speaker=speaker)
            random.shuffle(ds_spkr.sentences)
            sentence_iter = iter(ds_spkr.sentences)
            metadata['sorted'][sex][speaker] = []

            n = 0
            while n < n_samples:

                new_sentence = [new_blank()]
                L = len(new_sentence[-1])

                while L < duration * fs:

                    s = next(sentence_iter)
                    new_sentence.append(s.data)
                    L += s.data.shape[0]
                    new_sentence.append(new_blank())
                    L += len(new_sentence[-1])

                fn = filename.format(sex=sex, spkr=speaker, ind=n+1)
                wavfile.write(
                        os.path.join(output_dir, fn),
                        fs,
                        np.concatenate(new_sentence),
                        )
                metadata['sorted'][sex][speaker].append(fn)
                metadata['files'].append(fn)
                n += 1

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)

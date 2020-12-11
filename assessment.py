'''Script for the automatic assessment of the intonation of monophonic singing.

Expected usage:

assessment.py --performance performance_data_84.json --pitch pitch_84.json

'''

import sys
eps = sys.float_info.epsilon
import os
import json

import numpy as np
import mir_eval
import pandas as pd

import music21 as m21
import pretty_midi
import unidecode

from mir_eval.util import midi_to_hz, intervals_to_samples

import argparse
import urllib.request

# UTIL FUNCTIONS

# ------------------------------------------------------------------------------------------ #
def xml2midi(xmlfile):

    try:
        score = m21.converter.parseFile(xmlfile)

    except:
        raise ValueError("Can not parse the score. Aborting assessment...")

    if xmlfile.endswith('xml'):
        score.write('midi', xmlfile.replace('xml', 'mid'))

    elif xmlfile.endswith('mxl'):
        score.write('midi', xmlfile.replace('mxl', 'mid'))

    else:
        raise ValueError("Please input a valid score format: xml or mxl")

# ------------------------------------------------------------------------------------------ #

def midi_preparation(midifile):
    midi_data = dict()
    midi_data['onsets'] = dict()
    midi_data['offsets'] = dict()
    midi_data['midipitches'] = dict()  # midi notes?
    midi_data['hz'] = dict()

    patt = pretty_midi.PrettyMIDI(midifile)
    midi_data['downbeats'] = patt.get_downbeats()

    for instrument in patt.instruments:
        midi_data['onsets'][instrument.name] = []
        midi_data['offsets'][instrument.name] = []
        midi_data['midipitches'][instrument.name] = []

        for note in instrument.notes:
            midi_data['onsets'][instrument.name].append(note.start)
            midi_data['offsets'][instrument.name].append(note.end)
            midi_data['midipitches'][instrument.name].append(note.pitch)

        p = midi_data['midipitches'][instrument.name]
        midi_data['hz'][instrument.name] = midi_to_hz(np.array(p))

    return midi_data

# ------------------------------------------------------------------------------------------ #

def midi_to_trajectory(des_timebase, onsets, offsets, pitches):

    hop = des_timebase[2] - des_timebase[1]

    intervals = np.concatenate([np.array(onsets)[:, None], np.array(offsets)[:, None]], axis=1)

    timebase, midipitches = intervals_to_samples(intervals, list(pitches),
                                                 offset=des_timebase[0], sample_size=hop, fill_value=0)

    return np.array(timebase), np.array(midipitches)

# ------------------------------------------------------------------------------------------ #

def parse_midi(score_fname, voice_shortcut):

    voice_shortcut = unidecode.unidecode(voice_shortcut)

    if score_fname.endswith('xml'):
        midi_data = midi_preparation(score_fname.replace('xml', 'mid'))

    elif score_fname.endswith('mxl'):
        midi_data = midi_preparation(score_fname.replace('mxl', 'mid'))

    else:
        raise ValueError("Invalid score format. Found {} but expected {} or {}".format(score_fname[-3:], 'xml', 'mxl'))

    onsets = np.array(midi_data['onsets'][voice_shortcut])

    offsets = np.array(midi_data['offsets'][voice_shortcut])
    pitches = np.array(midi_data['hz'][voice_shortcut])

    return onsets, offsets, pitches, midi_data

# ------------------------------------------------------------------------------------------ #

def load_json_data(load_path):
    with open(load_path, 'r') as fp:
        data = json.load(fp)
    return data

# ------------------------------------------------------------------------------------------ #

def save_json_data(data, save_path):
    with open(save_path, 'w') as fp:
        json.dump(data, fp, indent=2)

# ------------------------------------------------------------------------------------------ #

def load_f0_contour(pitch_json_path, starttime):

    pitch = np.array(load_json_data(pitch_json_path)['pitch'])


    times_ = pitch[:, 0]
    freqs_ = pitch[:, 1]

    times_shift = times_ - np.abs(starttime)

    idxs_no = np.max(np.where(times_shift < 0)[0])

    times = times_shift[idxs_no + 1:]

    if not type(times[0]) == np.float64:
        raise ValueError("Problem with F0 contour")

    if times[0] != 0:
        offs = times[0]
        times -= offs

    freqs = freqs_[idxs_no + 1:]

    return times, freqs

# ------------------------------------------------------------------------------------------ #


def map_deviation_range(input_deviation, max_deviation=100):
    '''This function takes as input the deviation between the score and the performance in cents (as a ratio),
    and computes the output value mapping it into the range 0-1, (0 is bad intonation and 1 is good intonation).
    By default, we limit the deviation to max_deviation cents, which is one semitone. Values outside the range +-100 cents
    will be clipped and counted as intonation score = 0.
    '''

    score = np.clip(np.abs(input_deviation), 0, max_deviation) / float(max_deviation)

    assert score <= 1, "Score value is above 1"
    assert score >= 0, "Score value is below 0"

    return 1 - score


# ------------------------------------------------------------------------------------------ #

def intonation_assessment(startbar, endbar, offset, pitch_path, score_path, voice, dev_thresh=100):

    '''Automatic assessment of the intonation of singing performances from the CSP platform of the TROMPA project.


    Parameters
    ----------
    startbar : (int) indicates the first bar of the performance

    endbar : (int) indicates the last bar of the performance

    offset : (float) measured latency between audio and score

    pitch_json_path : (string) path to the json file with the pitch contour

    score_path : (string) path to the score in xml

    voice : (string) voice part as written in the score

    dev_thresh : (float) maximum allowed deviation in cents. Defaults to 100 cents

    Returns
    -------
    assessment : (dictionary) with the assessment results for each note in the 'pitchAssessment' field \n
    overall_score : (float) overall intonation score computed as the weighted sum of note intonation scores

    '''

    assessment = {}
    assessment['pitchAssessment'] = []

    try:

        '''STEP 1: parse xml score, convert to MIDI and save 
        '''
        # hack to deal with bariton voice with an accent, needs updating

        if voice == 'Baríton':
            xml_data = m21.converter.parse(score_path)
            xml_data.parts[2].getInstrument().partName = 'Bariton'
            xml_data.write('midi', score_path.replace('xml', 'mid'))

        else:
            xml2midi(score_path)

        '''STEP 2: parse MIDI file and arrange info
        '''
        onsets, offsets, pitches, midi_data = parse_midi(score_path, unidecode.unidecode(voice))

        '''STEP 3: parse the F0 contour and adjust according to latency
        '''
        # if latency is larger than 1 second it's likely and error, we set it to 0.3 by default
        if offset >= 1:
            offset = 0.3

        times, freqs = load_f0_contour(pitch_path, starttime=offset)

        '''STEP 4: Delimiting the performance in the score and the F0 curve
        '''
        starting = midi_data['downbeats'][int(startbar) - 1]  # bars start at 1, indices at 0

        # Account for the case of last bar being the last of the piece and size mismatch
        if int(endbar) >= len(midi_data['downbeats']):
            ending = offsets[-1]

        else:
            ending = midi_data['downbeats'][int(endbar)] - 0.005

        st_idx = np.where(onsets >= starting)[0][0]
        end_idx = np.where(offsets >= ending)[0][0]

        # getting info from notes according to the sung audio excerpt
        onsets, offsets, pitches = onsets[st_idx:end_idx + 1], offsets[st_idx:end_idx + 1], pitches[st_idx:end_idx + 1]

        # If all freqs are 0, there's no singing in the performance, we return 0
        if sum(freqs) == 0:
            assessment['pitchAssessment'] = [np.array([onset, 0]) for onset in onsets]
            overall_score = 0
            return assessment, overall_score


        try:
            st_idx = np.where(times + starting >= starting)[0][0]

        except:

            raise ValueError("Recording not valid, does not contain the performance.")

        try:
            end_idx = np.where(times + starting >= ending)[0][0]
        except:
            end_idx = -1

        '''STEP 5: Converting the MIDI info to a F0 trajectory for easier comparison. Resampling to a common 
        time base.
        '''
        ref_times, ref_freqs = midi_to_trajectory(times[st_idx:end_idx] + starting, onsets, offsets, pitches)
        times += ref_times[0]

        # Resample to the same timebase. We use the reference timebase
        freqs, voicing = mir_eval.melody.freq_to_voicing(freqs)
        est_freqs, _ = mir_eval.melody.resample_melody_series(times, freqs, voicing, ref_times, kind='nearest')


        '''STEP 6: Compute intonation score as the average (median) deviations for each note in the excerpt.
        '''
        note_deviations = []
        ratings = []
        for i in range(len(onsets)):

            # indices of the note region
            region_idxs = np.where((ref_times >= onsets[i]) & (ref_times < offsets[i]))[0]
            note_start, note_end = region_idxs[0], region_idxs[-1]

            # compute deviation frame-wise
            if pitches[i] < eps: pitches[i] = eps
            devs = 1200.0 * np.log2(est_freqs[note_start:note_end] / pitches[i])

            note_median_dev = np.median(devs)
            note_deviations.append(note_median_dev)

            # map deviation to range [0, 1]
            intonation_score = map_deviation_range(note_median_dev, max_deviation=dev_thresh)
            ratings.append(intonation_score)

            # store intonation score in the output dictionary
            # assessment['pitchAssessment'].append(
            #     np.array(
            #         [onsets[i], intonation_score]
            #     )
            # )
            assessment['pitchAssessment'].append(
                    [onsets[i], intonation_score]
            )

        assert len(onsets) == len(assessment['pitchAssessment']), "Number of ratings differs from number of notes."


        # Idea for a weighted overall score
        durations = offsets - onsets
        durations /= offsets[-1]
        overall_score = np.dot(ratings, durations)

        return assessment, overall_score

    except:
        print("Exception caught")
        assessment['error'] = 'Process failed.'
        overall_score = 0
        return assessment, overall_score


def main(args):
    performance_path = args.performance_path
    pitch_path = args.pitch_path

    data = load_json_data(performance_path)


    if not os.path.exists('./tmp'):
        os.mkdir('./tmp')


    startbar = data['startBar']
    endbar = data['endBar']
    voice = data['partName']
    latency = data['latencyOffset']

    score_url = data['score']
    try:
        _ = urllib.request.urlretrieve(score_url, './tmp/score.xml')
        score_path = './tmp/score.xml'

        assessment, _ = intonation_assessment(startbar, endbar, latency, pitch_path, score_path, voice, dev_thresh=100)

        #import pdb; pdb.set_trace()

        save_json_data(assessment, performance_path.replace('.json', '_output.json'))
        #pd.DataFrame(assessment).to_json(performance_path.replace('.json', '_output.json'))


    except Exception as e:
        print(e)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the intonation assessment algorithm given the json input with info from the performance "
                    "the json file with the F0 contour.")

    parser.add_argument("--performance",
                        dest='performance_path',
                        type=str,
                        help="Path to the json file with the performance data.")

    parser.add_argument("--pitch",
                        dest='pitch_path',
                        type=str,
                        help="Path to the json file with the F0 contour.")


    main(parser.parse_args())
'''Script for the automatic assessment of the intonation of monophonic singing.
Prototype for the CSP of the TROMPA project.

'''

import sys
eps = sys.float_info.epsilon
import json

import numpy as np
import mir_eval
import pandas as pd

import music21 as m21
import pretty_midi
import unidecode

from mir_eval.util import midi_to_hz, intervals_to_samples

import argparse


def xml2midi(xmlfile, format):

    try:
        score = m21.converter.parseFile(xmlfile, format=format)

    except:
        raise RuntimeError("Can not parse the {} score.".format(format))

    try:
        score.write('midi', '{}.mid'.format(xmlfile))

    # if xmlfile.endswith('xml'):
    #     score.write('midi', xmlfile.replace('xml', 'mid'))
    #
    # elif xmlfile.endswith('mxl'):
    #     score.write('midi', xmlfile.replace('mxl', 'mid'))

    except:
        raise RuntimeError("Could not convert {} to MIDI.".format(format))


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

def midi_to_trajectory(des_timebase, onsets, offsets, pitches):

    hop = des_timebase[2] - des_timebase[1]

    intervals = np.concatenate([np.array(onsets)[:, None], np.array(offsets)[:, None]], axis=1)

    timebase, midipitches = intervals_to_samples(intervals, list(pitches),
                                                 offset=des_timebase[0], sample_size=hop, fill_value=0)

    return np.array(timebase), np.array(midipitches)

def parse_midi(score_fname, voice_shortcut, format):

    voice_shortcut = unidecode.unidecode(voice_shortcut)

    try:
        midi_data = midi_preparation("{}.mid".format(score_fname))

    except:
        raise RuntimeError("Could not parse converted MIDI file".format(score_fname))

    onsets = np.array(midi_data['onsets'][voice_shortcut])

    offsets = np.array(midi_data['offsets'][voice_shortcut])
    pitches = np.array(midi_data['hz'][voice_shortcut])

    return onsets, offsets, pitches, midi_data

def load_json_data(load_path):
    with open(load_path, 'r') as fp:
        data = json.load(fp)
    return data

def save_json_data(data, save_path):
    with open(save_path, 'w') as fp:
        json.dump(data, fp, indent=2)

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


def map_deviation_range(input_deviation, max_deviation=100):
    '''This function takes as input the deviation between the score and the performance in cents (as a ratio),
    and computes the output value mapping it into the range 0-1, (0 is bad intonation and 1 is good intonation).
    By default, we limit the deviation to max_deviation cents, which is one semitone. Values outside the range +-100 cents
    will be clipped and counted as intonation score = 0.
    '''

    score = np.clip(np.abs(input_deviation), 0, max_deviation) / float(max_deviation)

    # assert score <= 1, "Score value is above 1"
    # assert score >= 0, "Score value is below 0"

    return 1 - score



def intonation_assessment(startbar, endbar, offset, pitch_json_file, score_file, voice, output_filename, dev_thresh=100, format='xml'):

    '''Automatic assessment of the intonation of singing performances from the CSP platform of the TROMPA project.


    Parameters
    ----------
    startbar : (int) indicates the first bar of the performance

    endbar : (int) indicates the last bar of the performance

    offset : (float) measured latency between audio and score

    pitch_json_file : (string) json file with the pitch contour

    score_file : (string) music score xml file

    voice : (string) voice part as written in the score

    output_filename : (string) output filename to use for the assessment results file

    dev_thresh : (float) maximum allowed deviation in cents. Defaults to 100 cents

    Returns
    -------

    assessment : (dictionary) the field 'pitchAssessment' contains a list of arrays with the results for each note in
    in the form [note_start_time, intonation_rating]. If the process fails, the list will be empty. The field 'error'
    will contain a string with an error message if the process fails, and will be None if it's successful.

    overall_score : (float) overall intonation score computed as the weighted sum of note intonation scores. Can be
    ignored because it's not used by the CSP.

    This function stores a json file with the assessment dictionary in the file indicated by the `output_filename`
    parameter.

    '''

    assessment = {}
    assessment['pitchAssessment'] = []
    assessment['error'] = None

    try:

        '''STEP 1: parse xml score, convert to MIDI and save 
        '''
        # quick hack to deal with accents in the voice parts, needs to be updated
        change_flag = 0
        xml_data = m21.converter.parse(score_file, format=format)

        for i in range(len(xml_data.parts)):
            name = xml_data.parts[i].getInstrument().partName
            if name != unidecode.unidecode(name):
                change_flag = 1
                xml_data.parts[i].getInstrument().partName = unidecode.unidecode(name)

        if change_flag != 0:

            try:
                xml_data.write('midi', "{}.mid".format(score_file))

            except:
                raise RuntimeError("Could not convert modified {} to MIDI.".format(format))

        else:
            xml2midi(score_file, format=format)

        import pdb; pdb.set_trace()
        #
        # if voice == 'Baríton':
        #     xml_data = m21.converter.parse(score_file)
        #     xml_data.parts[2].getInstrument().partName = 'Bariton'
        #     xml_data.write('midi', score_file.replace('xml', 'mid'))
        #
        # else:
        #     xml2midi(score_file)

        '''STEP 2: parse MIDI file and arrange info
        '''
        onsets, offsets, pitches, midi_data = parse_midi(score_file, unidecode.unidecode(voice), format)

        '''STEP 3: parse the F0 contour and adjust according to latency
        '''
        # if latency is larger than 1 second it's likely and error, we set it to 0.3 by default
        if offset >= 1:
            offset = 0.3

        times, freqs = load_f0_contour(pitch_json_file, starttime=offset)

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

        # resampling timebase because of irregular steps
        times = np.linspace(times[0], times[-1], len(times))

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
            # replace 0 by eps for the log

            if pitches[i] < eps: pitches[i] = eps

            est_freqs[note_start:note_end][est_freqs[note_start:note_end] <= 0] = eps

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



        # Idea for a weighted overall score
        durations = offsets - onsets
        durations /= offsets[-1]
        overall_score = np.dot(ratings, durations)

        '''Store the ratings in a json file
        '''
        save_json_data(assessment, output_filename)

        return assessment, overall_score

    except:

        assessment['error'] = 'Something went wrong during the assessment process.'
        overall_score = 0

        save_json_data(assessment, output_filename)

        return assessment, overall_score


def main(args):

    startbar = args.startbar
    endbar = args.endbar
    offset = args.offset
    pitch_json_file = args.pitch_json
    score_file = args.score_file
    voice = args.voice
    output_filename = args.output_filename
    dev_threshold = args.dev_threshold
    score_format = args.score_format

    _, _ = intonation_assessment(startbar, endbar, offset, pitch_json_file, score_file, voice,
                                 output_filename, dev_thresh=dev_threshold, format=score_format)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the intonation assessment algorithm given the json input with info from the performance "
                    "and the json file with the F0 contour.")

    parser.add_argument("--start_bar",
                        dest='startbar',
                        type=int,
                        help="Start bar of the performance.")

    parser.add_argument("--end_bar",
                        dest='endbar',
                        type=int,
                        help="End bar of the performance.")

    parser.add_argument("--latency",
                        dest='offset',
                        type=float,
                        help="Estimated latency offset between the score and the performance.")

    parser.add_argument("--pitch_json",
                        dest='pitch_json',
                        type=str,
                        help="Filename of the json file containing the pitch curve.")

    parser.add_argument("--score_file",
                        dest='score_file',
                        type=str,
                        help="Filename of the xmlfile containing with the score.")

    parser.add_argument("--voice",
                        dest='voice',
                        type=str,
                        help="Voice part of the singer.")

    parser.add_argument("--output_filename",
                        dest='output_filename',
                        type=str,
                        help="Filename of the results output file.")

    parser.add_argument("--dev_threshold",
                        dest='dev_threshold',
                        type=float,
                        default=100.0,
                        help="Maximum allowed deviation from the score in cents. Defaults to 100.")

    parser.add_argument("--format",
                        dest='score_format',
                        type=str,
                        default='xml',
                        help="Format of the score. Defaults to XML.")


    main(parser.parse_args())
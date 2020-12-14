# Intonation Assessment v0

This is a work in progress.

Helena Cuesta (helena.cuesta@upf.edu)

This repo contains (work in progress) code for the automatic assessment of the intonation of singing performances.
It's developed for the Choir Singers Platform of the TROMPA project.

## Documentation 

The main function is `intonation_assessment`, inside the `assessment.py` script.

This function has the following input parameters:

### Input parameters

    startbar : (int) indicates the first bar of the performance

    endbar : (int) indicates the last bar of the performance

    offset : (float) measured latency between audio and score

    pitch_json_file : (string) filename of the json file with the pitch contour

    score_file : (string) filename of the score xml file

    voice : (string) voice part as written in the score

    output_filename : (string) output filename to use for the assessment results file

    dev_thresh : (float) maximum allowed deviation in cents. Defaults to 100 cents


### Output parameters

    assessment : (dictionary) the field 'pitchAssessment' contains a list of arrays with the results for each note in
    in the form [note_start_time, intonation_rating]. If the process fails, the list will be empty. The field 'error'
    will contain a string with an error message if the process fails, and will be None if it's successful.

    overall_score : (float) overall intonation score computed as the weighted sum of note intonation scores. Can be
    ignored because it's not used by the CSP.
    
 
*Note: The `overall_score` output is a preliminary attempt to compute an intonation score the for whole performance. 
Since this is not used by the rehearsal platform, the variable can be ignored for now.*

### Output json file

In the current version, this script stores an **json file** as `output_filename.json` with the assessment results.
This json file has two fields: `pitchAssessment` and `error`.

The  `pitchAssessment` field contains a list of tuples `[note_start_time, pitch_value]`, with as many tuples as notes
in the performance. If the process fails, the list is empty.

The `error` field is None if the process is successful, and contains an error message string if it fails.

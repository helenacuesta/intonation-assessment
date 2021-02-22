# Intonation Assessment

This repo contains code for the automatic assessment of the intonation of singing performances.
It's developed for the Choir Singers Platform - Cantamus from the TROMPA project.

## Description 

Choir singers often rehearse their parts individually at home. 
While this is an excellent way to practice challenging parts in more depth, the conductor's figure is missing.
Therefore, the singer does not get any feedback about their performance.

In the context of the Cantamus platform from TROMPA (https://cantamus.app/), we developed a simple algorithm
to provide feedback to the singer in terms of their performance's intonation accuracy on a note basis.

This algorithm is integrated as part of the TROMPA Processing Library (TPL), and therefore its development and I/O
follow specific requirements.

The assessment algorithm is implemented in the core function `intonation_assessment`, inside the `assessment.py` script.
The algorithm takes several data from the score and the performance (see next section), and computes the average 
deviation of the singers' pitch from the reference pitch, given by the score.

After singing, the user receives the feedback in the form of a color-coded piano-roll, where the _transparency_ of 
each note refers to how _accurate_ each note was performed.

The `intonation_assessment` function has the following input parameters:

## Input parameters

    startbar : (int) indicates the first bar of the performance

    endbar : (int) indicates the last bar of the performance

    offset : (float) estimated latency between audio and score

    pitch_json_file : (string) filename of the json file with the pitch contour

    score_file : (string) filename of the score xml file

    voice : (string) voice part as written in the score

    output_filename : (string) output filename to use for the assessment results file

    dev_thresh : (float) maximum allowed deviation in cents. Defaults to 100 cents
    
    score_format: (string) specify the format of the input score if not XML
    
    tpl_output: (string) filename of the output config file with the path to the output results


### Output parameters

The main function generates two output files (real filenames are given as input parameters): 
`output_filename.json` and `tpl_output.ini`.

The JSON file (`output_filename.json`) contains the actual assessment results: a Python dictionary with two fields.
The first one, **'pitchAssessment'**, contains a list of arrays with the assessment results for each note in
in the form:
```
[note_start_time, intonation_rating]
```
If the process fails, the list will be empty. 

The field **'error'** will contain a string with an error message if the process fails, and will be None if it's successful.

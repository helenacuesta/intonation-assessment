# Intonation Assessment v0

This is a work in progress.

Helena Cuesta (helena.cuesta@upf.edu)

## Usage guide

#### Main script

**assessment.py** is the main script. It takes two json files as input: one containing information about the 
performance (measures, voice, score, latency estimation), and a second one with the pitch contour. This is the expected
usage:
```
python assessment.py --performance performance_data_84.json --pitch pitch_84.json
```

In the current version, this script generates an **output json file** named `performance_data_84_output.json` and stored 
in the same folder as the performance json. It contains a field *'pitchAssessment* that contains the output intonation 
rating for each note if the analysis is successful, and a field *'error'* with an error message if the analysis fails.

The *pitchAssessment* field contains a list of arrays `[note_start_time, pitch_value]`.

#### Requirements
**requirements.txt** contains the requirements to run the algorithm.





Dataset of paper "SRP-PHAT METHODS OF LOCATING SIMULTANEOUS MULTIPLE TALKERS USING A FRAME OF MICROPHONE ARRAY DATA"
====================================================================================================================

This page was originally created at the LEMS, Brown University, and the
[original content](https://web.archive.org/web/20100803034556/http://www.lems.brown.edu:80/array/data.html)
has been adapted by [Robin Scheibler](http://www.robinscheibler.org).

Dataset1
--------

This is the dataset used in the paper:

> H. Do and H. F. Silverman, “SRP-PHAT methods of locating simultaneous multiple talkers using a frame of microphone array data,” Proc. ICASSP, pp. 125–128, Mar. 2010.

![](icassp1.gif)

Real data recorded from 5 simultaneously talking, human sources

Talker 	Measured 3D-location (meters)

| speaker | location |
|:-:|:--------------------------:|
| 1 | [ 0.7296, -0.48, 2.5052] m  |
| 2 | [-0.7944, -0.29, 4.0292] m |
| 3 | [-0.2048, -0.40, 2.5052] m |
| 4 | [-0.7944, -0.44, 3.1148] m |
| 5 | [ 1.0344, -0.55, 3.7244] m  |



* The recording (24 channels, fs = 20KHz) in WAV format: [WAV file](dataset1/5talkers_fs20KHz_24mics.wav)

* 24 microphone locations (Matlab file): [dataset1/mic_loc.mat](dataset1/mic_loc.mat)

* Experimental conditions: T60=0.45s; Fs=20KHz; Framelength=102.4ms, advancing 25.6ms

* Boundaries of the focal volume (4m x 1m x 6m): Lowerbound = [-2, -1, 0] m, Upperbound = [2, 0, 6] m

* Hand-labeled data showing which talkers are active in each frame for the first 200 frames (Matlab MAT file): [dataset1/GS5tks.mat](dataset1/GS5tks.mat)

* An [AVI movie](dataset1/srpmov1fps_realtalkers.avi) (1 frame per second) created from Matlab showing a slice of the
  3-D SRP-PHAT surface through approximately the average talkers' height over
  400 frames (Please right click to save the file to your computer before
  playing it. If it still does not play, probably an Intel's IV50 codec is
  needed, which can be downloaded here ): Movie of 5 real talkers' SRP-PHAT
  surface

Dataset2
--------

This dataset is used in the paper:

> H. Do and H. F. Silverman, “Robust cross-correlation-based techniques for detecting and locating simultaneous, multiple sound sources,” Proc. ICASSP, Kyoto, Japan, Mar. 2012.

* A 120-second long recording of 10 real, human talkers (labeled from T1 to
  T10) using 181 microphones (plus 10 close-talking channels, so in total 191
  channels) was done with the HMA system in our laboratory.

* Experimental conditions: T60=0.45s, Fs=20KHz. Boundaries of the microphone
  array's focal volume (4m x 1m x 6m) are: Lowerbound = [-2, -1, 0] m,
  Upperbound = [2, 0, 6] m

* Among 10 talkers, 3 talkers (T7, T8, T9) were sitting around a table and
  having a conversation. The other 7 talkers divided into two groups. One group
  consisted of 5 talkers, among which 4 talkers (T3, T4, T5, T6) were having a
  conversation, and the other one (T10) was reciting a poem by himself. The
  second group had two talkers (T1, T2) facing each other and carrying on a
  conversation.

    | Talker |	Close-talking channel index | Measured 3D-location (meters) |
    |:------:|:----------------------------:|:-----------------------------:|
    |  T1    | 184                          | -0.7843, -0.5901, 4.6893      |
    |  T2    | 190                          | -1.2831, -0.2969, 4.6324      |
    |  T3    | 183                          | -0.7962, -0.3769, 2.7400      |
    |  T4    | 185                          |  0.2373, -0.2864, 2.7788      |
    |  T5    | 182                          | -0.2137, -0.4063, 1.8620      |
    |  T6    | 188                          | -1.0563, -0.3236, 1.9392      |
    |  T7    | 186                          |  0.7802, -0.7991, 3.4060      |
    |  T8    | 189                          |  0.1224, -0.8853, 4.0565      |
    |  T9    | 191                          |  0.6633, -0.7581, 4.3765      |
    | T10    | 187                          |  0.6914, -0.5672, 1.9318      |

* 3D-locations of 181 microphones (text file) can be found in [dataset2/micloc181.txt](dataset2/micloc181.txt).

* The 191-channel recording (WAV file) can be found in [dataset2/191mikes_10tks.wav](dataset2/191mikes_10tks.wav).



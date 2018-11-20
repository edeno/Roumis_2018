#!/bin/bash -l

DATE=20161114
DIRECTORY="/projectnb/braincom/Roumis_2018/Raw-Data/vx1_JZ1/raw"

echo "=========================================================="
echo Processing: $DIRECTORY/$DATE/*.h264

# Set SCC project
#$ -P braincom

# Specify hard time limit for the job.
#$ -l h_rt=24:00:00

# Give job a name
#$ -N run_position

# Combine output and error files into a single file
#$ -j y

# Specify the output file name
#$ -o position.log

# Request processor
#$ -pe omp 4

track_behavior "$DIRECTORY/$DATE/*.h264" "$DIRECTORY/position_config.json" --save_video --disable_progressbar

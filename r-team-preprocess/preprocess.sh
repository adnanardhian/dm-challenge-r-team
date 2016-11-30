#!/bin/bash
# Resizes all the training images to the same size and 
# saves them to PNG format using ImageMagick.
#
# Tasks:
# - resize the images
# - save to PNG format
#
# In addition, a label file at the image level is generated
# using information from the exams metadata table (see generate_labels.py).
#
# Author: Thomas Schaffter (thomas.schaff...@gmail.com)
# Last update: 2016-11-02

IMAGES_DIRECTORY="/dm_challenge/pilot_images"
EXAMS_METADATA_FILENAME="/dm_challenge/r-team-preprocess/metadata/exams_metadata_pilot_20160906.tsv"
IMAGES_CROSSWALK_FILENAME="/dm_challenge/r-team-preprocess/metadata/images_crosswalk_pilot_20160906.tsv"

PREPROCESS_DIRECTORY="/dm_challenge/r-team-preprocess"
PREPROCESS_IMAGES_DIRECTORY="$PREPROCESS_DIRECTORY/pilot_images_jpeg"
IMAGE_LABELS_FILENAME="$PREPROCESS_DIRECTORY/metadata/image_labels.txt"

mkdir -p $PREPROCESS_IMAGES_DIRECTORY

export SHELL=/bin/bash

echo "Resizing and converting $(find $IMAGES_DIRECTORY -name "*.dcm" | wc -l) DICOM images to JPEG format"
find $IMAGES_DIRECTORY/ -name "*.dcm" | parallel --will-cite "convert {} -resize 224x224! $PREPROCESS_IMAGES_DIRECTORY/{/.}.jpeg" # faster than mogrify
echo "JPEG images have been successfully saved to $PREPROCESS_IMAGES_DIRECTORY/."

echo "Generating image labels to $IMAGE_LABELS_FILENAME"
python /dm_challenge/r-team-preprocess/r-team-preprocess.py $EXAMS_METADATA_FILENAME $IMAGES_CROSSWALK_FILENAME $IMAGE_LABELS_FILENAME
# Replace the .dcm extension to .png
sed -i 's/.dcm/.jpeg/g' $IMAGE_LABELS_FILENAME

echo "Done"
ls

import numpy as np
import pandas as pd
import sys
import os

def generateImageLabels(images, metadata):
    metadata['examId'] = metadata.subjectId+("_")+metadata.examIndex.astype(str)
    images['examId'] = images.subjectId+("_")+images.examIndex.astype(str)
    examIdsCancerL = metadata.examId[metadata.cancerL == 1]
    examIdsCancerR = metadata.examId[metadata.cancerR == 1]
    images['cancer'] = 0
    images.loc[(images.examId.isin(examIdsCancerL)) & (images.laterality == "L"), 'cancer'] = 1
    images.loc[(images.examId.isin(examIdsCancerR)) & (images.laterality == "R"), 'cancer'] = 1
    return images[['filename', 'cancer']].copy()
    
if __name__ == '__main__':
    examsMetadataFilename = sys.argv[1]#"/dm_challenge/r-team-preprocess/metadata/exams_metadata_pilot_20160906.tsv"
    imagesCrosswalkFilename = sys.argv[2]#/dm_challenge/r-team-preprocess/metadata/images_crosswalk_pilot_20160906.tsv"
    labelsFilename = sys.argv[3]#"/dm_challenge/r-team-preprocess/metadata/image_labels.csv"

    # Read the label from the exams metadata
    fields = ['subjectId', 'examIndex', 'cancerL', 'cancerR']
    metadata = pd.read_csv(examsMetadataFilename, sep="\t", na_values='.',usecols=fields)
    
    #Read the data from the images crosswalk file
    fields = ['subjectId', 'examIndex', 'filename', 'laterality']
    images = pd.read_csv(imagesCrosswalkFilename, sep="\t", na_values='.', usecols=fields)
    
    # Convert subjectId to string
    metadata.subjectId = metadata.subjectId.astype(str)
    images.subjectId = images.subjectId.astype(str)
    
    labels = generateImageLabels(images, metadata)
    directory = os.path.dirname(labelsFilename)
    if not os.path.isdir(directory):
        os.makedirs(directory)
    labels.to_csv(labelsFilename, sep='\t', index=False, header=True)

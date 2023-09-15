# Neurosynth Montage Generator

This repository can be used to develop fNIRS Montages from Neurosyth association maps (or any .nii file). To get a list of fNIRS channels from the 10-20 headcap best suited to capture activation in a Neurosyth association map, follow these instructions:

1. Clone this repository.
2. Go to [Neurosyth.org](https://neurosynth.org/) and download an association map given a particular search term. For example, here is the [page associated with the term ''Social.''](https://neurosynth.org/analyses/terms/social/)
3. Place the downloaded .nii file in the root folder of the repository.
4. From the command line run `python montage_generator.py social_association-test_z_FDR_0.01.nii` replacing the name of the nii file with the name of the file you downloaded.
5. After running the script, you'll have a .csv file in the repository with the same name as the nii file (but with the .csv extension) containing a ranked list of channels based on how well they capture the activation shown in the association map.

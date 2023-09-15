import argparse
import numpy as np 
import os
import pandas as pd
import nibabel as nib
from re import split
from scipy.spatial.distance import squareform, pdist


class NeuroSynthImage(object):
	"""
	A class used to represent a 3D neuroSynth image downloaded from NeuroSynth

	Attributes
	----------
	filename : str
		name of '.nii' file in current directory downloaded from NeuroSynth
	img : NiBabel image
		Essentially a numpy array for storing the attribution map downloaded from NeuroSynth
	affine : numpy array
		3 x 3 tranformation matrix for transforming img into MNI space (see mni function for details of use)
	abc : numpy array
		3 x 1 tranformation matrix for transforming img into MNI space (see mni function for details of use)
	data : pandas dataframe
		Easier way of working with NiBabel data
	"""

	def __init__(self,filename):
		self.filename = filename

		self.img = nib.load(filename)			#Nifti file loaded

		self.affine = self.img.affine[:3, :3]	# affine transformation used for converting
		self.abc = self.img.affine[:3, 3]		# to mni coordinates according to Neurosynth website
												# "The images are all nominally in MNI152 
												# 2mm space (the default space in SPM and FSL)"
												# (https://neurosynth.org/faq/#q17)

		self.data = None						# We'll fill a dataframe which will be easier
												# work with when doing calculations.
		self.generate_dataframe()				# generate the dataframe by calling that function		
	
	def mni(self,i, j, k):
		""" Return X, Y, Z MNI coordinates for i, j, k """
		return self.affine.dot([i, j, k]) + self.abc

	def generate_dataframe(self):
		"""
		Generates pandas data frame with following columns:
		i : int
			first NiBabel index
		j : int
			second NiBabel index
		k : int
			third NiBabel index
		x : float
			first MNI152 coordinate
		y : float
			second MNI152 coordinate
		z : float
			third MNI152 coordinate
		weight : float
			weight assigned to that coordinate by NeuroSynth attribution map
		"""
		I,J,K = self.img.get_fdata().shape
		data = []
		
		# for each voxel in the data:
			# if weight is not zero (voxel matters when computing centroid)
				# calculate MNI coordinates (x,y,z)
				# append to list [i,j,k,x,y,z,weight]
		for i in range(I):
			for j in range(J):
				for k in range(K):
					weight = self.img.get_fdata()[i,j,k]
					
					if weight > 0:
						x,y,z = self.mni(i,j,k)
						data.append([i,j,k,x,y,z,self.img.get_fdata()[i,j,k]])

		# generate dataframe from data
		self.data = pd.DataFrame(data,columns=["i","j","k","x","y","z","weight"])


	def weighted_centroid(self,	x_min=0,x_max=100,
								y_min=0,y_max=100,
								z_min=0,z_max=100,
								extra_title=''):
		"""
		Computes weighted centroid in 3D range given x,y, and z constraints and saves list of fNIRS
		channels sorted based on ascending distance to centroid as csv file.

		params:
		x_min : int
			minimum x value in 3D range
		x_max : int
			maximum x value in 3D range
		y_min : int
			minimum y value in 3D range
		y_max : int
			maximum y value in 3D range
		z_min : int
			minimum z value in 3D range
		z_max : int
			maximum z value in 3D range
		extra_title : string
			Extra title used for saving csv file
		"""
		# slice out only voxels within bounding box given
		relevant_data = self.data[	(self.data['x'] >= x_min) & (self.data['x'] <= x_max) &
									(self.data['y'] >= y_min) & (self.data['y'] <= y_max) &
									(self.data['z'] >= z_min) & (self.data['z'] <= z_max)  ]

		print (relevant_data)

		# Compute the weighted centroid location. This is all the weights multiplied by the
		# x,y,z locations and divided by the sum of the weights.
		# Ex. In two dimensions, lets say we have two points:
			# One at location 1 with weight 1
			# The other at location 5 with weight (1/5)
			# The weighted centroid would be C = ((1*1) + (5*0.2)) / (1 + 0.2) = 1.6666
		Cx,Cy,Cz = [(relevant_data['x'] * relevant_data['weight']).sum(),
					(relevant_data['y'] * relevant_data['weight']).sum(),
					(relevant_data['z'] * relevant_data['weight']).sum()] / (relevant_data['weight']).sum()
		
		# Rank optodes according to distance from centroid
		self.rank_channels(Cx,Cy,Cz,extra_title)

		return Cx,Cy,Cz


	def rank_channels(self,x,y,z,extra_title):
		"""
		Sort channels by ascending distance from weighted centroid

		params:
		x : float
			x coordinate of weighted centroid
		y : float
			y coordinate of weighted centroid
		z : float
			z coordinate of weighted centroid
		extra_title : string
			extra title used for saving csv file
		"""
		# read in channel coordinates to dataframe
		channels = pd.read_csv("channel-coordinates.csv")
		
		# make new column "distance" with values equal to euclidian distance to centroid
		channels["distance"] = ((channels["X (mm)"] - x)**2 +
								(channels["Y (mm)"] - y)**2 +
								(channels["Z (mm)"] - z)**2 )**(0.5)
		
		# sort on column distance
		channels = channels.sort_values(by="distance")
		
		# write to file using same filename, but with csv extension
		channels.to_csv(split("\.nii",self.filename)[0] + extra_title + ".csv")


	def depth_sensitivity(self,depth):
		"""
		According to Strangman, Li, and Zhang (2013), the following function defines the sensitivity of a channel to tissue at depth D [1].

		[1] Strangman, G. E., Li, Z., & Zhang, Q. (2013). Depth sensitivity and source-detector separations for near infrared spectroscopy based on the Colin27 brain template. PloS one, 8(8), e66319.
		"""
		return 0.075*0.85**depth

	def rank_channels_global(self,extra_title):
		"""
		Ranks channels according to their weighted distance to all the voxels in the brain map.
		"""
		# read in channel coordinates to dataframe
		channels = pd.read_csv("channel-coordinates.csv")

		# only pay attention to non-zero weights
		relevant_data = self.data[self.data['weight']!=0]

		# get distance between every channel and every voxel (D)
		coordinates = pd.concat((relevant_data.iloc[:,3:6],channels.iloc[:,5:].rename({'X (mm)':'x', 'Y (mm)':'y', 'Z (mm)':'z'}, axis=1)))
		distances = pd.DataFrame(squareform(pdist(coordinates))).iloc[len(relevant_data):,:len(relevant_data)]

		# score each channel by depth_sensitivity(D) * weight summed across voxels
		channels["sensitivity"] = np.sum(distances.apply(lambda x: self.depth_sensitivity(x)).to_numpy() * relevant_data['weight'].to_numpy(),axis=1)

		# sort on column sensitivity
		channels = channels.sort_values(by="sensitivity",ascending=False)
		
		# write to file using same filename, but with csv extension
		channels.to_csv(split("\.nii",self.filename)[0] + extra_title + ".csv")


def main():
	argParser = argparse.ArgumentParser()
	argParser.add_argument("-f", "--file", help="path to nii file.")
	args = argParser.parse_args()

	print(args.file)

	NeuroSynthImage(args.file).rank_channels_global("")


if __name__ == "__main__":
    main()
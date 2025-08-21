import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nilearn import datasets, image, masking, plotting
from nilearn.glm.first_level import FirstLevelModel
from function.function_ import mask_from_labels
from nilearn.input_data import NiftiMasker
from .atlas_masking import AtlasMaskGenerator

class fMRIBOLD:
    def __init__(self, dir_path, t_r, out_path, atlas_type, **atlas_kwargs ):
        self.path = dir_path
        self.tr = t_r
        self.out_path = out_path
        self.image = image.load_img(self.path)
        self.n_scans = self.image.shape[-1]
        self.frame_time = np.arange(self.n_scans) * self.tr

        self.atlas_mask = AtlasMaskGenerator(atlas_type=atlas_type, fmri_path=self.path, output_dir=self.out_path, **atlas_kwargs)
        
    def create_masks(self, atlas_type, target_labels, mask_filename_prefix="mask", **atlas_kwargs):
        """
        Convenience method to generate masks using AtlasMaskGenerator.
        
        Parameters:
        atlas_type (str): Type of atlas (e.g., 'cort-maxprob-thr25-1mm')
        target_labels (list): List of labels to create masks for
        mask_filename_prefix (str): Prefix for mask filenames
        **atlas_kwargs: Additional arguments for atlas fetching
        
        Returns:
        dict: Dictionary of masks
        """
        mask_generator = AtlasMaskGenerator(atlas_type=atlas_type, fmri_path=self.path, output_dir=self.out_path, **atlas_kwargs)
        return mask_generator.generate_masks(target_labels,self.tr, mask_filename_prefix)


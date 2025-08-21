import os
from nilearn import datasets, image
import numpy as np
from nilearn import datasets, image, masking, plotting
from nilearn.input_data import NiftiMasker

class AtlasMaskGenerator:
    def __init__(self, atlas_type, output_dir, **atlas_kwargs):
        """
        Initialize the AtlasMaskGenerator with a specified atlas and output directory.
        
        Parameters:
        atlas_type (str): Type of atlas (e.g., 'cort-maxprob-thr25-1mm', 'sub-maxprob-thr25-1mm')
        output_dir (str): Directory where mask files will be saved
        **atlas_kwargs: Additional arguments to pass to datasets.fetch_atlas_harvard_oxford
        """
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.mkdir(output_dir)
        # Load the specified atlas
        self.atlas = datasets.fetch_atlas_harvard_oxford(atlas_type, **atlas_kwargs)
        self.atlas_img = image.load_img(self.atlas.maps)
        self.labels = self.atlas.labels


    def mask_from_labels(self, target_substrings, exact_match=False):
        """
        Create a binary mask from the atlas image for labels containing given substrings or exact matches.
        
        Parameters:
        target_substrings (list): List of substrings to match in label names (case-insensitive).
                                 If exact_match is True, these are treated as full label names.
        exact_match (bool): If True, require exact label matches instead of substring matching.
                           Default is False (substring matching).
        
        Returns:
        mask: Nifti image object representing the binary mask
        
        Raises:
        ValueError: If no valid labels are found or target_substrings is empty
        """
        if not target_substrings:
            raise ValueError("target_substrings cannot be empty")
        
        # Get the data array from the atlas image
        data = self.atlas_img.get_fdata()
        mask_data = np.zeros_like(data, dtype=np.uint8)
        matched_labels = []
        
        # Iterate over labels, skipping index 0 (usually 'Background')
        for idx, name in enumerate(self.labels):
            
            if idx == 0 or name is None:
                continue
            if exact_match:
                # Exact match mode
                if name in target_substrings:
                    mask_data[data == idx] = 1
                    matched_labels.append(name)
            else:
                # Substring match mode (case-insensitive)
                if any(s.lower() in str(name).lower() for s in target_substrings):
                    mask_data[data == idx] = 1
                    matched_labels.append(name)
        
        # Warn if no labels were matched
        if not matched_labels:
            print(f"Warning: No labels matched for substrings {target_substrings} "
                  f"(exact_match={exact_match}). Available labels: {self.labels[1:]}")
        
        # Create a new Nifti image for the mask
        return image.new_img_like(self.atlas_img, mask_data, affine=self.atlas_img.affine)

    

    def generate_masks(self,fmri_path, target_labels,t_r, mask_filename_prefix="mask"):
        """
        Generate masks for specified labels and resample to a target image.
        
        Parameters:
        target_labels (list): List of label names to create masks for
        img: Functional image to resample masks to
        mask_filename_prefix (str): Prefix for mask filenames (default: 'mask')
        
        Returns:
        dict: Dictionary containing the generated masks, keyed by label names
        """
        self.img = image.load_img(fmri_path)
        masks = {}
        ni_Maskers = {}
        label_timeseries = {}
        # Generate mask for the specified labels
        mask = self.mask_from_labels(target_labels)
        
        # Resample mask to the target image
        resampled_mask = image.resample_to_img(mask, self.img, interpolation="nearest", force_resample= True)
        ni_Masker = NiftiMasker(mask_img=resampled_mask, standardize=True, t_r=t_r)
        label_ts = ni_Masker.fit_transform(self.img).squeeze()

        # Save masks for each label
        for label in target_labels:
            # Find all labels in self.labels that contain the target label substring (case-insensitive)
            for atlas_label in self.labels:
                if atlas_label is None or atlas_label == "Background":
                    continue
                if label.lower() in str(atlas_label).lower():
                    safe_label = atlas_label.lower().replace(" ", "_")
                    mask_path = os.path.join(self.output_dir, f"{mask_filename_prefix}_{safe_label}.nii.gz")

                    resampled_mask.to_filename(mask_path)
                    masks[atlas_label] = resampled_mask
                    print(f"Mask for '{atlas_label}' saved to {mask_path}")
                    ni_Maskers[atlas_label] = ni_Masker
                    label_timeseries[atlas_label] = label_ts
        
        #df = np.array(list(label_timeseries[2].values())).transpose(1, 0, 2)  # New shape: (5470, 2, 189)
        #label_timeseries = df.reshape(df.shape[0], df.shape[1] * df.shape[2]) 
    
        return masks, ni_Maskers, label_timeseries
import torch
import torch.nn as nn

def integrate_camera_poses_to_positional_encoding(
    camera_poses, positional_encoding, image_size=518, patch_size=14, embed_dim=768
):
    """
    Integrates camera poses into the positional encoding for a vision transformer model.
    
    Parameters:
    - camera_poses (torch.Tensor): A tensor of shape (21, D), where D is the dimension of each pose (e.g., 6).
    - positional_encoding (torch.Tensor): A tensor of shape (1, num_patches, embed_dim) representing the initial positional encoding.
    - image_size (int): The size of the input image (assumed to be square). Default is 518.
    - patch_size (int): The size of each patch. Default is 14.
    - embed_dim (int): The embedding dimension of the vision transformer. Default is 768.
    
    Returns:
    - enhanced_positional_encoding (torch.Tensor): A tensor of shape (1, num_patches, embed_dim) with integrated camera pose information.
    """
    # Calculate the number of patches per dimension and total patches.
    num_patches_per_dim = image_size // patch_size  # e.g., 518 // 14 = 37 patches per dimension
    num_patches = num_patches_per_dim ** 2  # e.g., 37 * 37 = 1369 total patches

    # Define how the images from cameras are laid out within the 518x518 image:
    # Define ranges for each region based on how you concatenate the camera views.
    # Adjust these ranges based on the specific sizes of each region.
    camera_ranges = {
        'rgb_top': (0, num_patches_per_dim * (392 // patch_size)),  # Top portion (70px height)
        'rgb_left': (num_patches_per_dim * (392 // patch_size), num_patches_per_dim * (392 // patch_size) + num_patches_per_dim * (112 // patch_size) * 5),
        'rgb_back': (num_patches_per_dim * (392 // patch_size) + num_patches_per_dim * (112 // patch_size) * 5,
                     num_patches_per_dim * (392 // patch_size) + num_patches_per_dim * (112 // patch_size) * 10),
        'rgb_right': (num_patches_per_dim * (392 // patch_size) + num_patches_per_dim * (112 // patch_size) * 10,
                      num_patches_per_dim * (392 // patch_size) + num_patches_per_dim * (112 // patch_size) * 15),
        'rgb_front': (num_patches_per_dim * (392 // patch_size) + num_patches_per_dim * (112 // patch_size) * 15, num_patches),
    }

    # Map each camera index to its corresponding range.
    camera_indices = {
        'rgb_top': [2],
        'rgb_left': [5, 15, 16, 17, 18],
        'rgb_back': [4, 11, 12, 13, 14],
        'rgb_right': [6, 19, 20, 21, 22],
        'rgb_front': [3, 7, 8, 9, 10]
    }

    # Create a tensor for pose embeddings with the same length as the tokens (num_patches).
    pose_embedding = torch.zeros((num_patches, embed_dim))

    # Create an MLP layer for transforming poses into the embedding space.
    pose_to_embedding = nn.Linear(camera_poses.shape[-1], embed_dim)

    # Map the camera poses to the appropriate regions.
    for key, (start, end) in camera_ranges.items():
        indices = camera_indices[key]
        # Average the poses for the given cameras contributing to this section.
        avg_pose = torch.mean(torch.stack([camera_poses[idx] for idx in indices]), dim=0)
        
        # Transform the averaged pose to the embedding size (768).
        pose_embedded = pose_to_embedding(avg_pose)  # Shape (768,)
        
        # Assign this pose to all tokens in the corresponding range.
        pose_embedding[start:end] = pose_embedded

    # Add the pose embedding to the positional encoding.
    enhanced_positional_encoding = positional_encoding.clone()
    # the positional encoding corresponds to a class token at index 0, so we start from index 1
    enhanced_positional_encoding[0, 1:num_patches + 1] += pose_embedding

    return enhanced_positional_encoding

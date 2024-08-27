import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor


# use bfloat16 for the entire notebook
# torch.autocast(device_type="mps", dtype=torch.bfloat16).__enter__()

# if torch.cuda.get_device_properties(0).major >= 8:
# turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True


def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()


def show__best_mask(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    # Find the index of the mask with the lowest score
    min_score_index = np.argmin(scores)

    # Get the mask and score with the lowest score
    mask = masks[min_score_index]
    score = scores[min_score_index]

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(mask, plt.gca(), borders=borders)

    if point_coords is not None:
        assert input_labels is not None
        show_points(point_coords, input_labels, plt.gca())

    if box_coords is not None:
        show_box(box_coords, plt.gca())

    plt.title(f"Mask with Lowest Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()


def setup_segment():
    sam2_checkpoint = "/Users/stuartbman/GitHub/MRI_segment/checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="mps")

    predictor = SAM2ImagePredictor(sam2_model)
    return predictor


def setup_video_segment():
    sam2_checkpoint = "/Users/stuartbman/GitHub/MRI_segment/checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device="mps")
    return predictor


def segment_image(predictor, image, point_coords, point_labels, mask_input):
    predictor.set_image(image)

    input_point = np.array([point_coords])
    input_label = np.array([point_labels])

    if isinstance(mask_input, list) and len(mask_input) > 0:
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
            mask_input=np.array(mask_input)[None, :, :],
        )
    else:
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]

    return masks, logits


def segment_frame(predictor, inference, frame_index, points, point_labels, obj_id):
    """
    Segments a frame of a video
    :param predictor: SAM2VideoPredictor object
    :param inference: Inference State
    :param frame_index:
    :param points:
    :param point_labels:
    :param obj_id:
    :return:
    """
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference,
        frame_idx=frame_index,
        obj_id=obj_id,
        points=points,
        labels=point_labels,
    )

    out_mask = (out_mask_logits[0] > 0.0).cpu().numpy()

    return out_mask, out_mask_logits


def propagate_segment(predictor, inference_state, current_frame_idx):
    video_segments = {}  # video_segments contains the per-frame segmentation results

    # Forward propagation from the current frame to the end
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=current_frame_idx, reverse=False):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # Backward propagation from the current frame to the start
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=current_frame_idx, reverse=True):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    return video_segments

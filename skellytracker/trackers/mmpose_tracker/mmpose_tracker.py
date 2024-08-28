from mmpose.apis import inference_top_down_pose_model, init_pose_model
from skellytracker.trackers.base_tracker.base_tracker import BaseTracker

# correct/fill this out based on these docs: https://github.com/open-mmlab/mmpose/blob/main/docs/en/user_guides/inference.md


class MMPoseTracker(BaseTracker):
    def __init__(self, config_file, checkpoint_file):
        super().__init__(recorder=None, tracked_object_names=["human_pose"])
        self.model = init_pose_model(config_file, checkpoint_file, device="cuda:0")

    def process_image(self, image, **kwargs):
        pose_results, returned_outputs = inference_top_down_pose_model(
            self.model,
            image,
            bbox_thr=None,
            format="xyxy",
            dataset="TopDownCocoDataset",
            return_heatmap=False,
            outputs=None,
        )

        # Draw the poses on the image
        self.model.show_result(image, pose_results, show=False, out_file=None)

        # Update the tracking data
        self.tracking_data = {"human_pose": pose_results}
        self.annotated_image = image

        return {
            "tracking_data": self.tracking_data,
            "annotated_image": self.annotated_image,
            "raw_image": image,
        }


if __name__ == "__main__":
    config_file = "configs/top_down/hrnet/coco/hrnet_w48_coco_256x192.py"
    checkpoint_file = "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth"
    MMPoseTracker(config_file, checkpoint_file).demo()

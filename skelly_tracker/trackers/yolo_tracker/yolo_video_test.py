from skelly_tracker.trackers.yolo_tracker.yolo_tracker import YOLOPoseTracker


if __name__ == "__main__":
    video_path = "/Users/philipqueen/freemocap_data/recording_sessions/freemocap_sample_data/synchronized_videos/sesh_2022-09-19_16_16_50_in_class_jsm_synced_Cam2.mp4"
    YOLOPoseTracker().process_video(video_filepath=video_path)
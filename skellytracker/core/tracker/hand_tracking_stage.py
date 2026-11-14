"""Full-frame MediaPipe hand instances with independent per-camera tracking."""
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

from skellytracker.core.data_primitives.bounding_box import BoundingBox
from skellytracker.core.data_primitives.observation import StageObservation
from skellytracker.core.data_primitives.tracked_hand import HandSide, TrackedHand
from skellytracker.core.temporal_processing.bbox_policy import predict_bbox_from_keypoints
from skellytracker.core.temporal_processing.multi_person_config import MultiPersonTrackingConfig
from skellytracker.core.temporal_processing.track_association import associate
from skellytracker.core.tracker.detection_stage import DetectionStage
from skellytracker.core.tracker.person_track import PersonTrackState
from skellytracker.core.tracker.tracker_state import StageState


@dataclass
class HandTrack(PersonTrackState):
    # Handedness is a property of a track. A short vote tolerates an isolated
    # classification flip as the hand rotates, while allowing early errors to clear.
    timestamp_ms: int | None = None
    velocity: tuple[float, float] = (0., 0.)
    sides: deque[HandSide] = field(default_factory=lambda: deque(maxlen=5))

    def predicted(self, timestamp_ms):
        points, box = self.last_keypoints, self.last_bbox
        if timestamp_ms is None or self.timestamp_ms is None or not 0 < timestamp_ms - self.timestamp_ms <= 500:
            return points, box
        dt = timestamp_ms - self.timestamp_ms
        dx, dy = self.velocity[0] * dt, self.velocity[1] * dt
        return points.translated(dx, dy), BoundingBox(box.x1 + dx, box.y1 + dy, box.x2 + dx, box.y2 + dy)


@dataclass
class HandTracks:
    tracks: dict[int, HandTrack] = field(default_factory=dict)
    next_id: int = 0
    config: MultiPersonTrackingConfig = field(default_factory=MultiPersonTrackingConfig)

    def update(self, detections, timestamp_ms=None):
        candidates = []
        for side, points in detections:
            box = predict_bbox_from_keypoints(points, expansion_ratio=0.0)
            if box is not None:
                candidates.append((side, points, box))
        previous = list(self.tracks.values())
        predicted = [track.predicted(timestamp_ms) for track in previous]
        result = associate([box for _, box in predicted], [points for points, _ in predicted],
                           [box for _, _, box in candidates], [points for _, points, _ in candidates], self.config)
        assignments = {d: previous[t] for t, d in result.matches}
        for index in result.unmatched_tracks:
            track = previous[index]
            track.time_since_update += 1
            if track.time_since_update > self.config.max_age:
                del self.tracks[track.track_id]
        for index in result.unmatched_detections:
            track = HandTrack(track_id=self.next_id)
            self.next_id += 1
            self.tracks[track.track_id] = assignments[index] = track
        hands = []
        for index, (side, points, box) in enumerate(candidates):
            track = assignments[index]
            if track.timestamp_ms is not None and timestamp_ms is not None and 0 < timestamp_ms - track.timestamp_ms <= 500:
                motion = np.median(points.xy - track.last_keypoints.xy, axis=0)
                track.velocity = tuple(motion / (timestamp_ms - track.timestamp_ms))
            else:
                track.velocity = (0., 0.)
            track.timestamp_ms = timestamp_ms
            track.last_bbox, track.last_keypoints, track.time_since_update = box, points, 0
            track.sides.append(side)
            handedness = Counter(track.sides).most_common(1)[0][0]
            hands.append(TrackedHand(track.track_id, handedness, points))
        return tuple(hands)


@dataclass
class HandStageState(StageState):
    hand_tracks: HandTracks = field(default_factory=HandTracks)


class HandTrackingStage(DetectionStage):
    def _observe(self, detector, image, state, context):
        tracks = state.hand_tracks if isinstance(state, HandStageState) else HandTracks()
        hands = tracks.update(detector.detect_hands(image, context), context.timestamp_ms if context else None)
        return StageObservation(self.name, hands=hands), HandStageState(hand_tracks=tracks)

    def run(self, image, state, parent_keypoints=None, context=None):
        return self._observe(self.keypoint_detectors[0], image, state, context)

    def run_batch(self, images, states, context=None, parent_keypoints_per_cam=None):
        detectors = self._cam_kp_detectors[0]
        template = self.keypoint_detectors[0]
        pool = self._get_executor(len(images))
        futures = {}
        for camera, image in images.items():
            if camera not in detectors:
                detectors[camera] = type(template).create(template.config, template.session)
            futures[camera] = pool.submit(self._observe, detectors[camera], image, states.get(camera, StageState()), context)
        results = {camera: future.result() for camera, future in futures.items()}
        return ({camera: obs for camera, (obs, _) in results.items()},
                {camera: state for camera, (_, state) in results.items()})

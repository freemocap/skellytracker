use anyhow::Result;
use image::GrayImage;
use opencv::core::{Mat, Point2f, Size, Vector};
use opencv::objdetect;
use opencv::prelude::*;

use crate::core::observation::{Observation, ObservationPayload, TrackerKind};
use crate::core::point_cloud::PointCloud;
use crate::core::tracked_object_definition::TrackedObjectDefinition;
use crate::core::traits::Detect;
use crate::trackers::charuco::config::CharucoDetectorConfig;

/// Charuco board detector wrapping OpenCV's `cv::objdetect::CharucoDetector`.
pub struct CharucoDetector {
    _board: objdetect::CharucoBoard,
    detector: objdetect::CharucoDetector,
    aruco_marker_ids: Vec<i32>,
    all_charuco_ids: Vec<i32>,
    tracked_object_definition: TrackedObjectDefinition,
}

impl CharucoDetector {
    pub fn create(configuration: CharucoDetectorConfig) -> Result<Self> {
        let board_definition = &configuration.board_definition;

        // In opencv 0.98, get_predefined_dictionary takes PredefinedDictionaryType,
        // which is an enum with variants like DICT_4X4_250.
        let dictionary = objdetect::get_predefined_dictionary(
            objdetect::PredefinedDictionaryType::DICT_4X4_250,
        )?;

        let board = objdetect::CharucoBoard::new(
            Size::new(
                board_definition.squares_x as i32,
                board_definition.squares_y as i32,
            ),
            board_definition.square_length_millimeters as f32,
            board_definition.aruco_marker_length_millimeters() as f32,
            &dictionary,
            &Vector::<i32>::new(), // optional marker IDs — empty = auto-assign
        )?;

        let detector = objdetect::CharucoDetector::new_def(&board)?;

        // Collect marker IDs from the board (Vector<i32> in opencv 0.98)
        let board_ids = board.get_ids()?;
        let aruco_marker_ids: Vec<i32> = board_ids.to_vec();

        let number_of_corners = board_definition.number_of_corners();
        let all_charuco_ids: Vec<i32> = (0..number_of_corners as i32).collect();

        let yaml_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/trackers/charuco/charuco_tracked_object.yaml");
        let tracked_object_definition =
            TrackedObjectDefinition::from_yaml(&yaml_path)?;

        Ok(Self {
            _board: board,
            detector,
            aruco_marker_ids,
            all_charuco_ids,
            tracked_object_definition,
        })
    }
}

impl Detect for CharucoDetector {
    fn detect(&self, frame_number: u64, image: &GrayImage) -> Result<Observation> {
        let (_width, height) = image.dimensions();
        let data = image.as_raw();

        // Build an OpenCV Mat from the GrayImage pixels.
        let flat_mat = Mat::from_slice(data)?;
        let grey_mat = flat_mat.reshape(1, height as i32)?;

        // Run OpenCV Charuco detection
        let mut charuco_corners = Mat::default();
        let mut charuco_ids = Mat::default();
        let mut marker_corners = Vector::<Vector<Point2f>>::new();
        let mut marker_ids = Vector::<i32>::new();

        self.detector.detect_board(
            &grey_mat,
            &mut charuco_corners,
            &mut charuco_ids,
            &mut marker_corners,
            &mut marker_ids,
        )?;

        // Build PointCloud: one row per charuco corner ID, NaN if undetected
        let corner_names: Vec<String> = (0..self.all_charuco_ids.len())
            .map(|i| format!("CharucoCorner-{}", i))
            .collect();
        let mut point_cloud = PointCloud::new(corner_names);

        let mut detected_ids: Vec<i32> = Vec::new();
        let mut detected_coords: Vec<[f64; 2]> = Vec::new();
        let mut detected_marker_ids: Vec<i32> = Vec::new();
        let mut detected_marker_corners: Vec<[[f64; 2]; 4]> = Vec::new();

        // Extract charuco corners
        if !charuco_ids.empty() {
            let number_detected = charuco_ids.rows() as usize;

            for i in 0..number_detected {
                let corner_id = *charuco_ids.at_2d::<i32>(i as i32, 0)?;
                let x = *charuco_corners.at_2d::<f32>(i as i32, 0)? as f64;
                let y = *charuco_corners.at_2d::<f32>(i as i32, 1)? as f64;

                let index = corner_id as usize;
                if index < point_cloud.number_of_points() {
                    point_cloud.xyz[[index, 0]] = x;
                    point_cloud.xyz[[index, 1]] = y;
                    point_cloud.xyz[[index, 2]] = 0.0;
                    point_cloud.visibility[index] = 1.0;
                }

                detected_ids.push(corner_id);
                detected_coords.push([x, y]);
            }
        }

        // Extract detected aruco markers (filter to board markers only)
        if !marker_ids.is_empty() {
            for marker_index in 0..marker_ids.len() {
                let marker_id = marker_ids.get(marker_index)?; // i32, not &i32 in 0.98
                if self.aruco_marker_ids.contains(&marker_id) {
                    let corners_vec = marker_corners.get(marker_index)?; // Vector<Point2f>
                    let mut marker = [[0.0f64; 2]; 4];
                    for corner_index in 0..4 {
                        let point = corners_vec.get(corner_index)?;
                        marker[corner_index as usize] = [point.x as f64, point.y as f64];
                    }
                    detected_marker_ids.push(marker_id);
                    detected_marker_corners.push(marker);
                }
            }
        }

        if !detected_ids.is_empty() {
            println!(
                "  [opencv charuco] {} corners, {} markers, IDs: {:?}",
                detected_ids.len(),
                detected_marker_ids.len(),
                &detected_ids
            );
        }

        Ok(Observation {
            frame_number,
            tracker_kind: TrackerKind::Charuco,
            points: point_cloud,
            payload: ObservationPayload::Charuco {
                all_charuco_ids: self.all_charuco_ids.clone(),
                all_aruco_ids: self.aruco_marker_ids.clone(),
                detected_charuco_corner_ids: if detected_ids.is_empty() {
                    None
                } else {
                    Some(detected_ids)
                },
                detected_charuco_corners: if detected_coords.is_empty() {
                    None
                } else {
                    Some(detected_coords)
                },
                detected_aruco_marker_ids: if detected_marker_ids.is_empty() {
                    None
                } else {
                    Some(detected_marker_ids)
                },
                detected_aruco_marker_corners: if detected_marker_corners.is_empty() {
                    None
                } else {
                    Some(detected_marker_corners)
                },
                board_rotation_vector: None,
                board_translation_vector: None,
                detected_charuco_corners_in_camera_coordinates: None,
            },
        })
    }

    fn tracked_object_definition(&self) -> &TrackedObjectDefinition {
        &self.tracked_object_definition
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_construct_detector() {
        let config = CharucoDetectorConfig::default();
        let detector = CharucoDetector::create(config);
        assert!(detector.is_ok());
    }
}

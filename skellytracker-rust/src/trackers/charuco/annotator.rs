use std::collections::VecDeque;

use anyhow::Result;
use image::{DynamicImage, Rgba};
use imageproc::drawing::{draw_hollow_circle_mut, draw_line_segment_mut};

use crate::core::observation::{Observation, ObservationPayload};
use crate::core::traits::Annotate;
use crate::trackers::charuco::config::CharucoAnnotatorConfig;

/// Renders charuco detection results onto a color image.
///
/// Draws colored corner markers, ArUco marker boundary polylines, fading
/// trails from previous frames, and a list of undetected corner IDs.
pub struct CharucoAnnotator {
    configuration: CharucoAnnotatorConfig,
    observation_history: VecDeque<Observation>,
}

impl CharucoAnnotator {
    /// Creates a new annotator with the given configuration.
    pub fn create(configuration: CharucoAnnotatorConfig) -> Self {
        Self {
            configuration,
            observation_history: VecDeque::new(),
        }
    }
}

impl Annotate for CharucoAnnotator {
    fn annotate(
        &mut self,
        image: &DynamicImage,
        observation: &Observation,
    ) -> Result<DynamicImage> {
        // Convert to RGBA so we can use per-pixel alpha for fading trails.
        let mut annotated = image.to_rgba8();
        let image_width = annotated.width() as i32;
        let image_height = annotated.height() as i32;

        // --- observation history (for fading trails) ---
        self.observation_history.push_back(observation.clone());
        let max_history = self.configuration.show_tracks.unwrap_or(15);
        while self.observation_history.len() > max_history {
            self.observation_history.pop_front();
        }
        let history_len = self.observation_history.len();

        // Draw oldest first (lowest alpha), newest last (full opacity).
        for (history_offset, hist_obs) in self.observation_history.iter().rev().enumerate() {
            let scale = 1.0 - (history_offset as f64 / history_len as f64);
            let alpha = (255.0 * scale) as u8;

            let ObservationPayload::Charuco {
                ref detected_charuco_corners,
                ref detected_charuco_corner_ids,
                ref detected_aruco_marker_ids,
                ref detected_aruco_marker_corners,
                ..
            } = &hist_obs.payload;
                // =============================================
                // Charuco corner markers (circles)
                // =============================================
                if let (Some(ids), Some(corners)) =
                    (detected_charuco_corner_ids, detected_charuco_corners)
                {
                    let (r, g, b) = self.configuration.corner_marker_color;
                    let marker_color = Rgba([r, g, b, alpha]);
                    let base_radius =
                        std::cmp::max(1, (self.configuration.corner_marker_size as f64 * scale) as i32);
                    // Use corner_marker_thickness to draw concentric rings
                    let rings = std::cmp::max(1, self.configuration.corner_marker_thickness);

                    for (_corner_id, corner) in ids.iter().zip(corners.iter()) {
                        let (x, y) = (corner[0] as i32, corner[1] as i32);

                        // Clamp to image bounds to avoid drawing artifacts far off-screen
                        if x < -base_radius * 2
                            || y < -base_radius * 2
                            || x > image_width + base_radius * 2
                            || y > image_height + base_radius * 2
                        {
                            continue;
                        }

                        // Concentric rings for thickness
                        for t in 0..rings {
                            let radius = base_radius - t;
                            if radius > 0 {
                                draw_hollow_circle_mut(
                                    &mut annotated,
                                    (x, y),
                                    radius,
                                    marker_color,
                                );
                            }
                        }

                        // On the current frame, also draw a small marker at a
                        // nearby offset so each corner has a visible "label" dot.
                        if history_offset == 0 {
                            let (tr, tg, tb) = self.configuration.text_color;
                            let text_color = Rgba([tr, tg, tb, 255]);
                            let text_offset = (image_height as f64 * 0.01) as i32;
                            draw_hollow_circle_mut(
                                &mut annotated,
                                (x + text_offset, y + text_offset),
                                3,
                                text_color,
                            );
                        }
                    }
                }

                // =============================================
                // ArUco marker boundary polylines (current frame only)
                // =============================================
                if history_offset == 0 {
                    if let (Some(_marker_ids), Some(marker_corners)) =
                        (detected_aruco_marker_ids, detected_aruco_marker_corners)
                    {
                        let (r, g, b) = self.configuration.aruco_lines_color;
                        let aruco_color = Rgba([r, g, b, 255]);
                        let thick = std::cmp::max(1, self.configuration.aruco_lines_thickness);

                        for quad in marker_corners.iter() {
                            let pts: [(i32, i32); 4] = [
                                (quad[0][0] as i32, quad[0][1] as i32),
                                (quad[1][0] as i32, quad[1][1] as i32),
                                (quad[2][0] as i32, quad[2][1] as i32),
                                (quad[3][0] as i32, quad[3][1] as i32),
                            ];

                            // Draw each edge as a line segment; repeat with
                            // diagonal offsets for thickness.
                            for edge in 0..4 {
                                let (x1, y1) = pts[edge];
                                let (x2, y2) = pts[(edge + 1) % 4];

                                for t in 0..thick {
                                    let offset = t as i32 / 2;
                                    let sign = if t % 2 == 0 { 1 } else { -1 };
                                    let dx = sign * offset;
                                    let dy = sign * offset;
                                    draw_line_segment_mut(
                                        &mut annotated,
                                        ((x1 + dx) as f32, (y1 + dy) as f32),
                                        ((x2 + dx) as f32, (y2 + dy) as f32),
                                        aruco_color,
                                    );
                                }
                            }
                        }
                    }
                }
        }

        // =============================================
        // Undetected corner IDs
        // =============================================
        let ObservationPayload::Charuco {
            ref all_charuco_ids,
            ref detected_charuco_corner_ids,
            ..
        } = &observation.payload;

        let mut undetected = all_charuco_ids.clone();
        if let Some(detected) = detected_charuco_corner_ids {
            undetected.retain(|id| !detected.contains(id));
        }

        if !undetected.is_empty() {
            let (tr, tg, tb) = self.configuration.text_color;
            let text_color = Rgba([tr, tg, tb, 255]);
            let x_base = image_width - 30;

            for (i, _corner_id) in undetected.iter().enumerate() {
                let y = 30 + i as i32 * 16;
                if y < image_height {
                    draw_hollow_circle_mut(&mut annotated, (x_base, y), 4, text_color);
                }
            }
        }

        Ok(DynamicImage::ImageRgba8(annotated))
    }
}

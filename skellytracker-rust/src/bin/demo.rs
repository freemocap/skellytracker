use std::time::Instant;

use anyhow::Result;
use minifb::{Key, Scale, Window, WindowOptions};

use skellytracker::core::observation::ObservationPayload;
use skellytracker::core::tracker::Tracker;
use skellytracker::io::recorder::Recorder;
use skellytracker::io::webcam::WebcamCapture;
use skellytracker::trackers::charuco::annotator::CharucoAnnotator;
use skellytracker::trackers::charuco::config::{CharucoAnnotatorConfig, CharucoDetectorConfig};
use skellytracker::trackers::charuco::detector::CharucoDetector;

/// Fast nearest-neighbor downscale to fit within `maximum_dimension`.
/// Operates on raw `[u8]` — no per-pixel trait dispatch.
fn fast_downscale(image: &image::GrayImage, maximum_dimension: u32) -> image::GrayImage {
    let (w, h) = image.dimensions();
    let max_side = w.max(h);
    if max_side <= maximum_dimension {
        return image.clone();
    }
    let (nw, nh) = if w > h {
        (maximum_dimension, (h as u64 * maximum_dimension as u64 / w as u64) as u32)
    } else {
        ((w as u64 * maximum_dimension as u64 / h as u64) as u32, maximum_dimension)
    };
    let src = image.as_raw();
    let (sw, sh) = (w as usize, h as usize);
    let (dw, dh) = (nw as usize, nh as usize);
    let mut dst = vec![0u8; dw * dh];
    for y in 0..dh {
        let sy = y * sh / dh;
        let src_row = sy * sw;
        let dst_row = y * dw;
        for x in 0..dw {
            dst[dst_row + x] = src[src_row + x * sw / dw];
        }
    }
    image::GrayImage::from_raw(nw, nh, dst).expect("downscale dims must be non-zero")
}

fn main() -> Result<()> {
    println!("SkellyTracker Rust - Charuco Webcam Demo");
    println!("=========================================");
    println!("Default board: {}x{} letter-size ({}mm squares)", 5, 3, 54.0);
    println!("Press ESC to quit.\n");

    // Open webcam
    println!("Opening webcam...");
    let mut webcam = WebcamCapture::open_first()?;
    let width = webcam.width() as usize;
    let height = webcam.height() as usize;
    println!("Webcam ready: {}x{}\n", width, height);

    // Build Charuco tracker
    println!("Initializing Charuco detector (5x3 board)...");
    let detector_config = CharucoDetectorConfig::default();
    let board = &detector_config.board_definition;
    println!(
        "  Board: {}x{} squares, {}mm square length, {} corners",
        board.squares_x,
        board.squares_y,
        board.square_length_millimeters,
        board.number_of_corners()
    );

    let detector = CharucoDetector::create(detector_config)?;
    let annotator = CharucoAnnotator::create(CharucoAnnotatorConfig::default());
    let recorder = Recorder::new();

    let mut tracker = Tracker::new(
        Box::new(detector),
        Box::new(annotator),
        Box::new(recorder),
    );
    println!("Tracker ready. Starting loop...\n");

    // Create display window
    let mut window = Window::new(
        "SkellyTracker Rust - Charuco Demo (ESC to quit)",
        width,
        height,
        WindowOptions {
            resize: true,
            scale: Scale::FitScreen,
            ..WindowOptions::default()
        },
    )
    .expect("Failed to create window");
    window.set_target_fps(0); // unlimited — we control timing

    // Pre-allocated ARGB buffer (reused across frames, avoids per-frame allocation)
    let buffer_size = width * height;
    let mut argb_buffer: Vec<u32> = vec![0u32; buffer_size];

    let mut frame_count: u64 = 0;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let loop_start = Instant::now();

        // ── 1. Capture frame ──────────────────────────────────
        let t0 = Instant::now();
        let (frame_number, color_frame, gray_frame) = match webcam.read_frame()? {
            Some((frame_number, color)) => {
                let gray = color.to_luma8();
                (frame_number, color, gray)
            }
            None => {
                eprintln!("Camera returned no frame.");
                break;
            }
        };
        let capture_duration = t0.elapsed();

        // ── 2. Downscale for detection (640px max) ────────────
        let detection_image = fast_downscale(&gray_frame, 640);
        let scale_x = gray_frame.width() as f64 / detection_image.width() as f64;
        let scale_y = gray_frame.height() as f64 / detection_image.height() as f64;

        // ── 3. Detect ─────────────────────────────────────────
        let t1 = Instant::now();
        let mut observation = tracker.process_image(frame_number, &detection_image, true)?;
        let detect_duration = t1.elapsed();

        // Scale observation coordinates back to original resolution
        for point_index in 0..observation.points.number_of_points() {
            let x = observation.points.xyz[[point_index, 0]];
            let y = observation.points.xyz[[point_index, 1]];
            if !x.is_nan() {
                observation.points.xyz[[point_index, 0]] = x * scale_x;
                observation.points.xyz[[point_index, 1]] = y * scale_y;
            }
        }
        // Also scale the raw detection coordinates
        let skellytracker::core::observation::ObservationPayload::Charuco {
            ref mut detected_charuco_corners,
            ref mut detected_aruco_marker_corners,
            ..
        } = &mut observation.payload;

        if let Some(corners) = detected_charuco_corners {
            for c in corners.iter_mut() {
                c[0] *= scale_x;
                c[1] *= scale_y;
            }
        }
        if let Some(marker_corners) = detected_aruco_marker_corners {
            for marker in marker_corners.iter_mut() {
                for c in marker.iter_mut() {
                    c[0] *= scale_x;
                    c[1] *= scale_y;
                }
            }
        }

        // ── 4. Annotate (on original-size color frame) ──────────
        let t2 = Instant::now();
        let annotated = tracker.annotate_image(&color_frame, &observation)?;
        let annotate_duration = t2.elapsed();

        // ── 4. Convert RGB to ARGB buffer ─────────────────────
        let t3 = Instant::now();
        let annotated_rgb = annotated.to_rgb8();
        let raw_bytes = annotated_rgb.as_raw();
        let pixel_count = buffer_size.min(raw_bytes.len() / 3);
        unsafe {
            let source = raw_bytes.as_ptr();
            let destination = argb_buffer.as_mut_ptr();
            for i in 0..pixel_count {
                let offset = i * 3;
                let red = *source.add(offset);
                let green = *source.add(offset + 1);
                let blue = *source.add(offset + 2);
                *destination.add(i) =
                    0xFF00_0000 | ((red as u32) << 16) | ((green as u32) << 8) | (blue as u32);
            }
        }
        let convert_duration = t3.elapsed();

        // ── 5. Display ────────────────────────────────────────
        let t4 = Instant::now();
        window.update_with_buffer(&argb_buffer, width, height)?;
        let display_duration = t4.elapsed();

        // ── Extract detection stats ────────────────────────────
        let valid_points = observation.points.number_of_valid();
        let (marker_count, charuco_corner_count) = match &observation.payload {
            ObservationPayload::Charuco {
                ref detected_aruco_marker_ids,
                ref detected_charuco_corner_ids,
                ..
            } => (
                detected_aruco_marker_ids
                    .as_ref()
                    .map_or(0, |ids| ids.len()),
                detected_charuco_corner_ids
                    .as_ref()
                    .map_or(0, |ids| ids.len()),
            ),
        };

        frame_count += 1;
        let total_duration = loop_start.elapsed();
        let fps = 1.0 / total_duration.as_secs_f64();

        // Print every frame for the first 5 frames, then every 30 frames
        let print_verbose = frame_count <= 5 || frame_count % 30 == 0;
        if print_verbose || valid_points > 0 || marker_count > 0 {
            println!(
                "Frame {:>5} | {:>5.1} FPS | {:>3} valid corners from {} charuco detected | {:>2} markers | \
                 cap={:>4.0}ms detect={:>6.1}ms annotate={:>5.1}ms convert={:>4.1}ms display={:>4.1}ms",
                frame_count,
                fps,
                valid_points,
                charuco_corner_count,
                marker_count,
                capture_duration.as_secs_f64() * 1000.0,
                detect_duration.as_secs_f64() * 1000.0,
                annotate_duration.as_secs_f64() * 1000.0,
                convert_duration.as_secs_f64() * 1000.0,
                display_duration.as_secs_f64() * 1000.0,
            );

            // Print detected marker IDs if any were found
            if marker_count > 0 {
                let ObservationPayload::Charuco {
                    ref detected_aruco_marker_ids,
                    ref detected_charuco_corner_ids,
                    ..
                } = &observation.payload;
                if let Some(ids) = detected_aruco_marker_ids {
                    println!("  → Detected marker IDs: {:?}", ids);
                }
                if let Some(corner_ids) = detected_charuco_corner_ids {
                    println!("  → Charuco corner IDs: {:?}", corner_ids);
                }
            }
        }
    }

    println!("\nExiting...");

    // Save recorded data
    let output_path = std::path::Path::new("charuco_demo_output.npy");
    tracker.recorder.save_npy(output_path)?;
    println!(
        "Saved {} observations to {}",
        tracker.recorder.observation_count(),
        output_path.display()
    );

    Ok(())
}

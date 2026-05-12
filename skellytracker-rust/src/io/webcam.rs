use anyhow::Result;
use image::{DynamicImage, GrayImage, RgbImage};

/// Webcam capture via nokhwa (pure Rust, no C++ dependencies).
///
/// Uses `AbsoluteHighestFrameRate` to let nokhwa pick the best format the
/// camera actually supports — same approach as the working skellycam Rust app.
pub struct WebcamCapture {
    camera: nokhwa::Camera,
    frame_number: u64,
    width: u32,
    height: u32,
}

impl WebcamCapture {
    /// Open a webcam by index.
    ///
    /// Requests the highest available frame rate and then tries to set the
    /// desired resolution. Falls back to whatever the camera provides.
    pub fn open(camera_index: u32) -> Result<Self> {
        let requested = nokhwa::utils::RequestedFormat::new::<nokhwa::pixel_format::RgbFormat>(
            nokhwa::utils::RequestedFormatType::AbsoluteHighestFrameRate,
        );

        let mut camera = nokhwa::Camera::new(
            nokhwa::utils::CameraIndex::Index(camera_index),
            requested,
        )
        .map_err(|error| anyhow::anyhow!("Failed to open camera {camera_index}: {error}"))?;

        // Try to set 1280x720, but accept whatever we get
        let desired_resolution = nokhwa::utils::Resolution::new(1280, 720);
        let _ = camera.set_resolution(desired_resolution);

        camera
            .open_stream()
            .map_err(|error| anyhow::anyhow!("Failed to open stream: {error}"))?;

        let resolution = camera.resolution();
        let actual_width = resolution.width();
        let actual_height = resolution.height();
        println!(
            "Camera opened at {}x{}",
            actual_width, actual_height
        );

        Ok(Self {
            camera,
            frame_number: 0,
            width: actual_width,
            height: actual_height,
        })
    }

    /// Open the first available camera.
    pub fn open_first() -> Result<Self> {
        Self::open(0)
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    /// Read the next frame as a color DynamicImage.
    pub fn read_frame(&mut self) -> Result<Option<(u64, DynamicImage)>> {
        let nokhwa_buffer = self
            .camera
            .frame()
            .map_err(|error| anyhow::anyhow!("Failed to read frame: {error}"))?;

        let rgb_image = nokhwa_buffer
            .decode_image::<nokhwa::pixel_format::RgbFormat>()
            .map_err(|error| anyhow::anyhow!("Failed to decode frame: {error}"))?;

        // nokhwa's ImageBuffer has the same memory layout as image::RgbImage
        let (width, height) = rgb_image.dimensions();
        let raw_data = rgb_image.into_raw();
        let rgb = RgbImage::from_raw(width, height, raw_data)
            .ok_or_else(|| anyhow::anyhow!("Failed to construct RgbImage from raw data"))?;

        let current_frame = self.frame_number;
        self.frame_number += 1;

        Ok(Some((current_frame, DynamicImage::ImageRgb8(rgb))))
    }

    /// Read the next frame as a grayscale image (for detection).
    pub fn read_frame_gray(&mut self) -> Result<Option<(u64, GrayImage)>> {
        match self.read_frame()? {
            Some((frame_number, dynamic)) => {
                let gray = dynamic.to_luma8();
                Ok(Some((frame_number, gray)))
            }
            None => Ok(None),
        }
    }
}

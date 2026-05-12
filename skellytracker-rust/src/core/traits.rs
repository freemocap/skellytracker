use anyhow::Result;
use image::GrayImage;
use image::DynamicImage;

use super::observation::Observation;
use super::tracked_object_definition::TrackedObjectDefinition;

/// Trait for pose-estimation detectors.
///
/// Each tracker implements this to run inference on a single grayscale image frame.
/// The trait is object-safe so it can be used as `Box<dyn Detect>` for
/// runtime tracker switching.
pub trait Detect {
    /// Run detection on a single grayscale image frame.
    fn detect(&self, frame_number: u64, image: &GrayImage) -> Result<Observation>;

    /// The schema of tracked points and connections this detector produces.
    fn tracked_object_definition(&self) -> &TrackedObjectDefinition;
}

/// Trait for image annotators that draw detection results onto frames.
pub trait Annotate {
    /// Draw detection results onto a color image. Returns a new annotated image.
    fn annotate(
        &mut self,
        image: &DynamicImage,
        observation: &Observation,
    ) -> Result<DynamicImage>;
}

/// Trait for recording observations across frames.
///
/// Includes serialization methods so callers operating on `Box<dyn Record>`
/// can save results without knowing the concrete recorder type.
pub trait Record {
    /// Append an observation to the recording buffer.
    fn add_observation(&mut self, observation: Observation);

    /// Discard all recorded observations.
    fn clear(&mut self);

    /// Number of observations currently stored.
    fn observation_count(&self) -> usize;

    /// Serialize all observations to a JSON string.
    fn to_json_string(&self) -> Result<String>;

    /// Save observations to a .npy file.
    fn save_npy(&self, path: &std::path::Path) -> Result<()>;
}

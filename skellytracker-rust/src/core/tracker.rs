use anyhow::Result;
use image::{DynamicImage, GrayImage};

use super::observation::Observation;
use super::traits::{Annotate, Detect, Record};

/// Top-level orchestrator. Composes a detector, annotator, and recorder.
///
/// Uses `Box<dyn Trait>` (dynamic dispatch) to enable runtime tracker switching
/// in the demo without changing dispatch code when new trackers are added.
pub struct Tracker {
    pub detector: Box<dyn Detect>,
    pub annotator: Box<dyn Annotate>,
    pub recorder: Box<dyn Record>,
}

impl Tracker {
    /// Create a new Tracker from its components.
    pub fn new(
        detector: Box<dyn Detect>,
        annotator: Box<dyn Annotate>,
        recorder: Box<dyn Record>,
    ) -> Self {
        Self {
            detector,
            annotator,
            recorder,
        }
    }

    /// Run detection and optionally record the observation.
    pub fn process_image(
        &mut self,
        frame_number: u64,
        image: &GrayImage,
        record: bool,
    ) -> Result<Observation> {
        let observation = self.detector.detect(frame_number, image)?;
        if record {
            self.recorder.add_observation(observation.clone());
        }
        Ok(observation)
    }

    /// Annotate a color image with an observation. Returns a new annotated image.
    pub fn annotate_image(
        &mut self,
        image: &DynamicImage,
        observation: &Observation,
    ) -> Result<DynamicImage> {
        self.annotator.annotate(image, observation)
    }
}

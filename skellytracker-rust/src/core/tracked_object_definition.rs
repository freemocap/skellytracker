use anyhow::{bail, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

use super::point_cloud::PointCloud;

/// Schema definition for a tracked object: named points + skeleton connections.
///
/// Loaded from YAML. Replaces Python's Pydantic `TrackedObjectDefinition`.
/// The YAML composition system (`composed_of`) is deferred to the MediaPipe milestone.
#[derive(Debug, Clone, Deserialize)]
pub struct TrackedObjectDefinition {
    pub name: String,
    pub tracker_type: String,
    #[serde(default)]
    pub landmark_schema: String,
    pub tracked_points: Vec<String>,
    #[serde(default)]
    pub connections: Vec<(String, String)>,
}

impl TrackedObjectDefinition {
    /// Load a TrackedObjectDefinition from a YAML file.
    pub fn from_yaml(path: &Path) -> Result<Self> {
        let yaml_content = std::fs::read_to_string(path)?;
        let definition: Self = serde_yaml::from_str(&yaml_content)?;
        definition.validate()?;
        Ok(definition)
    }

    /// Validate that all connection names reference known tracked_points.
    fn validate(&self) -> Result<()> {
        let name_set: HashMap<&str, usize> = self
            .tracked_points
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_str(), i))
            .collect();

        for (from_name, to_name) in &self.connections {
            if !name_set.contains_key(from_name.as_str()) {
                bail!(
                    "Connection references unknown point '{}' in definition '{}'",
                    from_name,
                    self.name
                );
            }
            if !name_set.contains_key(to_name.as_str()) {
                bail!(
                    "Connection references unknown point '{}' in definition '{}'",
                    to_name,
                    self.name
                );
            }
        }

        // Check for duplicate point names
        if name_set.len() != self.tracked_points.len() {
            bail!(
                "Duplicate point names found in definition '{}'",
                self.name
            );
        }

        Ok(())
    }

    /// Resolve connection name-pairs to array indices (for drawing).
    pub fn connection_indices(&self) -> Result<Vec<(usize, usize)>> {
        let name_to_index: HashMap<&str, usize> = self
            .tracked_points
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_str(), i))
            .collect();

        self.connections
            .iter()
            .map(|(from_name, to_name)| {
                let from_index = *name_to_index
                    .get(from_name.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Unknown point: {}", from_name))?;
                let to_index = *name_to_index
                    .get(to_name.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Unknown point: {}", to_name))?;
                Ok((from_index, to_index))
            })
            .collect()
    }

    /// Factory for an all-NaN PointCloud sized to this definition.
    pub fn empty_point_cloud(&self) -> PointCloud {
        PointCloud::new(self.tracked_points.clone())
    }
}

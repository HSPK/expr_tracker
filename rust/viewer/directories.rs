use super::{blocking, invalid, Error, Result, Viewer};
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::Json;
use serde::{Deserialize, Serialize};
use std::io;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Deserialize)]
pub(super) struct RootQuery {
    pub root: Option<String>,
}

#[derive(Deserialize)]
pub(super) struct BrowseQuery {
    path: Option<String>,
}

#[derive(Serialize)]
struct Directory {
    name: String,
    path: String,
}

#[derive(Serialize)]
pub(super) struct Listing {
    path: String,
    parent: Option<String>,
    directories: Vec<Directory>,
}

fn path_text(path: &std::path::Path) -> Result<String> {
    path.to_str()
        .map(str::to_owned)
        .ok_or_else(|| invalid("The directory path cannot be represented as UTF-8"))
}

fn directory_error(error: io::Error) -> Error {
    let status = match error.kind() {
        io::ErrorKind::NotFound => StatusCode::NOT_FOUND,
        io::ErrorKind::PermissionDenied => StatusCode::FORBIDDEN,
        _ => StatusCode::BAD_REQUEST,
    };
    Error(status, format!("Cannot access server directory: {error}"))
}

impl Viewer {
    pub(super) fn resolve_root(&self, wanted: Option<&str>) -> Result<PathBuf> {
        let path = match wanted {
            None => self.root.clone(),
            Some("") => return Err(invalid("Directory path must not be empty")),
            Some(path) => self.root.join(path),
        };
        let path = path.canonicalize().map_err(directory_error)?;
        if !path.is_dir() {
            return Err(invalid("Selected path must be a directory"));
        }
        Ok(path)
    }

    fn browse(&self, wanted: Option<&str>) -> Result<Listing> {
        let path = self.resolve_root(wanted)?;
        let mut directories = Vec::new();
        for (at, entry) in path.read_dir().map_err(directory_error)?.enumerate() {
            if at >= 10_000 {
                return Err(invalid(
                    "Directory exceeds the 10000-entry browsing limit; enter a child path directly",
                ));
            }
            let entry = entry.map_err(directory_error)?;
            let kind = entry.file_type().map_err(directory_error)?;
            if kind.is_dir()
                || (kind.is_symlink() && entry.path().metadata().map_err(directory_error)?.is_dir())
            {
                directories.push(Directory {
                    name: entry
                        .file_name()
                        .into_string()
                        .map_err(|_| invalid("A directory name cannot be represented as UTF-8"))?,
                    path: path_text(&entry.path())?,
                });
            }
        }
        directories.sort_by(|left, right| left.name.cmp(&right.name));
        Ok(Listing {
            path: path_text(&path)?,
            parent: path.parent().map(path_text).transpose()?,
            directories,
        })
    }
}

pub(super) async fn browse(
    State(viewer): State<Arc<Viewer>>,
    Query(query): Query<BrowseQuery>,
) -> Result<Json<Listing>> {
    blocking(move || viewer.browse(query.path.as_deref())).await
}

pub(super) async fn location(
    State(viewer): State<Arc<Viewer>>,
    Query(query): Query<RootQuery>,
) -> Result<Json<serde_json::Value>> {
    blocking(move || {
        let path = viewer.resolve_root(query.root.as_deref())?;
        Ok(serde_json::json!({"path": path_text(&path)?}))
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oversized_listing_is_an_explicit_error() {
        let root = tempfile::tempdir().unwrap();
        for at in 0..10_001 {
            std::fs::write(root.path().join(at.to_string()), b"").unwrap();
        }
        let viewer = crate::viewer::tests::viewer(root.path());
        let error = match viewer.browse(None) {
            Ok(_) => panic!("Oversized listing must not succeed"),
            Err(error) => error,
        };
        assert_eq!(error.0, StatusCode::BAD_REQUEST);
        assert!(error.1.contains("10000-entry"));
    }
}

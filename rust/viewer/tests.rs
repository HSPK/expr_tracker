use super::*;
use http_body_util::BodyExt;
use tower::ServiceExt;

pub(super) fn viewer(root: &Path) -> Arc<Viewer> {
    Arc::new(Viewer {
        root: root.canonicalize().unwrap(),
        hosts: vec!["localhost:9110".into()],
        runs: Mutex::new(HashMap::new()),
        logs: RwLock::new(HashMap::new()),
        stopping: Arc::new(AtomicBool::new(false)),
    })
}

#[test]
fn cached_logs_query_independently() {
    use std::sync::mpsc;
    use std::time::Duration;

    let directory = tempfile::tempdir().unwrap();
    let first = directory.path().join("first.jsonl");
    let second = directory.path().join("second.jsonl");
    for path in [&first, &second] {
        std::fs::write(path, b"{\"_step\":0,\"v\":1}\n").unwrap();
    }
    let viewer = viewer(directory.path());
    for path in [&first, &second] {
        viewer.with_log(path, false, |_| Ok(())).unwrap();
    }
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let (done_tx, done_rx) = mpsc::channel();
    std::thread::scope(|scope| {
        let viewer = &viewer;
        let first = &first;
        let second = &second;
        let held = scope.spawn(move || {
            viewer.with_log(first, false, |_| {
                entered_tx.send(()).unwrap();
                release_rx.recv_timeout(Duration::from_secs(10)).unwrap();
                Ok(())
            })
        });
        entered_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        let other = scope.spawn(move || {
            viewer.with_log(second, false, |_| {
                done_tx.send(()).unwrap();
                Ok(())
            })
        });
        let progressed = done_rx.recv_timeout(Duration::from_millis(500));
        release_tx.send(()).unwrap();
        held.join().unwrap().unwrap();
        other.join().unwrap().unwrap();
        assert!(progressed.is_ok(), "Unrelated cached log was blocked");
    });
}

#[test]
fn same_log_queries_remain_exclusive() {
    use std::sync::mpsc;
    use std::time::Duration;

    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("metrics.jsonl");
    std::fs::write(&path, b"{\"_step\":0,\"v\":1}\n").unwrap();
    let viewer = viewer(directory.path());
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let (started_tx, started_rx) = mpsc::channel();
    let (done_tx, done_rx) = mpsc::channel();
    std::thread::scope(|scope| {
        let viewer = &viewer;
        let path = &path;
        let held = scope.spawn(move || {
            viewer.with_log(path, false, |_| {
                entered_tx.send(()).unwrap();
                release_rx.recv_timeout(Duration::from_secs(10)).unwrap();
                Ok(())
            })
        });
        entered_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        let other = scope.spawn(move || {
            started_tx.send(()).unwrap();
            viewer.with_log(path, false, |_| {
                done_tx.send(()).unwrap();
                Ok(())
            })
        });
        started_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        let overlapped = done_rx.recv_timeout(Duration::from_millis(50));
        release_tx.send(()).unwrap();
        held.join().unwrap().unwrap();
        other.join().unwrap().unwrap();
        assert!(overlapped.is_err());
        done_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    });
}

#[test]
fn registry_capacity_and_active_index_lifetime_remain_bounded() {
    use std::sync::mpsc;
    use std::time::Duration;

    let directory = tempfile::tempdir().unwrap();
    let paths: Vec<_> = (0..33)
        .map(|index| directory.path().join(format!("{index}.jsonl")))
        .collect();
    let viewer = viewer(directory.path());
    for path in &paths {
        std::fs::write(path, b"{\"_step\":0,\"v\":1}\n").unwrap();
    }
    for path in &paths[..32] {
        viewer.with_log(path, false, |_| Ok(())).unwrap();
    }
    assert_eq!(viewer.logs.read().len(), 32);
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let (started_tx, started_rx) = mpsc::channel();
    let (done_tx, done_rx) = mpsc::channel();
    std::thread::scope(|scope| {
        let viewer = &viewer;
        let first = &paths[0];
        let missing = &paths[32];
        let held = scope.spawn(move || {
            viewer.with_log(first, false, |_| {
                entered_tx.send(()).unwrap();
                release_rx.recv_timeout(Duration::from_secs(10)).unwrap();
                Ok(())
            })
        });
        entered_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        let other = scope.spawn(move || {
            started_tx.send(()).unwrap();
            viewer.with_log(missing, false, |_| {
                done_tx.send(()).unwrap();
                Ok(())
            })
        });
        started_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        let grew_while_pinned = done_rx.recv_timeout(Duration::from_millis(50));
        release_tx.send(()).unwrap();
        held.join().unwrap().unwrap();
        other.join().unwrap().unwrap();
        assert!(grew_while_pinned.is_err());
        done_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    });
    assert_eq!(viewer.logs.read().len(), 32);
}

#[test]
fn refresh_failure_drops_index_and_rebuilds_without_stale_records() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("metrics.jsonl");
    std::fs::write(&path, b"{\"_step\":0,\"old\":1}\n").unwrap();
    let viewer = viewer(directory.path());
    viewer.with_log(&path, false, |_| Ok(())).unwrap();
    std::fs::remove_file(&path).unwrap();
    std::fs::create_dir(&path).unwrap();
    assert!(viewer.with_log(&path, false, |_| Ok(())).is_err());
    assert!(viewer.logs.read()[&path].lock().is_none());
    std::fs::remove_dir(&path).unwrap();
    std::fs::write(&path, b"{\"_step\":2,\"new\":3}\n").unwrap();
    viewer
        .with_log(&path, false, |log| {
            assert!(!log.metrics.contains("old"));
            assert!(log.metrics.contains("new"));
            assert_eq!(
                log.db
                    .query_row::<i64, _, _>("SELECT count(*) FROM records", [], |row| row.get(0))?,
                1
            );
            Ok(())
        })
        .unwrap();
    assert!(viewer
        .with_log(&path, false, |_| Err::<(), _>(invalid("query error")))
        .is_err());
    assert!(viewer.logs.read()[&path].lock().is_some());
}

#[test]
fn invalidated_slots_do_not_displace_healthy_indices() {
    let directory = tempfile::tempdir().unwrap();
    let viewer = viewer(directory.path());
    for index in 0..32 {
        let path = directory.path().join(format!("{index}.jsonl"));
        std::fs::write(&path, b"{\"_step\":0,\"v\":1}\n").unwrap();
        viewer.with_log(&path, false, |_| Ok(())).unwrap();
    }
    let paths: Vec<_> = viewer.logs.read().keys().cloned().collect();
    let invalid = &paths[1];
    std::fs::remove_file(invalid).unwrap();
    std::fs::create_dir(invalid).unwrap();
    assert!(viewer.with_log(invalid, false, |_| Ok(())).is_err());
    let replacement = directory.path().join("replacement.jsonl");
    std::fs::write(&replacement, b"{\"_step\":0,\"v\":2}\n").unwrap();
    viewer.with_log(&replacement, false, |_| Ok(())).unwrap();
    let logs = viewer.logs.read();
    assert_eq!(logs.len(), 32);
    assert!(!logs.contains_key(invalid));
    assert!(paths
        .iter()
        .filter(|path| *path != invalid)
        .all(|path| logs.contains_key(path)));
}

#[test]
fn integer_step_order_is_exact_beyond_javascript_and_i64() {
    let values = [
        "-100000000000000000000",
        "-2",
        "0",
        "1",
        "9007199254740993",
        "100000000000000000000",
    ];
    let orders: Vec<_> = values
        .iter()
        .map(|value| bound(Some(value)).unwrap().unwrap())
        .collect();
    assert!(orders.windows(2).all(|pair| pair[0] < pair[1]));
    assert!(bound(Some("1.5")).is_err());
    assert!(bound(Some("true")).is_err());
}

#[test]
fn index_merges_patches_without_losing_null_or_numeric_zero() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("metrics.jsonl");
    std::fs::write(
        &path,
        b"{\"_step\":0,\"_time\":1,\"a\":0}\n{\"_step\":1,\"a\":2}\n{\"_step\":0,\"b\":null}\n",
    )
    .unwrap();
    let mut log = CachedLog::new(Arc::new(AtomicBool::new(false))).unwrap();
    log.refresh(&path, false).unwrap();
    let data: String = log
        .db
        .query_row("SELECT data FROM records WHERE step='0'", [], |row| {
            row.get(0)
        })
        .unwrap();
    let value: Value = serde_json::from_str(&data).unwrap();
    assert_eq!(value["a"], 0);
    assert!(value["b"].is_null());
    assert_eq!(value["_time"], 1);
    assert_eq!(
        log.db
            .query_row::<u64, _, _>("SELECT count(*) FROM records", [], |row| row.get(0))
            .unwrap(),
        2
    );
}

#[test]
fn downsampling_is_bounded_and_preserves_spikes_and_nonfinite_gaps() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("metrics.jsonl");
    let mut data = String::new();
    for step in 0..1000 {
        let metric = if step == 501 {
            "9999"
        } else if step == 600 {
            "NaN"
        } else {
            "1"
        };
        data.push_str(&format!("{{\"_step\":{step},\"v\":{metric}}}\n"));
    }
    std::fs::write(&path, data).unwrap();
    let mut log = CachedLog::new(Arc::new(AtomicBool::new(false))).unwrap();
    log.refresh(&path, false).unwrap();
    let query = HistoryQuery {
        root: None,
        run: String::new(),
        file: None,
        metrics: "[\"v\"]".into(),
        axis: None,
        max_points: Some(64),
        from_step: None,
        to_step: None,
    };
    let value = chart(&log, &query).unwrap();
    assert!(value["x"].as_array().unwrap().len() <= 64);
    let values = value["series"][0]["values"].as_array().unwrap();
    assert!(values.contains(&json!(9999.0)));
    assert!(values.contains(&Value::Null));
    assert_eq!(value["series"][0]["nonfinite"], 1);
    assert_eq!(value["total"], 1000);
}

#[tokio::test]
async fn viewer_lists_real_runs_and_rejects_unknown_paths_and_hosts() {
    let directory = tempfile::tempdir().unwrap();
    std::fs::write(
        directory.path().join("metrics.jsonl"),
        b"{\"_step\":0,\"v\":1}\n",
    )
    .unwrap();
    let app = router(viewer(directory.path()), directory.path().to_path_buf());
    let request = axum::http::Request::builder()
        .uri("/api/runs")
        .header("host", "localhost:9110")
        .body(axum::body::Body::empty())
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let runs: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(runs.as_array().unwrap().len(), 1);
    let request = axum::http::Request::builder()
        .uri("/api/run?run=../../outside")
        .header("host", "localhost:9110")
        .body(axum::body::Body::empty())
        .unwrap();
    assert_eq!(
        app.clone().oneshot(request).await.unwrap().status(),
        StatusCode::NOT_FOUND
    );
    let request = axum::http::Request::builder()
        .uri("/api/runs")
        .header("host", "external.example:9110")
        .body(axum::body::Body::empty())
        .unwrap();
    assert_eq!(
        app.oneshot(request).await.unwrap().status(),
        StatusCode::FORBIDDEN
    );
}

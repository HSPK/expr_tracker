use super::json::add_int;
use super::{visit_rows_from, HistoryQuery, JsonObject, SegmentedReader, Step};
use std::fs;

#[test]
fn lossless_extended_json_roundtrip() {
    let huge = "9".repeat(1000);
    let text = format!(
        r#"{{"_step":0,"big":{huge},"nan":NaN,"plus":Infinity,"minus":-Infinity,"\ud800":{{"nested":["\udcff",NaN]}},"\u0000literal":"ok"}}"#
    );
    let parsed = JsonObject::parse(&text).unwrap();
    let encoded = parsed.encode();
    assert!(encoded.contains(&huge));
    assert!(encoded.contains(r#""\ud800""#));
    assert!(encoded.contains(r#""\u0000literal""#));
    assert_eq!(JsonObject::parse(&encoded).unwrap().encode(), encoded);
}

#[test]
fn malformed_json_is_not_a_record() {
    for text in [
        r#"{"_step":0,"x":01}"#,
        r#"{"_step":0,"x":"\q"}"#,
        r#"{"_step":0,}"#,
        r#"{"_step":0}junk"#,
        r#"{"_step":0,"x":[1,]}"#,
    ] {
        assert!(JsonObject::parse(text).is_none(), "{text}");
    }
}

#[test]
fn integer_steps_and_addition_remain_exact() {
    let huge = format!("1{}", "0".repeat(200));
    assert!(Step(huge.clone()) > Step("999".into()));
    assert!(Step(format!("-{huge}")) < Step("-999".into()));
    assert_eq!(add_int(&huge, "-1"), "9".repeat(200));
    assert_eq!(
        add_int(&format!("-{huge}"), "1"),
        format!("-{}", "9".repeat(200))
    );
    assert_eq!(add_int(&huge, &format!("-{huge}")), "0");
    assert_eq!(add_int("999", "1"), "1000");
    let enormous = "9".repeat(10000);
    let record = JsonObject::parse(&format!("{{\"_step\":{enormous}}}")).unwrap();
    assert_eq!(record.step().unwrap().0, enormous);
}

#[test]
fn segmented_complete_prefix_is_incremental_and_snapshot_pinned() {
    let dir = tempfile::tempdir_in(".").unwrap();
    let path = dir.path().join("metrics.jsonl");
    let first = b"\n{\"_step\":0}\n";
    let second = b"{\"_step\":1}\n";
    fs::write(&path, first).unwrap();
    let parts = dir.path().join("metrics.jsonl.parts");
    fs::create_dir(&parts).unwrap();
    let active = parts.join("00000001.jsonl");
    let mut bytes = second.to_vec();
    bytes.extend_from_slice(b"{\"_step\":2");
    fs::write(&active, &bytes).unwrap();
    let captured = SegmentedReader::open(&path).unwrap();
    let mut rows = Vec::new();
    let boundary = visit_rows_from(&path, 0, |row| {
        rows.push(row.to_vec());
        Ok(())
    })
    .unwrap();
    assert_eq!(rows, vec![first[1..].to_vec(), second.to_vec()]);
    assert_eq!(boundary, (first.len() + second.len()) as u64);
    bytes.extend_from_slice(b"}\n");
    fs::write(&active, bytes).unwrap();
    let mut new = Vec::new();
    let next = visit_rows_from(&path, boundary, |row| {
        new.push(row.to_vec());
        Ok(())
    })
    .unwrap();
    assert_eq!(new, vec![b"{\"_step\":2}\n".to_vec()]);
    assert!(next > boundary);
    assert_eq!(captured.query(&HistoryQuery::default()).unwrap().len(), 2);
    assert_eq!(
        SegmentedReader::open(&path)
            .unwrap()
            .query(&HistoryQuery::default())
            .unwrap()
            .len(),
        3
    );
}

#[test]
fn patch_tail_selects_recency_and_timestamp_is_latest_write() {
    let dir = tempfile::tempdir_in(".").unwrap();
    let path = dir.path().join("metrics.jsonl");
    fs::write(&path,b"{\"_step\":100,\"_time\":1,\"a\":1}\n{\"_step\":2}\n{\"_step\":100,\"_time\":3,\"b\":2}\n{\"_step\":1}\n").unwrap();
    let reader = SegmentedReader::open(&path).unwrap();
    let rows = reader
        .query(&HistoryQuery {
            limit: Some(2),
            ..Default::default()
        })
        .unwrap();
    assert_eq!(
        rows.iter().map(|r| r.step().unwrap().0).collect::<Vec<_>>(),
        vec!["1", "100"]
    );
    assert_eq!(rows[1].get("_time"), Some("3"));
    assert_eq!(rows[1].get("a"), Some("1"));
    assert_eq!(rows[1].get("b"), Some("2"));
}

#[test]
fn encoded_queries_match_decoded_queries_for_ordered_and_patched_rows() {
    let dir = tempfile::tempdir_in(".").unwrap();
    let path = dir.path().join("metrics.jsonl");
    for bytes in [
        br#"{"_step":0,"v":NaN,"s":"\ud800"}
{"_step":1,"huge":1234567890123456789012345678901234567890}
"#
        .as_slice(),
        br#"{"_step":0,"_time":1,"a":1}
{"_step":1,"b":2}
{"_step":0,"_time":null,"a":3}
"#
        .as_slice(),
    ] {
        fs::write(&path, bytes).unwrap();
        let reader = SegmentedReader::open(&path).unwrap();
        let expected: Vec<_> = reader
            .query(&HistoryQuery::default())
            .unwrap()
            .iter()
            .map(super::JsonObject::encode)
            .collect();
        assert_eq!(
            reader
                .query_encoded(&HistoryQuery::default())
                .unwrap()
                .iter()
                .map(|text| JsonObject::parse(text).unwrap().encode())
                .collect::<Vec<_>>(),
            expected
        );
    }
}

#[test]
fn validated_step_matches_full_lossless_parser() {
    let huge = "9".repeat(10000);
    let mut cases = vec![format!(r#"{{"_step":{huge},"value":"\ud800"}}"#)];
    for step in ["0", "-0", "-17", "true", "false", "null", "1.0", "01"] {
        for key in [r#""_step""#, r#""\u005fstep""#, r#""_\u0073tep""#] {
            for body in [
                r#""nested":[{"x":[NaN,Infinity,-Infinity,"\ud800"]}]"#,
                r#""duplicate":1,"duplicate":2"#,
                r#""space":"\\\"\"" #,
                    r#""bad":[1,]"#,
                r#""bad":"\q""#,
            ] {
                cases.push(format!("{{{key}:{step},{body}}}"));
                cases.push(format!(r#"{{"_step":99,{key}:{step},{body}}}"#));
                cases.push(format!(r#"{{{key}:{step},"_step":99,{body}}}"#));
            }
        }
    }
    cases.extend(
        [
            r#"{"_step":0} trailing"#,
            r#"{"_step":0,"x":"unescaped
    newline"}"#,
            r#"{"_step":0,"x":1e}"#,
            r#"{"other":0}"#,
        ]
        .map(str::to_owned),
    );
    for text in cases {
        assert_eq!(
            JsonObject::validated_step(&text),
            JsonObject::parse(&text).and_then(|object| object.step()),
            "{text}"
        );
    }
}

#[test]
fn unmerged_encoded_queries_preserve_validated_source_spelling() {
    let dir = tempfile::tempdir_in(".").unwrap();
    let path = dir.path().join("metrics.jsonl");
    let rows = [
        r#"{"_step":false,"x":NaN,"s":"\ud800"}"#,
        r#"{"\u005fstep":true,"x":Infinity,"x":-Infinity}"#,
    ];
    fs::write(&path, rows.join("\n") + "\n").unwrap();
    let reader = SegmentedReader::open(&path).unwrap();
    assert_eq!(
        reader.query_encoded(&HistoryQuery::default()).unwrap(),
        rows.map(|row| format!("{row}\n"))
    );
}

#[test]
fn explicit_byte_boundary_never_completes_a_crossing_row() {
    let dir = tempfile::tempdir_in(".").unwrap();
    let path = dir.path().join("metrics.jsonl");
    let first = b"{\"_step\":0}\n";
    let mut bytes = first.to_vec();
    bytes.extend_from_slice(b"{\"_step\":1}\n");
    fs::write(&path, bytes).unwrap();
    let reader = SegmentedReader::open(&path)
        .unwrap()
        .snapshot(Some(first.len() as u64 + 4));
    assert_eq!(reader.query(&HistoryQuery::default()).unwrap().len(), 1);
}

#[test]
fn surrogate_key_patches_use_decoded_key_identity() {
    let mut record = JsonObject::parse(r#"{"_step":0,"a\uD800":1}"#).unwrap();
    record.merge(&JsonObject::parse(r#"{"_step":0,"\u0061\ud800":2}"#).unwrap());
    record.merge(&JsonObject::parse(r#"{"_step":0,"a\uD800":3}"#).unwrap());
    assert_eq!(record.0.len(), 2);
    assert!(record.encode().contains(r#""a\ud800": 3"#));
}

#[test]
fn shared_browser_encoding_tags_only_at_the_output_boundary() {
    let parsed = super::parse_compatible_json(
        br#"{"step":9007199254740993,"v":NaN,"s":"a\ud800","emoji":"\ud83d\ude00"}"#,
    )
    .unwrap();
    assert!(parsed.escaped_surrogates);
    assert!(parsed.value["step"].is_number());
    assert_eq!(parsed.value["emoji"], "\u{1f600}");
    let browser = super::encode_browser_value(parsed.value);
    assert_eq!(browser["step"]["$et.integer"], "9007199254740993");
    assert_eq!(browser["v"]["$et.nonfinite"], "NaN");
    assert_eq!(browser["s"], "a\\ud800");
}

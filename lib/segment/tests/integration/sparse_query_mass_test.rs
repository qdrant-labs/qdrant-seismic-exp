use std::collections::{BTreeMap, HashMap};

use common::ambient;
use ordered_float::OrderedFloat;
use segment::data_types::named_vectors::NamedVectors;
use segment::data_types::query_context::QueryContext;
use segment::data_types::vectors::{QueryVector, VectorRef};
use segment::entry::entry_point::SegmentEntry;
use segment::entry::{NonAppendableSegmentEntry, ReadSegmentEntry};
use segment::index::sparse_index::sparse_index_config::{SparseIndexConfig, SparseIndexType};
use segment::json_path::JsonPath;
use segment::payload_json;
use segment::segment::Segment;
use segment::segment_constructor::build_segment;
use segment::types::{
    Condition, DEFAULT_SPARSE_FULL_SCAN_THRESHOLD, ExtendedPointId, FieldCondition, Filter,
    PayloadSchemaType, SearchParams, SegmentConfig, SeqNumberType, SparseVectorDataConfig,
    SparseVectorStorageType, VectorStorageDatatype, WithPayload, WithVector,
};
use sparse::common::sparse_vector::SparseVector;
use tempfile::Builder;
use validator::Validate;

use crate::fixtures::segment::SPARSE_VECTOR_NAME;

/// Point id -> score, so that assertions don't depend on the order of tied scores.
type Scores = BTreeMap<u64, f32>;

/// Points of the test segment, all with weight 1.0 on the listed dimensions:
///
/// | point | sparse dims |
/// |-------|-------------|
/// | 0     | 0           |
/// | 1     | 1           |
/// | 2     | 0, 1        |
/// | 3     | 2           |
const POINTS: &[&[u32]] = &[&[0], &[1], &[0, 1], &[2]];

fn build_test_segment(path: &std::path::Path) -> Segment {
    let config = SegmentConfig {
        vector_data: Default::default(),
        sparse_vector_data: HashMap::from([(
            SPARSE_VECTOR_NAME.to_owned(),
            SparseVectorDataConfig {
                index: SparseIndexConfig {
                    memory: None,
                    full_scan_threshold: Some(DEFAULT_SPARSE_FULL_SCAN_THRESHOLD),
                    index_type: SparseIndexType::MutableRam,
                    datatype: Some(VectorStorageDatatype::Float32),
                },
                storage_type: SparseVectorStorageType::default(),
                modifier: None,
            },
        )]),
        payload_storage_type: Default::default(),
        id_tracker_memory: None,
    };

    let (mut segment, _) = build_segment(path, &config, None, true).unwrap();
    let _scope = ambient::test_guard();

    let mut op_num: SeqNumberType = 0;
    segment
        .create_field_index(
            op_num,
            &JsonPath::new("tenant"),
            Some(&PayloadSchemaType::Keyword.into()),
        )
        .unwrap();

    for (point_id, dims) in POINTS.iter().enumerate() {
        op_num += 1;
        let point_id = (point_id as u64).into();
        let vector = SparseVector::new(dims.to_vec(), vec![1.0; dims.len()]).unwrap();
        let vectors = NamedVectors::from_ref(SPARSE_VECTOR_NAME, VectorRef::Sparse(&vector));
        segment.upsert_point(op_num, point_id, vectors).unwrap();
        segment
            .set_payload(op_num, point_id, &payload_json! { "tenant": "a" }, &None)
            .unwrap();
    }

    segment
}

fn search(
    segment: &Segment,
    query: &SparseVector,
    filter: Option<&Filter>,
    query_mass: Option<f32>,
) -> Scores {
    let query_context = QueryContext::default();
    let segment_query_context = query_context.get_segment_query_context();
    let query_vector = QueryVector::from(query.clone());
    let params = SearchParams {
        query_mass: query_mass.map(OrderedFloat),
        ..Default::default()
    };

    let results = segment
        .search_batch(
            SPARSE_VECTOR_NAME,
            &[&query_vector],
            &WithPayload::from(false),
            &WithVector::Bool(false),
            filter,
            POINTS.len(),
            Some(&params),
            &segment_query_context,
        )
        .unwrap()
        .pop()
        .unwrap();

    results
        .into_iter()
        .map(|point| match point.id {
            ExtendedPointId::NumId(id) => (id, point.score),
            ExtendedPointId::Uuid(uuid) => panic!("unexpected uuid point id {uuid}"),
        })
        .collect()
}

fn assert_scores(actual: &Scores, expected: &[(u64, f32)], context: &str) {
    let expected: Scores = expected.iter().copied().collect();
    assert_eq!(
        actual.keys().collect::<Vec<_>>(),
        expected.keys().collect::<Vec<_>>(),
        "{context}: returned points differ"
    );
    for (id, score) in &expected {
        assert!(
            (actual[id] - score).abs() < 1e-6,
            "{context}: point {id} scored {}, expected {score}",
            actual[id]
        );
    }
}

/// Check `query_mass` through both sparse search paths: the inverted index (no filter) and the
/// plain scan (a filter matching so few points that the cardinality is below the full-scan
/// threshold).
#[test]
fn sparse_query_mass() {
    let dir = Builder::new().prefix("segment_dir").tempdir().unwrap();
    let segment = build_test_segment(dir.path());

    let tenant_a = Filter::new_must(Condition::Field(FieldCondition::new_match(
        JsonPath::new("tenant"),
        "a".to_string().into(),
    )));

    for (path, filter) in [("inverted index", None), ("plain", Some(&tenant_a))] {
        let check = |query: &SparseVector, query_mass, expected: &[(u64, f32)], what: &str| {
            let actual = search(&segment, query, filter, query_mass);
            assert_scores(&actual, expected, &format!("{path}, {what}"));
        };

        // Dim 1 has the largest weight (5.0), then dim 2 (1.0), then dim 0 (0.5). The total is 6.5,
        // so dim 1 alone makes up 77%, and dims 1 and 2 together 92%.
        let query = SparseVector::new(vec![0, 1, 2], vec![0.5, 5.0, 1.0]).unwrap();

        check(
            &query,
            None,
            &[(0, 0.5), (1, 5.0), (2, 5.5), (3, 1.0)],
            "no cut",
        );
        check(
            &query,
            Some(0.5),
            &[(1, 5.0), (2, 5.0)],
            "keep 50% of the weight",
        );
        check(
            &query,
            Some(0.8),
            &[(1, 5.0), (2, 5.0), (3, 1.0)],
            "keep 80% of the weight",
        );

        // Dims 1 and 2 reach 92%, so keeping 99% needs dim 0 as well. Keeping all changes nothing.
        let full = search(&segment, &query, filter, None);
        assert_eq!(search(&segment, &query, filter, Some(0.99)), full);
        assert_eq!(search(&segment, &query, filter, Some(1.0)), full);

        // Dimensions are ranked by absolute weight: -5.0 outranks 1.0.
        let negative = SparseVector::new(vec![0, 1], vec![1.0, -5.0]).unwrap();
        check(
            &negative,
            Some(0.5),
            &[(1, -5.0), (2, -5.0)],
            "cut keeps large negative weight",
        );
    }
}

/// `query_mass` is a fraction of the query weight: greater than 0 and at most 1.
#[test]
fn query_mass_validation() {
    let params = |query_mass: f32| SearchParams {
        query_mass: Some(OrderedFloat(query_mass)),
        ..Default::default()
    };

    for valid in [f32::MIN_POSITIVE, 0.1, 0.5, 0.999, 1.0] {
        assert!(params(valid).validate().is_ok(), "{valid} must be accepted");
    }
    for invalid in [
        0.0,
        -0.0,
        -0.5,
        1.0001,
        2.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ] {
        assert!(
            params(invalid).validate().is_err(),
            "{invalid} must be rejected"
        );
    }
    assert!(SearchParams::default().validate().is_ok());
}

/// The REST name of the parameter is `query_mass`, and its error message uses it too.
#[test]
fn query_mass_rest_name() {
    let params: SearchParams = serde_json::from_str(r#"{"query_mass": 0.5}"#).unwrap();
    assert_eq!(params.query_mass, Some(OrderedFloat(0.5)));
    assert_eq!(serde_json::to_value(&params).unwrap()["query_mass"], 0.5);
    assert!(params.validate().is_ok());

    let invalid: SearchParams = serde_json::from_str(r#"{"query_mass": 0.0}"#).unwrap();
    let message = invalid.validate().unwrap_err().to_string();
    assert!(message.contains("query_mass"), "{message}");
}
